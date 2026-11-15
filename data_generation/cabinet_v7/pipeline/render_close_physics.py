"""Native close-view rendering of mechanically solved continuous member wires.
All layout axes come from the supplied solver result. No post-solve spline/lift.
"""
import sys, importlib.util, argparse, json, hashlib, math, shutil
from pathlib import Path
sys.dont_write_bytecode=True
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cabinet_geometry
import numpy as np
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
LIB=Path(__file__).with_name('surface_library_v4.py')
spec=importlib.util.spec_from_file_location('surface_lib',LIB);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
B=R.B
SUPPORT_PATHS = []

def clear_segment(a,b,radius,bvh=None,margin=.0004):
 bvh=SUPPORT_BVH if bvh is None else bvh;a=np.asarray(a,float);b=np.asarray(b,float);length=float(np.linalg.norm(b-a));mid=(a+b)/2;distance=bvh.find_nearest(Vector(mid))[3]
 # Distance to a surface is 1-Lipschitz. This certifies the complete swept
 # capsule interval rather than relying on sparse point samples.
 if distance<=radius+margin:return False
 if distance>radius+margin+length/2:return True
 if length<.00020:return False
 return clear_segment(a,mid,radius,bvh,margin) and clear_segment(mid,b,radius,bvh,margin)

def load_subset(source, retained, device):
 """Build the cabinet around the original solved three-bundle scene."""
 read = lambda name: json.loads((source / name).read_text())
 bpy.ops.wm.open_mainfile(filepath=str(source / 'scene.blend'))
 R.PARTS = read('part_map.json')
 R.TOPO = read('topology.json')
 with np.load(source / 'centerline_gt.npz') as axes:
  R.AXES = {key: axes[key] for key in axes.files}
 R.FEATURES = read('feature_anchors.json')
 global SUPPORT_PATHS
 SUPPORT_PATHS = read('support_routes.json')
 import types
 state = types.SimpleNamespace(R=R, B=B, clear_segment=clear_segment, SUPPORT_PATHS=SUPPORT_PATHS)
 revision = cabinet_geometry.build_cabinet(state)
 SUPPORT_PATHS = state.SUPPORT_PATHS
 for part_id, metadata in list(R.PARTS.items()):
  owner = metadata['assembly_id'] or metadata.get('attached_assembly_id', 0)
  if owner and owner not in retained:
   bpy.data.objects.remove(bpy.data.objects[metadata['object_name']], do_unlink=True)
   del R.PARTS[part_id]
 R.TOPO = [entry for entry in R.TOPO if entry['assembly_id'] in retained]
 wire_ids = {wid for entry in R.TOPO for wid in entry['wire_ids']}
 R.AXES = {key: value for key, value in R.AXES.items()
           if int(key.split('_')[1]) in (retained if key.startswith('assembly_') else wire_ids)}
 R.FEATURES = [entry for entry in R.FEATURES if entry['assembly_id'] in retained]
 SUPPORT_PATHS = [entry for entry in SUPPORT_PATHS if entry['owner_assembly_id'] in retained]
 revision['clamps'] = [item for item in revision['clamps'] if item['assembly_id'] in retained]
 sc = bpy.context.scene
 outs = [node for node in sc.node_tree.nodes if node.type == 'OUTPUT_FILE']
 sc.cycles.device = device
 if device == 'GPU':
  preferences = bpy.context.preferences.addons['cycles'].preferences
  preferences.compute_device_type = 'CUDA'
  preferences.get_devices()
  for item in preferences.devices:
   item.use = item.type == 'CUDA'
  if not any(item.use for item in preferences.devices):
   raise RuntimeError('No CUDA device available; run with --device CPU')
 bpy.context.view_layer.update()
 return sc, sc.camera, outs, revision

def render(args):
 contract=json.loads(Path(args.contract).read_text());receipt=json.loads(Path(args.solution_receipt).read_text())
 assert receipt.get('status')=='passed_static_equilibrium_and_shape_gate_with_scope','refuse unconverged mechanical result'
 assert receipt.get('engineering_screen',{}).get('shape_gate_passed') is True,'refuse non-admitted mechanical shape'
 assert receipt.get('output_sha256')==R.sha(args.solution),'mechanical output SHA mismatch'
 assert receipt.get('contract_sha256')==R.sha(args.contract),'mechanical contract SHA mismatch'
 assert receipt.get('min_surface_clearance_m',-1)>=0,'refuse penetrated member rods'
 retained = set(args.keep_assemblies)
 source = Path(args.source_scene)
 sc,cam,outs,revision = load_subset(source, retained, args.device)
 active_assemblies = [asm for asm in contract['assemblies'] if asm['assembly_id'] in retained]
 assert {asm['assembly_id'] for asm in active_assemblies} == retained
 sd=Path(args.output)/args.scene_id;sd.mkdir(parents=True,exist_ok=True)
 if (sd/'COMPLETE.json').exists():raise RuntimeError('Refuse overwrite completed data')
 snap=sd/'source_snapshot';snap.mkdir(exist_ok=True)
 for source_file in [Path(__file__),LIB,R.OLD,Path(cabinet_geometry.__file__)]:shutil.copy2(source_file,snap/source_file.name)
 assert (sc.render.resolution_x, sc.render.resolution_y, sc.cycles.samples) == (args.size, args.size, args.samples)
 bvh,surfaceids=R.surface_bvh();np.savez_compressed(sd/'centerline_gt.npz',**R.AXES);R.dump(sd/'topology.json',R.TOPO);R.dump(sd/'feature_anchors.json',R.FEATURES);R.dump(sd/'support_routes.json',SUPPORT_PATHS);R.dump(sd/'cabinet_revision.json',revision)
 for ob in bpy.context.scene.objects:
  part=R.PARTS.get(str(ob.pass_index))
  if part:part['world_matrix']=np.asarray(ob.matrix_world).tolist();part['bounds_world_m']=[list(ob.matrix_world@Vector(v)) for v in ob.bound_box]
 R.dump(sd/'part_map.json',R.PARTS);R.dump(sd/'object_meta.json',list(R.PARTS.values()));shutil.copy2(args.contract,sd/'mechanical_input.json');shutil.copy2(args.solution_receipt,sd/'mechanical_receipt.json')
 meta={'scene_id':args.scene_id,'prototype':args.prototype,'units':'m','is_fixed_scene':True,'bundle_count':len(active_assemblies),'wire_count':sum(len(asm['wire_ids']) for asm in active_assemblies),'retained_assembly_ids':sorted(retained),'source_scene':str(source.resolve()),'source_blend_sha256':R.sha(source/'scene.blend'),'source_policy':'Inherited solved harness geometry, camera and original endpoint fittings. Replaced bridge clamps with cushioned circular fasteners and added a three-dimensional control cabinet.','semantic_classes':R.SEM,'target_semantics':['cloth_cover','braid_cover','tape_cover','wire_pvc','heatshrink'],'cabinet_geometry_sha256':R.sha(Path(cabinet_geometry.__file__)),'feature_classes':['round_cushioned_cable_clamp','industrial_cabinet_component','cloth_folds','woven_surface','spiral_overlap','opening_exposed_wires','fixed_clip_and_tie','connector_label_heatshrink'],'mechanical_solution_source':str(Path(args.solution).resolve()),'mechanical_solution_sha256':R.sha(args.solution),'mechanical_input_sha256':R.sha(args.contract),'mechanical_receipt_sha256':R.sha(args.solution_receipt),'axis_definition':'strict solved member polylines; covers follow member arithmetic means at corresponding material nodes; no postsolve spline or visual lift','radius_definition':'wire radii are true member radius; assembly radius arrays are passive sleeve nominal base profiles only where radius_valid is true, excluding shell modulation/seams; actual surface geometry is authoritative','hardware_support_definition':'New circular clamp standoffs were checked against all three original target assemblies with conservative swept clearance. Original connector supports are retained.','camera_definition':'native %d-pixel close-local frames, fixed50mm/36mm horizontal sensor, no digital crop, synthetic calibration only'%args.size,'renderer_sha256':R.sha(__file__),'surface_library_sha256':R.sha(LIB),'helper_sha256':R.sha(R.OLD),'branch_valid_definition':'exclude junction>0; branch0 at junction means unassigned, not background','depth_noise_definition':'illustrative uncalibrated Gaussian and missingness, not measured sensor'};R.dump(sd/'scene.json',meta)
 results=[];rng=np.random.default_rng(20261114)
 for vi in range(args.views):
  vd=sd/f'view_{vi:03d}'
  vd.mkdir(exist_ok=True)
  source_camera=json.loads((source/vd.name/'camera.json').read_text())
  distance=source_camera['nominal_cable_distance_m']
  yaw=source_camera['azimuth_deg']
  pitch=source_camera['pitch_deg']
  target=np.array(source_camera['observation_target_world_m'])
  cam.matrix_world=Matrix((np.array(source_camera['T_world_from_camera'])@np.diag([1.,-1.,-1.,1.])).tolist())
  bpy.context.view_layer.update()
  for node in outs:node.base_path=str(vd/'raw_passes')
  sc.render.filepath=str(vd/'rgb.png');sc.frame_set(1)
  if vi==0:bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(sd/'scene.blend'))
  bpy.ops.render.render(write_still=True);raw=B.read_exr(vd/'raw_passes/depth_raw_0001.exr');parts=np.rint(B.read_exr(vd/'raw_passes/instance_0001.exr')).astype(np.uint16)
  fx=args.size*50/36;K=np.array([[fx,0,args.size/2],[0,fx,args.size/2],[0,0,1.]]);T=np.array(cam.matrix_world)@np.diag([1.,-1.,-1.,1.]);kind,diagnostic,_=B.ray_diagnostic(sc,cam,raw,parts,K,T);maps={}
  for filename,key in [('assembly','assembly_id'),('branch','branch_id'),('wire','wire_id'),('junction','junction_id'),('semantic','semantic_id')]:
   assert set(np.unique(parts)).issubset({0,*map(int,R.PARTS)}), 'Unexpected rendered part ID'
   lut=np.zeros(max(map(int,R.PARTS))+1,np.uint16)
   for k,v in R.PARTS.items():lut[int(k)]=v[key]
   maps[filename]=lut[parts];np.save(vd/(filename+'.npy'),maps[filename])
  np.save(vd/'part_id.npy',parts);np.save(vd/'instances.npy',maps['assembly']);yy,xx=np.mgrid[:args.size,:args.size];norm=np.sqrt(1+((xx+.5-K[0,2])/fx)**2+((yy+.5-K[1,2])/fx)**2);valid=np.isfinite(raw)&(raw>0)&(raw<10);z=np.where(valid,raw if kind=='z' else raw/norm,0).astype(np.float32);dv=valid&(rng.random(z.shape)>.025);edge=np.zeros(z.shape,bool);edge[1:]|=abs(np.diff(z,axis=0))>.012;edge[:,1:]|=abs(np.diff(z,axis=1))>.012;dv&=~(edge&(rng.random(z.shape)<.55));dep=np.where(dv,np.maximum(.01,z+rng.normal(0,.0005+.0007*z*z,z.shape)),0).astype(np.float32)
  for name,arr in [('depth_z_gt',z),('depth_range_gt',(z*norm).astype(np.float32)),('depth_input',dep),('depth_valid',dv),('annotation_valid',np.ones_like(dv)),('branch_annotation_valid',maps['junction']==0)]:np.save(vd/(name+'.npy'),arr)
  hit=bvh.find_nearest(cam.location);visible=np.unique(maps['assembly']);visible=visible[visible>0].tolist();border=np.unique(np.r_[maps['assembly'][0],maps['assembly'][-1],maps['assembly'][:,0],maps['assembly'][:,-1]]);border=border[border>0].tolist();pixels={str(i):int((maps['assembly']==i).sum()) for i in visible}
  roi_lo=int(.15*args.size);roi_hi=int(.85*args.size);center_ids=maps['assembly'][roi_lo:roi_hi,roi_lo:roi_hi];center_counts={str(i):int((center_ids==i).sum()) for i in visible};diam=[]
  for asm in active_assemblies:
   for k in [24,40,48]:
    pp=R.AXES['assembly_%d_trunk'%asm['assembly_id']][k];pc=(np.linalg.inv(T)@np.r_[pp,1])[:3];uv=K@pc;uv=uv[:2]/uv[2];diam.append({'assembly_id':asm['assembly_id'],'axis_node':k,'pixel_xy':uv.tolist(),'camera_z_m':float(pc[2]),'nominal_diameter_px':float(fx*2*R.AXES['assembly_%d_trunk_radius'%asm['assembly_id']][k]/pc[2]),'nominal_diameter_image_fraction':float(fx*2*R.AXES['assembly_%d_trunk_radius'%asm['assembly_id']][k]/pc[2]/args.size),'in_image':bool(np.all(uv>=0) and np.all(uv<args.size)),'definition':'front-parallel nominal diameter approximation; actual visible diameter QA independent'})
  camera={'K':K.tolist(),'T_world_from_camera':T.tolist(),'T_camera_from_world':np.linalg.inv(T).tolist(),'image_wh':[args.size,args.size],'pixel_coordinates':'top-left; centers index+0.5','units':'m','depth_convention':'z','raw_blender_depth_convention':kind,'raycast_diagnostic':diagnostic,'role':'close_local','no_digital_crop':True,'lens_mm':50,'sensor_width_mm':36,'nominal_cable_distance_m':distance,'actual_nearest_cable_surface_distance_m':float(hit[3]),'nearest_surface_point_world_m':list(hit[0]),'nearest_assembly_id':surfaceids[hit[2]],'visible_assembly_ids':visible,'visible_assembly_pixels':pixels,'central_inspection_window_xyxy':[roi_lo,roi_lo,roi_hi,roi_hi],'central_inspection_assembly_pixels':center_counts,'central_inspection_junction_pixels':int((maps['junction'][roi_lo:roi_hi,roi_lo:roi_hi]>0).sum()),'border_touching_assembly_ids':border,'azimuth_deg':yaw,'pitch_deg':pitch,'observation_target_world_m':target.tolist(),'projected_diameter_samples':diam};R.dump(vd/'camera.json',camera);R.dump(vd/'part_visibility.json',{str(k):int((parts==k).sum()) for k in np.unique(parts)});results.append({'view_id':vi,'role':'close_local','visible_assembly_ids':visible,'visible_assembly_pixels':pixels,'actual_distance_m':float(hit[3]),'border_touching_assembly_ids':border});print('CLOSE_VIEW_READY '+str(vd),flush=True)
 R.dump(sd/'COMPLETE.json',{'scene_id':args.scene_id,'complete':True,'bundle_count':len(active_assemblies),'views':results,'renderer_sha256':R.sha(__file__),'config':{'size':args.size,'samples':args.samples,'views':len(results)}})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-scene',required=True);ap.add_argument('--keep-assemblies',type=int,nargs='+',required=True);ap.add_argument('--contract',required=True);ap.add_argument('--solution',required=True);ap.add_argument('--solution-receipt',required=True);ap.add_argument('--output',required=True);ap.add_argument('--scene-id',default='prototype_medium3');ap.add_argument('--prototype',action='store_true');ap.add_argument('--device',choices=['CPU','GPU'],default='CPU');ap.add_argument('--size',type=int,default=1024);ap.add_argument('--samples',type=int,default=128);ap.add_argument('--views',type=int,default=1);args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);render(args)
if __name__=='__main__':main()
