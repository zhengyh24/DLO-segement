"""Independent final v5 RGB-D, solved-axis provenance and close-framing checks."""
import sys
sys.dont_write_bytecode=True
import argparse,json,struct
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from qa_common import sha,digest,fingerprint,worldmesh
from qa_annotation_layers import audit_layers
from qa_pixel_geometry import audit_view_geometry
from qa_local_frame import frame_evidence
def read(p):return json.loads(Path(p).read_text())
def audit(sd,out,nviews):
 failures=[];warnings=[];meta=read(sd/'scene.json');topo=read(sd/'topology.json');parts=read(sd/'part_map.json');complete=read(sd/'COMPLETE.json');axes=np.load(sd/'centerline_gt.npz')
 solution=Path(meta['mechanical_solution_source']);phys=np.load(solution);mechanical=read(sd/'mechanical_input.json')
 if sha(solution)!=meta['mechanical_solution_sha256']:failures.append('physics_solution_hash_changed')
 if sha(sd/'mechanical_input.json')!=meta['mechanical_input_sha256']:failures.append('mechanical_input_hash_disagrees')
 if sha(sd/'mechanical_receipt.json')!=meta['mechanical_receipt_sha256']:failures.append('mechanical_result_hash_disagrees')
 source=sd/'source_snapshot/render_close_physics.py'
 if not source.exists() or sha(source)!=meta['renderer_sha256'] or meta['renderer_sha256']!=complete['renderer_sha256']:failures.append('renderer_source_snapshot_hash_disagrees')
 aids=sorted(t['assembly_id'] for t in topo)
 if len(aids)!=meta['bundle_count'] or len(set(aids))!=len(aids):failures.append('physical_assembly_count_inconsistent')
 axis_rows=[];wire_by_id={w['wire_id']:w for w in mechanical['wires']}
 for t in topo:
  aid=t['assembly_id'];ws=[]
  for wid in t['wire_ids']:
   actual=axes['wire_'+str(wid)];solved=phys['wire_'+str(wid)];err=float(np.max(abs(actual-solved)));ws.append(solved)
   if actual.shape!=solved.shape or err>1e-12:failures.append({'postsolve_wire_geometry_changed':wid,'max_m':err})
   if abs(float(axes['wire_%d_radius'%wid])-wire_by_id[wid]['radius_m'])>1e-12:failures.append({'wire_radius_changed_from_physical_model':wid})
   axis_rows.append({'wire_id':wid,'max_absolute_position_change_m':err,'nodes':len(actual)})
  expected={'trunk':np.mean([p[:61] for p in ws],axis=0),'left':np.mean([p[60:] for p in ws[:2]],axis=0),'right':np.mean([p[60:] for p in ws[2:]],axis=0)}
  for role,p in expected.items():
   if not np.allclose(axes['assembly_%d_%s'%(aid,role)],p,atol=1e-12,rtol=0):failures.append({'cover_mean_axis_not_derived_from_solved_members':[aid,role]})
 canonical={}
 for pid,m in parts.items():
  canonical[int(pid)]={k:int(m[k+'_id']) for k in ['assembly','branch','wire','junction']}
  if bool(m['target'])!=(m['assembly_id']>0):failures.append({'target_owner_inconsistent':pid})
  if not m['target'] and any(m[k+'_id'] for k in ['assembly','branch','wire','junction']):failures.append({'aux_has_target_labels':pid})
  if m['semantic_name']!='wire_pvc' and m['wire_id']>0:failures.append({'cover_or_aux_has_visible_individual_wire_ID':pid})
 bpy.ops.wm.open_mainfile(filepath=str(sd/'scene.blend'));sc=bpy.context.scene;deps=bpy.context.evaluated_depsgraph_get();meshes={}
 for ob in sc.objects:
  if ob.type not in ['MESH','CURVE','FONT','SURFACE','META']:continue
  ev=ob.evaluated_get(deps);me=ev.to_mesh();valid=len(me.vertices)>0 and len(me.polygons)>0;ev.to_mesh_clear()
  if valid:meshes[ob.name]=worldmesh(ob,deps)
 for pid,m in parts.items():
  if m['object_name'] not in meshes:failures.append({'part_has_no_evaluated_mesh':pid})
  elif meshes[m['object_name']]['id']!=int(pid):failures.append({'saved_mesh_part_ID_mismatch':pid})
 targets=[m['object_name'] for m in parts.values() if m['target'] and m['object_name'] in meshes]
 # Conservative radial envelope bounds supplement actual mesh contact tests.
 fit=[];missing_fit=[]
 for pid,m in parts.items():
  if not m['target'] or m.get('shell_thickness_m',0)<=0:continue
  source_axis=m.get('source_axis_key');members=m.get('member_wire_ids')
  if not source_axis or not members:missing_fit.append(int(pid));continue
  lo,hi=m['centerline_node_interval'];center=axes[source_axis][lo:hi+1];offset=0 if source_axis.endswith('_trunk') else 60
  env=np.max(np.stack([np.linalg.norm(axes['wire_'+str(wid)][lo+offset:hi+offset+1]-center,axis=1)+float(axes['wire_%d_radius'%wid]) for wid in members]),axis=0)
  negative={'cloth_cover':m.get('surface_modulation_bound_m',0),'braid_cover':m.get('surface_modulation_bound_m',0),'tape_cover':0,'heatshrink':0}.get(m['semantic_name'],0)
  inner=np.array(m['radius_base_profile_m'])-m['shell_thickness_m']-negative;gap=inner-env
  fit.append({'part_id':int(pid),'object':m['object_name'],'axis':source_axis,'member_wire_ids':members,'minimum_conservative_radial_gap_m':float(gap.min()),'maximum_conservative_radial_gap_m':float(gap.max()),'maximum_member_envelope_radius_m':float(env.max())})
  if gap.min()< -1e-6:failures.append({'own_members_outside_conservative_inner_sleeve_envelope':int(pid),'gap_m':float(gap.min())})
 if missing_fit:warnings.append({'parts_without_explicit_axis_member_fit_metadata':missing_fit,'geometry_contact_tests_still_required':True})
 static=fingerprint();views=[];viewdirs=sorted(v for v in sd.glob('view_*') if (v/'camera.json').exists())
 if len(viewdirs)!=nviews:failures.append({'view_count':len(viewdirs),'expected':nviews})
 for vi,vd in enumerate(viewdirs):
  c=read(vd/'camera.json');names=['part_id','assembly','branch','wire','junction','semantic','depth_z','depth_range','depth_input','depth_valid'];arrays={k:np.load(vd/({'depth_z':'depth_z_gt','depth_range':'depth_range_gt'}.get(k,k)+'.npy')) for k in names}
  al=audit_layers(arrays,canonical,aids,primary=False);h,w=arrays['part_id'].shape;png=(vd/'rgb.png').read_bytes()
  if png[:8]!=b'\x89PNG\r\n\x1a\n' or list(struct.unpack('>II',png[16:24]))!=[w,h]:al['failures'].append('RGB_dimensions_disagree')
  if c['image_wh']!=[w,h] or c.get('units')!='m' or c.get('depth_convention')!='z':al['failures'].append('camera_or_depth_contract_disagrees')
  if not c.get('no_digital_crop') or [sc.render.resolution_x,sc.render.resolution_y]!=[w,h] or sc.render.resolution_percentage!=100:al['failures'].append('not_native_uncropped_render_size')
  if abs(c['K'][0][0]-w*sc.camera.data.lens/sc.camera.data.sensor_width)>1e-5:al['failures'].append('K_does_not_match_saved_native_camera')
  if np.max(abs(np.array(c['T_camera_from_world'])@np.array(c['T_world_from_camera'])-np.eye(4)))>1e-5:al['failures'].append('pose_inverse_disagrees')
  if not np.array_equal(np.load(vd/'instances.npy'),arrays['assembly']):al['failures'].append('instances_alias_not_assembly')
  if not np.array_equal(np.load(vd/'branch_annotation_valid.npy').astype(bool),arrays['junction']==0):al['failures'].append('junction_branch_validity_wrong')
  if not np.load(vd/'annotation_valid.npy').all():al['failures'].append('unexpected_assembly_annotation_invalidity')
  expected_sem=np.zeros((h,w),np.uint16)
  for pid in np.unique(arrays['part_id']):
   if str(pid) in parts:expected_sem[arrays['part_id']==pid]=parts[str(pid)]['semantic_id']
  if not np.array_equal(expected_sem,arrays['semantic']):al['failures'].append('semantic_not_first_visible_part_semantics')
  al['status']='passed' if not al['failures'] else 'failed'
  geom=audit_view_geometry(sc,deps,meshes,targets,c,arrays);evidence=frame_evidence(arrays,c,axes,topo,parts,meshes)
  if set(c['visible_assembly_ids'])!=set(geom['visible_assembly_ids']):geom['failures'].append('visible_ID_metadata_disagrees')
  if set(c['border_touching_assembly_ids'])!=set(geom['border_touching_assembly_ids']):geom['failures'].append('truncation_metadata_disagrees')
  if abs(c['actual_nearest_cable_surface_distance_m']-geom['nearest_target_surface_distance_m'])>.001:geom['failures'].append('independent_surface_distance_disagrees')
  if set(evidence['meaningful_visible_assembly_ids'])!=set(aids):geom['failures'].append('retained_assemblies_not_all_distinguishable')
  if vi==0 and len(aids)>1 and not evidence['verified_crossing_windows']:geom['failures'].append('primary_lacks_verified_local_crossing')
  if vi==0 and not evidence['verified_branch_windows']:geom['failures'].append('primary_lacks_visible_branch_junction_and_two_exits')
  geom['status']='passed' if not geom['failures'] else 'failed'
  for e in evidence['verified_crossing_windows']+evidence['verified_branch_windows']:e['source_rgb']=str(vd/'rgb.png')
  feature_pixels={}
  for pid in np.unique(arrays['part_id']):
   pm=parts.get(str(int(pid)),{});npix=int(np.count_nonzero(arrays['part_id']==pid))
   for tag in pm.get('feature_tags',[]):feature_pixels[tag]=feature_pixels.get(tag,0)+npix
  evidence['visible_feature_tag_pixels_from_first_visible_part']=feature_pixels
  views.append({'view':vd.name,'array_checks':al,'geometry_checks':geom,'local_frame_evidence':evidence,'file_sha256':{p.name:sha(p) for p in sorted(vd.iterdir()) if p.is_file() and p.suffix in ['.npy','.json','.png']}})
  if al['failures'] or geom['failures']:failures.append({'view':vd.name,'array':al['failures'],'geometry':geom['failures']})
 rec={'status':'passed_with_explicit_scope' if not failures else 'failed','scene_id':sd.name,'num_assemblies':len(aids),'num_views':len(views),'source_sha256':meta['renderer_sha256'],'blend_sha256':sha(sd/'scene.blend'),'source_solution_sha256':sha(solution),'scene_metadata_sha256':sha(sd/'scene.json'),'topology_sha256':sha(sd/'topology.json'),'part_map_sha256':sha(sd/'part_map.json'),'centerline_sha256':sha(sd/'centerline_gt.npz'),'script_sha256':sha(__file__),'static_fingerprint':{k:digest(v) for k,v in static.items()},'wire_axis_provenance':axis_rows,'cover_fit_bounds':fit,'views':views,'warnings':warnings,'failures':failures,'scope':['All-pixel stored hierarchy/semantic mapping, branch invalidity and depth array checks.','Sampled complete-scene first-hit ID/depth rays; true nearest target-mesh distance and native image/camera calibration.','Visible trunk mask-width samples and source-pixel crossing/branch evidence windows; truncation allowed.','Rendered wire nodes/radii compared exactly with canonical mechanics output; derived sleeve mean axes checked.'],'limits':['4-8% width is an approximate framing reference; per-assembly actual samples and warnings are retained.','Array consistency is exhaustive for stored layers; 3D ray tests are sampled and do not validate all points.','Conservative nodewise sleeve radial bounds need actual mesh contact checks, including other-branch wires and inter-assembly surfaces.','No measured material, sensor accuracy, manufacture compliance or learned E3 performance claim.']};out.write_text(json.dumps(rec,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()));print(json.dumps({'scene':sd.name,'status':rec['status'],'views':len(views),'failures':failures,'warnings':warnings}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--views',type=int,default=4);p.add_argument('--out',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);audit(a.scene,a.out,a.views)
