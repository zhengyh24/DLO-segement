"""Native close-view rendering of mechanically solved continuous member wires.
All layout axes come from the supplied solver result. No post-solve spline/lift.
"""
import sys, importlib.util, argparse, json, hashlib, math, shutil
from pathlib import Path
sys.dont_write_bytecode=True
import bpy
import numpy as np
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
LIB=Path(__file__).with_name('surface_library_v4.py')
spec=importlib.util.spec_from_file_location('surface_lib',LIB);R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
B=R.B
def tube(name,p,r,ma,sem,asm=0,branch=0,wire=0,lo=0,hi=None,junction=False,shell=False,style=None,tags=None,wall=.00065,axis_key=None,members=None,physical_branch_role=None):
 p=np.asarray(p);hi=len(p)-1 if hi is None else hi;n,b=R.frame(p);sides=24;theta=np.arange(sides)*2*np.pi/sides;arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];rbase=np.broadcast_to(np.asarray(r,dtype=float),(len(p),));rad=np.repeat(rbase[:,None],sides,axis=1)
 if style=='cloth':
  fold_weight=.20+.80*np.exp(-((arc-arc[len(p)//2])/.020)**2)
  rad+=fold_weight[:,None]*(.00010*np.sin(arc[:,None]*410+2*theta)+.00006*np.cos(arc[:,None]*173-5*theta))
 if style=='braid':rad+=.00017*np.sin(arc[:,None]*830+8*theta)*np.cos(arc[:,None]*830-8*theta)
 if style=='tape':rad+=.0003*(.5+.5*np.cos(2*np.pi*arc[:,None]/.018-theta))**10
 holes=set()
 if shell and style=='cloth' and name.startswith('main cloth') and lo==21:
  for kk in range(23,29):
   half=1 if kk in [23,28] else (2 if kk%3 else 3)
   for jj in range(18-half,18+half):holes.add((kk,jj%sides))
  tags=list(tags or [])+['local_torn_opening'];R.FEATURES.append({'assembly_id':asm,'feature':'local_torn_opening','point_world_m':p[24].tolist()})
 dirs=n[:,None,:]*np.cos(theta)[None,:,None]+b[:,None,:]*np.sin(theta)[None,:,None];verts=p[:,None,:]+rad[:,:,None]*dirs;faces=[]
 for k in range(lo,hi):
  for j in range(sides):
   if (k,j) not in holes:faces.append((k*sides+j,k*sides+(j+1)%sides,(k+1)*sides+(j+1)%sides,(k+1)*sides+j))
 if shell:
  inn=p[:,None,:]+(rad-wall)[:,:,None]*dirs;off=len(p)*sides;vv=np.concatenate([verts.reshape(-1,3),inn.reshape(-1,3)])
  for k in range(lo,hi):
   for j in range(sides):
    if (k,j) not in holes:faces.append((off+k*sides+j,off+(k+1)*sides+j,off+(k+1)*sides+(j+1)%sides,off+k*sides+(j+1)%sides))
  edges={}
  for k,j in holes:
   quad=[k*sides+j,k*sides+(j+1)%sides,(k+1)*sides+(j+1)%sides,(k+1)*sides+j]
   for a,c in zip(quad,quad[1:]+quad[:1]):
    key=tuple(sorted((a,c)));edges[key]=(edges.get(key,(0,a,c))[0]+1,a,c)
  for key,(times,a,c) in edges.items():
   if times==1:faces.append((a,c,off+c,off+a))
  for k in [lo,hi]:
   for j in range(sides):
    face=(k*sides+j,k*sides+(j+1)%sides,off+k*sides+(j+1)%sides,off+k*sides+j)
    faces.append(tuple(reversed(face)) if k==lo else face)
 else:
  vv=verts.reshape(-1,3)
  if lo==0:faces.append(tuple(reversed(range(sides))))
  if hi==len(p)-1:faces.append(tuple(hi*sides+j for j in range(sides)))
 used=sorted(set(i for f in faces for i in f));remap={old:new for new,old in enumerate(used)};vv=vv[used];faces=[tuple(remap[i] for i in f) for f in faces]
 me=bpy.data.meshes.new(name);me.from_pydata(vv.tolist(),[],faces);me.update();o=bpy.data.objects.new(name,me);bpy.context.collection.objects.link(o);o.data.materials.append(ma)
 for f in me.polygons:f.use_smooth=True
 R.reg(o,sem,asm,branch,wire,junction,tags)
 R.PARTS[str(o.pass_index)].update(source_axis_key=axis_key or ('wire_'+str(wire) if wire else None),member_wire_ids=members or ([wire] if wire else []),physical_branch_role=physical_branch_role,shell_thickness_m=wall if shell else 0.,radius_base_profile_m=rbase[lo:hi+1].tolist(),centerline_node_interval=[lo,hi],surface_modulation_bound_m=.00016 if style=='cloth' else (.00017 if style=='braid' else (.0003 if style=='tape' else 0.)))
 return o

SUPPORT_BVH=None
SUPPORT_PATHS=[]
FOREIGN_CACHE=(None,None)
def foreign_target_bvh(owner):
 global FOREIGN_CACHE
 if FOREIGN_CACHE[0]==owner:return FOREIGN_CACHE[1]
 dg=bpy.context.evaluated_depsgraph_get();verts=[];faces=[]
 for ob in bpy.context.scene.objects:
  part=R.PARTS.get(str(ob.pass_index),{})
  if not part.get('target') or part.get('assembly_id')==owner or ob.type!='MESH':continue
  ev=ob.evaluated_get(dg);me=ev.to_mesh();me.calc_loop_triangles();off=len(verts);verts.extend([ob.matrix_world@v.co for v in me.vertices]);faces.extend([tuple(off+int(k) for k in t.vertices) for t in me.loop_triangles]);ev.to_mesh_clear()
 result=BVHTree.FromPolygons(verts,faces,all_triangles=True);FOREIGN_CACHE=(owner,result);return result

def clear_segment(a,b,radius,bvh=None,margin=.0004):
 bvh=SUPPORT_BVH if bvh is None else bvh;a=np.asarray(a,float);b=np.asarray(b,float);length=float(np.linalg.norm(b-a));mid=(a+b)/2;distance=bvh.find_nearest(Vector(mid))[3]
 # Distance to a surface is 1-Lipschitz. This certifies the complete swept
 # capsule interval rather than relying on sparse point samples.
 if distance<=radius+margin:return False
 if distance>radius+margin+length/2:return True
 if length<.00020:return False
 return clear_segment(a,mid,radius,bvh,margin) and clear_segment(mid,b,radius,bvh,margin)

def choose_support_route(tip,radius,foot_sphere_radius=0):
 tip=np.asarray(tip,float);candidates=[tip[:2].copy()]
 # Prefer real free board-side corridors, then progressively smaller local detours.
 for x in [-.195,.195,-.18,.18]:
  for y in [tip[1],-.18,-.12,-.06,0,.06,.12,.18]:candidates.append(np.array([x,y]))
 for rad in [.015,.030,.050,.075,.10]:
  for ang in np.linspace(0,2*np.pi,24,endpoint=False):candidates.append(tip[:2]+rad*np.array([np.cos(ang),np.sin(ang)]))
 for xy in candidates:
  if abs(xy[0])>.205 or abs(xy[1])>.19:continue
  elbow=np.r_[xy,tip[2]];foot=np.r_[xy,.006]
  if foot_sphere_radius and SUPPORT_BVH.find_nearest(Vector((xy[0],xy[1],.018)))[3]<=foot_sphere_radius+.0005:continue
  if clear_segment(tip,elbow,radius) and clear_segment(elbow,foot,radius):return [tip.tolist(),elbow.tolist(),foot.tolist()]
 raise RuntimeError('No target-free support path to real panel for tip '+str(tip.tolist()))

def make_support(name,tip,radius,asm,sem,blue_foot=False):
 route=choose_support_route(tip,radius,.024 if blue_foot else .008)
 for j,(a,b) in enumerate(zip(route[:-1],route[1:])):
  if np.linalg.norm(np.asarray(a)-b)>.0001:R.cyl(name+' segment'+str(j),a,b,radius,B.MATS['steel'],sem,attached=asm,tags=['physical_support','all_target_clear_route'])
 foot=np.asarray(route[-1]);R.box(name+' mounting foot',(foot[0],foot[1],.006),(.014,.014,.004),B.MATS['gold'],sem=sem,attached=asm,tags=['panel_attachment'])
 if blue_foot:R.box('blue support base on panel',(foot[0],foot[1],.018),(.032,.027,.022),B.MATS['blue'],sem='connector',attached=asm,tags=['connector_support','panel_mounted_blue_block'])
 SUPPORT_PATHS.append({'owner_assembly_id':asm,'name':name,'axis_polyline_world_m':route,'radius_m':radius,'target_clearance_certified_m':.0004,'method':'adaptive BVH surface distance Lipschitz bound for whole swept cylinder capsule; both lateral and vertical legs checked','blue_base_on_panel':blue_foot})
 return route

def add_port(endpoint,tangent,asm,label):
 ma=B.MATS;p=np.asarray(endpoint);d=np.asarray(tangent);d/=np.linalg.norm(d);c=p+d*.012;foreign=foreign_target_bvh(asm);connector_radius=.0085 if label=='root' else .006
 if not clear_segment(p,c,connector_radius,foreign,margin=.0002):raise RuntimeError('Connector body intersects foreign target: '+str((asm,label)))
 R.cyl('installed endpoint fitting '+label,p,c,connector_radius,ma['steel'],'connector',attached=asm,tags=['connector','endpoint_mount'])
 center=c-np.array([0,0,.014]);box_radius=float(np.linalg.norm([.016,.0135,.010]))
 if foreign.find_nearest(Vector(center))[3]>box_radius+.0005:
  R.box('blue interface module '+label,center,(.032,.027,.020),ma['blue'],sem='connector',attached=asm,tags=['connector','endpoint_mount'])
  make_support('connector support bracket '+label,c-np.array([0,0,.023]),.0035,asm,'connector')
  R.box('endpoint identification label',(c[0],c[1],c[2]-.0035),(.018,.007,.001),ma['paper'],sem='label',attached=asm,tags=['printed_label'])
 else:
  make_support('connector remote mounting bracket '+label,c,.0035,asm,'connector',blue_foot=True)

def segment_distance_rows(a0,a1,b0,b1):
 u=a1-a0;v=b1-b0;w=a0-b0;aa=float(u@u);bb=v@u;cc=np.einsum('ij,ij->i',v,v);dd=w@u;ee=np.einsum('ij,ij->i',v,w);den=aa*cc-bb*bb
 ss=np.clip(np.divide(bb*ee-cc*dd,den,out=np.zeros_like(den),where=den>1e-18),0,1);tt=(bb*ss+ee)/cc
 low=tt<0;high=tt>1;ss[low]=np.clip(-dd[low]/aa,0,1);ss[high]=np.clip((bb[high]-dd[high])/aa,0,1);tt=np.clip(tt,0,1)
 return np.linalg.norm(w+ss[:,None]*u-tt[:,None]*v,axis=1)

def passive_branch_start(legs,envs,radii,members):
 # Conservative capsules bound every swept shell/boot; pick an exit only after
 # the two physical branches have enough space for both actual outer surfaces.
 cross=np.stack([segment_distance_rows(a,b,legs[1][:-1],legs[1][1:]) for a,b in zip(legs[0][:-1],legs[0][1:])])
 otherdist=[]
 for li in range(2):
  others=members[2:] if li==0 else members[:2];b0=np.concatenate([q[:-1] for q in others]);b1=np.concatenate([q[1:] for q in others]);otherdist.append(np.stack([segment_distance_rows(a,b,b0,b1) for a,b in zip(legs[li][:-1],legs[li][1:])]))
 margin=.00015
 for start in range(14,49):
  bounds=[]
  for env,radius in zip(envs,radii):
   rad=env+.0006;rad=rad.copy();rad[start+2:56]=radius[start+2:56]+.00016;bounds.append(np.maximum(rad[:-1],rad[1:]))
  shell_gap=float(np.min(cross[start:,start:]-bounds[0][start:,None]-bounds[1][None,start:]))
  wire_gap=min(float(np.min(otherdist[li][start:]-bounds[li][start:,None]-.00135)) for li in range(2))
  if shell_gap>=margin and wire_gap>=margin:return start,{'branch_leg_start_node':start,'global_wire_start_node':start+60,'cloth_global_start_node':start+62,'minimum_sibling_shell_capsule_gap_m':shell_gap,'minimum_other_branch_wire_capsule_gap_m':wire_gap,'required_conservative_gap_m':margin,'policy':'first candidate after global74 satisfying actual solved-branch shell-shell and sibling-wire capsule separation; axis unchanged'}
 raise RuntimeError('No nonintersecting passive branch sleeve start before leg48')

def clamp_on_curve(p,asm,r,top_world_z):
 ma=B.MATS;k=27;n,_=R.frame(p);q=p[k];nn=n[k];rr=float(np.max(np.asarray(r)[26:29]))
 for sign in [-1,1]:
  side=q+sign*(rr+.004)*nn
  make_support('panel fixed clamp bracket',(side[0],side[1],top_world_z+.002),.0022,asm,'clip_tie')
 strap=R.box('metal bridge clamp',(q[0],q[1],top_world_z+.0012),(2*rr+.012,.007,.0025),ma['steel'],sem='clip_tie',attached=asm,tags=['fixed_clip']);strap.rotation_euler.z=math.atan2(nn[1],nn[0])
 ob=tube('colored tie',p,np.asarray(r)+.00035,ma['tie_yellow' if asm%2==0 else 'tie_red'],'clip_tie',lo=42,hi=44,shell=True,tags=['zip_tie'],wall=.0003);R.PARTS[str(ob.pass_index)]['attached_assembly_id']=asm
 R.FEATURES.append({'assembly_id':asm,'feature':'fixed_clip_and_tie','point_world_m':p[43].tolist()})

def build(contract,solution):
 global SUPPORT_BVH,SUPPORT_PATHS,FOREIGN_CACHE
 SUPPORT_PATHS=[];FOREIGN_CACHE=(None,None);ports=[];clamps=[]
 bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False);R.PARTS={};R.AXES={};R.TOPO=[];R.FEATURES=[];R.initial_materials();ma=B.MATS
 # The environment is built around actual fixed ports, not photobacked scenery.
 R.box('cream equipment mounting panel',(0,0,-.014),(.46,.43,.036),ma['panel'],bevel=.004)
 for x in [-.19,.19]:
  R.box('aluminum stiffener',(x,0,.012),(.016,.41,.016),ma['steel'])
 for x in np.arange(-.20,.21,.025):
  for y in np.arange(-.18,.19,.025):R.cyl('black panel fastener recess',(float(x),float(y),.004),(float(x),float(y),.005),.0013,ma['ink'])
 for i,c in enumerate([(.55,.54,.48),(.20,.014,.010),(.014,.017,.020),(.018,.050,.11)]):
  m=ma['wire'+str(i)];bs=m.node_tree.nodes.get('Principled BSDF');
  for link in list(bs.inputs['Base Color'].links):m.node_tree.links.remove(link)
  bs.inputs['Base Color'].default_value=(*c,1);bs.inputs['Roughness'].default_value=.38
  tc=m.node_tree.nodes.new('ShaderNodeTexCoord');no=m.node_tree.nodes.new('ShaderNodeTexNoise');no.inputs['Scale'].default_value=2600;m.node_tree.links.new(tc.outputs['Object'],no.inputs['Vector']);bu=m.node_tree.nodes.new('ShaderNodeBump');bu.inputs['Strength'].default_value=.10;bu.inputs['Distance'].default_value=.000015;m.node_tree.links.new(no.outputs['Fac'],bu.inputs['Height']);m.node_tree.links.new(bu.outputs[0],bs.inputs['Normal'])
 for asm in contract['assemblies']:
  ai=asm['assembly_id'];ids=asm['wire_ids'];ws=[np.asarray(solution['wire_'+str(wid)],float) for wid in ids]
  assert all(p.shape==(121,3) and np.isfinite(p).all() for p in ws),'solver member-node contract mismatch'
  trunk=np.mean(np.stack([p[:61] for p in ws]),axis=0);left=np.mean(np.stack([p[60:] for p in ws[:2]]),axis=0);right=np.mean(np.stack([p[60:] for p in ws[2:]]),axis=0);style=asm['style'];tr=ai*10+1;lb=ai*10+2;rb=ai*10+3
  env=np.max(np.linalg.norm(np.stack([w[:61] for w in ws])-trunk[None],axis=2),axis=0)+.00135;neg={'cloth':.00016,'braid':.00017,'tape':0}[style];rp=env+.00065+neg+.00012;leg_env=[np.max(np.linalg.norm(np.stack([w[60:] for w in pair])-axis[None],axis=2),axis=0)+.00135 for pair,axis in [(ws[:2],left),(ws[2:],right)]];leg_r=[ee+.00065+.00016+.00010 for ee in leg_env];branch_start,start_receipt=passive_branch_start([left,right],leg_env,leg_r,ws)
  R.AXES['assembly_%d_trunk'%ai]=trunk;R.AXES['assembly_%d_left'%ai]=left;R.AXES['assembly_%d_right'%ai]=right;R.AXES['assembly_%d_trunk_radius'%ai]=rp;R.AXES['assembly_%d_branch_radius'%ai]=np.stack(leg_r);R.AXES['assembly_%d_trunk_radius_valid'%ai]=(np.arange(61)>=3)&(np.arange(61)<=55);R.AXES['assembly_%d_branch_radius_valid'%ai]=np.repeat(((np.arange(61)>=branch_start+2)&(np.arange(61)<=55))[None],2,axis=0)
  for wi,(wid,p) in enumerate(zip(ids,ws)):
   R.AXES['wire_'+str(wid)]=p;R.AXES['wire_%d_radius'%wid]=np.array(.00135)
   for lo,hi,br,junc in [(0,50,tr,False),(50,74,0,True),(74,120,lb if wi<2 else rb,False)]:tube('solved PVC wire %d interval%d'%(wid,lo),p,.00135,ma['wire'+str(wi)],'wire_pvc',ai,br,wid,lo,hi,junc,tags=['exposed_wire','opening_exposed_wires','mechanically_solved_member'])
  # The passive sleeve follows solved-member envelope; it is not a separate shell mechanics solve.
  surface_ranges=[(3,21,tr,False),(21,50,tr,False),(50,55,0,True)] if style=='cloth' else [(3,50,tr,False),(50,55,0,True)]
  covers=[]
  for lo,hi,br,junc in surface_ranges:
   covers.append(tube('main %s sleeve assembly%d'%(style,ai),trunk,rp,ma[style],style+'_cover',ai,br,lo=lo,hi=hi,junction=junc,shell=True,style=style,tags=[style+'_wrapping','real_thickness','cloth_folds' if style=='cloth' else ('woven_surface' if style=='braid' else 'spiral_overlap')],axis_key='assembly_%d_trunk'%ai,members=ids,physical_branch_role='trunk'))
  for li,(leg,br) in enumerate([(left,lb),(right,rb)]):
   role='left' if li==0 else 'right';axis_key='assembly_%d_%s'%(ai,role);own_ids=ids[:2] if li==0 else ids[2:]
   tube('branch cloth solved assembly%d'%ai,leg,leg_r[li],ma['cloth'],'cloth_cover',ai,br,lo=branch_start+2,hi=55,shell=True,style='cloth',tags=['cloth_wrapping','real_thickness','cloth_folds'],axis_key=axis_key,members=own_ids,physical_branch_role=role)
   for lo,hi in [(branch_start,branch_start+2),(55,58)]:tube('branch heatshrink',leg,leg_env[li]+.0006,ma['heat'],'heatshrink',ai,br,lo=lo,hi=hi,shell=True,tags=['heatshrink'],wall=.0005,axis_key=axis_key,members=own_ids,physical_branch_role=role)
  # Existing physical seam geometry is densely sampled; it follows solved cover axis only.
  if style in ['braid','tape']:
   for lo,hi in [(3,55)]:
    pts=trunk[lo:hi+1];ar=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))];ss=np.linspace(0,ar[-1],max(30,int(ar[-1]/.0005)+1));pp=np.stack([np.interp(ss,ar,pts[:,k]) for k in range(3)],1);n,b=R.frame(pp)
    for direction in ([1] if style=='tape' else [1,-1]):
     for phase in ([0] if style=='tape' else [0,2.094,4.189]):
      theta=direction*2*np.pi*ss/(.018 if style=='tape' else .022)+phase;rad=np.interp(ss,ar,rp[lo:hi+1])+.00038+(direction<0)*.00040;hp=pp+rad[:,None]*(np.cos(theta)[:,None]*n+np.sin(theta)[:,None]*b)
      cut=int(np.searchsorted(ss,ar[50-lo]));
      for slo,shi,sbr,sj in [(0,cut,tr,False),(cut,len(hp)-1,0,True)]:tube('helical tape seam' if style=='tape' else 'braid surface thread',hp,.00040 if style=='tape' else .00018,ma[style],style+'_cover',ai,sbr,lo=slo,hi=shi,junction=sj,tags=['spiral_overlap' if style=='tape' else 'woven_surface'],axis_key='assembly_%d_trunk'%ai,members=ids,physical_branch_role='trunk')
  topz=max(float(v.co.z) for ob in covers for v in ob.data.vertices if np.linalg.norm(np.array(v.co)[:2]-trunk[27,:2])<.009);clamps.append((trunk,ai,rp+(.00016 if style=='cloth' else .00055),topz))
  label=tube('harness paper identification',trunk,rp+neg+.00020,ma['paper'],'label',lo=38,hi=40,shell=True,tags=['printed_label'],wall=.00015);R.PARTS[str(label.pass_index)]['attached_assembly_id']=ai
  ports.append((trunk[0],trunk[0]-trunk[2],ai,'root'))
  for li,(name,leg,br) in enumerate([('left',left,lb),('right',right,rb)]):
   ports.append((leg[-1],leg[-1]-leg[-3],ai,name));tube('port heatshrink',leg,leg_env[li]+.0006,ma['heat'],'heatshrink',ai,br,lo=58,hi=60,shell=True,tags=['heatshrink'],wall=.0005,axis_key='assembly_%d_%s'%(ai,name),members=ids[:2] if li==0 else ids[2:],physical_branch_role=name)
  tube('root heatshrink',trunk,env+.0006,ma['heat'],'heatshrink',ai,tr,lo=0,hi=3,shell=True,tags=['heatshrink'],wall=.0005,axis_key='assembly_%d_trunk'%ai,members=ids,physical_branch_role='trunk')
  R.FEATURES.extend([{'assembly_id':ai,'feature':style+'_wrapping','point_world_m':trunk[26].tolist()},{'assembly_id':ai,'feature':'opening_exposed_wires','point_world_m':trunk[58].tolist()},{'assembly_id':ai,'feature':'junction','point_world_m':trunk[60].tolist()}])
  asm_export=dict(asm)
  for legacy in ['outer_cover_base_radius_m','branch_outer_cover_base_radius_m','covered_trunk_node_intervals','covered_branch_node_intervals','uncovered_junction_node_interval']:
   if legacy in asm_export:asm_export['initial_planned_'+legacy]=asm_export.pop(legacy)
  R.TOPO.append(dict(asm_export,junction_label_wire_node_interval=[50,74],passive_branch_start_receipt=start_receipt,actual_radius_source='centerline_gt.npz assembly_%d_trunk_radius/branch_radius with radius_valid; part_map shell thickness/modulation'%ai,axis_provenance='member means from mechanical solution, no postsolve artistic smoothing',branch_ids=[tr,lb,rb],junction_id=ai,wire_radius_m=.00135,cover_radius_derived_from_solved_member_envelope=True,rendered_main_covered_node_interval=[3,55],rendered_branch_covered_wire_node_interval=[branch_start+60,120],cover_is_passive_render_geometry_not_independent_shell_dynamics=True,cover_radial_clearance_at_worst_inward_modulation_m=.00012,cloth_radius_modulation_bound_m=.00016))
 bpy.context.view_layer.update();SUPPORT_BVH=R.surface_bvh()[0]
 for task in clamps:clamp_on_curve(*task)
 for task in ports:add_port(*task)
 R.box('real background wall',(0,0,-.08),(4,4,.04),B.mat('wall',(.11,.12,.13),.9),sem='background',bevel=0)



def render(args):
 contract=json.loads(Path(args.contract).read_text());receipt=json.loads(Path(args.solution_receipt).read_text())
 assert receipt.get('status')=='passed_static_equilibrium_and_shape_gate_with_scope','refuse unconverged mechanical result'
 assert receipt.get('engineering_screen',{}).get('shape_gate_passed') is True,'refuse non-admitted mechanical shape'
 assert receipt.get('output_sha256')==R.sha(args.solution),'mechanical output SHA mismatch'
 assert receipt.get('contract_sha256')==R.sha(args.contract),'mechanical contract SHA mismatch'
 assert receipt.get('min_surface_clearance_m',-1)>=0,'refuse penetrated member rods'
 solution=np.load(args.solution);build(contract,solution);sd=Path(args.output)/args.scene_id;sd.mkdir(parents=True,exist_ok=True)
 if (sd/'COMPLETE.json').exists():raise RuntimeError('Refuse overwrite completed data')
 snap=sd/'source_snapshot';snap.mkdir(exist_ok=True)
 for source in [Path(__file__),LIB,R.OLD]:shutil.copy2(source,snap/source.name)
 sc,cam,outs=R.configure(args.size,args.samples,'CPU');cam.data.lens=50;bpy.context.view_layer.update();np.savez_compressed(sd/'centerline_gt.npz',**R.AXES);R.dump(sd/'topology.json',R.TOPO);R.dump(sd/'feature_anchors.json',R.FEATURES);R.dump(sd/'support_routes.json',SUPPORT_PATHS)
 for ob in bpy.context.scene.objects:
  part=R.PARTS.get(str(ob.pass_index))
  if part:part['world_matrix']=np.asarray(ob.matrix_world).tolist();part['bounds_world_m']=[list(ob.matrix_world@Vector(v)) for v in ob.bound_box]
 R.dump(sd/'part_map.json',R.PARTS);R.dump(sd/'object_meta.json',list(R.PARTS.values()));shutil.copy2(args.contract,sd/'mechanical_input.json');shutil.copy2(args.solution_receipt,sd/'mechanical_receipt.json')
 meta={'scene_id':args.scene_id,'prototype':args.prototype,'units':'m','is_fixed_scene':True,'bundle_count':len(contract['assemblies']),'wire_count':len(contract['wires']),'semantic_classes':R.SEM,'target_semantics':['cloth_cover','braid_cover','tape_cover','wire_pvc','heatshrink'],'feature_classes':['cloth_folds','woven_surface','spiral_overlap','opening_exposed_wires','fixed_clip_and_tie','connector_label_heatshrink'],'mechanical_solution_source':str(Path(args.solution).resolve()),'mechanical_solution_sha256':R.sha(args.solution),'mechanical_input_sha256':R.sha(args.contract),'mechanical_receipt_sha256':R.sha(args.solution_receipt),'axis_definition':'strict solved member polylines; covers follow member arithmetic means at corresponding material nodes; no postsolve spline or visual lift','radius_definition':'wire radii are true member radius; assembly radius arrays are passive sleeve nominal base profiles only where radius_valid is true, excluding shell modulation/seams; actual surface geometry is authoritative','hardware_support_definition':'targets completed first, then whole swept support paths checked against all actual target meshes; blocked port boxes moved to safe panel support bases, no target holes or hidden fixtures','camera_definition':'native %d-pixel close-local frames, fixed50mm/36mm horizontal sensor, no digital crop, synthetic calibration only'%args.size,'renderer_sha256':R.sha(__file__),'surface_library_sha256':R.sha(LIB),'helper_sha256':R.sha(R.OLD),'branch_valid_definition':'exclude junction>0; branch0 at junction means unassigned, not background','depth_noise_definition':'illustrative uncalibrated Gaussian and missingness, not measured sensor'};R.dump(sd/'scene.json',meta)
 cam.location = (0, .025, .45)
 cam.rotation_euler = (Vector((0, .025, .075)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
 bpy.context.preferences.filepaths.save_version = 0
 bpy.ops.wm.save_as_mainfile(filepath=str(sd / 'scene.blend'))

def main():
 parser = argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--contract',required=True)
 parser.add_argument('--solution',required=True)
 parser.add_argument('--solution-receipt',required=True)
 parser.add_argument('--output',required=True)
 parser.set_defaults(size=1024,samples=256,scene_id='easy3',prototype=False)
 render(parser.parse_args(sys.argv[sys.argv.index('--')+1:]))

if __name__=='__main__':
 main()
