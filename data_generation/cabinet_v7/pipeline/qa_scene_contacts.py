"""Scoped independent gross-intersection audit, on saved evaluated meshes.
Triangle intersections are real surface contacts; not every contact is a manufacturing defect.
Only specified same-assembly sleeves/clamps and near-end connectors are reported as allowed contacts.
"""
import argparse,json,sys,hashlib
sys.dont_write_bytecode=True
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mesh(o,deps):
 ev=o.evaluated_get(deps);me=ev.to_mesh();me.calc_loop_triangles()
 v=np.array([ev.matrix_world@x.co for x in me.vertices],float);tri=np.array([list(t.vertices) for t in me.loop_triangles],int);ev.to_mesh_clear()
 if not len(v) or not len(tri):return None
 used=np.unique(tri);actual=v[used];bv=BVHTree.FromPolygons(v.tolist(),tri.tolist(),all_triangles=True)
 ed=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1);counts=np.unique(ed,axis=0,return_counts=True)[1]
 vol=float(np.einsum('ij,ij->i',v[tri[:,0]],np.cross(v[tri[:,1]],v[tri[:,2]])).sum()/6)
 return {'bvh':bv,'v':v,'tri':tri,'lo':actual.min(0),'hi':actual.max(0),'closed':bool((counts==2).all() and abs(vol)>1e-12)}
def overlap(a,b):return np.all(a['hi']>=b['lo']) and np.all(b['hi']>=a['lo'])
def inside_by_parity(b,p):
    odd=[]
    for raw in [(1.,.371,.219),(.173,1.,.417),(.313,.211,1.)]:
        direction=Vector(raw).normalized();origin=Vector(p);hits=0
        for step in range(128):
            q,n,face,d=b['bvh'].ray_cast(origin,direction,100.)
            if q is None:break
            hits+=1;origin=q+direction*1e-6
        odd.append(hits%2==1)
    return all(odd),odd
def audit(sd,out):
 meta=json.loads((sd/'part_map.json').read_text());topo=json.loads((sd/'topology.json').read_text());axes=np.load(sd/'centerline_gt.npz');bpy.ops.wm.open_mainfile(filepath=str(sd/'scene.blend'));deps=bpy.context.evaluated_depsgraph_get()
 infos={};byname={v['object_name']:v for v in meta.values()};failures=[];contacts=[];pairchecks=0
 for o in bpy.context.scene.objects:
  if o.type not in ['MESH','FONT','CURVE']:continue
  m=mesh(o,deps)
  if m:infos[o.name]=m
 targets=[name for name in infos if byname.get(name,{}).get('target')]
 fixtures=[name for name in infos if not byname.get(name,{}).get('target')]
 endpoints={t['assembly_id']:np.array([axes['assembly_%d_trunk'%t['assembly_id']][0],axes['assembly_%d_left'%t['assembly_id']][-1],axes['assembly_%d_right'%t['assembly_id']][-1]]) for t in topo}
 for i,na in enumerate(targets):
  a=infos[na];ma=byname[na];aid=ma['assembly_id']
  for nb in targets[i+1:]:
   mb=byname[nb]
   if not overlap(a,infos[nb]):continue
   if mb['assembly_id']==aid:
    aw=ma.get('semantic_name')=='wire_pvc';bw=mb.get('semantic_name')=='wire_pvc'
    if not aw and not bw and ma.get('source_axis_key') and ma.get('source_axis_key')==mb.get('source_axis_key'):continue # Only declared layers on the same physical axis; separate branch shells are tested.
    if aw and bw and ma['wire_id']==mb['wire_id']:
     ia=ma.get('centerline_node_interval',[0,0]);ib=mb.get('centerline_node_interval',[0,0])
     if ia[1]==ib[0] or ib[1]==ia[0]:continue # Joined render parts; nonlocal self-capsules are checked independently.
   pairchecks+=1;hits=a['bvh'].overlap(infos[nb]['bvh'])
   if hits:failures.append({'issue':'member_or_cross_assembly_surface_intersection','objects':[na,nb],'assemblies':[aid,mb['assembly_id']],'triangle_intersections':len(hits)})
  for nb in fixtures:
   b=infos[nb]
   if not overlap(a,b):continue
   pairchecks+=1;hits=a['bvh'].overlap(b['bvh'])
   if not hits:continue
   mb=byname.get(nb,{});same=mb.get('attached_assembly_id')==aid;sem=mb.get('semantic_name','unknown');allowed=False;reason=None
   if same and sem in ['clip_tie','label'] and ma.get('semantic_name')!='wire_pvc':allowed=True;reason='same-assembly auxiliary clamp/tie/label surface contact; interior-axis test remains active for clamp'
   if same and sem=='connector':
    ti=np.array([x for x,y in hits],int);vv=a['v'][a['tri'][ti]].reshape(-1,3);nearest=np.linalg.norm(vv[:,None]-endpoints[aid][None],axis=2).min(1)
    if float(nearest.max())<.025:allowed=True;reason='same-assembly endpoint connector contact within 25mm of declared endpoint'
   rec={'issue':'target_auxiliary_surface_intersection','target':na,'assembly_id':aid,'fixture':nb,'fixture_owner':mb.get('attached_assembly_id',0),'fixture_semantics':sem,'triangle_intersections':len(hits),'allowed_contact':allowed,'reason':reason}
   (contacts if allowed else failures).append(rec)
 # Finite dense samples along the actual four wire axes detect embedding, including in own clamp.
 for t in topo:
  aid=t['assembly_id']
  for wire in [{'wire_id':i,'path_npz_key':'wire_'+str(i)} for i in t['wire_ids']]:
   ax=axes[wire['path_npz_key']];wid=wire['wire_id'];ax=np.concatenate([ax,*(ax[:-1]*(1-f)+ax[1:]*f for f in [.25,.5,.75])])
   for name in fixtures:
    b=infos[name]
    if not b['closed']:continue
    inside_box=np.where(np.all(ax>=b['lo'],axis=1)&np.all(ax<=b['hi'],axis=1))[0]
    if not len(inside_box):continue
    mb=byname.get(name,{});sem=mb.get('semantic_name');owner=mb.get('attached_assembly_id',0);bad=[]
    for j in inside_box:
     p=ax[j];q,n,face,d=b['bvh'].find_nearest(Vector(p))
     if q is not None and float(d)>.0002 and inside_by_parity(b,p)[0]:
      near=float(np.linalg.norm(endpoints[aid]-p,axis=1).min())
      allowed=(sem=='connector' and owner==aid and near<.025)
      if not allowed:bad.append(float(d))
    if bad:failures.append({'issue':'wire_axis_inside_closed_auxiliary','wire_id':wid,'assembly_id':aid,'fixture':name,'fixture_owner':owner,'sample_count':len(bad),'maximum_sampled_depth_inside_m':max(bad)})
 rec={'status':'passed_with_explicit_scope' if not failures else 'failed','scene_id':sd.name,'blend_sha256':sha(sd/'scene.blend'),'script_sha256':sha(__file__),'target_mesh_parts':len(targets),'non_target_mesh_parts':len(fixtures),'AABB_candidate_pairs_actually_tested_with_BVH':pairchecks,'failures':failures,'declared_scoped_contacts':contacts,'limits':['Only declared COVER-to-COVER layers sharing one physical source_axis_key may overlap/abut; different branch shells remain tested. Different member wires and member-to-cover surfaces remain tested within the same assembly. Adjacent render parts of one continuous member share boundaries; independent centerline capsule checks cover nonlocal self-contact.','Same-assembly clamps/ties and printed-label collars may touch or slightly compress a jacket; recorded contacts are not certified fits.','Centerline-inside checks use finite samples and only closed auxiliary meshes, requiring odd intersection parity on all three fixed non-axis-aligned rays and nearest surface distance >0.2mm. This avoids relying on inconsistent cap normals; not an exhaustive continuous proof.','AABB only selects candidates; surface failures use actual evaluated triangle intersections.','No electrical correctness, mechanical strength, bend-radius compliance, dynamics or manufacturer conformity claim.']}
 out.write_text(json.dumps(rec,indent=2));print(json.dumps({'scene':sd.name,'status':rec['status'],'failures':len(failures),'first_failures':failures[:12]}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);audit(a.scene,a.out)
if __name__=='__main__':main()
