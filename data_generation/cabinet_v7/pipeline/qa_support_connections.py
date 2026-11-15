"""Check explicit hardware-to-panel connectivity; no load/stiffness certification."""
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
sys.path.insert(0,str(Path(__file__).parent))
from qa_scene_contacts import mesh

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(sd,out):
 parts=json.loads((sd/'part_map.json').read_text());parts={v['object_name']:v for v in parts.values()};bpy.ops.wm.open_mainfile(filepath=str(sd/'scene.blend'));deps=bpy.context.evaluated_depsgraph_get();names=[];infos={};tol=.0001
 tags={'physical_support','panel_attachment','fixed_clip','endpoint_mount','panel_mounted_blue_block'}
 for ob in bpy.context.scene.objects:
  m=parts.get(ob.name,{})
  if m.get('target'):continue
  if ob.name=='cream equipment mounting panel' or tags.intersection(m.get('feature_tags',[])):
   info=mesh(ob,deps)
   if info:infos[ob.name]=info;names.append(ob.name)
 panel='cream equipment mounting panel';fail=[]
 if panel not in infos:raise ValueError('No actual mounting-panel mesh')
 graph={n:set() for n in names};edges=[];tested=0
 for i,a in enumerate(names):
  ma=infos[a];oa=parts[a]['attached_assembly_id']
  for b in names[i+1:]:
   mb=infos[b];ob=parts[b]['attached_assembly_id']
   if a!=panel and b!=panel and oa!=ob:continue
   if not(np.all(ma['hi']+tol>=mb['lo']) and np.all(mb['hi']+tol>=ma['lo'])):continue
   tested+=1;hits=ma['bvh'].overlap(mb['bvh']);distance=None
   if not hits:
    distance=min([mb['bvh'].find_nearest(Vector(v))[3] for v in ma['v']]+[ma['bvh'].find_nearest(Vector(v))[3] for v in mb['v']])
   if hits or distance<=tol:
    graph[a].add(b);graph[b].add(a);edges.append({'objects':[a,b],'triangle_intersection_pairs':len(hits),'sampled_vertex_to_surface_distance_m':None if distance is None else float(distance)})
 seen={panel};todo=[panel]
 while todo:
  current=todo.pop()
  for n in graph[current]-seen:seen.add(n);todo.append(n)
 disconnected=[n for n in names if n not in seen]
 if disconnected:fail.append({'hardware_without_same_owner_geometric_connection_to_panel':disconnected})
 feet=[]
 for name in names:
  if 'panel_attachment' not in parts[name].get('feature_tags',[]):continue
  gap=float(infos[name]['lo'][2]-infos[panel]['hi'][2]);feet.append({'object':name,'bottom_to_panel_top_gap_m':gap})
  if abs(gap)>tol:fail.append({'mounting_foot_not_on_panel_surface':name,'gap_m':gap})
 rec={'status':'passed_with_explicit_scope' if not fail else 'failed','scene_id':sd.name,'blend_sha256':sha(sd/'scene.blend'),'script_sha256':sha(__file__),'hardware_count':len(names)-1,'hardware_connected_to_panel':len(seen)-1,'pair_candidates_checked':tested,'contact_tolerance_m':tol,'mounting_feet':feet,'contact_graph_edges':edges,'failures':fail,'scope':'Independent actual evaluated-mesh connectivity among explicitly declared hardware parts of each owner and the mounting panel; no target cable is used to complete a support path.','limits':['A triangle intersection or vertex-to-surface proximity proves geometric connection evidence only, not welded/bolted joint strength, static support reaction distribution, or manufacturing compliance.','Auxiliary-auxiliary support contacts are intentional here; wrong-owner target collisions remain separately forbidden.']}
 out.write_text(json.dumps(rec,indent=2));print(json.dumps({'scene':sd.name,'status':rec['status'],'hardware':len(names)-1,'connected':len(seen)-1,'failures':fail}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);audit(a.scene,a.out)
