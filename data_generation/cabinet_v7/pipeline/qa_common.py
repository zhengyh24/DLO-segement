"""Independent evaluated-scene fingerprint and world-mesh helpers. Derived from previously audited v3 QA; no renderer import."""
import hashlib,json
from pathlib import Path
import bpy,numpy as np
from mathutils.bvhtree import BVHTree
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def val(v):
 if isinstance(v,(str,int,float,bool)) or v is None:return v
 try:return [val(x) for x in v]
 except:return str(getattr(v,'name',type(v).__name__))
def scalar_rna(o,exclude=()):
 out={}
 for p in o.bl_rna.properties:
  k=p.identifier
  if k in exclude or k in ['rna_type','session_uid','users','tag','is_updated','is_updated_data','is_evaluated','is_runtime_data'] or p.type not in ['BOOLEAN','INT','FLOAT','STRING','ENUM']:continue
  try:out[k]=val(getattr(o,k))
  except:pass
 return out
def nodes(t):
 if t is None:return None
 out=[]
 for n in t.nodes:
  d={'name':n.name,'type':n.bl_idname,'props':scalar_rna(n,('select','location','width','height','dimensions','label','hide','color','use_custom_color','show_options','show_preview','show_texture')),'inputs':[(s.identifier,val(s.default_value)) for s in n.inputs if hasattr(s,'default_value')],'outputs':[(s.identifier,val(s.default_value)) for s in n.outputs if hasattr(s,'default_value')]}
  if hasattr(n,'color_ramp'):d['color_ramp']={'props':scalar_rna(n.color_ramp),'elements':[(e.position,list(e.color)) for e in n.color_ramp.elements]}
  if getattr(n,'image',None):
   im=n.image;path=Path(bpy.path.abspath(im.filepath));d['image']={'name':im.name,'path':str(path),'colorspace':im.colorspace_settings.name,'sha256':sha(path) if path.is_file() else None,'packed_sha256':hashlib.sha256(im.packed_file.data).hexdigest() if im.packed_file else None}
  if getattr(n,'node_tree',None):d['group']=nodes(n.node_tree)
  out.append(d)
 return {'nodes':sorted(out,key=lambda x:x['name']),'links':sorted((l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier) for l in t.links)}
def fingerprint():
 sc=bpy.context.scene;deps=bpy.context.evaluated_depsgraph_get();objects={}
 for o in sc.objects:
  if o.type=='CAMERA':continue
  d={'type':o.type,'matrix':np.array(o.matrix_world).tolist(),'hide_render':o.hide_render,'pass_index':o.pass_index,'materials':[s.material.name if s.material else None for s in o.material_slots],'ray_visibility':{k:getattr(o,k,None) for k in ['visible_camera','visible_diffuse','visible_glossy','visible_shadow','visible_transmission','visible_volume_scatter']}}
  if o.type in ['MESH','CURVE','SURFACE','FONT','META']:
   e=o.evaluated_get(deps);m=e.to_mesh();d['vertices_sha']=hashlib.sha256(np.array([v.co[:] for v in m.vertices],dtype='<f8').tobytes()).hexdigest();d['polygons_sha']=digest([(list(p.vertices),p.material_index,p.use_smooth) for p in m.polygons]);e.to_mesh_clear()
  if o.type=='LIGHT':d['light']=scalar_rna(o.data);d['light_nodes']=nodes(o.data.node_tree)
  objects[o.name]=d
 materials={m.name:{'properties':scalar_rna(m),'nodes':nodes(m.node_tree)} for m in bpy.data.materials if m.users>0}
 return {'objects':objects,'materials':materials,'world':{'properties':scalar_rna(sc.world),'nodes':nodes(sc.world.node_tree)},'color_management':scalar_rna(sc.view_settings),'unit_settings':scalar_rna(sc.unit_settings),'render_invariants':{k:getattr(sc.render,k,None) for k in ['engine','film_transparent','use_motion_blur']},'cycles_invariants':{k:getattr(sc.cycles,k,None) for k in ['use_denoising','max_bounces','diffuse_bounces','glossy_bounces','transmission_bounces','volume_bounces']}}
def worldmesh(o,deps):
 e=o.evaluated_get(deps);m=e.to_mesh();m.calc_loop_triangles();v=np.array([e.matrix_world@p.co for p in m.vertices],float);tri=np.array([list(t.vertices) for t in m.loop_triangles],int);e.to_mesh_clear();b=BVHTree.FromPolygons(v.tolist(),tri.tolist(),all_triangles=True);edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1);count=np.unique(edges,axis=0,return_counts=True)[1];vol=np.einsum('ij,ij->i',v[tri[:,0]],np.cross(v[tri[:,1]],v[tri[:,2]])).sum()/6
 return {'bvh':b,'min':v.min(0),'max':v.max(0),'closed':bool((count==2).all() and abs(vol)>1e-12),'id':o.pass_index}
