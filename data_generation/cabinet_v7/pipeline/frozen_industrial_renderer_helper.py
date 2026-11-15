"""Blender primitives, materials, camera, and raw passes shared by the v7 pipeline."""
import bpy
import numpy as np
from mathutils import Vector

MATS = {}
META = {}

def mat(name,col,rough=.5,metal=0,noise=.0,emission=0):
 if name in MATS:return MATS[name]
 m=bpy.data.materials.new(name);m.use_nodes=True;n=m.node_tree.nodes;b=n.get('Principled BSDF');b.inputs['Base Color'].default_value=(*col,1);b.inputs['Roughness'].default_value=rough;b.inputs['Metallic'].default_value=metal
 if emission:b.inputs['Emission Color'].default_value=(*col,1);b.inputs['Emission Strength'].default_value=emission
 if noise:
  t=n.new('ShaderNodeTexNoise');t.inputs['Scale'].default_value=140;t.inputs['Detail'].default_value=2
  r=n.new('ShaderNodeValToRGB');r.color_ramp.elements[0].color=(*(np.array(col)*(1-noise)),1);r.color_ramp.elements[1].color=(*(np.minimum(np.array(col)*(1+noise),1)),1)
  m.node_tree.links.new(t.outputs['Fac'],r.inputs[0]);m.node_tree.links.new(r.outputs[0],b.inputs['Base Color'])
  bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.14;bump.inputs['Distance'].default_value=.00013;m.node_tree.links.new(t.outputs['Fac'],bump.inputs['Height']);m.node_tree.links.new(bump.outputs[0],b.inputs['Normal'])
 MATS[name]=m;return m

def track(o,cat,extra=None):
 o.pass_index=0;META[o.name]={'category':cat,**(extra or {})};return o

def box(name,loc,dim,material,bevel=.001,cat='fixture'):
 bpy.ops.mesh.primitive_cube_add(size=1,location=loc);o=bpy.context.object;o.name=name;o.dimensions=dim;bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
 o.data.materials.append(material)
 if bevel:
  b=o.modifiers.new('manufactured edge radius','BEVEL');b.width=bevel;b.segments=2
  o.modifiers.new('weighted normals','WEIGHTED_NORMAL')
 return track(o,cat)

def cyl(name,a,b,r,material,vertices=24,cat='fixture'):
 a=np.array(a);b=np.array(b);v=b-a;bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=float(np.linalg.norm(v)),location=(a+b)/2)
 o=bpy.context.object;o.name=name;o.rotation_euler=Vector(v).to_track_quat('Z','Y').to_euler();o.data.materials.append(material)
 for f in o.data.polygons:f.use_smooth=True
 bevel=o.modifiers.new('edge rounding','BEVEL');bevel.width=min(r*.1,.0005);bevel.segments=2
 return track(o,cat)

def text_obj(body,loc,size=.01,col='ink',align='LEFT'):
 cu=bpy.data.curves.new('label '+body,'FONT');cu.body=body;cu.size=size;cu.align_x=align;cu.extrude=.00001
 o=bpy.data.objects.new('label '+body,cu);bpy.context.collection.objects.link(o);o.location=loc;o.data.materials.append(MATS[col]);return track(o,'label')

def configure(size,samples,device,seed):
 sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.samples=samples;sc.cycles.use_denoising=False;sc.render.use_persistent_data=True
 sc.render.resolution_x=size;sc.render.resolution_y=size;sc.render.resolution_percentage=100;sc.render.image_settings.file_format='PNG';sc.render.image_settings.color_mode='RGB';sc.render.threads_mode='FIXED';sc.render.threads=8
 if device=='GPU':
  pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='CUDA';pref.get_devices();found=False
  for d in pref.devices:d.use=d.type=='CUDA';found|=d.use
  if not found:raise RuntimeError('No CUDA device')
  sc.cycles.device='GPU'
 else:sc.cycles.device='CPU'
 sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast';sc.view_settings.exposure=-.9
 sc.world.use_nodes=True;sc.world.node_tree.nodes['Background'].inputs[0].default_value=(.25,.29,.36,1);sc.world.node_tree.nodes['Background'].inputs[1].default_value=.17
 for name,loc,energy,size,col in [('overhead softbox',(-.6,.8,1.2),55,.8,(.78,.87,1)),('warm right fill',(.65,.0,.75),28,.65,(1,.87,.70)),('front worklight',(-.10,-.40,1.0),16,.45,(1,.97,.9))]:
  bpy.ops.object.light_add(type='AREA',location=loc);l=bpy.context.object;l.name=name;l.data.energy=energy;l.data.shape='DISK';l.data.size=size;l.data.color=col;l.rotation_euler=(Vector((0,0,0))-l.location).to_track_quat('-Z','Y').to_euler()
 bpy.ops.object.camera_add();cam=bpy.context.object;cam.data.type='PERSP';cam.data.lens=43;cam.data.sensor_width=36;cam.data.sensor_fit='HORIZONTAL';cam.data.clip_start=.01;cam.data.clip_end=10;sc.camera=cam
 layer=sc.view_layers[0];layer.use_pass_z=True;layer.use_pass_object_index=True;sc.use_nodes=True;tree=sc.node_tree;tree.nodes.clear();rl=tree.nodes.new('CompositorNodeRLayers');comp=tree.nodes.new('CompositorNodeComposite');tree.links.new(rl.outputs['Image'],comp.inputs[0]);outs=[]
 for slot,name in [('Depth','depth_raw_'),('IndexOB','instance_')]:
  node=tree.nodes.new('CompositorNodeOutputFile');node.format.file_format='OPEN_EXR';node.format.color_depth='32';node.format.color_mode='RGB';node.file_slots[0].path=name;tree.links.new(rl.outputs[slot],node.inputs[0]);outs.append(node)
 return sc,cam,outs

def read_exr(path):
 im=bpy.data.images.load(str(path),check_existing=False);im.colorspace_settings.name='Non-Color';w,h=im.size;a=np.array(im.pixels[:],np.float32).reshape(h,w,4)[::-1,:,0].copy();bpy.data.images.remove(im);return a

def ray_diagnostic(sc,cam,raw,ids,K,T):
 h,w=ids.shape;grad=np.zeros((h,w),bool);grad[1:]|=abs(np.diff(raw,axis=0))>.001;grad[:,1:]|=abs(np.diff(raw,axis=1))>.001
 samples=[]
 for y in range(12,h-12,47):
  for x in range(12,w-12,47):
   if not grad[y-1:y+2,x-1:x+2].any():samples.append((y,x))
 for ident in np.unique(ids[ids>0]):
  yy,xx=np.where((ids==ident)&~grad)
  for k in np.linspace(0,len(yy)-1,min(12,len(yy))).astype(int):samples.append((int(yy[k]),int(xx[k])))
 deps=bpy.context.evaluated_depsgraph_get();out=[]
 for y,x in samples:
  rc=np.array([(x+.5-K[0,2])/K[0,0],(y+.5-K[1,2])/K[1,1],1.]);rw=T[:3,:3]@rc;rw/=np.linalg.norm(rw)
  hit,loc,normal,index,obj,matrix=sc.ray_cast(deps,Vector(T[:3,3]),Vector(rw),distance=10)
  if hit:
   z=(np.linalg.inv(T)@np.r_[loc,1.])[2];rn=np.linalg.norm(np.array(loc)-T[:3,3]);out.append((y,x,float(raw[y,x]),float(z),float(rn),int(ids[y,x]),int(obj.pass_index)))
 a=np.array(out);ze=abs(a[:,2]-a[:,3]);re=abs(a[:,2]-a[:,4]);kind='z' if np.median(ze)<np.median(re) else 'range';err=ze if kind=='z' else re
 good=err<.0015
 receipt={'method':'first-hit raycast of complete evaluated 3D scene at calibrated pixel centers; no background-plane assumption','sample_count':len(a),'z_error_p50_m':float(np.median(ze)),'z_error_p95_m':float(np.quantile(ze,.95)),'range_error_p50_m':float(np.median(re)),'range_error_p95_m':float(np.quantile(re,.95)),'raw_depth_convention':kind,'selected_error_p95_m':float(np.quantile(err,.95)),'center_id_match_fraction':float((a[:,5]==a[:,6]).mean()),'foreground_sample_count':int((a[:,5]>0).sum())}
 if receipt['center_id_match_fraction']<.99 or receipt['foreground_sample_count']<24:raise RuntimeError('Insufficient raycast ID agreement '+str(receipt))
 if np.quantile(err,.95)>.002:raise RuntimeError('Depth/raycast mismatch '+str(receipt))
 return kind,receipt,a
