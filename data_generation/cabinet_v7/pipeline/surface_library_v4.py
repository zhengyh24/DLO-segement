"""Scene semantics and reusable surface primitives."""
import sys,importlib.util,argparse,json,math,hashlib,shutil
from pathlib import Path
sys.dont_write_bytecode=True
import bpy
import numpy as np
from mathutils import Vector,Matrix
from mathutils.bvhtree import BVHTree
OLD=Path(__file__).resolve().with_name('frozen_industrial_renderer_helper.py')
spec=importlib.util.spec_from_file_location('frozen_helpers',OLD);B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
PARTS={};AXES={};TOPO=[];FEATURES=[]
SEM={'background':0,'cloth_cover':1,'braid_cover':2,'tape_cover':3,'wire_pvc':4,'heatshrink':5,'clip_tie':6,'connector':7,'label':8,'panel_device':9,'aux_pipe_rod':10}
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2),encoding='utf8')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reg(o,sem='panel_device',asm=0,branch=0,wire=0,junction=False,tags=None,attached=0):
 pid=len(PARTS)+1;o.pass_index=pid;PARTS[str(pid)]={'part_id':pid,'object_name':o.name,'semantic_id':SEM[sem],'semantic_name':sem,'assembly_id':asm,'branch_id':branch,'wire_id':wire,'junction_id':asm if junction else 0,'target':bool(asm),'attached_assembly_id':attached or asm,'feature_tags':tags or []};return o

def box(name,p,d,ma,sem='panel_device',bevel=.001,**kw):return reg(B.box(name,p,d,ma,bevel),sem,**kw)

def cyl(name,a,b,r,ma,sem='panel_device',**kw):return reg(B.cyl(name,a,b,r,ma,24),sem,**kw)

def frame(p):
 t=np.gradient(p,axis=0);t/=np.linalg.norm(t,axis=1,keepdims=True).clip(1e-12);n=[];last=np.array([1.,0,0])
 for v in t:
  q=last-v*np.dot(v,last)
  if np.linalg.norm(q)<1e-6:q=np.cross(v,[0,0,1])
  q/=np.linalg.norm(q);n.append(q);last=q
 n=np.array(n);return n,np.cross(t,n)

def fabric(name,color):
 # Object coordinates are metres: all swept mesh vertices and scene units are metric.
 # Crossed thread modulation uses explicit 0.8mm periods, not Generated bounds.
 ma=B.mat(name,color,.88,0,.10);nodes=ma.node_tree.nodes;links=ma.node_tree.links;bs=nodes.get('Principled BSDF')
 coord=nodes.new('ShaderNodeTexCoord');sep=nodes.new('ShaderNodeSeparateXYZ');links.new(coord.outputs['Object'],sep.inputs[0])
 def mathnode(op,a,b=None):
  n=nodes.new('ShaderNodeMath');n.operation=op
  for k,v in enumerate([a,b]):
   if v is None:continue
   if isinstance(v,(int,float)):n.inputs[k].default_value=v
   else:links.new(v,n.inputs[k])
  return n.outputs[0]
 u=mathnode('ADD',sep.outputs['X'],sep.outputs['Y']);v=mathnode('SUBTRACT',sep.outputs['X'],sep.outputs['Y'])
 warp=mathnode('SINE',mathnode('MULTIPLY',u,2*math.pi/(.0008*math.sqrt(2))))
 weft=mathnode('SINE',mathnode('MULTIPLY',v,2*math.pi/(.0008*math.sqrt(2))))
 cross=mathnode('MULTIPLY',warp,weft)
 height=mathnode('ADD',mathnode('MULTIPLY',mathnode('ADD',warp,weft),.20),mathnode('MULTIPLY',cross,.10))
 height=mathnode('ADD',height,.5)
 tex=nodes.new('ShaderNodeTexNoise');links.new(coord.outputs['Object'],tex.inputs['Vector']);tex.inputs['Scale'].default_value=1600;tex.inputs['Detail'].default_value=2
 height=mathnode('ADD',mathnode('MULTIPLY',height,.88),mathnode('MULTIPLY',tex.outputs['Fac'],.12))
 ramp=nodes.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.12;ramp.color_ramp.elements[0].color=tuple(c*.63 for c in color)+(1,);ramp.color_ramp.elements[1].position=.88;ramp.color_ramp.elements[1].color=tuple(c*1.55 for c in color)+(1,);links.new(height,ramp.inputs[0]);links.new(ramp.outputs['Color'],bs.inputs['Base Color'])
 bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.68;bump.inputs['Distance'].default_value=.00018;links.new(height,bump.inputs['Height']);links.new(bump.outputs[0],bs.inputs['Normal'])
 links.new(mathnode('ADD',.84,mathnode('MULTIPLY',tex.outputs['Fac'],.12)),bs.inputs['Roughness'])
 if 'Sheen Weight' in bs.inputs:bs.inputs['Sheen Weight'].default_value=.32
 ma['surface_provenance']='procedural metric woven shader, period0.8mm; geometry folds retained; not measured cloth BRDF'
 return ma

def initial_materials():
 B.MATS={};B.META={}
 B.mat('paper',(.76,.75,.65),.7);B.mat('ink',(.006,.008,.009),.5);B.mat('steel',(.43,.47,.50),.27,.9,.05);B.mat('gold',(.45,.32,.13),.42,.65,.12);B.mat('blue',(.017,.073,.14),.43,.35,.1);B.mat('panel',(.61,.61,.51),.68,.12,.08);B.mat('rubber',(.015,.019,.021),.65);B.mat('heat',(.015,.018,.02),.45);B.mat('tie_red',(.45,.012,.012),.5);B.mat('tie_yellow',(.63,.38,.012),.5);B.mat('tie_white',(.65,.64,.52),.64)
 fabric('cloth',(.026,.029,.033));fabric('braid',(.055,.064,.07));B.mat('tape',(.02,.024,.026),.39,0,.08)
 for i,c in enumerate([(.40,.012,.015),(.012,.08,.24),(.018,.2,.045),(.48,.40,.025)]):B.mat('wire'+str(i),c,.42,0,.05)

def surface_bvh():
 dg=bpy.context.evaluated_depsgraph_get();verts=[];faces=[];ids=[]
 for obj in bpy.context.scene.objects:
  if not PARTS.get(str(obj.pass_index),{}).get('target') or obj.type!='MESH':continue
  ev=obj.evaluated_get(dg);me=ev.to_mesh();me.calc_loop_triangles();off=len(verts);verts.extend([obj.matrix_world@v.co for v in me.vertices]);faces.extend([tuple(off+int(k) for k in t.vertices) for t in me.loop_triangles]);ids.extend([PARTS[str(obj.pass_index)]['assembly_id']]*len(me.loop_triangles));ev.to_mesh_clear()
 return BVHTree.FromPolygons(verts,faces,all_triangles=True),ids

def configure(size,samples,device):
 sc,cam,outs=B.configure(size,samples,device,20261114)
 cam.data.lens=28;sc.view_settings.exposure=-.9
 # Existing helper creates three physical lights; their position/energy stays static per scene.
 return sc,cam,outs
