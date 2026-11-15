#!/usr/bin/env python3
"""Factual visual report of procedural panel-harness pilot; no model execution."""
from __future__ import annotations
import argparse,hashlib,json,html,math,re,shutil,time,zipfile,colorsys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
BG='#101923';PANEL='#192633';FG='#edf4fa';MUTED='#a9bac8'
REQUIRED=['rgb.png','assembly.npy','branch.npy','wire.npy','junction.npy','semantic.npy','part_id.npy','depth_z_gt.npy','depth_range_gt.npy','depth_input.npy','depth_valid.npy','camera.json']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def js(p):return json.loads(Path(p).read_text())

def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def text(draw,xy,s,size=20,color=FG,bold=False):
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
    except OSError:
        font = ImageFont.load_default(size=size)
    draw.text(xy,str(s),font=font,fill=color)

def fit(im,wh):
 im=im.convert('RGB');im.thumbnail(wh,Image.Resampling.LANCZOS);c=Image.new('RGB',wh,BG);c.paste(im,((wh[0]-im.width)//2,(wh[1]-im.height)//2));return c

def tile(im,title,subtitle,width=512,foot=None):
 c=Image.new('RGB',(width,width+(98 if foot else 74)),PANEL);c.paste(fit(im,(width,width)),(0,0));d=ImageDraw.Draw(c);text(d,(12,width+9),title,21,bold=True);text(d,(12,width+40),subtitle,16,MUTED)
 if foot:text(d,(12,width+65),foot,15,MUTED)
 return c

def seg_image(ids):
 palette=np.zeros((int(ids.max())+1,3),np.uint8)
 for i in range(1,len(palette)):palette[i]=np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.68,1))*255
 return Image.fromarray(palette[ids])

def depth_image(z,valid,limits):
 lo,hi=limits;z=np.nan_to_num(z,nan=0,posinf=0,neginf=0);q=np.clip((z-lo)/max(hi-lo,1e-6),0,1)*3;i=np.minimum(q.astype(int),2);f=(q-i)[...,None];colors=np.array([[55,24,107],[41,116,142],[51,181,120],[249,232,90]],float);rgb=colors[i]*(1-f)+colors[i+1]*f;rgb[~valid]=[17,23,31];return Image.fromarray(rgb.astype(np.uint8))

def camera_distance(cam):
 close=cam.get('close_range',{})
 for key in ['actual_nearest_target_surface_distance_m','actual_nearest_cable_surface_distance_m','measured_mesh_distance_m']:
  if key in cam:return float(cam[key]),key
  if key in close:return float(close[key]),'close_range.'+key
 return None,None

def inspect_view(view,count,expected_ids=None):
 for n in REQUIRED:
  if not (view/n).is_file():raise ValueError(f'Missing required output {view/n}')
 with Image.open(view/'rgb.png') as im:w,h=im.size
 arrays={n:np.load(view/(n+'.npy')) for n in ['assembly','branch','wire','junction','semantic','part_id','depth_z_gt','depth_range_gt','depth_input','depth_valid']}
 for n,a in arrays.items():
  if a.shape!=(h,w):raise ValueError(f'{view}: {n} image shape mismatch')
 for name in ['assembly','branch','wire','junction','semantic','part_id']:
  if not np.issubdtype(arrays[name].dtype,np.integer):raise ValueError(f'{view}: {name} labels are not integer IDs')
 ann=np.load(view/'annotation_valid.npy').astype(bool) if (view/'annotation_valid.npy').is_file() else np.ones((h,w),bool)
 target_sem=np.isin(arrays['semantic'],[1,2,3,4,5])
 if not np.array_equal(target_sem[ann],(arrays['assembly']>0)[ann]):raise ValueError(f'{view}: assembly target disagrees with semantic policy')
 if np.any((arrays['wire']>0)&(arrays['semantic']!=4)):raise ValueError(f'{view}: visible wire labels outside PVC class')
 if np.any((arrays['branch']>0)&((arrays['branch']//10!=arrays['assembly'])|~np.isin(arrays['branch']%10,[1,2,3]))):raise ValueError(f'{view}: branch-to-assembly hierarchy conflict')
 if np.any((arrays['junction']>0)&(arrays['junction']!=arrays['assembly'])):raise ValueError(f'{view}: junction ownership conflict')
 ids=arrays['assembly'];ids=np.where(ann,ids,0);visible=np.unique(ids[ids>0]).astype(int).tolist();expected=list(range(1,count+1)) if expected_ids is None else expected_ids
 if any(i not in expected for i in visible):raise ValueError(f'{view}: unexpected assembly IDs {visible}')
 edge=np.unique(np.concatenate([ids[0],ids[-1],ids[:,0],ids[:,-1]]));border=[int(i) for i in edge if i>0];z=arrays['depth_z_gt'];clean=np.isfinite(z)&(z>0)&(z<1e6)&ann;dep=arrays['depth_input'];valid=arrays['depth_valid'].astype(bool)&np.isfinite(dep)&(dep>0)&ann;cam=js(view/'camera.json')
 if cam.get('units')!='m' or cam.get('depth_convention')!='z':raise ValueError(f'{view}: expected metric camera-axis Z contract')
 measured,field=camera_distance(cam)
 m={'view_id':view.name,'image_shape':[h,w],'physical_assembly_count':count,'visible_assembly_ids':visible,'visible_assembly_count':len(visible),'border_touching_assembly_ids':border,'visible_branch_ids':np.unique(arrays['branch'][arrays['branch']>0]).astype(int).tolist(),'visible_wire_ids':np.unique(arrays['wire'][arrays['wire']>0]).astype(int).tolist(),'junction_assembly_ids':np.unique(arrays['junction'][arrays['junction']>0]).astype(int).tolist(),'target_pixel_fraction':float((ids>0).mean()),'depth_input_valid_fraction':float(valid.mean()),'measured_distance_m':measured,'distance_field':field,'file_sha256':{n:sha(view/n) for n in REQUIRED}}
 if (view/'annotation_valid.npy').is_file():m['file_sha256']['annotation_valid.npy']=sha(view/'annotation_valid.npy')
 if (view/'branch_annotation_valid.npy').is_file():
  branch_valid=np.load(view/'branch_annotation_valid.npy').astype(bool)
  if branch_valid.shape!=(h,w) or not np.array_equal(branch_valid,arrays['junction']==0):raise ValueError('Branch annotation-valid mask does not match junction exclusion')
  m['file_sha256']['branch_annotation_valid.npy']=sha(view/'branch_annotation_valid.npy')
 return {'path':view,'camera':cam,'metrics':m}
