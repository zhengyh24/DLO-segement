#!/usr/bin/env python3
"""Cloud-only viewer for actual v5 renders; no generated or repainted pixels."""
import argparse,importlib.util,json,hashlib,re,shutil,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
HELPER=Path(__file__).with_name('panel_report_helpers.py')
spec=importlib.util.spec_from_file_location('old_visual_helpers',HELPER);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--qa',type=Path);p.add_argument('--preview',action='store_true');a=p.parse_args()
 data=a.data.resolve();out=a.output.resolve();root=data.parent;assets=out/'assets';assets.mkdir(parents=True,exist_ok=True)
 qa=H.js(a.qa) if a.qa else {}
 if not a.preview and qa.get('status') not in ['passed','passed_with_advisories','passed_with_explicit_scope']:raise ValueError('Final report requires explicit passed QA')
 manifest=data/'manifest.jsonl';rows=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()] if manifest.is_file() else []
 if qa and (qa.get('manifest_sha256')!=H.sha(manifest) or Path(qa['artifact_root']).resolve()!=data):raise ValueError('Final QA and dataset manifest binding mismatch')
 scenes=[];all_frames=[];heroes=[];cards=[]
 for scene in sorted(data.iterdir(),key=lambda q:({'easy1':0,'medium2':1,'easy3':2}.get(q.name,99),q.name)):
  if not scene.is_dir() or not (scene/'scene.json').is_file():continue
  meta=H.js(scene/'scene.json');count=int(meta.get('bundle_count',meta.get('physical_assembly_count',0)))
  if not count:raise ValueError('Scene needs explicit physical assembly count')
  expected_ids=meta.get('retained_assembly_ids',list(range(1,count+1)))
  tier={'easy1':'简单档 · 1 束','medium2':'中等档 · 2 束','easy3':'困难档 · 3 束'}[scene.name]
  views=[H.inspect_view(v,count,expected_ids) for v in sorted(scene.glob('view_*')) if (v/'rgb.png').is_file()]
  if not views:continue
  entries=[];gifframes=[]
  for v in views:
   vp=v['path'];m=v['metrics'];stem=scene.name+'_'+vp.name;rgb=Image.open(vp/'rgb.png').convert('RGB');shutil.copyfile(vp/'rgb.png',assets/(stem+'_rgb.png'))
   ids=np.load(vp/'assembly.npy');wire=np.load(vp/'wire.npy');z=np.load(vp/'depth_z_gt.npy');dep=np.load(vp/'depth_input.npy');valid=np.load(vp/'depth_valid.npy').astype(bool);clean=np.isfinite(z)&(z>0)&(z<1e6);limits=np.percentile(z[clean],[1,99]).tolist()
   modes={'rgb':'assets/'+stem+'_rgb.png'}
   for name,im in [('assembly',H.seg_image(ids)),('wire',H.seg_image(wire)),('branch',H.seg_image(np.load(vp/'branch.npy'))),('junction',H.seg_image(np.load(vp/'junction.npy'))),('depth_gt',H.depth_image(z,clean,limits)),('depth_input',H.depth_image(dep,valid,limits))]:
    im.save(assets/(stem+'_'+name+'.png'));modes[name]='assets/'+stem+'_'+name+'.png'
   entry={'scene':scene.name,'tier':tier,'view':vp.name,'physical_assemblies':count,'metrics':m,'camera':v['camera'],'images':modes,'depth_display_range_m':limits};entries.append(entry);all_frames.append(entry)
   d=m['measured_distance_m'];subtitle=f'{len(m["visible_assembly_ids"])} visible / {count} in scene; '+(f'{d*100:.1f} cm' if d is not None else 'see camera metadata')
   gifframes.append(H.tile(rgb,scene.name+' / '+vp.name,subtitle,640).quantize(colors=256))
  first=entries[0];primary=Image.open(scene/'view_000/rgb.png');heroes.append(H.tile(primary,{'easy1':'Easy / 1 harness','medium2':'Medium / 2 harnesses','easy3':'Hard / 3 harnesses'}[scene.name],f'{len(first["metrics"]["visible_assembly_ids"])} visible / {count} physical assemblies',640))
  gifname=scene.name+'_actual_views.gif';gifframes[0].save(assets/gifname,save_all=True,append_images=gifframes[1:],duration=900,loop=0,disposal=2)
  with Image.open(assets/gifname) as gi:assert gi.n_frames==len(views)
  sheet=Image.new('RGB',(512*len(views),586),H.BG)
  for i,v in enumerate(views):sheet.paste(H.tile(Image.open(v['path']/'rgb.png'),v['path'].name,f'{v["metrics"]["visible_assembly_count"]} visible assemblies',512),(512*i,0))
  sheet.save(assets/(scene.name+'_all_views.jpg'),quality=94,subsampling=0)
  scene_record={'scene_id':scene.name,'physical_assemblies':count,'frame_count':len(entries),'source_metadata':meta,'tier':tier,'frames':entries,'gif':'assets/'+gifname};scenes.append(scene_record)
  cards.append('<section><h2>'+tier+'</h2><p>'+str(count)+' 套物理线束；局部近景允许出画。每个视角内的可见数量按目标标签实际统计。</p><img src="assets/'+scene.name+'_all_views.jpg"><details><summary>查看实际多视角帧动画</summary><img style="max-width:640px" src="assets/'+gifname+'"><p>每帧 900ms；没有插帧，不代表采集帧率。</p></details></section>')
 if not scenes:raise ValueError('No rendered scenes found')
 if not a.preview:
  if len(scenes)!=3 or sorted(s['physical_assemblies'] for s in scenes)!=[1,2,3] or any(s['frame_count']!=4 for s in scenes):raise ValueError('Expected 3 scenes (1/2/3), 4 views each')
  if len(rows)!=len(all_frames):raise ValueError('Manifest/frame coverage mismatch')
 cover=Image.new('RGB',(640*len(heroes),714),H.BG)
 for i,hero in enumerate(heroes):cover.paste(hero,(640*i,0))
 cover.save(assets/'close_inspection_comparison.jpg',quality=96,subsampling=0)
 summary={'status':'preview' if a.preview else 'delivered_close_physics_pilot','scene_count':len(scenes),'view_count':len(all_frames),'physical_harness_count':sum(s['physical_assemblies'] for s in scenes),'scenes':scenes,'training_performed':False,'material_parameters_calibrated':False,'sensor_noise_calibrated':False,'manifest_sha256':H.sha(manifest) if manifest.is_file() else None,'qa_sha256':H.sha(a.qa) if a.qa else None,'report_source_sha256':H.sha(__file__),'helper_sha256':H.sha(HELPER)}
 H.dump(out/'summary.json',summary)
 for fn in ['manifest.jsonl','dataset_card.json']:
  if (data/fn).is_file():shutil.copyfile(data/fn,out/fn)
 if a.qa:shutil.copyfile(a.qa,out/'final_qa.json')
 evid=out/'evidence';evid.mkdir(exist_ok=True)
 for pth in sorted((root/'receipts').glob('*.json')):shutil.copyfile(pth,evid/pth.name)
 physiology=root/'physics'
 for pth in sorted(physiology.rglob('*receipt*.json')):
  if pth.stat().st_size<5000000:shutil.copyfile(pth,evid/(pth.parent.name+'_'+pth.name))
 state='首批预览，尚未完成终版检查' if a.preview else '已校验全部视角的深度和标注'
 page=r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>控制柜近景线束 · 1–3 束</title><style>body{margin:0;background:#111b24;color:#edf4fa;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1440px;margin:auto;padding:32px 24px 70px}h1{font-size:36px;line-height:1.3}h2{font-size:25px}.muted{color:#b2c4d3}section{background:#1c2a37;padding:24px;border-radius:14px;margin:24px 0}img{max-width:100%;height:auto;border-radius:8px}button,select{background:#2b4355;color:white;border:1px solid #496274;border-radius:6px;padding:9px 14px;margin:4px;cursor:pointer}button.active{background:#136c6d}a,summary{color:#7ce7df}#viewer{display:block;max-width:900px;width:100%;margin:18px auto}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #496274;text-align:left;padding:10px}.notice{border-left:4px solid #4bd7cc;padding-left:16px}</style></head><body><main><p style="color:#7ce7df">PANEL HARNESS / CLOSE INSPECTION / RGB-D</p><h1>控制柜近景线束</h1><p class="muted">__STATE__</p><p>__COUNTS__。三个档位分别有 1、2、3 束。圆形卡箍带有金属环、缓冲衬垫和安装螺栓。控制柜背景包括 DIN 导轨、端子排、开槽线槽和设备模块。</p><img src="assets/close_inspection_comparison.jpg"><section><h2>圆形卡箍细节</h2><img src="assets/clamp_detail.png" alt="困难档主视角渲染中的圆形卡箍"></section><section><h2>逐帧检查</h2><label>场景与视角 <select id="frame"></select></label><div id="modes"></div><p id="caption" class="muted"></p><img id="viewer"><p class="muted">标签颜色仅表示身份；深度图是仿真输出，可在同一色阶下比较干净深度和模拟输入。此处不是模型预测。</p></section>__CARDS__<section><h2>标签与使用范围</h2><p>圆形卡箍是辅助部件，整束分割只标注线束的可见表面。全部三维部件参与同一相机的深度计算。</p><p>主要目标为整束及分支；可见单线保留独立 ID。不透明包覆内的导线不标成可见像素。分叉交汇区单独标记，不强行归给某一条支路。夹具、扎带、标签、接头及设备背景单独标注。局部出画不等于断线；同一装配的多个视角必须保持在同一数据划分中。</p><p>当前为设计验证样本；三个配置共用设计基底，属于同一相关场景家族，不能拆到训练和测试两边。尚未按真实 AR 眼镜的内参、深度噪声和材料外观标定。本轮未训练模型，也未验证真实场景迁移效果。完整三维模型、RGB-D、相机与轴线数据保留在云端。</p><p><a href="summary.json">逐视角数量、相机与来源</a>__QA_LINK__</p></section></main><script>const frames=__FRAMES__;let current=0,mode='rgb';const names={rgb:'RGB',assembly:'整束 ID',wire:'可见单线 ID',branch:'主干与分支 ID',junction:'分叉归属',depth_gt:'干净深度',depth_input:'模拟输入深度'};const select=document.getElementById('frame');frames.forEach((f,i)=>{let o=document.createElement('option');o.value=i;o.textContent=f.tier+' / '+f.view;select.appendChild(o)});for(const [k,n] of Object.entries(names)){let b=document.createElement('button');b.textContent=n;b.dataset.mode=k;b.onclick=()=>{mode=k;show()};document.getElementById('modes').appendChild(b)}select.onchange=()=>{current=Number(select.value);show()};function show(){let f=frames[current],m=f.metrics;document.getElementById('viewer').src=f.images[mode];let distance=m.measured_distance_m===null?'参见相机记录':(m.measured_distance_m*100).toFixed(1)+' cm';document.getElementById('caption').textContent='场景 '+f.physical_assemblies+' 束 / 当前可见 '+m.visible_assembly_count+' 束；相机到目标最近表面 '+distance+'；原生 '+m.image_shape.join(' × ')+' 像素';document.querySelectorAll('button[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode))}show();</script></body></html>"""
 page=page.replace('__STATE__',state).replace('__COUNTS__',f'{len(scenes)} 个装配场景、{len(all_frames)} 个实际视角').replace('__CARDS__',''.join(cards)).replace('__FRAMES__',json.dumps(all_frames,ensure_ascii=False).replace('</',r'<\/')).replace('__QA_LINK__',' · <a href="final_qa.json">最终检查记录</a>' if a.qa else '')
 (out/'index.html').write_text(page,encoding='utf-8')
 for link in re.findall(r'(?:src|href)="([^"]+)"',page):
  if '://' not in link and not (out/link).is_file():raise ValueError('Broken link '+link)
 for fr in all_frames:
  for link in fr['images'].values():assert (out/link).is_file()
 files=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':H.sha(f)} for f in sorted(out.rglob('*')) if f.is_file() and f.suffix!='.zip' and f.name!='delivery_receipt.json']
 H.dump(out/'delivery_receipt.json',{'status':summary['status'],'files':files,'raw_rgbd_or_scene_assets_included':False})
 if not a.preview:
  with zipfile.ZipFile(out/'visuals.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for f in sorted(out.rglob('*')):
    if f.is_file() and f.suffix!='.zip':z.write(f,str(f.relative_to(out)))
  with zipfile.ZipFile(out/'visuals.zip') as z:assert z.testzip() is None
 print(json.dumps({'status':summary['status'],'output':str(out),'views':len(all_frames),'files':len(files),'archive_sha256':H.sha(out/'visuals.zip') if not a.preview else None}))
if __name__=='__main__':main()
