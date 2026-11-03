"""Matched local serial latency: same six RGB files, three passes, one warmup.
Time includes RGB loading, preprocessing, complete prediction and final ID PNG write.
Exclude model loading, initial warmup, ground truth, scoring, plots, downloads.
"""
from pathlib import Path
import sys,os,json,time,argparse,hashlib
WORK=Path(__file__).resolve().parents[2];OUT=Path(os.environ["PADLOS_RUN"]).resolve();OUT.mkdir(parents=True, exist_ok=True)
CACHE=Path(os.environ["PADLOS_CACHE"])
BDIR=WORK/"results/sam_tracker/full_wireseg_BC"
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
os.environ["MPLBACKEND"]="Agg";os.environ["MPLCONFIGDIR"]=str(CACHE/"matplotlib")
sys.path.insert(0,str(WORK/"third_party/padlos_deps"))
sys.path.insert(0,str(WORK/"datasets/PADLOS_official"))
sys.path.insert(0,str(WORK/"cable_seg/versions/full_wireseg_20261029"))
import numpy as np,torch,cv2
from PIL import Image
p=argparse.ArgumentParser();p.add_argument("--method",choices=["B","PADLOS"],required=True);args=p.parse_args()
torch.set_num_threads(1);cv2.setNumThreads(1)
selected=json.loads((BDIR/"selected_samples.json").read_text(encoding="utf-8"))["selected"]
ids=[str(r["record"]["id"]) for r in selected]
for r in selected:
 image=CACHE/"data/rgb"/(str(r["record"]["id"])+".png")
 assert hashlib.sha256(image.read_bytes()).hexdigest()==r["record"]["image_sha256"]
root=CACHE/"matched_latency"/args.method;root.mkdir(parents=True,exist_ok=False)
if args.method=="B":
 from sam_path_v1.checkpoint import build_from_checkpoint
 from sam_path_v1.predict import load_rgb
 from sam_path_v1.inference import decode_instances
 model,payload=build_from_checkpoint(Path(os.environ["TRACKER_B_BEST"]),device="cuda",sam_repo=os.environ["SAM3_REPO"],sam_checkpoint=Path(os.environ["SAM3_BASE"]))
 config=payload["train_config"]
else:
 from PDSegmentor.benchmark_pgm import load_model,build_transform,postprocess
 from PDSegmentor.depthinfer import build_model
 from instance_extractor.acm import LeeCCNumbaNeighborWireSegmenter,initialize_endpoint_nodes,reconstruct_instances,warmup_numba
 warmup_numba()
 model=load_model(WORK/".models/padlos/pgm.pth",torch.device("cuda"),"vits",1)
 depth=build_model("vits",WORK/".models/padlos/depth_anything_v2_vits.pth",torch.device("cuda"))
 transform=build_transform(518)
def one(ident,label):
 dest=root/label/ident;dest.mkdir(parents=True,exist_ok=False)
 image=CACHE/"data/rgb"/(ident+".png")
 torch.cuda.synchronize();start=time.perf_counter()
 if args.method=="B":
  rgb,images,valid=load_rgb(image,1036);images,valid=images.to("cuda"),valid.to("cuda")
  outputs=model(images,valid)
  assert outputs["calls"]=={"encoder":1,"decoder":1}
  result=decode_instances(outputs,images.shape[-2:],valid,object_threshold=config["object_threshold"],mask_threshold=config["mask_threshold"])[0]
  idsmap=result["instances"][:rgb.shape[0],:rgb.shape[1]].astype(np.uint16)
 else:
  rgb=np.array(Image.open(image).convert("RGB"))
  tensor=torch.from_numpy(transform({"image":rgb.astype(np.float32)/255.})["image"]).unsqueeze(0).to("cuda").float()
  mask=postprocess(model(tensor),rgb.shape[:2],.05)
  raw=depth.infer_image(np.ascontiguousarray(rgb[:,:,::-1]),518)
  gray=((raw-raw.min())/max(float(raw.max()-raw.min()),1e-8)*255).astype(np.uint8)
  Image.fromarray(mask).save(dest/"pgm_mask.png");Image.fromarray(gray).save(dest/"depth.png")
  segmenter=LeeCCNumbaNeighborWireSegmenter(image.as_posix(),(dest/"pgm_mask.png").as_posix(),(dest/"depth.png").as_posix())
  segmenter.preprocess();segmenter.build_graph();initialize_endpoint_nodes(segmenter)
  _,paths=segmenter.optimize_wires_global(w_bend=.4,w_color=.3,w_width=.1,w_len=.1,k_paths=100)
  rendered=reconstruct_instances(segmenter,paths)
  colors,inverse=np.unique(rendered.reshape(-1,3),axis=0,return_inverse=True);lookup=np.zeros(len(colors),np.uint16);count=0
  for i,color in enumerate(colors):
   if np.any(color):count+=1;lookup[i]=count
  idsmap=lookup[inverse].reshape(rendered.shape[:2])
 Image.fromarray(idsmap).save(dest/"instance_ids.png")
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 return {"id":ident,"seconds":elapsed,"predicted_ids":int(idsmap.max()),"prediction_pixels_sha256":hashlib.sha256(idsmap.tobytes()).hexdigest()}
records=[]
with torch.inference_mode():
 warmup=one("008412","warmup")
 for repeat in range(1,4):
  for ident in ids:
   r=one(ident,f"repeat_{repeat}");r["repeat"]=repeat;records.append(r);print(json.dumps(r),flush=True)
times=[r["seconds"] for r in records]
receipt={"method":args.method,"hardware":torch.cuda.get_device_name(0),"torch":torch.__version__,"cpu_threads":1,"acm_workers":1 if args.method=="PADLOS" else 0,"ids":ids,"repeats":3,"timed_predictions":len(records),"one_full_image_warmup_excluded":warmup,"mean_seconds":float(np.mean(times)),"median_seconds":float(np.median(times)),"p95_seconds":float(np.percentile(times,95)),"serial_fps":len(times)/sum(times),"timing_scope":"RGB file read + preprocessing + all model/ACM steps + final instance-ID PNG write; excludes model loading, warmup, GT, scoring, figures, downloads.","scope_limit":"Six fixed qualitative examples only, NOT a full-1800 latency estimate. Includes the one valid empty PADLOS result; no CPU worker parallelism.","records":records}
(OUT/("matched_latency_"+args.method+".json")).write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print("RESULT",json.dumps({k:v for k,v in receipt.items() if k not in ("records","one_full_image_warmup_excluded")}),flush=True)
