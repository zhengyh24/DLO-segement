"""Fixed PADLOS RGB-only benchmark against the existing WireSeg test1800.
Only orchestration/logging are new. Released PGM/depth/ACM code is unchanged.
"""
from __future__ import annotations
import os,sys,json,time,hashlib,traceback,datetime,shutil,multiprocessing as mp,subprocess,gzip
from pathlib import Path
WORK=Path(__file__).resolve().parents[2]
OUT=Path(os.environ["PADLOS_RUN"]).resolve()
CACHE=Path(os.environ["PADLOS_CACHE"])
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[name]="1"
os.environ["MPLBACKEND"]="Agg"
os.environ["MPLCONFIGDIR"]=str(CACHE/"matplotlib")
sys.path.insert(0,str(WORK/"third_party/padlos_deps"))
sys.path.insert(0,str(WORK/"datasets/PADLOS_official"))
sys.path.insert(0,str(WORK/"cable_seg/versions/full_wireseg_20261029"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(p)
def acm_worker(conn):
    import numpy as np,cv2
    from PIL import Image
    from instance_extractor.acm import LeeCCNumbaNeighborWireSegmenter,initialize_endpoint_nodes,reconstruct_instances,warmup_numba
    import matplotlib.pyplot as plt
    cv2.setNumThreads(1)
    warmup_numba()
    conn.send({"ready":True,"pid":os.getpid()})
    while True:
        job=conn.recv()
        if job is None:return
        ident=job["id"];case=CACHE/"predictions"/ident
        started=time.perf_counter()
        with (case/"acm.log").open("w",encoding="utf-8") as log:
            oldout,olderr=sys.stdout,sys.stderr;sys.stdout=sys.stderr=log
            try:
                image=(CACHE/"data/rgb"/(ident+".png")).as_posix()
                segmenter=LeeCCNumbaNeighborWireSegmenter(image,(case/"pgm_mask.png").as_posix(),(case/"depth.png").as_posix())
                segmenter.preprocess();segmenter.build_graph();initialize_endpoint_nodes(segmenter)
                _,paths=segmenter.optimize_wires_global(w_bend=.4,w_color=.3,w_width=.1,w_len=.1,k_paths=100)
                rendered=reconstruct_instances(segmenter,paths)
                assert tuple(rendered.shape)==tuple(job["shape"]+[3]),(rendered.shape,job["shape"])
                colors,inverse=np.unique(rendered.reshape(-1,3),axis=0,return_inverse=True)
                table=np.zeros(len(colors),np.uint16);count=0
                for i,color in enumerate(colors):
                    if np.any(color):count+=1;table[i]=count
                ids=table[inverse].reshape(rendered.shape[:2])
                Image.fromarray(ids).save(case/"instance_ids.png")
                if ident in job["visual_ids"]:Image.fromarray(rendered).save(case/"instances_rgb.png")
                palette=np.asarray(plt.cm.tab20(np.linspace(0,1,len(paths))))[:,:3]
                palette=(palette*255).astype(np.uint8)
                palette_count=len(np.unique(palette,axis=0)) if len(paths) else 0
                result={"id":ident,"status":"ok" if count else "empty","path_count":len(paths),"rendered_id_count":count,"palette_collision":palette_count<len(paths),"acm_seconds":time.perf_counter()-started,"ground_truth_used":False}
            except Exception as exc:
                traceback.print_exc()
                result={"id":ident,"status":"failed","error":repr(exc),"acm_seconds":time.perf_counter()-started,"ground_truth_used":False}
            finally:sys.stdout,sys.stderr=oldout,olderr
        conn.send(result)
def main():
    os.chdir(WORK)
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,"reconfigure"):stream.reconfigure(encoding="utf-8",errors="replace",line_buffering=True)
    import numpy as np,torch,cv2
    from PIL import Image
    from PDSegmentor.benchmark_pgm import load_model,build_transform,postprocess
    from PDSegmentor.depthinfer import build_model
    from sam_path_v1.data import load_record
    from sam_path_v1.metrics import score_image,summarize
    torch.set_num_threads(1);cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    published=WORK/"results/sam_tracker"
    with gzip.open(published/"padlos_test1800/test_view.json.gz","rt",encoding="utf-8") as f: view=json.load(f)
    rows=view["records"];assert len(rows)==1800
    with gzip.open(published/"full_wireseg_BC/test/B/best/metrics.json.gz","rt",encoding="utf-8") as f: best=json.load(f)
    assert {str(r["id"]) for r in rows}=={str(r["id"]) for r in best["per_image"]}
    assert view["source_manifest_sha256"]==best["manifest_sha256"]
    visual_ids=json.loads((published/"padlos_test1800/protocol.json").read_text(encoding="utf-8"))["visual_ids"]
    for row in rows:
        for rel in ("rgb/"+str(row["id"])+".png", "masks/"+str(row["id"])+".npz"):
            if not (CACHE/"data"/rel).is_file(): raise FileNotFoundError(CACHE/"data"/rel)
    byid={str(r["id"]):r for r in rows}
    sources={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/"datasets/PADLOS_official").rglob("*.py"))}
    sources["cable_seg/versions/full_wireseg_20261029/sam_path_v1/metrics.py"]=sha(WORK/"cable_seg/versions/full_wireseg_20261029/sam_path_v1/metrics.py")
    weights={str(p.relative_to(WORK)):sha(p) for p in [WORK/".models/padlos/pgm.pth",WORK/".models/padlos/depth_anything_v2_vits.pth"]}
    protocol={"material_passport":{"origin_skill":"academic-research-suite/experiment-agent","mode":"run","date":"2026-10-30","verification_status":"RUNNING","version":"padlos_wireseg_test1800_v1"},"method":"released PADLOS PGM(vits)+DepthAnythingV2(vits)+Lee/ACM","source_commit":"1b951197f177202df28a16f0cb97a344355c5448","source_sha256":sources,"weights_sha256":weights,"no_training_or_tuning":True,"input_size":518,"pgm_threshold":.05,"postprocess":"threshold before native-size nearest-neighbor restoration, per released code","acm":{"w_bend":.4,"w_color":.3,"w_width":.1,"w_len":.1,"k_paths":100},"depth":"released BGR infer_image(518), per-image uint8 min/max normalization","scoring":"unchanged sam_path_v1.metrics: GT-normalized Hungarian IoU, strict IoU>0.5 PQ, native1024; overlaps ignored","failure_policy":"Every test image retained, failures score zero; no automatic retry of failed images","acm_hard_timeout_seconds":180,"workers":4,"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"manifest_sha256":view["source_manifest_sha256"],"ids":[str(r["id"]) for r in rows],"visual_ids":visual_ids,"data_cache":str(CACHE),"script_sha256":sha(__file__),"limitations":["Uses pretrained PADLOS without WireSeg fine-tuning versus B trained on 32400 WireSeg images.","Public PGM internally uses 518 whereas B pads native1024 to1036; not an equal-input-resolution architecture ablation.","Synthetic image holdout, not verified scene isolation. Real Wire300 is excluded.","Official RGB palette reconstruction retained; palette collisions logged."]}
    assert not (OUT/"protocol.json").exists(),"Existing run: never silently overwrite"
    dump(OUT/"protocol.json",protocol)
    (OUT/"run.lock").write_text(str(os.getpid()))
    device=torch.device("cuda")
    print("Loading released PADLOS checkpoints.",flush=True)
    model=load_model(WORK/".models/padlos/pgm.pth",device,"vits",1)
    depth=build_model("vits",WORK/".models/padlos/depth_anything_v2_vits.pth",device)
    transform=build_transform(518)
    ctx=mp.get_context("spawn");slots=[]
    def spawn():
        parent,child=ctx.Pipe();proc=ctx.Process(target=acm_worker,args=(child,),daemon=False);proc.start()
        return dict(process=proc,conn=parent,job=None,ready=False,started=time.monotonic())
    for _ in range(4):slots.append(spawn())
    start=time.monotonic();next_image=0;pending=[];scored=[];gpu_times=[];last_status=0
    def evaluate_result(result):
        ident=result["id"];row=byid[ident];case=CACHE/"predictions"/ident
        pgm=json.loads((case/"pgm.json").read_text())
        localrow={**row,"image":"rgb/"+ident+".png","masks":"masks/"+ident+".npz"}
        rgb,source=load_record(CACHE/"data",localrow)
        ignore=source.sum(0,dtype=np.int32)>1;gt=source&~ignore[None];gt=gt[gt.reshape(len(gt),-1).any(1)]
        failed=result["status"] not in ("ok","empty")
        if failed:
            ids=np.zeros(rgb.shape[:2],np.uint16)
            Image.fromarray(ids).save(case/"instance_ids.png")
        else:ids=np.asarray(Image.open(case/"instance_ids.png"))
        values=np.unique(ids);values=values[values!=0]
        pred=np.stack([ids==v for v in values]) if len(values) else np.zeros((0,*ids.shape),bool)
        score=score_image(pred,gt,~ignore,failed=failed)
        foreground=np.asarray(Image.open(case/"pgm_mask.png"))>0
        g=gt.any(0);valid=~ignore;u=(foreground|g)&valid
        front_iou=float(np.count_nonzero(foreground&g&valid)/np.count_nonzero(u)) if u.any() else 1.
        if pgm["status"]!="ok":front_iou=0.
        score.update(id=ident,domain=row["domain"],condition=row["source_condition_group"],pgm_union_iou=front_iou,status=result["status"],error=result.get("error"),pgm_seconds=pgm["seconds"],acm_seconds=result.get("acm_seconds"),palette_collision=result.get("palette_collision",False),path_count=result.get("path_count"),source_rgb_sha256=row["rgb_sha256"])
        dump(case/"receipt.json",{"inference":result,"pgm":pgm,"score":score})
        with (OUT/"scores.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(score,ensure_ascii=False,allow_nan=False)+"\n")
        scored.append(score)
        if len(scored)<=6 or len(scored)%50==0:print(json.dumps({"completed":len(scored),"planned":1800,"id":ident,"status":result["status"],"instance_iou":score["instance_iou"],"elapsed_seconds":round(time.monotonic()-start,1)}),flush=True)
    try:
        with torch.inference_mode():
            while len(scored)<len(rows):
                if shutil.disk_usage(CACHE).free<3*2**30:raise RuntimeError("Data drive below 3GiB free")
                for slot in slots:
                    if slot["conn"].poll():
                        msg=slot["conn"].recv()
                        if msg.get("ready"):slot["ready"]=True
                        else:evaluate_result(msg);slot["job"]=None
                    if not slot["process"].is_alive():
                        raise RuntimeError("ACM worker exited unexpectedly: "+str(slot["process"].pid))
                    if slot["job"] and time.monotonic()-slot["started"]>180:
                        ident=slot["job"]
                        # Kill only this benchmark-owned timed-out worker tree, never another task.
                        subprocess.run(["taskkill","/PID",str(slot["process"].pid),"/T","/F"],capture_output=True)
                        slot["process"].join(5)
                        evaluate_result({"id":ident,"status":"timeout","error":"ACM exceeded frozen 180s per-image budget","acm_seconds":180.})
                        slot.clear();slot.update(spawn())
                    if not slot["ready"] and time.monotonic()-slot["started"]>120:raise RuntimeError("ACM initialization timeout")
                    if slot["ready"] and slot["job"] is None and pending:
                        ident=pending.pop(0);slot["conn"].send({"id":ident,"shape":byid[ident]["shape"],"visual_ids":visual_ids})
                        slot["job"]=ident;slot["started"]=time.monotonic()
                if next_image<len(rows):
                    row=rows[next_image];ident=str(row["id"])
                    if (CACHE/"data/rgb"/(ident+".png")).is_file():
                        case=CACHE/"predictions"/ident;case.mkdir(parents=True,exist_ok=False)
                        began=time.perf_counter()
                        try:
                            image=CACHE/"data/rgb"/(ident+".png")
                            assert sha(image)==row["image_sha256"]
                            rgb=np.array(Image.open(image).convert("RGB"))
                            assert list(rgb.shape[:2])==row["shape"] and hashlib.sha256(rgb.tobytes()).hexdigest()==row["rgb_sha256"]
                            assert np.array_equal(rgb,cv2.imread(image.as_posix())[:,:,::-1])
                            tensor=torch.from_numpy(transform({"image":rgb.astype(np.float32)/255.})["image"]).unsqueeze(0).to(device).float()
                            torch.cuda.synchronize();t=time.perf_counter()
                            mask=postprocess(model(tensor),rgb.shape[:2],.05)
                            raw=depth.infer_image(np.ascontiguousarray(rgb[:,:,::-1]),518)
                            gray=((raw-raw.min())/max(float(raw.max()-raw.min()),1e-8)*255).astype(np.uint8)
                            torch.cuda.synchronize();seconds=time.perf_counter()-t
                            Image.fromarray(mask).save(case/"pgm_mask.png");Image.fromarray(gray).save(case/"depth.png")
                            dump(case/"pgm.json",{"status":"ok","seconds":seconds,"total_seconds":time.perf_counter()-began,"image_sha256":row["image_sha256"],"ground_truth_used":False,"input_size":518,"threshold":.05})
                            gpu_times.append(seconds);pending.append(ident)
                        except Exception as exc:
                            zeros=np.zeros(row["shape"],np.uint8)
                            Image.fromarray(zeros).save(case/"pgm_mask.png");Image.fromarray(zeros).save(case/"depth.png")
                            dump(case/"pgm.json",{"status":"failed","seconds":time.perf_counter()-began,"error":repr(exc),"ground_truth_used":False})
                            evaluate_result({"id":ident,"status":"pgm_failed","error":traceback.format_exc()})
                        next_image+=1
                if (OUT/"download_failure.json").exists():raise RuntimeError("Dataset download failed")
                if time.monotonic()-last_status>=10:
                    dump(OUT/"status.json",{"status":"running","pid":os.getpid(),"elapsed_seconds":time.monotonic()-start,"planned":1800,"pgm_completed":next_image,"completed":len(scored),"failures":sum(r["failed"] for r in scored),"pending_acm":len(pending),"active_acm":[s["job"] for s in slots if s["job"]],"workers":[s["process"].pid for s in slots],"gpu_mean_seconds":float(np.mean(gpu_times)) if gpu_times else None})
                    last_status=time.monotonic()
                time.sleep(.01)
        for slot in slots:slot["conn"].send(None)
        for slot in slots:slot["process"].join(15)
        assert len({r["id"] for r in scored})==1800
        report=summarize(scored)
        report.update(method="PADLOS released PGM+ACM",split="test",manifest_sha256=view["source_manifest_sha256"],per_image=scored,by_condition={c:summarize([r for r in scored if r["condition"]==c]) for c in sorted({r["condition"] for r in scored})},pgm_union_iou_macro=float(np.mean([r["pgm_union_iou"] for r in scored])),elapsed_seconds=time.monotonic()-start,gpu_mean_seconds=float(np.mean(gpu_times)) if gpu_times else None,palette_collision_images=sum(r["palette_collision"] for r in scored),protocol_sha256=sha(OUT/"protocol.json"))
        dump(OUT/"metrics.json",report)
        unchanged=all(sha(WORK/p)==h for p,h in sources.items()) and all(sha(WORK/p)==h for p,h in weights.items())
        assert unchanged
        dump(OUT/"completion.json",{"status":"completed" if not report["failures"] else "completed_with_failures","images":1800,"failures":report["failures"],"source_weights_unchanged":unchanged,"elapsed_seconds":time.monotonic()-start,"metrics_sha256":sha(OUT/"metrics.json"),"finished_at":datetime.datetime.now().isoformat()})
        dump(OUT/"status.json",{"status":"completed","completed":1800,"failures":report["failures"],"elapsed_seconds":time.monotonic()-start})
        print("COMPLETED",json.dumps({k:v for k,v in report.items() if not isinstance(v,(list,dict))}),flush=True)
    except Exception:
        dump(OUT/"failure.json",{"error":traceback.format_exc(),"completed":len(scored),"pgm_completed":next_image,"time":datetime.datetime.now().isoformat()})
        raise
if __name__=="__main__":
    mp.freeze_support();main()
