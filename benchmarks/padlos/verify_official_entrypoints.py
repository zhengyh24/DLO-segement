"""Two fixed successful samples verify the wrapper against released entrypoints."""
from pathlib import Path
import os,sys,json,shutil
WORK=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
CACHE=Path(os.environ["PADLOS_CACHE"])
os.environ["MPLBACKEND"]="Agg";os.environ["MPLCONFIGDIR"]=str(CACHE/"matplotlib")
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[k]="1"
sys.path.insert(0,str(WORK/"third_party/padlos_deps"));sys.path.insert(0,str(WORK/"datasets/PADLOS_official"))
import numpy as np,torch,cv2
from PIL import Image
from PDSegmentor.benchmark_pgm import run_mask_inference
from PDSegmentor.depthinfer import run_depth_inference
import PDSegmentor.benchmark_pgm as pgm_module
import PDSegmentor.depthinfer as depth_module
from instance_extractor.acm import run_instance_extraction
torch.set_num_threads(1);cv2.setNumThreads(1)
root=CACHE/"official_entrypoint_parity_device_override";root.mkdir(exist_ok=False)
# The released guard requires sm_89 to appear literally in get_arch_list;
# this Torch build executes sm_89 via compatible kernels despite that test.
# Only choose the already-exercised CUDA device; all algorithm functions stay original.
device_override={"reason":"released GPU guard rejects sm_89 absent from literal arch list; 1800 real CUDA forwards ran successfully","capability":list(torch.cuda.get_device_capability(0)),"compiled_arch_list":torch.cuda.get_arch_list(),"device":"cuda"}
pgm_module.resolve_device=lambda arg: torch.device("cuda")
depth_module.resolve_device=lambda arg: torch.device("cuda")
inputs=root/"inputs";inputs.mkdir()
ids=["008412","004137"]
for i in ids:shutil.copyfile(CACHE/"data/rgb"/(i+".png"),inputs/(i+".png"))
run_mask_inference(input_path=inputs,checkpoint=WORK/".models/padlos/pgm.pth",save_dir=root/"masks",device_arg="cuda",input_size=518,threshold=.05,encoder="vits",num_classes=1,warmup=0,limit=0,log_output=False,benchmark=False)
run_depth_inference(img_path=inputs,checkpoint=WORK/".models/padlos/depth_anything_v2_vits.pth",outdir=root/"depth",encoder="vits",device_arg="cuda",input_size=518,grayscale=True,show=False,save_npy=False,log_output=False)
run_instance_extraction(img_path=inputs,mask_dir=root/"masks",depth_dir=root/"depth",save_dir=root/"instances",limit=0)
rows=[]
for i in ids:
 case=CACHE/"predictions"/i
 mask=np.asarray(Image.open(root/"masks"/(i+"_mask.png")))
 depth=np.asarray(Image.open(root/"depth"/(i+"_depth_gray.png")))
 # Official cv2.imwrite receives its RGB-labelled array as BGR bytes.
 # cv2.imread reverses that encoding, recovering the array fed into the writer.
 official=cv2.imread(str(root/"instances"/(i+"_instances.png")))
 original=np.asarray(Image.open(case/"instances_rgb.png"))
 row={"id":i,"pgm_mask_exact":np.array_equal(mask,np.asarray(Image.open(case/"pgm_mask.png"))),"depth_gray_exact":np.array_equal(depth,np.asarray(Image.open(case/"depth.png"))),"acm_rendered_array_exact":np.array_equal(official,original)}
 rows.append(row)
receipt={"sample_ids":ids,"rows":rows,"all_exact":all(all(r[k] for k in ("pgm_mask_exact","depth_gray_exact","acm_rendered_array_exact")) for r in rows),"runtime_device_override":device_override,"scope":"Two sample algorithm parity checks using released entrypoints with CUDA device selection override; not a full test1800 rerun. Formal outputs unchanged."}
(OUT/"official_entrypoint_parity.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
print(json.dumps(receipt),flush=True)
assert receipt["all_exact"]
