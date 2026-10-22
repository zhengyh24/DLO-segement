"""Implementation verification, no scientific accuracy claim."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from sacm_frontend.model import CableForeground
from sacm_residual_fusion_20261015 import add_fusion,pack,load_model,sha
def main():
 out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(20261015);torch.set_num_threads(2)
 cfg=dict(backbone='tiny_test',detail=True,recompute=False);model=CableForeground(**cfg).eval();base=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},model_config=cfg);torch.save(base,out/'base.pt');rgb=torch.rand(1,3,48,64)
 with torch.no_grad():before=model(rgb)
 add_fusion(model);model.eval()
 with torch.no_grad():after=model(rgb)
 assert torch.equal(before,after),'Parent logits changed at initialization'
 model.train();optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
 loss=torch.nn.functional.binary_cross_entropy_with_logits(model(rgb),torch.rand(1,1,48,64));loss.backward()
 grad=model.head.residual[-1].weight.grad;assert torch.isfinite(grad).all() and grad.abs().sum()>0
 optim.step();model.eval();delta=pack(model,base,out/'base.pt',sha(out/'base.pt'));delta['model_config']=dict(cfg,residual_fusion=True);torch.save(delta,out/'delta.pt');restored,_=load_model(out/'delta.pt','cpu');restored.eval()
 assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
 with torch.no_grad():assert torch.equal(model(rgb),restored(rgb))
 key=next(k for k,p in model.named_parameters() if not p.requires_grad)
 with torch.no_grad():dict(model.named_parameters())[key].add_(1)
 try:pack(model,base,out/'base.pt',sha(out/'base.pt'));raise RuntimeError('Modified frozen tensor accepted')
 except AssertionError:pass
 delta['base_sha256']='bad';torch.save(delta,out/'bad_base_delta.pt')
 try:load_model(out/'bad_base_delta.pt','cpu');raise RuntimeError('Wrong base accepted')
 except AssertionError:pass
 result=dict(initial_logits_exact=True,compact_state_exact=True,compact_prediction_exact=True,finite_nonzero_residual_gradient=True,modified_frozen_rejected=True,wrong_base_rejected=True,added_trainable_parameters=sum(p.numel() for p in model.head.residual.parameters()),scope='Tiny-model deterministic mechanics only; no segmentation accuracy inference.')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
