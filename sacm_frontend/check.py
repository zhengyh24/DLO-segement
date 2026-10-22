"""Software checks only; random tiny encoder is never an accuracy result."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from sacm_frontend.model import CableForeground
from sacm_frontend.losses import foreground_loss

def main():
    torch.set_num_threads(2);torch.manual_seed(7)
    model=CableForeground('tiny_test',detail=True)
    x=torch.rand(1,3,35,47);masks=torch.zeros(2,35,47,dtype=torch.bool)
    masks[0,8:10,:]=True;masks[1,:,20:22]=True
    valid=torch.ones(35,47,dtype=torch.bool);logits=model(x)
    assert logits.shape==(1,1,35,47)
    loss,_=foreground_loss(logits,masks,valid,True,.1);loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.patch_embed.proj.weight.grad is None
    assert model.encoder.blocks[0].mlp.adapter.up.weight.grad.abs().sum()>0
    assert model.external[0].up.weight.grad.abs().sum()>0
    assert model.detail[0].weight.grad.abs().sum()>0
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3).step()
    empty,_=foreground_loss(logits.detach(),masks[:0],valid,True,.1)
    assert torch.isfinite(empty)
    padded=torch.zeros(1,1,40,50,requires_grad=True);padmask=torch.zeros(1,40,50,dtype=torch.bool)
    padvalid=torch.zeros(40,50,dtype=torch.bool);padvalid[:35,:47]=True
    loss,_=foreground_loss(padded,padmask,padvalid,True,.1);loss.backward()
    assert padded.grad[0,0,35:,:].abs().sum()==0
    from sacm_frontend.predict import predict_native
    import numpy as np
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.zeros(x.shape[0],1,*x.shape[-2:],device=x.device)
    prob=predict_native(Constant(),np.zeros((71,89,3),np.uint8),32,24,torch.device('cpu'))
    assert prob.shape==(71,89) and np.all(prob==.5)
    assert torch.allclose(foreground_loss(logits.detach(),masks,valid)[0],foreground_loss(logits.detach(),masks.flip(0),valid)[0])
    # Full CLI training/checkpoint/reload/inference using generated test fixtures.
    import tempfile,subprocess,hashlib
    from PIL import Image
    from sacm_frontend.data import read_manifest
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);rows=[]
        for i,split in enumerate(('train','validation')):
            rgb=np.random.default_rng(i).integers(0,256,(35,47,3),dtype=np.uint8)
            Image.fromarray(rgb).save(root/f'{i}.png');np.savez_compressed(root/f'{i}.npz',masks=masks.numpy())
            rows.append(dict(id=str(i),image=f'{i}.png',masks=f'{i}.npz',split=split,group_id=f'scene{i}',rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        manifest=dict(grouping_kind='verified_scene',independent_scene_split=True,records=rows)
        mp=root/'manifest.json';mp.write_text(json.dumps(manifest));read_manifest(mp)
        rows[1]['group_id']='scene0';mp.write_text(json.dumps(manifest))
        try:read_manifest(mp)
        except ValueError:pass
        else:raise AssertionError('Split leakage accepted')
        rows[1]['group_id']='scene1';mp.write_text(json.dumps(manifest))
        cfg=json.loads((Path(__file__).parent/'configs/D.json').read_text())
        cfg.update(backbone='tiny_test',tile=32,stride=24,epochs=1,accumulation=2)
        cp=root/'config.json';cp.write_text(json.dumps(cfg));run=root/'run'
        command=[sys.executable,str(Path(__file__).parent/'train.py'),'--manifest',str(mp),'--config',str(cp),'--out',str(run),'--device','cpu','--software-smoke']
        subprocess.run(command,check=True)
        subprocess.run(command+['--resume',str(run/'last.pt')],check=True)
        # Manufactured interrupted fixture: epoch 0 from the same two-epoch schedule.
        # Epoch count does not change epoch-0 optimizer/RNG behavior (no scheduler).
        cfg['epochs']=2;cp.write_text(json.dumps(cfg))
        state=torch.load(run/'last.pt',weights_only=True);state['config']=cfg
        torch.save(state,run/'last.pt')
        subprocess.run(command+['--resume',str(run/'last.pt')],check=True)
        direct=command.copy();direct[direct.index('--out')+1]=str(root/'uninterrupted')
        subprocess.run(direct,check=True)
        resumed=torch.load(run/'last.pt',weights_only=True)['model']
        full=torch.load(root/'uninterrupted/last.pt',weights_only=True)['model']
        assert all(torch.equal(v,full[k]) for k,v in resumed.items()),'Resume differs from uninterrupted training'
        invalid=command.copy();invalid[invalid.index('--out')+1]=str(root/'uninterrupted')
        rejected=subprocess.run(invalid+['--resume',str(run/'last.pt')],capture_output=True)
        assert rejected.returncode!=0,'Conflicting resume output was accepted'
        good=torch.load(run/'last.pt',weights_only=True)
        bad=dict(good);bad['best_epoch']=good['best_epoch']+1
        torch.save(bad,run/'last.pt')
        rejected=subprocess.run(command+['--resume',str(run/'last.pt')],capture_output=True)
        assert rejected.returncode!=0 and b'Checkpoint pair mismatch' in rejected.stderr
        torch.save(good,run/'last.pt')
        subprocess.run([sys.executable,str(Path(__file__).parent/'predict.py'),'--checkpoint',str(run/'best.pt'),'--image',str(root/'0.png'),'--out',str(root/'prediction'),'--device','cpu'],check=True)
        assert np.load(root/'prediction/probability.npy').shape==(35,47)
    out = Path(__file__).parent / 'checks'
    out.mkdir(exist_ok=True)
    (out/'software_smoke.json').write_text(json.dumps({'stage':'F2','passed':True,'accuracy_claim':False,'checks':['frozen_backbone','adapter_gradients','detail_gradients','empty_gt','padding_zero_gradient','native_tile_coverage','instance_permutation','split_leak_rejected','cli_training','checkpoint_resume_exact_vs_uninterrupted','conflicting_resume_rejected','interrupted_checkpoint_pair_rejected','rgb_only_inference']}))
    print('MODEL_LOSS_CHECK_PASS')

if __name__=='__main__':main()
