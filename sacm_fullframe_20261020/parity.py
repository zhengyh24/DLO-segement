"""Trained-weight equivalence on real native crops, separate from accuracy evaluation."""
import argparse,json
from pathlib import Path
import torch
from sacm_frontend.model import CableForeground
from .model import NativeForeground
from .common import write,now
from .data import sample
from .preflight import PARENT,REAL

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(17);device=torch.device('cuda')
    ck=torch.load(PARENT,map_location='cpu',weights_only=True)
    old=CableForeground(**ck['model_config']).to(device).eval();old.load_state_dict(ck['model'])
    new=NativeForeground(**ck['model_config']).to(device).eval();new.load_state_dict(ck['model'])
    rows=json.loads(Path(REAL).read_text())['records'];receipts=[]
    selected=[next(r for r in rows if r['split']=='train' and r['dataset']==d) for d in ('c135','wire300')]
    for row in selected:
        x,_,_,_=sample(Path(REAL).parent,row,dict(crop_seed=0,flip_h=False,flip_v=False),'crop');x=x[None].to(device)
        for amp in (False,True):
            with torch.inference_mode(),torch.autocast('cuda',enabled=amp):u=old(x).float();v=new(x).float()
            diff=float((u.sigmoid()-v.sigmoid()).abs().max())
            disagreement=float(((u>=0)!=(v>=0)).float().mean())
            if not amp:torch.testing.assert_close(u,v,rtol=2e-4,atol=2e-4)
            else:assert diff<.015 and disagreement<.001,(diff,disagreement)
            receipts.append(dict(id=row['id'],amp=amp,max_logit_diff=float((u-v).abs().max()),max_probability_diff=diff,threshold05_pixel_disagreement=disagreement))
    write(a.out,dict(status='passed',time=now(),records=receipts,scope='native512 crop arithmetic witness, not full-image accuracy',
                    note='Algebra equivalent; AMP matrix kernel shapes can change rounding. Same engine is used in both arms.'))

if __name__=='__main__':main()
