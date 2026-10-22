"""R16 dual-adapter inspiration with custom decoder; NOT official SACM."""
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .vendor.image_encoder import ImageEncoderViT

class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down=nn.Linear(dim,max(4,int(dim*.1)))
        self.up=nn.Linear(max(4,int(dim*.1)),dim)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):return self.up(F.gelu(self.down(x)))

class InternalMLP(nn.Module):
    def __init__(self,base,dim):
        super().__init__();self.base=base;self.adapter=Adapter(dim)
    def forward(self,x):return self.base(x)+self.adapter(x)

class CableForeground(nn.Module):
    def __init__(self,backbone='vit_b',detail=False,recompute=True):
        super().__init__()
        dim,depth,heads,indices={'vit_b':(768,12,12,(2,5,8,11)),
            'vit_l':(1024,24,16,(5,11,17,23)),'tiny_test':(32,2,4,(1,))}[backbone]
        self.backbone,self.has_detail,self.recompute=backbone,detail,recompute
        self.encoder=ImageEncoderViT(img_size=1024,embed_dim=dim,depth=depth,num_heads=heads,
            out_chans=256,norm_layer=partial(nn.LayerNorm,eps=1e-6),use_rel_pos=True,
            window_size=14,global_attn_indexes=indices)
        self.encoder.requires_grad_(False)
        for block in self.encoder.blocks:block.mlp=InternalMLP(block.mlp,dim)
        self.external=nn.ModuleList([Adapter(dim) for _ in range(depth)])
        self.layer_score=nn.Linear(dim,1)
        self.projections=nn.ModuleList([nn.Conv2d(dim,32,1) for _ in range(depth)])
        self.semantic=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,16,3,padding=1),nn.GELU())
        if detail:self.detail=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),nn.Conv2d(16,16,3,padding=1),nn.GELU())
        self.head=nn.Conv2d(32 if detail else 16,1,1)
        self.register_buffer('mean',torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))
    def load_encoder(self,path):
        state=torch.load(path,map_location='cpu',weights_only=True)
        state=state.get('model',state.get('state_dict',state))
        state={k[len('image_encoder.'):]:v for k,v in state.items() if k.startswith('image_encoder.')}
        if not state:raise ValueError('Expected SAM/HQ image_encoder.* checkpoint')
        mapped={k.replace('.mlp.','.mlp.base.'):v for k,v in state.items()}
        missing,extra=self.encoder.load_state_dict(mapped,strict=False)
        missing=[k for k in missing if '.mlp.adapter.' not in k]
        if missing or extra:raise ValueError(f'Encoder mismatch: {missing}, {extra}')
        return {'loaded_encoder_tensors':len(mapped),'source':str(path),'note':'Encoder only; not official SACM weights'}
    def forward(self,rgb):
        if rgb.ndim!=4 or rgb.shape[1]!=3:raise ValueError('Expected B,3,H,W RGB in [0,1]')
        h,w=rgb.shape[-2:];x=(rgb*255-self.mean)/self.std
        x=self.encoder.patch_embed(F.pad(x,(0,(-w)%16,0,(-h)%16)))
        pos=F.interpolate(self.encoder.pos_embed.permute(0,3,1,2),size=x.shape[1:3],mode='bicubic',align_corners=False).permute(0,2,3,1)
        x=x+pos;maps=[];scores=[]
        for block,external,projection in zip(self.encoder.blocks,self.external,self.projections):
            x=checkpoint(block,x,use_reentrant=False) if self.training and self.recompute else block(x)
            x=x+external(x);scores.append(self.layer_score(x.mean((1,2))))
            maps.append(projection(x.permute(0,3,1,2)))
        weights=torch.softmax(torch.stack(scores,1),1)
        fused=sum(m*weights[:,i,:,None,None] for i,m in enumerate(maps))
        semantic=F.interpolate(self.semantic(fused),size=(h,w),mode='bilinear',align_corners=False)
        features=torch.cat([semantic,self.detail(rgb)],1) if self.has_detail else semantic
        return self.head(features)
