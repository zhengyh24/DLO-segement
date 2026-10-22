"""Small shared CNN for SAME INSTANCE scoring, not foreground segmentation."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,cin,cout,stride):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(cin,cin,3,stride,1,groups=cin,bias=False),
            nn.GroupNorm(1,cin),nn.SiLU(),nn.Conv2d(cin,cout,1,bias=False),
            nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,1,1,groups=cout,bias=False),
            nn.GroupNorm(8,cout),nn.SiLU())
    def forward(self,x):return self.layers(x)

class PairNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(7,32,3,2,1,bias=False),nn.GroupNorm(8,32),nn.SiLU(),
                                   Block(32,64,2),Block(64,96,2),Block(96,128,2))
        self.visual=nn.Sequential(nn.Linear(128*3*2,128),nn.SiLU())
        self.geometry=nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(160,64),nn.SiLU(),nn.Linear(64,1))
    def pool(self,x):
        f=self.encoder(x);parts=[f.mean((2,3))]
        for channel in (5,6):
            mask=F.adaptive_max_pool2d(x[:,channel:channel+1],f.shape[-2:])
            parts.append((f*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1))
        return torch.cat(parts,1)
    def forward(self,small,large,geometry):
        if small.ndim!=4 or large.ndim!=4 or small.shape[1]!=7 or large.shape[1]!=7 or geometry.shape!=(small.shape[0],16):
            raise ValueError('Expected Bx7xHxW native crops and Bx16 geometry')
        v=self.visual(torch.cat([self.pool(small),self.pool(large)],1))
        return self.head(torch.cat([v,self.geometry(geometry)],1)).squeeze(1)
