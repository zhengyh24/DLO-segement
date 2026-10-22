"""Same learned parameters; query-chunked EXACT global attention and pad/unpad geometry.

Every query still attends to every key. This is not windowed/approximate attention.
Sources: existing sacm_frontend (R16/R29), vendor SAM-HQ encoder (R14).
"""
import types
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from sacm_frontend.model import CableForeground
from sacm_frontend.vendor.image_encoder import get_rel_pos


def chunked_attention(self, x):
    b, h, w, c = x.shape
    qkv = self.qkv(x).reshape(b, h*w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, b*self.num_heads, h*w, -1).unbind(0)
    rh = get_rel_pos(h, h, self.rel_pos_h)
    rw = get_rel_pos(w, w, self.rel_pos_w)
    pieces = []
    for start in range(0, h*w, self.query_chunk):
        end = min(start+self.query_chunk, h*w)
        positions = torch.arange(start, end, device=x.device)
        # Arguments, rather than loop closures, keep checkpoint recomputation correct.
        def attend(qq, kk, vv, hr, wr):
            a = (qq*self.scale) @ kk.transpose(-2, -1)
            bh = torch.einsum('bqc,qkc->bqk', qq, hr)
            bw = torch.einsum('bqc,qkc->bqk', qq, wr)
            a = (a.reshape(a.shape[0], a.shape[1], h, w)
                 + bh[..., :, None] + bw[..., None, :]).flatten(2)
            return a.softmax(-1) @ vv
        args = (q[:, start:end], k, v, rh[positions//w], rw[positions%w])
        z = checkpoint(attend, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else attend(*args)
        pieces.append(z)
    z = torch.cat(pieces, 1).view(b, self.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, c)
    return self.proj(z)


class NativeForeground(CableForeground):
    def __init__(self, backbone='vit_b', detail=True, recompute=True, query_chunk=128):
        super().__init__(backbone, detail, recompute)
        self.query_chunk = query_chunk
        if query_chunk:
            for block in self.encoder.blocks:
                if block.window_size == 0:
                    block.attn.query_chunk = query_chunk
                    block.attn.forward = types.MethodType(chunked_attention, block.attn)

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError('Expected B,3,H,W RGB, no image resize')
        h, w = rgb.shape[-2:]
        ph, pw = h+(-h)%16, w+(-w)%16
        x = self.encoder.patch_embed(F.pad((rgb*255-self.mean)/self.std, (0,pw-w,0,ph-h)))
        pos = F.interpolate(self.encoder.pos_embed.permute(0,3,1,2), size=x.shape[1:3], mode='bicubic', align_corners=False).permute(0,2,3,1)
        x = x+pos
        maps, scores = [], []
        for block, external, projection in zip(self.encoder.blocks, self.external, self.projections):
            x = checkpoint(block, x, use_reentrant=False) if self.training and self.recompute else block(x)
            x = x+external(x)
            scores.append(self.layer_score(x.mean((1,2))))
            maps.append(projection(x.permute(0,3,1,2)))
        weights = torch.softmax(torch.stack(scores,1),1)
        fused = sum(m*weights[:,i,:,None,None] for i,m in enumerate(maps))
        # Recover padded spatial coordinates BEFORE removing padding.
        semantic = F.interpolate(self.semantic(fused), size=(ph,pw), mode='bilinear', align_corners=False)[...,:h,:w]
        features = torch.cat([semantic,self.detail(rgb)],1) if self.has_detail else semantic
        return self.head(features)
