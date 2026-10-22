"""Continuation needs independent geometric or SAM support, or a visible seam."""
import time
import numpy as np
from cable_combination_backend import identity_description,continuation_edges,signed_support,choose_continuations
from cable_evidence_fusion import UnionFind,compact


def geometry_signature(tip,geometry):
    pts=tip.sample;values=geometry[pts[:,1],pts[:,0]]
    observed=values>0
    if observed.mean()<.65:return None
    counts=np.bincount(values[observed]);ident=int(counts.argmax())
    return ident if counts[ident]/max(counts.sum(),1)>=.85 else None


def dual_stitch(rgb,visible,ids,proposals,geometry,agreement_only=False):
    began=time.perf_counter();tips,features=identity_description(rgb,ids)
    tips=sorted(tips,key=lambda t:(-features[t.ident]['spine_pixels'],t.ident,t.xy[1],t.xy[0]))[:160]
    geometric=[geometry_signature(t,geometry) for t in tips]
    candidates=continuation_edges(tips,proposals,use_evidence=False)
    edges=[];rejected=[]
    for e in candidates:
        a,b=tips[e['i']],tips[e['j']]
        pos,neg,support=signed_support(a,b,proposals)
        ga,gb=geometric[e['i']],geometric[e['j']]
        geometry_agrees=ga is not None and ga==gb
        sam_agrees=pos>=.60 and pos>=neg+.25
        points=np.rint(np.linspace(a.xy,b.xy,max(3,int(e['distance'])+1))).astype(int)
        corridor=float(visible[points[:,1],points[:,0]].mean())
        seam=e['distance']<=1.5*max(a.width,b.width) and corridor>=.75
        contradiction=neg>=1.25 and neg>pos+.75
        allowed=not contradiction and (geometry_agrees or sam_agrees or (seam and not agreement_only))
        audit=dict(support,geometry_ids=[ga,gb],geometry_agrees=bool(geometry_agrees),sam_agrees=bool(sam_agrees),
                   visible_corridor_fraction=corridor,visible_seam=bool(seam))
        if not allowed:
            rejected.append(dict(e,dual_support=audit));continue
        score=e['benefit']+.10*geometry_agrees+.15*sam_agrees-.10*min(neg,1.)
        if score>.45:edges.append(dict(e,benefit=float(score),dual_support=audit))
    chosen,solver=choose_continuations(edges,len(tips),True)
    uf=UnionFind(int(ids.max())+1)
    for e in chosen:uf.join(*e['ids'])
    lut=np.array([uf.find(i) for i in range(int(ids.max())+1)],np.uint16)
    out=compact(lut[ids]);out[~visible]=0
    return out,{'seconds':time.perf_counter()-began,'candidate_links':len(candidates),'accepted_candidates':len(edges),
                'selected_links':chosen,'rejected_links':rejected,'matching_solver':solver,'agreement_only':agreement_only,
                'foreground_changed_pixels':int(np.count_nonzero((out>0)!=(ids>0))),
                'ground_truth_used_for_inference':False,'hidden_pixels_added':0}
