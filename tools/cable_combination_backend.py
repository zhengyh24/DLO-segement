"""Small-scale cable combinations: continuation matching and path cover.

Inference only. Existing identities and independent masks are evidence; no GT,
learned thresholds or amodal pixels are consumed/created here. Inspiration does
not imply faithful reproduction of HANDLOOM, E-SAM or DeFiNe.
"""
from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, eye
from skimage.morphology import skeletonize
from cable_evidence_fusion import UnionFind, compact, complete_visible, proposal_features


@dataclass
class Tip:
    ident: int
    xy: np.ndarray
    direction: np.ndarray
    width: float
    color: np.ndarray
    sample: np.ndarray


def identity_description(rgb, ids):
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
    tips=[];features={}
    for ident in np.unique(ids[ids>0]):
        yy,xx=np.where(ids==ident)
        x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
        mask=(ids[y0:y1,x0:x1]==ident)
        sk=skeletonize(np.pad(mask,1))
        dt=cv2.distanceTransform(np.pad(mask.astype(np.uint8),1),cv2.DIST_L2,5)
        sy,sx=np.where(sk);gy,gx=sy+y0-1,sx+x0-1
        widths=2*dt[sk];width=max(1.,float(np.median(widths)))
        degree=cv2.filter2D(sk.astype(np.uint8),cv2.CV_16U,np.ones((3,3),np.uint8))-sk
        endpoint=np.column_stack(np.where(sk & (degree==1)))
        branch=cv2.connectedComponents((sk & (degree>=3)).astype(np.uint8),8)[0]-1
        components=cv2.connectedComponents(sk.astype(np.uint8),8)[0]-1
        colors=lab[gy,gx].copy();colors[:,0]*=.35
        deviation=float(np.minimum(np.linalg.norm(colors-np.median(colors,axis=0),axis=1),50).mean()) if len(colors) else 0.
        width_cv=float(np.std(widths)/max(np.mean(widths),1)) if len(widths) else 1.
        slenderness=len(sx)/max(width,1)
        shape=min(slenderness/12.,1.)/(1+.12*branch+.25*width_cv)
        features[int(ident)]={'pixels':int(len(xx)),'spine_pixels':int(len(sx)),'width':width,
                             'shape_quality':float(shape),'color_dispersion':deviation,
                             'branches':int(branch),'components':int(components),'ends':int(len(endpoint)),
                             'spine_xy':np.column_stack([gx,gy])}
        if len(sx)<max(12,width*3):continue
        for ey,ex in endpoint:
            visited={(int(ey),int(ex))};walk=[(int(ey),int(ex))]
            cy,cx=int(ey),int(ex)
            # Walk along the actual segment. Euclidean neighbors on a loop can
            # lie on the other arm and would corrupt the tangent estimate.
            for _ in range(int(min(80,max(16,width*4)))):
                next_=[]
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        p=(cy+dy,cx+dx)
                        if (dy or dx) and p not in visited and 0<=p[0]<sk.shape[0] and 0<=p[1]<sk.shape[1] and sk[p]:next_.append(p)
                if len(next_)!=1:break
                cy,cx=next_[0];visited.add((cy,cx));walk.append((cy,cx))
            if len(walk)<6:continue
            walk=np.array(walk);global_=np.column_stack([walk[:,1]+x0-1,walk[:,0]+y0-1])
            direction=global_[-min(5,len(global_)):].mean(axis=0)-global_[0]
            norm=np.linalg.norm(direction)
            if norm<2:continue
            samples=global_[max(2,len(global_)//3):]
            color=np.median(lab[samples[:,1],samples[:,0]],axis=0);color[0]*=.35
            tips.append(Tip(int(ident),global_[0].astype(float),direction/norm,
                            max(1.,float(np.median(dt[walk[:,0],walk[:,1]])*2)),color,samples))
    return tips,features


def tip_support(p,tip):
    x0,y0,x1,y1=p.box
    pts=tip.sample;inside=(pts[:,0]>=x0)&(pts[:,0]<x1)&(pts[:,1]>=y0)&(pts[:,1]<y1)
    if inside.mean()<.8:return None
    pts=pts[inside]
    return float(p.mask[pts[:,1]-y0,pts[:,0]-x0].mean())


def signed_support(a,b,proposals):
    positive={};negative={}
    for p in proposals:
        aa,bb=tip_support(p,a),tip_support(p,b)
        if aa is None or bb is None:continue
        key=p.box  # repeated prompts in one crop do not become independent votes
        q=float(p.quality)
        if min(aa,bb)>=.65:positive[key]=max(positive.get(key,0.),q*min(aa,bb))
        elif max(aa,bb)>=.75 and min(aa,bb)<=.15:negative[key]=max(negative.get(key,0.),q*max(aa,bb))
    pos,neg=sum(positive.values()),sum(negative.values())
    return pos,neg,{'positive_crop_votes':len(positive),'negative_crop_votes':len(negative),
                    'positive':float(pos),'negative':float(neg)}


def continuation_edges(tips,proposals,gap_factor=5.,use_evidence=True):
    edges=[]
    for i,a in enumerate(tips):
        for j in range(i+1,len(tips)):
            b=tips[j]
            if a.ident==b.ident:continue
            delta=b.xy-a.xy;distance=float(np.linalg.norm(delta));width=max(a.width,b.width)
            if distance>max(12.,gap_factor*width+6):continue
            cosine=float(np.dot(a.direction,b.direction))
            if cosine>-.55:continue
            facing=0.
            if distance>max(2.,width*.6):
                toward=delta/distance
                fa,fb=float(np.dot(a.direction,toward)),float(np.dot(b.direction,-toward))
                if max(fa,fb)>-.35:continue
                facing=(2+fa+fb)/2
            color=float(np.linalg.norm(a.color-b.color))/35.
            width_delta=abs(np.log(max(a.width,1.)/max(b.width,1.)))
            if color>.75 or width_delta>1.:continue
            benefit=1.-(.35*(1+cosine)+.20*facing+.20*color+.10*width_delta+.15*distance/max(12.,gap_factor*width+6))
            support={};hard_conflict=False
            if use_evidence:
                pos,neg,support=signed_support(a,b,proposals)
                hard_conflict=neg>=1.25 and neg>pos+.75
                if hard_conflict or (pos<.25 and neg>.65):continue
                benefit+=.35*min(pos,1.5)-.35*min(neg,1.5)-.20
            if benefit>.45:
                edges.append({'i':i,'j':j,'benefit':float(benefit),'distance':distance,
                              'ids':[a.ident,b.ident],'support':support})
    return sorted(edges,key=lambda e:(-e['benefit'],e['ids'],e['i'],e['j']))


def choose_continuations(edges,n_tips,global_match=True):
    if not edges:return [],{'status':'no_candidates'}
    if not global_match:
        used=set();chosen=[]
        for e in edges:
            if e['i'] not in used and e['j'] not in used:
                chosen.append(e);used.update([e['i'],e['j']])
        return chosen,{'status':'greedy'}
    rr=[];cc=[]
    for k,e in enumerate(edges):rr.extend([e['i'],e['j']]);cc.extend([k,k])
    matrix=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(n_tips,len(edges)))
    result=milp(-np.array([e['benefit']-.45 for e in edges]),integrality=np.ones(len(edges)),
                bounds=Bounds(np.zeros(len(edges)),np.ones(len(edges))),
                constraints=LinearConstraint(matrix,np.zeros(n_tips),np.ones(n_tips)),
                options={'time_limit':3.,'mip_rel_gap':.001})
    if result.x is None:
        chosen,audit=choose_continuations(edges,n_tips,False)
        audit.update(solver_status=int(result.status),fallback=True);return chosen,audit
    return [e for e,x in zip(edges,result.x) if x>.5],{'status':'matching','solver_status':int(result.status),
                'optimal_objective':float(result.fun),'fallback':False}


def stitch_ids(rgb,visible,ids,proposals,use_evidence=True,global_match=True,only_proposal=False,selection=None):
    if only_proposal and selection!='proposal_graph':return ids.copy(),{'skipped':'stable_geometric_backend'}
    began=time.perf_counter()
    for p in proposals:
        if p.quality==0:proposal_features(p)
    tips,features=identity_description(rgb,ids)
    # Bound pathological micro-fragment scenes without truncating image geometry.
    tips=sorted(tips,key=lambda t:(-features[t.ident]['spine_pixels'],t.ident,t.xy[1],t.xy[0]))[:160]
    edges=continuation_edges(tips,proposals,use_evidence=use_evidence)
    chosen,solver=choose_continuations(edges,len(tips),global_match)
    uf=UnionFind(int(ids.max())+1)
    for e in chosen:uf.join(*e['ids'])
    lut=np.array([uf.find(i) for i in range(int(ids.max())+1)],np.uint16)
    output=compact(lut[ids]);output[~visible]=0
    return output,{'tips':len(tips),'candidate_links':len(edges),'selected_links':chosen,'solver':solver,
                   'seconds':time.perf_counter()-began,'identity_groups_before':len(features),
                   'identity_groups_after':len(np.unique(output[output>0])),
                   'foreground_changed_pixels':int(np.count_nonzero((output>0)!=(ids>0))),
                   'hidden_pixels_added':0,'ground_truth_used_for_inference':False}


def proposal_agreement(mask_ids,ident,proposals):
    by_crop={}
    for p in proposals:
        x0,y0,x1,y1=p.box
        m=mask_ids[y0:y1,x0:x1]==ident
        area=int(m.sum())
        if area<64:continue
        inter=int(np.count_nonzero(m&p.mask))
        fit=inter/max(int(np.count_nonzero(m|p.mask)),1)
        by_crop[p.box]=max(by_crop.get(p.box,0.),fit)
    return float(np.mean(list(by_crop.values()))) if by_crop else .5


def global_path_cover(rgb,visible,geometry,proposal,proposals,current,use_evidence=False):
    """Whole geometric/proposal paths compete in a sampled foreground cover.

    Every sampled spine atom must be covered. Duplicate coverage is penalized.
    Pixel competition is still performed on the full native masks.
    """
    began=time.perf_counter();candidates=[]
    for source,ids in [('geometry',geometry),('proposal',proposal)]:
        _,features=identity_description(rgb,ids)
        for ident,f in features.items():
            quality=f['shape_quality']/(1+f['color_dispersion']/25.)
            if use_evidence:quality=.65*quality+.35*proposal_agreement(ids,ident,proposals)
            candidates.append({'source':source,'ident':ident,'mask':ids==ident,'quality':float(quality),
                               'pixels':f['pixels'],'spine_pixels':f['spine_pixels']})
    if not candidates:return current.copy(),{'status':'empty'}
    if len(candidates)>96:return current.copy(),{'status':'candidate_cap_96','fallback':True}
    spine=skeletonize(visible);yy,xx=np.where(spine)
    step=max(1,int(np.ceil(len(xx)/16000)));yy,xx=yy[::step],xx[::step]
    gallery=np.stack([c['mask'][yy,xx] for c in candidates],axis=1)
    covered=gallery.any(axis=1);gallery=gallery[covered]
    if not len(gallery):return current.copy(),{'status':'no_observed_support','fallback':True}
    atoms,counts=np.unique(gallery,axis=0,return_counts=True)
    weight=counts/max(counts.sum(),1);A=csr_matrix(atoms.astype(float));n,m=len(candidates),len(atoms)
    # Penalize a fragmented explanation as well as a broad incoherent bundle.
    costs=np.array([.025+np.mean(gallery[:,i])*.70*(1-c['quality']) for i,c in enumerate(candidates)])
    objective=np.r_[costs,.5*weight]
    lower=hstack([A,csr_matrix((m,m))],format='csr')
    upper=hstack([A,-eye(m)],format='csr')
    result=milp(objective,integrality=np.r_[np.ones(n),np.zeros(m)],
        bounds=Bounds(np.zeros(n+m),np.r_[np.ones(n),np.full(m,n)]),
        constraints=[LinearConstraint(lower,np.ones(m),np.full(m,np.inf)),
                     LinearConstraint(upper,np.full(m,-np.inf),np.ones(m))],
        options={'time_limit':3.,'mip_rel_gap':.005})
    if result.x is None:return current.copy(),{'status':'solver_failed','solver_status':int(result.status),'fallback':True}
    selected=[c for c,v in zip(candidates,result.x[:n]) if v>.5]
    ids=np.zeros(visible.shape,np.uint16);best=np.full(visible.shape,-1.,np.float32)
    for ident,c in enumerate(selected,1):
        dt=cv2.distanceTransform(c['mask'].astype(np.uint8),cv2.DIST_L2,5)
        width=max(1.,float(np.median(dt[c['mask']])*2))
        score=c['quality']+.15*np.minimum(2*dt/width,1.)
        update=c['mask']&(score>best+1e-7)
        ids[update]=ident;best[update]=score[update]
    ids=complete_visible(ids,visible)
    fallback=np.count_nonzero(ids)<.99*np.count_nonzero(current)
    audit={'status':'ok','solver_status':int(result.status),'candidate_paths':n,'sampled_atoms':m,
           'selected_paths':[{k:c[k] for k in ['source','ident','quality','pixels']} for c in selected],
           'objective':float(result.fun),'use_proposal_fit':use_evidence,'seconds':time.perf_counter()-began,
           'fallback':bool(fallback),'ground_truth_used_for_inference':False,'hidden_pixels_added':0}
    return (current.copy() if fallback else compact(ids)),audit
