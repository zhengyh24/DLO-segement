"""Inference-only reliability gate for proposal and geometric identities."""
from __future__ import annotations
import cv2
import numpy as np
from skimage.morphology import skeletonize


def color_dispersion(rgb, ids, visible):
    """Length-weighted robust within-path Lab dispersion on the visible spine.

    Color is supporting evidence only. Uniform/similarly-colored cables can have
    equal energies under different identity assignments.
    """
    spine = skeletonize(visible)
    yy,xx=np.where(spine & (ids>0))
    if not len(xx): return 0.
    yy,xx=yy[::2],xx[::2]
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)[yy,xx].astype(np.float32)
    lab[:,0]*=.35
    labels=ids[yy,xx]
    total=0.
    for ident in np.unique(labels):
        samples=lab[labels==ident]
        center=np.median(samples,axis=0)
        residual=np.linalg.norm(samples-center,axis=1)
        total+=float(np.minimum(residual,50).sum())
    return total/max(len(xx),1)


def select_identity(rgb, visible, original_ids, completed_ids, proposal_ids, crop_count,
                    policy='coverage98'):
    coverage=float(np.count_nonzero(original_ids)/max(visible.sum(),1))
    audit={'original_visible_coverage':coverage,'threshold':.98,'policy':policy}
    choose_proposal=coverage < .98
    if policy=='coverage_color' and not choose_proposal and crop_count>1:
        geometric=color_dispersion(rgb,completed_ids,visible)
        proposal=color_dispersion(rgb,proposal_ids,visible)
        ng=len(np.unique(completed_ids[completed_ids>0]));np_=len(np.unique(proposal_ids[proposal_ids>0]))
        audit.update(geometric_color_dispersion=geometric,proposal_color_dispersion=proposal,
                     geometric_count=ng,proposal_count=np_)
        choose_proposal=(geometric >= 3 and proposal < geometric*.8 and np_ <= max(ng*1.5,ng+1))
    elif policy not in ('coverage98','coverage_color'):
        raise ValueError(policy)
    audit['selected_backend']='proposal_graph' if choose_proposal else 'fill_geodesic'
    audit['ground_truth_used_for_inference']=False
    return (proposal_ids if choose_proposal else completed_ids).copy(),audit
