"""Fill only a bounded band of unassigned, already-visible foreground pixels."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def fill_observed_ids(ids, visible):
    if ids.shape!=visible.shape:raise ValueError('Geometry mismatch')
    out=ids.copy();out[~visible]=0
    count,labels,stats,_=cv2.connectedComponentsWithStats(visible.astype(np.uint8),8)
    added=0
    for component in range(1,count):
        x,y,w,h,_=map(int,stats[component])
        region=labels[y:y+h,x:x+w]==component
        local=out[y:y+h,x:x+w]
        anchors=(local>0)&region
        if not anchors.any():continue
        width_dt=cv2.distanceTransform(region.astype(np.uint8),cv2.DIST_L2,5)
        width=max(2.,2*float(np.percentile(width_dt[anchors],75)))
        distance,nearest=distance_transform_edt(~anchors,return_indices=True)
        allowed=region&(local==0)&(distance<=width)
        local[allowed]=local[nearest[0][allowed],nearest[1][allowed]]
        added+=int(allowed.sum())
    return out,{'filled_visible_pixels':added,'unresolved_visible_pixels':int(np.count_nonzero(visible&(out==0))),
                'hidden_pixels_added':0,'method':'same_foreground_component_nearest_ID_with_width_distance_cap'}
