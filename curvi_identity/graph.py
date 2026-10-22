"""Inference-only native graph extraction. No annotation imports or arguments."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from .config import Config

@dataclass
class Graph:
    visible: np.ndarray
    skeleton: np.ndarray
    arcs: np.ndarray
    partition: np.ndarray
    nodes: dict
    candidates: list
    audit: dict

def _tip(coords, point, pixels, width):
    walk = [tuple(point)]; seen = {tuple(point)}
    for _ in range(max(6, min(48, int(width * 3)))):
        y, x = walk[-1]
        neighbors = [(y+dy, x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                     if (dy or dx) and (y+dy,x+dx) in pixels and (y+dy,x+dx) not in seen]
        if len(neighbors) != 1:
            break
        walk.append(neighbors[0]); seen.add(neighbors[0])
    if len(walk) < 3:
        return None
    xy = np.asarray(point[::-1], dtype=float)
    inner = np.asarray(walk[-min(4,len(walk)):])[:, ::-1].mean(axis=0)
    direction = xy-inner
    norm = np.linalg.norm(direction)
    if norm < .1:
        return None
    return {'xy': xy.tolist(), 'direction': (direction/norm).tolist(), 'width': float(width)}

def build_graph(visible, config=Config()):
    visible = np.asarray(visible, dtype=bool)
    if visible.ndim != 2:
        raise ValueError('Expected native HxW foreground')
    skeleton = skeletonize(visible)
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3,3), np.uint8), mode='constant')-skeleton
    junction = skeleton & (degree > 2)
    cut = ndi.binary_dilation(junction, iterations=config.junction_radius) if config.junction_radius else junction
    raw, count = ndi.label(skeleton & ~cut, structure=np.ones((3,3)))
    arcs = np.zeros(visible.shape, np.int32); nodes = {}
    radius = ndi.distance_transform_edt(visible)
    # find_objects avoids repeatedly scanning a high-resolution image per fragment.
    for old, slices in enumerate(ndi.find_objects(raw), 1):
        if slices is None:
            continue
        yy, xx = np.where(raw[slices] == old)
        coords = np.column_stack((yy+slices[0].start,xx+slices[1].start))
        if len(coords) < config.min_arc:
            continue
        ident = len(nodes)+1; arcs[coords[:,0],coords[:,1]] = ident
        pixels = set(map(tuple, coords.tolist()))
        ends = [p for p in pixels if sum((p[0]+dy,p[1]+dx) in pixels
                for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx) == 1]
        width = max(1., float(np.median(radius[coords[:,0],coords[:,1]])*2))
        tips = [_tip(coords, p, pixels, width) for p in sorted(ends)] if len(ends) <= 2 else []
        nodes[ident] = {'coords': coords, 'tips': [t for t in tips if t is not None], 'width': width}
    # Keep unseeded foreground components as separate identities, never attach
    # them across background via an unrestricted Euclidean nearest-seed fill.
    markers = arcs.copy()
    components, _ = ndi.label(visible, structure=np.ones((3,3)))
    next_id = len(nodes)+1
    for cid,slices in enumerate(ndi.find_objects(components),1):
        m = components[slices] == cid
        if not np.any(markers[slices][m]):
            yy,xx = np.where(m); markers[yy[0]+slices[0].start,xx[0]+slices[1].start] = next_id
            next_id += 1
    partition = watershed(-radius, markers, mask=visible, connectivity=np.ones((3,3))) if visible.any() else markers
    tips = [(ident,e,t) for ident,n in nodes.items() for e,t in enumerate(n['tips'])]
    candidates = []; outside = 0
    for i,(aid,ae,a) in enumerate(tips):
        for bid,be,b in tips[i+1:]:
            if aid == bid:
                continue
            delta = np.asarray(b['xy'])-a['xy']; gap = float(np.linalg.norm(delta))
            if not 0 < gap <= config.max_gap:
                continue
            if np.max(np.abs(delta))+16 > config.small_size:
                outside += 1; continue
            center = (np.asarray(a['xy'])+b['xy'])/2
            candidates.append({'key':f'{aid}:{ae}-{bid}:{be}', 'a':aid,'b':bid,
                               'a_end':ae,'b_end':be,'a_tip':a,'b_tip':b,
                               'center':center.tolist(),'gap':gap})
    candidates.sort(key=lambda c:(c['gap'],c['key']))
    before = len(candidates); candidates = candidates[:config.max_candidates]
    if partition.max(initial=0) > 65535:
        raise ValueError('Too many native identities for uint16 output')
    return Graph(visible,skeleton,arcs,partition.astype(np.uint16),nodes,candidates,
                 {'arcs':len(nodes),'junction_pixels':int(junction.sum()),
                  'candidates_before_cap':before,'candidates':len(candidates),
                  'candidates_capped':before-len(candidates),'outside_context':outside,
                  'unseeded_components':next_id-len(nodes)-1,
                  'ground_truth_used':False,'native_shape':list(visible.shape)})

def decode(graph, scores, config=Config(), baseline=None):
    """Bidirectional rejection + endpoint matching + group conflict veto.

    Scores are (forward, reverse) for each candidate. Only identity labels
    change; hypotheses crossing background never paint foreground pixels.
    """
    scores = np.asarray(scores,dtype=float).reshape(-1,2)
    if len(scores) != len(graph.candidates) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Expected finite bidirectional probabilities for every candidate')
    if baseline is not None:
        baseline = np.asarray(baseline)
        if baseline.shape != graph.visible.shape or not np.array_equal(baseline>0,graph.visible):
            raise ValueError('Baseline must share the exact native visible union')
        if not np.issubdtype(baseline.dtype,np.integer) or baseline.min(initial=0)<0 or baseline.max(initial=0)>65535:
            raise ValueError('Baseline must contain uint16-compatible nonnegative identities')
    if not graph.candidates and baseline is not None:
        return baseline.astype(np.uint16).copy(), {'fallback':'no_candidates','selected':[], 'hidden_pixels_added':0}
    confidence = scores.min(axis=1)
    endpoints = [((c['a'],c['a_end']),(c['b'],c['b_end'])) for c in graph.candidates]
    retained=[]; reasons={}
    for i,c in enumerate(graph.candidates):
        rivals=[confidence[j] for j in range(len(confidence)) if i!=j and set(endpoints[i])&set(endpoints[j])
                and {c['a'],c['b']}!={graph.candidates[j]['a'],graph.candidates[j]['b']}]
        if confidence[i] < config.threshold:
            reasons[c['key']]='low_bidirectional_score'
        elif rivals and confidence[i]-max(rivals) < config.margin:
            reasons[c['key']]='ambiguous_alternative'
        else:
            retained.append(i)
    chosen=[]; solver='no_candidates'
    if retained:
        resources = sorted({p for i in retained for p in endpoints[i]})
        positions = {p:i for i,p in enumerate(resources)}
        matrix=lil_matrix((len(resources),len(retained)))
        for col,i in enumerate(retained):
            for ep in endpoints[i]: matrix[positions[ep],col]=1
        result=milp(-confidence[retained],integrality=np.ones(len(retained)),
                    bounds=Bounds(np.zeros(len(retained)),np.ones(len(retained))),
                    constraints=LinearConstraint(matrix.tocsr(),0,1),
                    options={'time_limit':config.matching_seconds})
        if result.x is not None:
            chosen=[i for i,v in zip(retained,result.x) if v>.5];solver=f'milp_status_{result.status}'
        else:
            used=set()
            for i in sorted(retained,key=lambda j:(-confidence[j],j)):
                if not used.intersection(endpoints[i]):chosen.append(i);used.update(endpoints[i])
            solver='greedy_fallback'
    parent=list(range(int(graph.partition.max(initial=0))+1))
    members={i:{i} for i in range(1,len(parent))}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    # Low scores concern same-ID membership, not merely which adjacent tip to use.
    # They are heuristic vetoes; model errors can over-reject true long paths.
    negative={(min(c['a'],c['b']),max(c['a'],c['b'])) for c,p in zip(graph.candidates,confidence)
              if p < config.conflict_threshold}
    selected=[];vetoed=[]
    for i in sorted(chosen,key=lambda j:(-confidence[j],j)):
        c=graph.candidates[i];a,b=find(c['a']),find(c['b'])
        if a!=b and any((min(x,y),max(x,y)) in negative for x in members[a] for y in members[b]):
            vetoed.append(c['key']);continue
        if a!=b:parent[b]=a;members[a]|=members.pop(b)
        selected.append({'key':c['key'],'a':c['a'],'b':c['b'],'score':float(confidence[i]),
                         'kind':'identity_link_only','gap':c['gap']})
    lut=np.zeros(len(parent),np.uint16); mapping={}
    for i in range(1,len(parent)):
        root=find(i);mapping.setdefault(root,len(mapping)+1);lut[i]=mapping[root]
    out=lut[graph.partition]
    if not np.array_equal(out>0,graph.visible):raise AssertionError('Visible union changed')
    return out,{'solver':solver,'selected':selected,'rejected':reasons,'transitive_vetoed':vetoed,
                'ids_before':len(np.unique(graph.partition[graph.partition>0])),
                'ids_after':len(np.unique(out[out>0])),'hidden_pixels_added':0,
                'foreground_changed_pixels':0,'ground_truth_used':False}
