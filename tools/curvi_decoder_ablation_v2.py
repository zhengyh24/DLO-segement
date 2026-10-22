"""Standalone inference-only frozen-score decoder ablation.

Copied decode semantics from curvi_identity.graph; never changes graph/candidates,
weights or foreground. Switches disable each stage independently. In particular,
use_margin=False removes competition entirely (margin=0 does NOT do this).
No GT annotations are accepted. Source modules remain unmodified.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curvi_identity.config import Config

def decode(graph, scores, config=Config(), *, use_margin=True, use_matching=True, use_conflict=True, baseline=None):
    """Bidirectional rejection + endpoint matching + group conflict veto.

    Scores are (forward, reverse) for each candidate. Only identity labels
    change; hypotheses crossing background never paint foreground pixels.
    """
    for flag in (use_margin, use_matching, use_conflict):
        if type(flag) is not bool:
            raise ValueError('Decoder switches must be booleans')
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
        elif use_margin and rivals and confidence[i]-max(rivals) < config.margin:
            reasons[c['key']]='ambiguous_alternative'
        else:
            retained.append(i)
    chosen=[]; solver='no_candidates'
    if retained and not use_matching:
        chosen=retained.copy();solver='matching_disabled'
    elif retained:
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
    for i in retained:
        if i not in chosen:
            reasons[graph.candidates[i]['key']]='endpoint_matching'
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
        if use_conflict and a!=b and any((min(x,y),max(x,y)) in negative for x in members[a] for y in members[b]):
            vetoed.append(c['key']);reasons[c['key']]='transitive_conflict';continue
        if a!=b:parent[b]=a;members[a]|=members.pop(b)
        selected.append({'key':c['key'],'a':c['a'],'b':c['b'],'score':float(confidence[i]),
                         'kind':'identity_link_only','gap':c['gap']})
    lut=np.zeros(len(parent),np.uint16); mapping={}
    for i in range(1,len(parent)):
        root=find(i);mapping.setdefault(root,len(mapping)+1);lut[i]=mapping[root]
    out=lut[graph.partition]
    if not np.array_equal(out>0,graph.visible):raise AssertionError('Visible union changed')
    selected_keys={r['key'] for r in selected}
    stages=[dict(key=c['key'],score=float(confidence[i]),
                 threshold_pass=bool(confidence[i]>=config.threshold),
                 retained_after_margin=i in retained,chosen_after_matching=i in chosen,
                 selected=c['key'] in selected_keys,
                 rejection_reason=reasons.get(c['key']))
            for i,c in enumerate(graph.candidates)]
    return out,{'switches':dict(margin=use_margin,endpoint_matching=use_matching,conflict_veto=use_conflict),
                'threshold':config.threshold,'margin':config.margin,'conflict_threshold':config.conflict_threshold,
                'stages':stages,'solver':solver,'selected':selected,'rejected':reasons,'transitive_vetoed':vetoed,
                'ids_before':len(np.unique(graph.partition[graph.partition>0])),
                'ids_after':len(np.unique(out[out>0])),'hidden_pixels_added':0,
                'foreground_changed_pixels':0,'ground_truth_used':False}
