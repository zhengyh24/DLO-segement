"""Observed cable identity experiments. No labels, model calls or hidden painting.

The proposal graph uses agreement in shared crop coordinates. The tracer splits
the 2-D skeleton at junctions and pairs incident segments rather than treating
every crossing as a physical connection. These are experimental implementations,
not reproductions of E-SAM, HANDLOOM or DeFiNe.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.segmentation import watershed


@dataclass
class Proposal:
    box: tuple
    mask: np.ndarray
    source: str
    score: float
    key: str
    quality: float = 0.
    width: float = 1.


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        i, j = self.find(i), self.find(j)
        if i != j:
            self.parent[max(i, j)] = min(i, j)


def compact(ids):
    values = np.unique(ids)
    values = values[values > 0]
    lut = np.zeros(int(ids.max()) + 1, np.uint16)
    lut[values] = np.arange(1, len(values) + 1, dtype=np.uint16)
    return lut[ids]


def load_cached(source, dataset, name):
    """Only saved inference evidence. HQ rows inherit their original crop box."""
    case = Path(source) / dataset
    coarse = json.loads((case / 'ours_coarse' / name / 'inference.json').read_text(encoding='utf-8'))
    refined = json.loads((case / 'ours_hq' / name / 'inference.json').read_text(encoding='utf-8'))
    visible = np.asarray(__import__('PIL.Image', fromlist=['Image']).open(case / 'ours_hq' / name / 'mask.png')) > 0
    base = np.asarray(__import__('PIL.Image', fromlist=['Image']).open(case / 'ours_mbest' / name / 'instance_ids.png')).astype(np.uint16)
    if base.shape != visible.shape:
        raise ValueError('Cached geometry mismatch')
    proposals = []
    rows = [(r, r, 'text') for r in coarse['text_proposals'] if r.get('accepted') and r.get('mask_path')]
    if len(refined['proposals']) != len(coarse['point_proposals']):
        raise ValueError('HQ proposal correspondence changed')
    rows += [(new, old, 'hq_point') for new, old in zip(refined['proposals'], coarse['point_proposals']) if new.get('accepted')]
    from PIL import Image
    for row, origin, kind in rows:
        box = tuple(map(int, origin['crop_xyxy']))
        x0, y0, x1, y1 = box
        mask = np.asarray(Image.open(row['mask_path']).convert('L')) > 0
        if mask.shape != (y1-y0, x1-x0) or not (0 <= x0 < x1 <= visible.shape[1] and 0 <= y0 < y1 <= visible.shape[0]):
            raise ValueError('Proposal crop coordinates invalid')
        mask &= visible[y0:y1, x0:x1]
        if mask.sum() < 16:
            continue
        key = hashlib.sha256(str(box).encode() + np.packbits(mask).tobytes()).hexdigest()
        score = float(row.get('score', row.get('probability', .5)))
        proposals.append(Proposal(box, mask, kind, score, key))
    return visible, base, proposals


def proposal_features(p):
    yy, xx = np.where(p.mask)
    local = p.mask[max(0, yy.min()-1):yy.max()+2, max(0, xx.min()-1):xx.max()+2]
    dt = cv2.distanceTransform(local.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(local)
    widths = dt[sk] * 2
    width = max(1., float(np.median(widths))) if len(widths) else 1.
    degree = cv2.filter2D(sk.astype(np.uint8), cv2.CV_16U, np.ones((3, 3), np.uint8))
    branches = cv2.connectedComponents(((degree >= 4) & sk).astype(np.uint8), 8)[0] - 1
    variation = float(np.std(widths) / max(float(np.mean(widths)), 1)) if len(widths) else 1.
    p.width = width
    p.quality = float(.8 * min(sk.sum() / max(width * 12, 1), 1.) / (1 + .12*branches + .25*variation) + .2*np.clip(p.score, 0, 1))


def pair_overlap(a, b):
    x0, y0 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
    x1, y1 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
    if x0 >= x1 or y0 >= y1:
        return 0., False
    aa = a.mask[y0-a.box[1]:y1-a.box[1], x0-a.box[0]:x1-a.box[0]]
    bb = b.mask[y0-b.box[1]:y1-b.box[1], x0-b.box[0]:x1-b.box[0]]
    na, nb = int(aa.sum()), int(bb.sum())
    inter = int(np.count_nonzero(aa & bb))
    overlap = inter / max(na + nb - inter, 1)
    # A handful of crossing pixels must not link two whole cables.
    significant = inter >= max(32., 3*min(a.width, b.width)**2)
    conflict = min(na, nb) >= max(64., 3*min(a.width, b.width)**2) and overlap < .2
    return overlap if significant else 0., conflict


def proposal_graph(proposals, shape, merge_iou=.55):
    for p in proposals:
        proposal_features(p)
    proposals = sorted(proposals, key=lambda p: (-p.quality, p.key))
    n = len(proposals)
    overlap = np.zeros((n, n), np.float32)
    conflict = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i+1, n):
            overlap[i, j], conflict[i, j] = pair_overlap(proposals[i], proposals[j])
            overlap[j, i], conflict[j, i] = overlap[i, j], conflict[i, j]
    uf = UnionFind(n)
    edges = sorted(((float(overlap[i, j]), i, j) for i in range(n) for j in range(i+1, n) if overlap[i, j] >= merge_iou), reverse=True)
    for _, i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        first = [k for k in range(n) if uf.find(k) == ri]
        second = [k for k in range(n) if uf.find(k) == rj]
        if not conflict[np.ix_(first, second)].any():
            uf.join(i, j)
    groups = []
    for root in sorted({uf.find(i) for i in range(n)}):
        members = [i for i in range(n) if uf.find(i) == root]
        boxes = [proposals[i].box for i in members]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        mask = np.zeros((y1-y0, x1-x0), bool)
        score = np.full(mask.shape, -1., np.float32)
        support = np.zeros_like(mask)
        for i in members:
            p = proposals[i]
            a, b, c, d = p.box
            sl = np.s_[b-y0:d-y0, a-x0:c-x0]
            mask[sl] |= p.mask
            support[sl] = True
            dt = cv2.distanceTransform(p.mask.astype(np.uint8), cv2.DIST_L2, 5)
            evidence = p.quality + .15*np.minimum(2*dt/p.width, 1.)
            np.maximum(score[sl], np.where(p.mask, evidence, -1), out=score[sl])
        groups.append(dict(box=(x0, y0, x1, y1), mask=mask, support=support, score=score,
                           quality=float(np.mean([proposals[i].quality for i in members])),
                           width=float(np.median([proposals[i].width for i in members])),
                           members=len(members), sources=[proposals[i].source for i in members]))
    groups.sort(key=lambda g: (g['box'][1], g['box'][0], -int(g['mask'].sum())))
    ids = np.zeros(shape, np.uint16)
    best = np.full(shape, -1., np.float32)
    for i, g in enumerate(groups, 1):
        x0, y0, x1, y1 = g['box']
        sl = np.s_[y0:y1, x0:x1]
        update = g['score'] > best[sl] + 1e-7
        ids[sl][update] = i
        best[sl][update] = g['score'][update]
    return ids, groups, {'proposal_count': n, 'proposal_groups': len(groups), 'merge_edges': len(edges), 'shared_crop_merge_iou': merge_iou}


def complete_visible(markers, visible):
    """Geodesic competition inside observed foreground; no Euclidean bridges."""
    if not np.any(markers):
        return markers.copy()
    distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 5)
    return watershed(-distance, markers=markers.astype(np.int32), mask=visible, connectivity=np.ones((3, 3), bool)).astype(np.uint16)


def fuse_residual(base, visible, proposal_ids, groups, merge=False):
    out = base.copy()
    out[~visible] = 0
    uf = UnionFind(int(base.max()) + 1)
    assignments, merge_events = {}, []
    for i, g in enumerate(groups, 1):
        x0, y0, x1, y1 = g['box']
        local = base[y0:y1, x0:x1]
        counts = np.bincount(local[g['mask']], minlength=int(base.max()) + 1)
        counts[0] = 0
        occupied = int(counts.sum())
        order = np.argsort(-counts)
        if occupied and counts[order[0]] / occupied >= .80:
            assignments[i] = int(order[0])
        if merge and occupied:
            candidates = [int(k) for k in order if k > 0 and counts[k] >= max(64, g['width']**2*3)]
            good = [k for k in candidates if counts[k] / max(np.count_nonzero((local == k) & g['support']), 1) >= .75]
            if len(good) >= 2 and sum(int(counts[k]) for k in good) / occupied >= .9 and g['quality'] >= .45:
                # A shared proposal must explain most of each path in its FOV.
                for k in good[1:]:
                    uf.join(good[0], k)
                assignments[i] = good[0]
                merge_events.append({'proposal_group': i, 'path_ids': good})
    lut = np.array([uf.find(i) for i in range(int(base.max()) + 1)], np.uint16)
    out = lut[out]
    next_id = int(base.max()) + 1
    new_groups = []
    for i, g in enumerate(groups, 1):
        exclusive = (proposal_ids == i) & visible & (out == 0)
        if not exclusive.any():
            continue
        target = assignments.get(i)
        if target is not None:
            out[exclusive] = uf.find(target)
        elif exclusive.sum() >= max(80, g['width']**2*5) and exclusive.sum() / max(int(g['mask'].sum()), 1) >= .45 and g['quality'] >= .35:
            out[exclusive] = next_id
            new_groups.append({'proposal_group': i, 'new_id': next_id, 'pixels': int(exclusive.sum())})
            next_id += 1
    out = complete_visible(out, visible)
    return compact(out), {'path_merges': merge_events, 'evidence_supported_new_paths': new_groups}


def best_pairs(tips, use_evidence=True):
    """Exact local matching with explicit unpaired choice (not a branch union)."""
    n = len(tips)
    cost = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            a, b = tips[i], tips[j]
            if a['edge'] == b['edge']:
                continue
            cosine = float(np.dot(a['direction'], b['direction']))
            if cosine > -.35:
                continue
            bend = (1 + cosine) / 2
            width = min(abs(np.log(max(a['width'], 1.) / max(b['width'], 1.))), 2.)
            color = min(float(np.linalg.norm(a['color'] - b['color'])) / 50., 2.)
            proposal = 0.
            if use_evidence:
                shared = set(a['signature']) & set(b['signature'])
                proposal = -max((min(a['signature'][k], b['signature'][k]) for k in shared), default=0.)
                if a['signature'] and b['signature'] and not shared:
                    proposal = .35
            cost[i, j] = cost[j, i] = .85*bend + .25*width + .20*color + .55*proposal
    # A maximum of twelve incident arms bounds matching enumeration.
    if n > 12:
        used, pairs = set(), []
        for v, i, j in sorted((cost[i, j], i, j) for i in range(n) for j in range(i+1, n)):
            if v < .8 and i not in used and j not in used:
                used.update((i, j)); pairs.append((i, j))
        return pairs
    @lru_cache(None)
    def solve(bits):
        if bits == 0:
            return 0., ()
        i = (bits & -bits).bit_length() - 1
        rest = bits & ~(1 << i)
        value, pairs = solve(rest)
        best = (value + .4, pairs)
        for j in range(i+1, n):
            if rest & (1 << j) and np.isfinite(cost[i, j]):
                value, pairs = solve(rest & ~(1 << j))
                candidate = (value + float(cost[i, j]), pairs + ((i, j),))
                if candidate[0] < best[0] - 1e-10:
                    best = candidate
        return best
    return list(solve((1 << n) - 1)[1])


def trace_segments(rgb, visible, groups=(), use_evidence=True):
    began = time.perf_counter()
    sk = skeletonize(visible)
    degree = cv2.filter2D(sk.astype(np.uint8), cv2.CV_16U, np.ones((3, 3), np.uint8)) - sk
    junctions = sk & (degree >= 3)
    junction_band = cv2.dilate(junctions.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    nj, junction_labels = cv2.connectedComponents(junction_band.astype(np.uint8), 8)
    ne, edge_labels, stats, _ = cv2.connectedComponentsWithStats((sk & ~junction_band).astype(np.uint8), 8)
    dt = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 5)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge_pixels, tips_by_junction, tips_all = {}, {}, []
    uf = UnionFind(ne)
    for e in range(1, ne):
        x, y, w, h, area = map(int, stats[e])
        yy, xx = np.where(edge_labels[y:y+h, x:x+w] == e)
        yy, xx = yy+y, xx+x
        if len(xx) < 4:
            continue
        edge_pixels[e] = (yy, xx)
        local = (edge_labels[y:y+h, x:x+w] == e).astype(np.uint8)
        local_degree = cv2.filter2D(local, cv2.CV_16U, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT) - local
        ey, ex = np.where((local > 0) & (local_degree <= 1))
        width = max(1., float(np.median(dt[yy, xx]) * 2))
        for py, px in zip(ey+y, ex+x):
            distances = np.hypot(xx-px, yy-py)
            nearby = distances <= max(12, 3*width)
            outer = nearby & (distances >= max(3, width))
            if not outer.any():
                outer = nearby & (distances >= 1)
            direction = np.array([np.mean(xx[outer])-px, np.mean(yy[outer])-py])
            if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1:
                continue
            direction /= np.linalg.norm(direction)
            color = np.median(lab[yy[outer], xx[outer]], axis=0)
            signature = {}
            if use_evidence:
                sample_y, sample_x = yy[nearby], xx[nearby]
                for gidx, g in enumerate(groups):
                    x0, y0, x1, y1 = g['box']
                    inside = (sample_x >= x0) & (sample_x < x1) & (sample_y >= y0) & (sample_y < y1)
                    if inside.sum() < max(3, len(sample_x)*.4):
                        continue
                    support = float(np.mean(g['mask'][sample_y[inside]-y0, sample_x[inside]-x0]))
                    if support >= .6:
                        signature[gidx] = support*g['quality']
            tip = dict(edge=e, xy=(int(px), int(py)), direction=direction, width=width, color=color, signature=signature)
            tips_all.append(tip)
            neighborhood = junction_labels[max(0,py-2):py+3, max(0,px-2):px+3]
            js = np.unique(neighborhood)
            js = js[js > 0]
            if len(js):
                tips_by_junction.setdefault(int(js[0]), []).append(tip)
    pair_audit = []
    for j, tips in sorted(tips_by_junction.items()):
        for a, b in best_pairs(tips, use_evidence):
            uf.join(tips[a]['edge'], tips[b]['edge'])
            pair_audit.append({'junction': j, 'edges': [tips[a]['edge'], tips[b]['edge']]})
    markers = np.zeros(visible.shape, np.uint16)
    roots = sorted({uf.find(e) for e in edge_pixels})
    root_ids = {r: i+1 for i, r in enumerate(roots)}
    for e, (yy, xx) in edge_pixels.items():
        markers[yy, xx] = root_ids[uf.find(e)]
    ids = complete_visible(markers, visible)
    return compact(ids), {'traced_segments': len(edge_pixels), 'junction_clusters': nj-1,
                          'paired_arms': pair_audit, 'use_proposal_evidence': use_evidence,
                          'seconds': time.perf_counter()-began}


def predict_methods(rgb, visible, base, proposals, methods):
    if rgb.shape[:2] != visible.shape or base.shape != visible.shape:
        raise ValueError('Native coordinate mismatch')
    pid, groups, graph_audit = None, [], {}
    if any(m not in ('fill_geodesic', 'trace_geometry') for m in methods):
        pid, groups, graph_audit = proposal_graph(proposals, visible.shape)
    for method in methods:
        began = time.perf_counter()
        audit = {}
        if method == 'fill_geodesic':
            ids = complete_visible(base, visible)
        elif method == 'proposal_graph':
            ids = complete_visible(pid, visible)
        elif method in ('hybrid_residual', 'hybrid_merge'):
            ids, audit = fuse_residual(base, visible, pid, groups, method == 'hybrid_merge')
        elif method in ('trace_geometry', 'trace_evidence'):
            ids, audit = trace_segments(rgb, visible, groups, method == 'trace_evidence')
        else:
            raise ValueError(method)
        ids[~visible] = 0
        ids = compact(ids)
        audit.update(graph=graph_audit if method not in ('fill_geodesic', 'trace_geometry') else {},
                     seconds_after_shared_preprocessing=time.perf_counter()-began,
                     assigned_visible_pixels=int(np.count_nonzero(ids)),
                     unknown_visible_pixels=int(np.count_nonzero(visible & (ids == 0))),
                     hidden_pixels_added=0, ground_truth_used_for_inference=False)
        yield method, ids, audit
