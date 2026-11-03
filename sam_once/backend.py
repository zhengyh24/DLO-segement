"""Fresh-evidence adapter to the frozen coverage_color identity policy.

Never disguises new proposals as old HQ caches. mBEST failure is recorded and
the existing zero-geometry/proposal selection path remains explicit.
"""
from __future__ import annotations
import time
import hashlib
import numpy as np
from .runtime import bootstrap
bootstrap()


def assign_instances(rgb, records, crop_count, geometry_predict=None):
    from cable_evidence_fusion import Proposal, predict_methods
    from cable_identity_selector import select_identity
    from refine_observed_ids import fill_observed_ids
    if geometry_predict is None:
        from mbest_backend import predict as geometry_predict
    start = time.perf_counter()
    visible = np.zeros(rgb.shape[:2], bool)
    proposals = []
    for record in records:
        record.validate()
        x0, y0, x1, y1 = record.crop.xyxy
        mask = record.current_logits > 0
        visible[y0:y1, x0:x1] |= mask
        if mask.sum() >= 16:  # same minimum proposal area as frozen cache adapter
            key = hashlib.sha256(str(record.crop.xyxy).encode() + np.packbits(mask).tobytes()).hexdigest()
            proposals.append(Proposal(record.crop.xyxy, mask.copy(), "sam31_once_refined" if record.metadata.get("refined") else "sam31_once",
                                      record.score, key))
    try:
        base, geometric_audit = geometry_predict(rgb, visible)
    except Exception as exc:
        base = np.zeros(visible.shape, np.uint16)
        geometric_audit = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                           "fallback": "zero_geometry_then_frozen_proposal_selection"}
    if base.shape != visible.shape:
        raise ValueError("Geometry backend changed native dimensions")
    base, fill = fill_observed_ids(base, visible)
    evidence = {method: (ids, audit) for method, ids, audit in
                predict_methods(rgb, visible, base, proposals, ["fill_geodesic", "proposal_graph"])}
    ids, selection = select_identity(rgb, visible, base, evidence["fill_geodesic"][0],
                                     evidence["proposal_graph"][0], crop_count, "coverage_color")
    ids[~visible] = 0
    unknown = visible & (ids == 0)
    return ids.astype(np.uint16), visible, unknown, {
        "policy": "coverage_color", "status": "empty" if not visible.any() else
        ("geometry_failed" if geometric_audit.get("status") == "failed" else "ok"),
        "geometry": geometric_audit, "visible_band_fill": fill, "selection": selection,
        "graph": evidence["proposal_graph"][1]["graph"],
        "seconds": time.perf_counter() - start, "unknown_visible_pixels": int(unknown.sum()),
        "identity_outside_visible_pixels": 0, "ground_truth_used_for_inference": False}
