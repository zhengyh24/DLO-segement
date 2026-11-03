"""Native tiled inference with one global query identity set, explicitly multi-call.

Each RGB tile is encoded once. Overlap-weighted low features and locally pooled
FPN features condition global queries. Tile decodes retain these same query IDs;
there is no GT-driven matching or independently generated instance-ID stitching.
This is an engineering expansion protocol, not single-image single-encode SAM.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import atomic_json, build_from_checkpoint, new_directory, sha256
from .data import read_manifest, manifest_root, load_record
from .metrics import score_image, summarize
from .identity import visible_identity, summarize_identity


def tile_origins(length, tile_size, stride):
    if min(length, tile_size, stride) < 1 or stride > tile_size:
        raise ValueError("Invalid tile geometry")
    result = [0]
    while result[-1] + tile_size < length:
        result.append(result[-1] + stride)
    return result


def blending_window(side):
    coordinate = (torch.arange(side, dtype=torch.float32)+.5)/side
    weights = 1-(2*coordinate-1).abs()
    return weights[:, None]*weights[None, :]


def factor_tile(factor, query_start, query_end, y, x, grid_side):
    """Materialize only a query chunk on a single tile's feature grid."""
    spatial = factor["spatial"][:, query_start:query_end, y:y+grid_side, x:x+grid_side]
    spatial = F.pad(spatial, (0, grid_side-spatial.shape[-1], 0, grid_side-spatial.shape[-2]))
    channels = factor["channels"][:, query_start:query_end]
    return channels[..., None, None] * spatial[:, :, None]


@torch.no_grad()
def predict_native(model, rgb, *, device="cuda", tile_size=1024, stride=784,
                   patch_size=14, query_chunk=4, object_threshold=.5, mask_threshold=.5):
    if model.config.backend not in ("sam31", "toy"):
        raise ValueError("Native expansion is audited for SAM3.1 automatic-query arms only")
    if model.config.num_queries != 80 and model.config.backend != "toy":
        raise ValueError("Use a trained Q80 checkpoint; never append untrained query slots")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be native uint8 HWC, with no GT inputs")
    canvas = model.config.canvas_size
    if canvas % patch_size or stride % patch_size or tile_size > canvas or query_chunk < 1:
        raise ValueError("Tiles must align to the backbone feature grid")
    if not 0 <= object_threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("Invalid confidence thresholds")
    h, w = rgb.shape[:2]
    gh, gw, grid = math.ceil(h/patch_size), math.ceil(w/patch_size), canvas//patch_size
    origins = [(y, x) for y in tile_origins(h, tile_size, stride) for x in tile_origins(w, tile_size, stride)]
    sums = [torch.zeros(1, 256, gh, gw) for _ in range(3)]
    weights = torch.zeros(gh, gw)
    pixel_weights = np.zeros((h, w), np.float32)
    blend = blending_window(tile_size)
    cache, calls = [], dict(encoder=0, decoder=0)
    encoder = model.base.trunk if model.config.backend == "sam31" else model.base.encoder
    hooks = [encoder.register_forward_pre_hook(lambda *unused: calls.__setitem__("encoder", calls["encoder"]+1))]
    if model.config.backend != "toy":
        hooks.append(model.base.decoder.register_forward_pre_hook(lambda *unused: calls.__setitem__("decoder", calls["decoder"]+1)))
    was_training = model.training
    model.eval()
    start = time.perf_counter()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    try:
        for y, x in origins:
            ih, iw = min(tile_size, h-y), min(tile_size, w-x)
            image = torch.full((1, 3, canvas, canvas), .5)
            image[0, :, :ih, :iw] = torch.from_numpy(rgb[y:y+ih, x:x+iw].transpose(2, 0, 1).copy()).float()/255.
            features = model.base.encode(image.to(device))
            if features["low"].shape[-2:] != (grid, grid):
                raise RuntimeError("Backbone feature stride differs from the frozen tile geometry")
            feature_weight = torch.zeros(1, 1, canvas, canvas)
            feature_weight[0, 0, :ih, :iw] = blend[:ih, :iw]
            feature_weight = F.adaptive_avg_pool2d(feature_weight, (grid, grid))[0, 0]
            fy, fx = y//patch_size, x//patch_size
            fh, fw = min(grid, gh-fy), min(grid, gw-fx)
            local = [features["low"]] + [F.adaptive_avg_pool2d(v, (grid, grid)) for v in features["high"]]
            for destination, value in zip(sums, local):
                destination[..., fy:fy+fh, fx:fx+fw] += value.cpu()[..., :fh, :fw]*feature_weight[:fh, :fw]
            weights[fy:fy+fh, fx:fx+fw] += feature_weight[:fh, :fw]
            pixel_weights[y:y+ih, x:x+iw] += blend[:ih, :iw].numpy()
            cache.append(dict(y=y, x=x, h=ih, w=iw,
                              features={"low": features["low"].cpu(), "high": [v.cpu() for v in features["high"]]}))
        if not bool((weights > 0).all()) or not bool((pixel_weights > 0).all()):
            raise RuntimeError("Native tile coverage has a hole")
        merged = [(value/weights[None, None]).to(device) for value in sums]
        shared = {"low": merged[0], "high": merged[1:]}
        valid = (weights > 0)[None].to(device)
        queries, condition_factor, object_logits = model.condition.forward_factors(shared["low"], valid)
        factors, tokens, trace = [condition_factor], queries[:, :, None], {}
        if model.tracker is not None:
            descriptor = shared["low"] if model.descriptor is None else model.descriptor(shared)[0]
            path_tokens, path_factor, trace = model.tracker(descriptor, queries, valid, factorized=True)
            factors.append(path_factor)
            tokens = torch.cat((tokens, path_tokens), 2)
        confidence = object_logits.sigmoid()[0].cpu().numpy()
        best_scores = np.full((h, w), -1., np.float32)
        global_ids = np.zeros((h, w), np.int32)
        query_count = model.config.num_queries
        # Chunk outer loop bounds native logit storage to chunk*H*W, not Q*H*W.
        for begin in range(0, query_count, query_chunk):
            end = min(begin+query_chunk, query_count)
            numerator = np.zeros((end-begin, h, w), np.float32)
            for tile in cache:
                y, x, ih, iw = (tile[k] for k in ("y", "x", "h", "w"))
                features = {"low": tile["features"]["low"].to(device),
                            "high": [v.to(device) for v in tile["features"]["high"]]}
                dense = sum(factor_tile(factor, begin, end, y//patch_size, x//patch_size, grid) for factor in factors)
                logits, _, _ = model.base.decode(features, tokens[:, begin:end], dense)
                if model.config.backend == "toy":
                    calls["decoder"] += 1
                native = F.interpolate(logits.float(), (canvas, canvas), mode="bilinear", align_corners=False)[0, :, :ih, :iw].cpu().numpy()
                numerator[:, y:y+ih, x:x+iw] += native*blend[:ih, :iw].numpy()
            probability = torch.from_numpy(numerator/pixel_weights[None]).sigmoid().numpy()
            eligible = (probability >= mask_threshold) & (confidence[begin:end, None, None] >= object_threshold)
            score = np.where(eligible, probability*confidence[begin:end, None, None], -1.)
            winner = score.argmax(0)
            value = np.take_along_axis(score, winner[None], axis=0)[0]
            update = value > best_scores  # equal scores retain the smaller global query
            best_scores[update] = value[update]
            global_ids[update] = (winner[update]+begin+1)
        expected = {"encoder": len(cache), "decoder": len(cache)*math.ceil(query_count/query_chunk)}
        if calls != expected:
            raise RuntimeError(f"Native expansion actual call mismatch: {calls} != {expected}")
        retained = np.unique(global_ids)
        retained = retained[retained > 0]
        masks = (global_ids[None] == retained[:, None, None]) if len(retained) else np.zeros((0, h, w), bool)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        return dict(masks=masks, instances=global_ids, queries=(retained-1).tolist(),
            confidences=confidence[retained-1].tolist(), calls=calls, native_size=[h, w],
            encoded_views=len(cache), processed_encoder_pixels=len(cache)*canvas*canvas,
            geometry_transform="native aligned tiles; no resize; common global query identities",
            high_feature_policy="each tile cached on CPU; locally pooled high features blended for global descriptor",
            path_positions=trace["path_positions"][0].cpu().tolist() if trace else [],
            seconds=time.perf_counter()-start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.device(device).type == "cuda" else None)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root"); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--split", default="test")
    parser.add_argument("--sam-repo"); parser.add_argument("--sam-checkpoint")
    args = parser.parse_args(argv)
    root = new_directory(args.output); (root / "predictions").mkdir()
    model, payload = build_from_checkpoint(args.checkpoint, device=args.device,
        sam_repo=args.sam_repo, sam_checkpoint=args.sam_checkpoint)
    manifest = read_manifest(args.manifest, allow_condition=True)
    data_root = manifest_root(args.manifest, manifest, args.data_root)
    selected = [row for row in manifest["records"] if row["split"] == args.split]
    if not selected:
        raise ValueError("Empty native evaluation split")
    rows = []
    for index, row in enumerate(selected):
        masks = None
        identity, diagnostic_error = None, None
        sample_start = time.perf_counter()
        try:
            rgb, source_masks = load_record(data_root, row)
            valid = source_masks.sum(0) <= 1
            masks = source_masks & valid[None]
            masks = masks[masks.reshape(len(masks), rgb.shape[0]*rgb.shape[1]).any(1)]
            prediction = predict_native(model, rgb, device=args.device,
                object_threshold=payload["train_config"]["object_threshold"],
                mask_threshold=payload["train_config"]["mask_threshold"])
            score = score_image(prediction["masks"], masks, valid)
            try:
                identity = visible_identity(prediction["masks"], masks, valid)
            except Exception as exc:
                diagnostic_error = f"{type(exc).__name__}: {exc}"
            name = f"{index:04d}.npz"
            np.savez_compressed(root / "predictions" / name, instances=prediction["instances"])
            score.update(prediction={key:value for key,value in prediction.items() if key not in ("masks", "instances")},
                         prediction_file=f"predictions/{name}", error=None)
        except Exception as exc:
            if masks is None:
                score = dict(instance_iou=0., union_iou=0., tp=0, fp=0, fn=0, matched_iou=0.,
                             n_pred=0, n_gt=None, empty_prediction=True, failed=True, gt_available=False)
            else:
                empty = np.zeros((0, *masks.shape[1:]), bool)
                score = score_image(empty, masks, valid, failed=True)
                try:
                    identity = visible_identity(empty, masks, valid, failed=True)
                except Exception as diagnostic_exc:
                    diagnostic_error = f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
            score["error"] = f"{type(exc).__name__}: {exc}"
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        score.update(id=row["id"], native_size=row["shape"], domain=row.get("domain", "unspecified"),
                     identity=identity, diagnostic_error=diagnostic_error,
                     evaluation_seconds=time.perf_counter()-sample_start)
        rows.append(score)
        atomic_json(root / "progress.json", dict(images=len(rows), planned=len(selected), last=score))
    report = summarize(rows)
    report["identity_diagnostics"] = summarize_identity(rows)
    report["diagnostic_failures"] = sum(row["diagnostic_error"] is not None for row in rows)
    report.update(per_image=rows, protocol="native_tiles_global_queries_q80", tile_size=1024, stride=784,
        query_chunk=4, checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest),
        no_single_pass_claim=True, no_hidden_resize=True, no_gt_forward_inputs=True,
        timing_note="prediction.seconds covers native inference; evaluation_seconds also includes file/GT IO, scoring and artifact storage")
    atomic_json(root / "metrics.json", report)
    clean = report["complete"] and report["failures"] == 0 and report["diagnostic_failures"] == 0
    atomic_json(root / "completion.json", dict(status="completed" if clean else "completed_with_failures",
                                               images=len(rows), failures=report["failures"],
                                               diagnostic_failures=report["diagnostic_failures"]))
    print(json.dumps({key:value for key,value in report.items() if key != "per_image"}))
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
