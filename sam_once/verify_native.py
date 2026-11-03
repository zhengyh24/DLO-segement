"""OFFLINE ONLY: independent native-head parity and one-batch capacity receipts.

This verifier deliberately runs SAM repeatedly for tests. It is never imported
by deployment inference. Synthetic locations are interface tests, not cable
accuracy measurements. Native reference uses the same fixed learned padding;
unpadded reference differences are reported separately as a diagnostic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch

from sam_once.sam_adapter import NativeSamAdapter, SamInferenceError
from sam_once.types import CropSpec, PromptSpec, PromptPlan


def read_plan(path, rgb):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "plan" in data:
        data = data["plan"]
    crops = [CropSpec(row["crop_id"], tuple(row["xyxy"])) for row in data["crops"]]
    prompts = [PromptSpec(row["prompt_id"], row["crop_id"],
                         tuple(tuple(p) for p in row["positive"]),
                         tuple(tuple(p) for p in row.get("negative", [])),
                         tuple(row["box"]) if row.get("box") is not None else None,
                         row.get("metadata", {})) for row in data["prompts"]]
    return PromptPlan(crops, prompts, np.zeros(rgb.shape[:2], np.float32), data.get("audit", {}))


def diagnostic_plan(rgb, crop_count, prompt_count):
    """Deterministic coordinate cases; never inspect GT or SAM predictions."""
    h, w = rgb.shape[:2]
    if min(h, w) < 8 or min(crop_count, prompt_count) < 1 or prompt_count < crop_count:
        raise ValueError("Need image >=8 pixels and prompt-count >= crop-count >=1")
    crops = []
    for index in range(crop_count):
        # Slightly varied border extents also test rectangular padding on small images.
        fraction = index / max(crop_count - 1, 1)
        x0, y0 = int(max(0, w-1008)*fraction), int(max(0, h-1008)*(1-fraction))
        x1, y1 = min(w, x0+1008), min(h, y0+1008)
        if index and w <= 1008 and h <= 1008:
            x0 = min(index, w-8)
            y0 = min(index, h-8)
        crops.append(CropSpec(f"probe_{index:03d}", (x0, y0, x1, y1)))
    prompts = []
    for index in range(prompt_count):
        crop = crops[index % crop_count]
        cw, ch = crop.width, crop.height
        if index % 3 == 0:
            positive, negative, box = ((.45*cw, .45*ch),), (), None
        elif index % 3 == 1:
            positive = ((.35*cw, .35*ch), (.65*cw, .65*ch))
            negative, box = ((.1*cw, .9*ch),), (.2*cw, .2*ch, .8*cw, .8*ch)
        else:
            positive, negative = ((1., 1.),), ((cw-2., ch-2.),)
            box = (0., 0., float(cw-1), float(ch-1))
        prompts.append(PromptSpec(f"probe_prompt_{index:03d}", crop.crop_id,
                                  positive, negative, box, {"diagnostic_only": True}))
    return PromptPlan(crops, prompts, np.zeros((h, w), np.float32),
                      {"synthetic_prompt_locations": True, "not_accuracy_evaluation": True})


def native_reference(adapter, rgb, crop, prompt, *, padded=True):
    """Fresh independent native backbone + official native interactive head.

    No adapter encode/decode reuse. Reproduces the upstream projection sequence,
    then invokes native _prepare_backbone_features, _get_interactive_pix_mem,
    and _forward_sam_heads. The reference is stateless, without video memory.
    """
    from sam3.model.io_utils import load_resource_as_video_frames
    from sam3.model.data_misc import NestedTensor
    tracker = adapter.tracker
    canvas = np.full((1008, 1008, 3), 127, np.uint8)
    x0, y0, x1, y1 = crop.xyxy
    canvas[:crop.height, :crop.width] = rgb[y0:y1, x0:x1]
    with torch.cuda.device(adapter.device), torch.inference_mode(), adapter._autocast():
        images, _, _ = load_resource_as_video_frames(
            [Image.fromarray(canvas)], image_size=1008, offload_video_to_cpu=False)
        features = adapter.model.detector.backbone.forward_image(
            images, need_sam3_out=False, need_interactive_out=True, need_propagation_out=False)
        raw = features["interactive"]["backbone_fpn"]
        with adapter._head_precision():
            features["interactive"]["backbone_fpn"] = [
                NestedTensor(tracker.interactive_sam_mask_decoder.conv_s0(raw[0].tensors.float()), None),
                NestedTensor(tracker.interactive_sam_mask_decoder.conv_s1(raw[1].tensors.float()), None),
                NestedTensor(raw[2].tensors.float(), None)]
            prepared = tracker._prepare_backbone_features(features)["interactive"]
            low = tracker._get_interactive_pix_mem(prepared["vision_feats"], prepared["feat_sizes"])
        high = [x.permute(1, 2, 0).reshape(x.size(1), x.size(2), *size)
                for x, size in zip(prepared["vision_feats"][:-1], prepared["feat_sizes"][:-1])]
        # Build reference prompts independently, including explicitly documented
        # fixed padding; this is not a variable-length legacy-click equivalence.
        coords = list(prompt.positive) + list(prompt.negative)
        labels = [1]*len(prompt.positive) + [0]*len(prompt.negative)
        if prompt.box is not None:
            bx0, by0, bx1, by1 = prompt.box
            coords.extend([(bx0, by0), (bx1, by1)])
            labels.extend([2, 3])
        if padded:
            extra = adapter.prompt_slots-len(coords)
            coords.extend([(0., 0.)]*extra)
            labels.extend([-1]*extra)
        points = {"point_coords": torch.tensor([coords], device=adapter.device, dtype=torch.float32),
                  "point_labels": torch.tensor([labels], device=adapter.device, dtype=torch.int32)}
        state = tracker.multiplex_controller.get_state(
            num_valid_entries=1, device=adapter.device, dtype=torch.float32,
            random=False, object_ids=[1])
        with adapter._head_precision():
            out = tracker._forward_sam_heads(backbone_features=low.float(), point_inputs=points,
                mask_inputs=None, interactive_high_res_features=[x.float() for x in high],
                propagation_high_res_features=None, multimask_output=False,
                multiplex_state=state, objects_to_interact=[0])
        logits = out["high_res_masks"][0, 0, :crop.height, :crop.width].float().cpu().numpy()
        return logits, float(out["ious"][0, 0].float().cpu())


def compare(a, b, atol, rtol, mask_tolerance):
    difference = np.abs(a-b)
    disagreement = float(np.mean((a > 0) != (b > 0)))
    return {"max_absolute_logit_error": float(difference.max()),
            "mean_absolute_logit_error": float(difference.mean()),
            "mask_disagreement_fraction": disagreement,
            "passed": bool(np.allclose(a, b, atol=atol, rtol=rtol)
                           and disagreement <= mask_tolerance)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="Existing RGB-only plan JSON; overrides diagnostic counts")
    parser.add_argument("--checkpoint", type=Path, default=Path("D:/sam31/weights/sam3.1_multiplex_fp16.pt"))
    parser.add_argument("--repo", type=Path, default=Path("D:/sam31/repo"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-count", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=2)
    parser.add_argument("--max-crops", type=int)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--prompt-slots", type=int, default=8)
    parser.add_argument("--interactive-only-residency", action="store_true",
                        help="Use the experimental interactive-only residency; bind capacity/parity receipts to this policy")
    parser.add_argument("--mode", choices=("parity", "capacity"), default="parity")
    parser.add_argument("--capacity-24gb", action="store_true",
                        help="Certify tested K/Q only on an actual 23-25GiB GPU in capacity mode")
    parser.add_argument("--output", type=Path, required=True, help="New JSON receipt path; never overwritten")
    parser.add_argument("--atol", type=float, default=.15)
    parser.add_argument("--rtol", type=float, default=.02)
    parser.add_argument("--mask-tolerance", type=float, default=.002)
    parser.add_argument("--unpadded-diagnostic", action="store_true",
                        help="Extra offline runs quantify learned-padding change; not a pass gate")
    args = parser.parse_args(argv)
    if args.capacity_24gb and args.mode != "capacity":
        parser.error("--capacity-24gb requires --mode capacity")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite verification receipt: {args.output}")
    receipt = {"schema": 2, "offline_verification_only": True,
               "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "status": "started", "mode": args.mode,
               "capacity_24gb_requested": args.capacity_24gb,
               "parity_scope": "fresh native interactive head, identical fixed padding and single-mask policy",
               "not_legacy_unpadded_click_equivalence": True,
               "tolerances": {"atol": args.atol, "rtol": args.rtol, "mask": args.mask_tolerance}}
    adapter = None
    try:
        if args.capacity_24gb:
            if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
                raise RuntimeError("24GB capacity verification requires an actual CUDA GPU")
            actual_memory = torch.cuda.get_device_properties(torch.device(args.device)).total_memory
            receipt["device_total_memory_bytes"] = actual_memory
            if not 23*1024**3 <= actual_memory <= 25*1024**3:
                raise RuntimeError("24GB capacity receipt requires an actual 23-25GiB GPU; no emulation claim")
        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        plan = (read_plan(args.plan, rgb) if args.plan else
                diagnostic_plan(rgb, args.crop_count, args.prompt_count))
        receipt["image"] = str(args.image.resolve())
        receipt["native_shape"] = list(rgb.shape)
        receipt["plan"] = plan.metadata()
        adapter = NativeSamAdapter(args.checkpoint, args.repo, args.device,
            args.max_crops if args.max_crops is not None else len(plan.crops),
            args.max_prompts if args.max_prompts is not None else len(plan.prompts),
            prompt_slots=args.prompt_slots, interactive_only_residency=args.interactive_only_residency)
        batch, audit = adapter.predict(rgb, plan)
        receipt["batch_audit"] = audit
        receipt["capacity"] = {"tested_K": len(plan.crops), "tested_Q": len(plan.prompts),
                               "passed": True, "verified_24gb": bool(args.capacity_24gb),
                               "device_total_memory_bytes": audit["device_total_memory_bytes"],
                               "not_a_24gb_capacity_claim": not args.capacity_24gb}
        passed = True
        if args.mode == "parity":
            reverse_plan = PromptPlan(plan.crops[::-1], plan.prompts[::-1], plan.ridge, plan.audit)
            reverse, reverse_audit = adapter.predict(rgb, reverse_plan)
            receipt["permuted_batch_audit"] = reverse_audit
            by_id = {r.prompt.prompt_id: r for r in reverse}
            receipt["permutation_isolation"] = []
            for record in batch:
                row = compare(record.initial_logits, by_id[record.prompt.prompt_id].initial_logits,
                              args.atol, args.rtol, args.mask_tolerance)
                row["prompt_id"] = record.prompt.prompt_id
                receipt["permutation_isolation"].append(row)
                passed &= row["passed"]
            receipt["independent_native_reference"] = []
            for record in batch:
                reference, score = native_reference(adapter, rgb, record.crop, record.prompt)
                row = compare(record.initial_logits, reference, args.atol, args.rtol, args.mask_tolerance)
                row.update(prompt_id=record.prompt.prompt_id,
                           score_absolute_error=abs(record.score-score))
                row["passed"] &= bool(np.isclose(record.score, score, atol=args.atol, rtol=args.rtol))
                receipt["independent_native_reference"].append(row)
                passed &= row["passed"]
                if args.unpadded_diagnostic:
                    unpadded, _ = native_reference(adapter, rgb, record.crop, record.prompt, padded=False)
                    diagnostic = compare(reference, unpadded, args.atol, args.rtol, args.mask_tolerance)
                    diagnostic.pop("passed")
                    row["unpadded_change_diagnostic_only"] = diagnostic
        receipt["status"] = "passed" if passed else "failed_numerical_parity"
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
        if isinstance(exc, SamInferenceError):
            receipt["failed_call_audit"] = exc.audit
        elif adapter is not None:
            receipt["last_audit"] = adapter.last_audit
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"status": receipt["status"], "receipt": str(args.output)}, ensure_ascii=False))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
