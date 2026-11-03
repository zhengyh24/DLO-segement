"""One frozen SAM3.1 image batch and one interactive prompt/decoder batch.

This is a stateless adapter to the *multiplex checkpoint's* interactive branch,
not the SAM3 image predictor. It does not run detection, tracking, or HQ-SAM.
Fixed-length prompt padding is intentional: SAM's -1 token is learned, not an
attention mask. Batch-dependent padding would break batch isolation.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from sam_once.types import PromptPlan, ProposalRecord


class SamInferenceError(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(repo):
    """Stable code identity for parity/capacity receipts and deployment gates."""
    files = {"sam_once/sam_adapter.py": sha256_file(__file__)}
    for relative in ("sam3/model_builder.py", "sam3/model/vl_combiner.py",
                     "sam3/model/necks.py", "sam3/model/vitdet.py",
                     "sam3/model/video_tracking_multiplex.py", "sam3/model/position_encoding.py",
                     "sam3/sam/prompt_encoder.py", "sam3/sam/mask_decoder.py",
                     "sam3/sam/transformer.py"):
        path = Path(repo) / relative
        files[relative] = sha256_file(path) if path.is_file() else None
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"source_fingerprint": hashlib.sha256(encoded).hexdigest(),
            "source_files_sha256": files}


class NativeSamAdapter:
    """No microbatches, retries, model-result feedback, or implicit resizing.

    ``model`` injection is exclusively for offline CPU contract tests. A real
    model is loaded lazily so empty/admission-rejected plans use no GPU.
    Prompt coordinates are crop-local native pixels. ``prompt_slots`` includes
    the two box corners; the native encoder appends its own final -1 token.
    """

    def __init__(self, checkpoint, repo, device, max_crops, max_prompts, *,
                 prompt_slots=8, model=None, canvas_size=1008, dtype=None,
                 interactive_only_residency=False):
        if min(max_crops, max_prompts, prompt_slots) < 1:
            raise ValueError("Positive explicit K/Q/prompt-token budgets are required")
        if model is None and canvas_size != 1008:
            raise ValueError("The native SAM3.1 canvas is fixed to 1008")
        self.checkpoint = Path(checkpoint)
        self.repo = Path(repo)
        self.device = torch.device(device)
        self.max_crops, self.max_prompts = int(max_crops), int(max_prompts)
        self.prompt_slots, self.canvas_size = int(prompt_slots), int(canvas_size)
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.model = model
        self.injected = model is not None
        self.interactive_only_residency = bool(interactive_only_residency)
        self.last_audit = None
        self._checkpoint_sha256 = None
        self._weight_receipt = None
        self._builder_log = None
        self._residency_receipt = None
        if model is not None:
            self._freeze()

    def _freeze(self):
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "parameters"):
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.tracker.interactive_sam_prompt_encoder.float()
        self.tracker.interactive_sam_mask_decoder.float()

    def _ensure_model(self):
        if self.model is not None:
            return
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Native multiplex builder requires CUDA; CPU is for injected tests only")
        if not self.checkpoint.is_file() or not (self.repo / "sam3/model_builder.py").is_file():
            raise FileNotFoundError("Explicit local SAM3.1 checkpoint and repository are required")
        self._checkpoint_sha256 = sha256_file(self.checkpoint)
        sys.path.insert(0, str(self.repo.resolve()))
        build_log = io.StringIO()
        with torch.cuda.device(self.device), redirect_stdout(build_log):
            if self.interactive_only_residency:
                self.model = self._build_interactive_only()
            else:
                from sam3.model_builder import build_sam3_multiplex_video_predictor
                predictor = build_sam3_multiplex_video_predictor(
                    checkpoint_path=str(self.checkpoint.resolve()), max_num_objects=1,
                    use_fa3=False, use_rope_real=False, compile=False, warm_up=False,
                    async_loading_frames=False)
                # Upstream enters persistent CUDA autocast contexts. Own bounded
                # contexts below instead, and exit the nested contexts in order.
                predictor.bf16_context.__exit__(None, None, None)
                predictor.model.tracker.bf16_context.__exit__(None, None, None)
                self.model = predictor.model
        log_text = build_log.getvalue()
        self._builder_log = {"characters": len(log_text), "lines": len(log_text.splitlines()),
                             "sha256": hashlib.sha256(log_text.encode("utf-8")).hexdigest()}
        # Upstream deliberately loads with strict=False. Require every parameter
        # used by this adapter to be present, so missing heads cannot run random.
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("model", checkpoint)
        keys = set()
        for key in state:
            if key.startswith("sam3_model."):
                key = "detector." + key[len("sam3_model."):]
            elif key.startswith("sam2_predictor."):
                key = "tracker." + key[len("sam2_predictor."):]
            keys.add(key)
        prefixes = ("detector.backbone.vision_backbone.",
                    "tracker.model.interactive_sam_prompt_encoder.",
                    "tracker.model.interactive_sam_mask_decoder.",
                    "tracker.model.interactivity_no_mem_embed")
        required = [name for name, _ in self.model.named_parameters() if name.startswith(prefixes)]
        missing = sorted(set(required) - keys)
        if not required or missing:
            self.model = None
            raise RuntimeError(f"SAM3.1 used-weight coverage failed; missing={missing[:12]}")
        self._weight_receipt = {**(self._weight_receipt or {}),
                               "used_parameters_found": len(required), "missing": missing}
        del checkpoint, state
        self._freeze()

    @staticmethod
    def _load_exact_cpu_state(model, state):
        """Load every retained native tensor, checking keys, shapes and values."""
        expected = model.state_dict()
        missing = sorted(set(expected) - set(state))
        if any(value.device.type != "cpu" for value in expected.values()):
            raise RuntimeError("Strict checkpoint validation must precede GPU placement")
        derived = []
        buffers, parameters = dict(model.named_buffers()), dict(model.named_parameters())
        for key in missing:
            # Native ViT registers this deterministic complex RoPE table as a
            # persistent buffer, but the published checkpoint omits it. Never
            # exempt a parameter, another buffer, or another module's same name.
            if not re.fullmatch(r"detector\.backbone\.vision_backbone\.trunk\.blocks\.\d+\.attn\.freqs_cis", key):
                continue
            owner = model.get_submodule(key.rsplit(".", 1)[0])
            if (key not in buffers or key in parameters or
                type(owner).__module__ != "sam3.model.vitdet" or type(owner).__name__ != "Attention" or
                not getattr(owner, "use_rope", False) or getattr(owner, "use_rope_real", True) or
                getattr(owner, "use_ve_rope", True) or not torch.is_complex(expected[key])):
                continue
            original = expected[key].clone()
            owner._setup_rope_freqs()
            if not torch.equal(original, owner.freqs_cis):
                raise RuntimeError(f"Native derived RoPE buffer is not reproducible: {key}")
            derived.append(key)
        unresolved = sorted(set(missing)-set(derived))
        if not expected or unresolved:
            raise RuntimeError(f"Retained native state missing checkpoint keys: {unresolved[:12]}")
        selected = {key: (expected[key] if key in derived else state[key]) for key in expected}
        model.load_state_dict(selected, strict=True)
        for key, value in model.state_dict().items():
            reference = selected[key].to(device="cpu", dtype=value.dtype)
            if not torch.equal(value, reference):
                raise RuntimeError(f"Native checkpoint value mismatch: {key}")
        return {"retained_state_tensors": len(expected), "strict_keys_shapes_values": True,
                "checkpoint_state_tensors": len(expected)-len(derived),
                "native_initialized_derived_buffers": derived,
                "derived_buffer_validation": "native Attention._setup_rope_freqs exact CPU recomputation",
                "omitted_checkpoint_tensors": len(state)-len(expected)+len(derived)}

    def _build_interactive_only(self):
        """Experimental residency; native modules and math remain unchanged.

        The upstream CPU builder still creates CUDA position caches. We retain
        and audit them; no global CUDA monkeypatch or third-party edit is used.
        """
        from sam3.model_builder import build_sam3_multiplex_video_model
        from sam3.model.vl_combiner import SAM3VLBackboneTri
        tracker = build_sam3_multiplex_video_model(
            checkpoint_path=None, load_from_HF=False, device="cpu",
            use_fa3=False, use_rope_real=False, compile=False)
        visual = tracker.backbone.vision_backbone
        tracker.backbone = None
        holder = torch.nn.Module()
        holder.detector = torch.nn.Module()
        holder.detector.backbone = SAM3VLBackboneTri(visual=visual, text=None, scalp=0)
        holder.tracker = torch.nn.Module()
        holder.tracker.model = tracker
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("model", checkpoint)
        remapped = {}
        for key, value in state.items():
            if key.startswith("sam3_model."):
                key = "detector." + key[len("sam3_model."):]
            elif key.startswith("sam2_predictor."):
                key = "tracker." + key[len("sam2_predictor."):]
            if key in remapped:
                raise RuntimeError(f"Ambiguous native checkpoint key: {key}")
            remapped[key] = value
        self._weight_receipt = self._load_exact_cpu_state(holder, remapped)
        self._residency_receipt = self._place_interactive_components(holder)
        return holder

    def _place_interactive_components(self, model):
        """Move only interactive computation and native-reference pointer heads."""
        visual = model.detector.backbone.vision_backbone
        tracker = getattr(model.tracker, "model", model.tracker)
        modules = [visual.trunk, visual.interactive_convs,
                   tracker.interactive_sam_prompt_encoder, tracker.interactive_sam_mask_decoder]
        # Native _forward_sam_heads returns object pointers even though deployment
        # does not use them. Retain that reference API without mixed-device errors.
        for name in ("interactive_obj_ptr_proj", "no_obj_ptr_linear"):
            module = getattr(tracker, name, None)
            if module is not None:
                modules.append(module)
        for module in modules:
            module.to(device=self.device)
        tracker.interactivity_no_mem_embed = torch.nn.Parameter(
            tracker.interactivity_no_mem_embed.detach().to(self.device), requires_grad=False)
        cache_rows = []
        for name, module in model.named_modules():
            cache = getattr(module, "cache", None)
            if isinstance(cache, dict):
                for key, value in cache.items():
                    if torch.is_tensor(value):
                        if module is getattr(visual, "position_encoding", None):
                            cache[key] = value = value.to(self.device)
                        cache_rows.append({"module": name, "shape_key": str(key),
                            "device": str(value.device), "bytes": value.numel()*value.element_size()})
        active = list(p for module in modules for p in module.parameters())
        active.append(tracker.interactivity_no_mem_embed)
        return {"experimental": True, "text_and_detector_heads_constructed": False,
                "native_reference_pointer_heads_resident": True,
                "used_parameter_bytes": sum(p.numel()*p.element_size() for p in active),
                "gpu_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()
                                           if p.device.type == "cuda"),
                "cpu_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()
                                           if p.device.type == "cpu"),
                "position_caches": cache_rows,
                "native_builder_position_cache_precompute_uses_cuda": True}

    @property
    def tracker(self):
        return getattr(self.model.tracker, "model", self.model.tracker)

    def _audit(self, plan):
        return {
            "status": "pending", "sam_version": "3.1_multiplex_interactive",
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": self._checkpoint_sha256,
            "repo": str(self.repo), "device": str(self.device), "torch": torch.__version__,
            "dtype": str(self.dtype), "injected_test_model": self.injected,
            "encoder_dtype": str(self.dtype), "decoder_dtype": "torch.float32",
            "precision_policy": "bf16_backbone_fp32_interactive_head" if self.dtype == torch.bfloat16
                                else "fp32_backbone_fp32_interactive_head",
            "head_tf32": False,
            "residency_policy": "experimental_interactive_only" if self.interactive_only_residency
                                else "native_full_predictor",
            **source_identity(self.repo),
            "requested_crops": len(plan.crops), "requested_prompts": len(plan.prompts),
            "max_crops": self.max_crops, "max_prompts": self.max_prompts,
            "canvas_size": self.canvas_size, "prompt_slots": self.prompt_slots,
            "prompt_padding": "fixed learned -1 tokens; native encoder appends one more",
            "multimask_output": False, "input_resize": False, "padding_rgb": 127,
            "microbatches": 0, "retries": 0, "calls": {
                "backbone_forward_image": 0, "image_encoder": 0,
                "prompt_encoder": 0, "mask_decoder": 0,
                "detector": 0, "propagation_decoder": 0, "text_encoder": 0},
        }

    def _admit(self, rgb, plan):
        plan.validate(rgb, self.canvas_size)
        if len(plan.crops) > self.max_crops or len(plan.prompts) > self.max_prompts:
            raise ValueError("Plan exceeds explicit single-batch K/Q admission budget")
        for prompt in plan.prompts:
            n = len(prompt.positive) + len(prompt.negative) + (2 if prompt.box else 0)
            if n > self.prompt_slots:
                raise ValueError(f"Prompt {prompt.prompt_id} exceeds fixed token budget {self.prompt_slots}")

    def image_batch(self, rgb, crops):
        """Identical normalization order to native image loader (fp16 storage)."""
        canvas = np.full((len(crops), self.canvas_size, self.canvas_size, 3), 127, np.uint8)
        for index, crop in enumerate(crops):
            x0, y0, x1, y1 = crop.xyxy
            canvas[index, :crop.height, :crop.width] = rgb[y0:y1, x0:x1]
        # Native loader first divides RGB/255, then stores half, then normalizes.
        batch = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        batch = batch.to(device=self.device, dtype=torch.float16)
        batch.sub_(0.5).div_(0.5)
        # The native PIL loader stacks CHW tensors into contiguous NCHW storage.
        # Preserve that layout too: NHWC strides can select different low-
        # precision convolution kernels despite identical normalized pixels.
        batch = batch.contiguous()
        return batch if self.device.type == "cuda" else batch.float()

    def point_batch(self, prompts):
        coords = torch.zeros((len(prompts), self.prompt_slots, 2), device=self.device)
        labels = torch.full((len(prompts), self.prompt_slots), -1,
                            dtype=torch.int32, device=self.device)
        for index, prompt in enumerate(prompts):
            xy = list(prompt.positive) + list(prompt.negative)
            lab = [1] * len(prompt.positive) + [0] * len(prompt.negative)
            if prompt.box is not None:
                x0, y0, x1, y1 = prompt.box
                xy.extend([(x0, y0), (x1, y1)])
                lab.extend([2, 3])
            coords[index, :len(xy)] = torch.tensor(xy, device=self.device, dtype=torch.float32)
            labels[index, :len(lab)] = torch.tensor(lab, device=self.device, dtype=torch.int32)
        return {"point_coords": coords, "point_labels": labels}

    @contextmanager
    def _instrument(self, audit):
        """Pre-hooks measure entered modules, including a call that raises OOM.

        The VL wrapper invokes vision_backbone.forward directly, bypassing hooks
        on that wrapper. Hook the actual ViT trunk and wrap forward_image instead.
        """
        backbone = self.model.detector.backbone
        tracker = self.tracker
        hooks, restores = [], []
        def hook(key, forbidden=False):
            def count(module, inputs):
                audit["calls"][key] += 1
                if forbidden:
                    raise RuntimeError(f"Forbidden SAM branch entered: {key}")
                if audit["calls"][key] > 1:
                    raise RuntimeError(f"More than one forward entered: {key}")
            return count
        def wrap_method(owner, name, key, forbidden=False):
            original = getattr(owner, name)
            had_local = name in owner.__dict__
            previous_local = owner.__dict__.get(name)
            def wrapped(*args, **kwargs):
                audit["calls"][key] += 1
                if forbidden or audit["calls"][key] > 1:
                    raise RuntimeError(f"Forbidden/repeated model call: {key}")
                return original(*args, **kwargs)
            setattr(owner, name, wrapped)
            restores.append((owner, name, had_local, previous_local))
        try:
            trunk = backbone.vision_backbone.trunk
            hooks.append(trunk.register_forward_pre_hook(hook("image_encoder")))
            hooks.append(tracker.interactive_sam_prompt_encoder.register_forward_pre_hook(hook("prompt_encoder")))
            hooks.append(tracker.interactive_sam_mask_decoder.register_forward_pre_hook(hook("mask_decoder")))
            for owner, name, key in ((tracker, "sam_mask_decoder", "propagation_decoder"),
                                     (backbone, "language_backbone", "text_encoder")):
                module = getattr(owner, name, None)
                if module is not None:
                    hooks.append(module.register_forward_pre_hook(hook(key, True)))
            wrap_method(backbone, "forward_image", "backbone_forward_image")
            if hasattr(self.model.detector, "forward_grounding"):
                wrap_method(self.model.detector, "forward_grounding", "detector", True)
            yield
        finally:
            for handle in hooks:
                handle.remove()
            for owner, name, had_local, previous_local in reversed(restores):
                if had_local:
                    setattr(owner, name, previous_local)
                else:
                    delattr(owner, name)

    def _autocast(self):
        return (torch.autocast("cuda", dtype=self.dtype) if self.device.type == "cuda"
                else nullcontext())

    @contextmanager
    def _head_precision(self):
        """Small interactive head stays true FP32, independent of batch GEMM shape."""
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast(device_type=self.device.type, enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def encode(self, images):
        """One backbone forward; includes native projection/no-memory additions."""
        tracker = self.tracker
        out = self.model.detector.backbone.forward_image(
            images, need_sam3_out=False, need_interactive_out=True, need_propagation_out=False)
        raw = [item.tensors for item in out["interactive"]["backbone_fpn"]]
        if len(raw) != 3 or not tracker.directly_add_no_mem_embed:
            raise RuntimeError("Unexpected native SAM3.1 interactive feature contract")
        with self._head_precision():
            high = [tracker.interactive_sam_mask_decoder.conv_s0(raw[0].float()),
                    tracker.interactive_sam_mask_decoder.conv_s1(raw[1].float())]
            # Native _get_interactive_pix_mem adds [1,1,C] to [(HW),B,C].
            low = raw[-1].float() + tracker.interactivity_no_mem_embed.float().reshape(1, -1, 1, 1)
        return low, high

    def _decode(self, low, high, points):
        with self._head_precision():
            return self._decode_fp32(low.float(), [x.float() for x in high], points)

    def _decode_fp32(self, low, high, points):
        tracker = self.tracker
        sparse, dense = tracker.interactive_sam_prompt_encoder(
            points=(points["point_coords"], points["point_labels"]), boxes=None, masks=None)
        masks, scores, _, object_logits = tracker.interactive_sam_mask_decoder(
            image_embeddings=low, image_pe=tracker.interactive_sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=high)
        if tracker.pred_obj_scores:
            visible = object_logits > tracker.object_score_logit_threshold
            masks = torch.where(visible[:, None, None], masks, -1024.0)
        return masks.float(), scores.float(), object_logits.float()

    def predict(self, rgb: np.ndarray, plan: PromptPlan):
        audit = self._audit(plan)
        self.last_audit = audit
        started = time.perf_counter()
        try:
            self._admit(rgb, plan)
            if not plan.prompts:
                audit.update(status="empty_prompts", seconds=time.perf_counter() - started)
                return [], audit
            self._ensure_model()
            audit["checkpoint_sha256"] = self._checkpoint_sha256
            audit["weight_coverage"] = self._weight_receipt
            audit["builder_stdout_summary"] = self._builder_log
            audit["residency"] = self._residency_receipt
            audit["device_name"] = (torch.cuda.get_device_name(self.device)
                                    if self.device.type == "cuda" else "CPU injected test")
            audit["device_total_memory_bytes"] = (torch.cuda.get_device_properties(self.device).total_memory
                                                  if self.device.type == "cuda" else 0)
            backbone = self.model.detector.backbone
            if getattr(backbone, "forward_in_chunk_for_eval", False):
                raise RuntimeError("Native backbone internal chunking must be disabled")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                audit["free_vram_before"] = torch.cuda.mem_get_info(self.device)[0]
            infer_start = time.perf_counter()
            with torch.inference_mode(), self._autocast(), self._instrument(audit):
                images = self.image_batch(rgb, plan.crops)
                low, high = self.encode(images)
                mapping = {crop.crop_id: index for index, crop in enumerate(plan.crops)}
                indices = torch.tensor([mapping[p.crop_id] for p in plan.prompts], device=self.device)
                low = low.index_select(0, indices)
                high = [value.index_select(0, indices) for value in high]
                points = self.point_batch(plan.prompts)
                masks, scores, object_logits = self._decode(low, high, points)
                full = F.interpolate(masks, (self.canvas_size, self.canvas_size),
                                     mode="bilinear", align_corners=False).cpu().numpy()
                quality = scores.cpu().numpy()
                presence = object_logits.cpu().numpy().reshape(-1)
            expected = {"backbone_forward_image": 1, "image_encoder": 1,
                        "prompt_encoder": 1, "mask_decoder": 1,
                        "detector": 0, "propagation_decoder": 0, "text_encoder": 0}
            if audit["calls"] != expected:
                raise RuntimeError(f"Single-batch instrumentation contract failed: {audit['calls']}")
            crop_map = {crop.crop_id: crop for crop in plan.crops}
            records = []
            for index, prompt in enumerate(plan.prompts):
                crop = crop_map[prompt.crop_id]
                logits = np.ascontiguousarray(full[index, 0, :crop.height, :crop.width])
                records.append(ProposalRecord(prompt, crop, logits.copy(), logits.copy(),
                    float(quality[index, 0]), {"object_score_logit": float(presence[index]),
                    "sam_version": "3.1_multiplex_interactive", "fixed_prompt_slots": self.prompt_slots}).validate())
            audit.update(status="ok", microbatches=1, inference_seconds=time.perf_counter()-infer_start)
            return records, audit
        except Exception as exc:
            audit.update(status="oom_no_retry" if isinstance(exc, torch.cuda.OutOfMemoryError)
                         else "failed_no_retry", error=f"{type(exc).__name__}: {exc}")
            raise SamInferenceError(str(exc), audit) from exc
        finally:
            # Keep identity/loading evidence even when admission to the first
            # forward fails; a loading error must not erase the computed SHA.
            audit["checkpoint_sha256"] = self._checkpoint_sha256
            audit["weight_coverage"] = self._weight_receipt
            audit["residency"] = self._residency_receipt
            audit["seconds"] = time.perf_counter() - started
            if self.device.type == "cuda" and torch.cuda.is_initialized():
                audit["peak_allocated_vram"] = torch.cuda.max_memory_allocated(self.device)
                audit["peak_reserved_vram"] = torch.cuda.max_memory_reserved(self.device)

