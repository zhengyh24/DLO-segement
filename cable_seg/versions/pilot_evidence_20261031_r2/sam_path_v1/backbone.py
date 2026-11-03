"""Tensor-only SAM3.1 interactive branch; never calls a video predictor.

The official ViT, interactive FPN convolutions and mask decoder are retained.
For 1024-pixel data, a 1036 canvas pads 12 pixels, rather than resizing RGB.
RoPE is generated for the configured patch grid by the official constructor.
This resolution adaptation is intentional, not a claim of 1008-input parity.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Sam31Backbone(nn.Module):
    """Frozen encoder + trainable original decoder, with exact weight admission."""

    def __init__(self, repo, checkpoint, canvas_size=1036, encoder_amp="bf16"):
        super().__init__()
        if canvas_size < 28 or canvas_size % 14:
            raise ValueError("SAM3.1 canvas_size must be a positive multiple of patch size 14")
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if not (repo / "sam3/model/vitdet.py").is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Supply --sam-repo and the SAM3.1 multiplex --sam-checkpoint")
        sys.path.insert(0, str(repo))
        from sam3.model.vitdet import ViT
        from sam3.model.necks import Sam3TriViTDetNeck
        from sam3.sam.mask_decoder import MaskDecoder
        from sam3.sam.prompt_encoder import PromptEncoder
        from sam3.sam.transformer import TwoWayTransformer

        # Guard against a previously imported different SAM installation.
        import sam3.model.vitdet as vit_module
        if Path(vit_module.__file__).resolve() != repo / "sam3/model/vitdet.py":
            raise RuntimeError("Another sam3 installation was imported; start a fresh Python process")
        self.canvas_size, self.encoder_amp = int(canvas_size), encoder_amp
        self.trunk = ViT(
            img_size=canvas_size, pretrain_img_size=336, patch_size=14,
            embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625,
            norm_layer="LayerNorm", drop_path_rate=0.1, qkv_bias=True,
            use_abs_pos=True, tile_abs_pos=True, global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(), use_rope=True, use_interp_rope=True, window_size=24,
            pretrain_use_cls_token=True, retain_cls_token=False, ln_pre=True,
            ln_post=False, return_interm_layers=False, bias_patch_embed=False,
            compile_mode=None, use_fa3=False, use_rope_real=False,
        )
        # PositionEmbeddingSine is unused by this branch: the decoder uses its
        # pretrained PromptEncoder's random PE. Avoid unused CUDA caches/heads.
        neck = Sam3TriViTDetNeck(self.trunk, None, 256, scale_factors=[4., 2., 1.])
        self.neck = neck.interactive_convs
        del neck
        grid = canvas_size // 14
        self.prompt_encoder = PromptEncoder(256, (grid, grid), (canvas_size, canvas_size), 16)
        self.decoder = MaskDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            num_multimask_outputs=3, iou_head_depth=3, iou_head_hidden_dim=256,
            use_high_res_features=True, iou_prediction_use_sigmoid=False,
            pred_obj_scores=True, pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True, dynamic_multimask_via_stability=False,
        )
        self.no_mem = nn.Parameter(torch.zeros(1, 1, 256), requires_grad=False)
        self.register_buffer("pixel_mean", torch.tensor([.5, .5, .5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([.5, .5, .5]).view(1, 3, 1, 1), persistent=False)
        self.receipt = self._load_checkpoint(checkpoint)
        self.receipt.update(repo=str(repo), checkpoint=str(checkpoint),
                            checkpoint_sha256=file_sha256(checkpoint), canvas_size=canvas_size,
                            patch_grid=grid, native_rgb_resize=False,
                            resolution_adaptation="official ViT RoPE regenerated; RGB only right/bottom padded",
                            source_sha256={name: file_sha256(repo / name) for name in (
                                "sam3/model/vitdet.py", "sam3/model/necks.py",
                                "sam3/sam/mask_decoder.py", "sam3/sam/prompt_encoder.py",
                                "sam3/sam/transformer.py")})
        for module in (self.trunk, self.neck, self.prompt_encoder):
            module.requires_grad_(False)
            module.eval()

    def _load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        source = payload.get("model", payload)
        normalized = {}
        for key, value in source.items():
            if key.startswith("sam3_model."):
                key = "detector." + key[len("sam3_model."):]
            if key.startswith("sam2_predictor."):
                key = "tracker." + key[len("sam2_predictor."):]
            if key in normalized:
                raise ValueError(f"Duplicate normalized checkpoint key: {key}")
            normalized[key] = value
        prefixes = {
            "trunk.": "detector.backbone.vision_backbone.trunk.",
            "neck.": "detector.backbone.vision_backbone.interactive_convs.",
            "prompt_encoder.": "tracker.model.interactive_sam_prompt_encoder.",
            "decoder.": "tracker.model.interactive_sam_mask_decoder.",
        }
        selected, derived, rope_audit = {}, [], []
        parameters = dict(self.named_parameters())
        for name, expected in self.state_dict().items():
            # RoPE is deterministic geometry, not a learned checkpoint weight.
            if name.startswith("trunk.blocks.") and name.endswith(".attn.freqs_cis"):
                if name in parameters or not expected.is_complex():
                    raise RuntimeError(f"Unexpected RoPE buffer: {name}")
                owner = self.get_submodule(name.rsplit(".", 1)[0])
                before = expected.detach().clone()
                owner._setup_rope_freqs()
                regenerated = owner.freqs_cis
                if (regenerated.shape != before.shape or regenerated.dtype != before.dtype
                        or not torch.equal(regenerated, before)):
                    raise RuntimeError(f"Native RoPE recomputation mismatch: {name}")
                source_key = prefixes["trunk."] + name[len("trunk."):]
                stored = normalized.get(source_key)
                rope_audit.append({
                    "name": name, "shape": list(regenerated.shape), "dtype": str(regenerated.dtype),
                    "sha256": hashlib.sha256(regenerated.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                    "input_size": list(owner.input_size), "pretraining_size": list(owner.rope_pt_size),
                    "native_recompute_exact": True,
                    "checkpoint_tensor_present": stored is not None,
                    "checkpoint_shape": list(stored.shape) if stored is not None else None,
                    "checkpoint_dtype": str(stored.dtype) if stored is not None else None,
                    "checkpoint_buffer_ignored": stored is not None,
                })
                selected[name] = regenerated
                derived.append(name)
                continue
            key = "tracker.model.interactivity_no_mem_embed" if name == "no_mem" else None
            for local, remote in prefixes.items():
                if name.startswith(local):
                    key = remote + name[len(local):]
                    break
            if key is None or key not in normalized:
                raise RuntimeError(f"Required pretrained tensor missing: {name} / {key}")
            if normalized[key].shape != expected.shape:
                raise RuntimeError(f"Pretrained shape mismatch: {name}")
            selected[name] = normalized[key]
        self.load_state_dict(selected, strict=True)
        return {"strict_retained_state": True, "loaded_tensor_count": len(selected)-len(derived),
                "regenerated_rope_buffers": derived, "rope_buffer_audit": rope_audit,
                "unused_checkpoint_tensors_not_loaded": True}

    def train(self, mode=True):
        super().train(mode)
        self.trunk.eval(); self.neck.eval(); self.prompt_encoder.eval()
        return self

    def encode(self, images):
        if images.shape[-2:] != (self.canvas_size, self.canvas_size):
            raise ValueError("Input must already be padded to configured canvas; no implicit resize")
        use_amp = images.device.type == "cuda" and self.encoder_amp == "bf16"
        context = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        # no_grad is safe only around the frozen encoder, never around decode.
        with torch.no_grad(), context:
            x = self.trunk(((images - self.pixel_mean) / self.pixel_std).contiguous())[-1]
            x = getattr(x, "tensors", x)
            features = [conv(x).float() for conv in self.neck]
        return {"low": features[-1] + self.no_mem.reshape(1, 256, 1, 1), "high": features[:2]}

    def decode(self, features, tokens, guidance):
        b, q, n, c = tokens.shape
        indices = torch.arange(b, device=tokens.device).repeat_interleave(q)
        low = features["low"].index_select(0, indices)
        high = [self.decoder.conv_s0(features["high"][0]),
                self.decoder.conv_s1(features["high"][1])]
        high = [x.index_select(0, indices) for x in high]
        dense = guidance.reshape(b*q, c, *low.shape[-2:])
        dense = dense + self.prompt_encoder.no_mask_embed.weight.reshape(1, c, 1, 1)
        masks, quality, _, presence = self.decoder(
            image_embeddings=low, image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=tokens.reshape(b*q, n, c),
            dense_prompt_embeddings=dense, multimask_output=False,
            repeat_image=False, high_res_features=high,
        )
        return (masks[:, 0].reshape(b, q, *masks.shape[-2:]),
                quality[:, 0].reshape(b, q), presence.reshape(b, q))


class ToyBackbone(nn.Module):
    """Small CPU test double. Never presented as SAM or used in real results."""

    def __init__(self, canvas_size=64, **unused):
        super().__init__()
        self.canvas_size = canvas_size
        self.encoder = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
                                     nn.Conv2d(32, 256, 3, 2, 1), nn.GELU())
        self.encoder.requires_grad_(False)
        self.decoder = nn.ModuleDict({
            "image": nn.Conv2d(256, 32, 1), "token": nn.Linear(256, 32),
            "quality": nn.Linear(256, 1), "presence": nn.Linear(256, 1),
        })
        self.receipt = {"backend": "toy", "not_sam": True, "canvas_size": canvas_size}

    def encode(self, images):
        with torch.no_grad():
            x = self.encoder(images)
        return {"low": x, "high": [x, x]}

    def decode(self, features, tokens, guidance):
        b, q = tokens.shape[:2]
        pooled = tokens.mean(2)
        maps = features["low"][:, None] + guidance
        maps = self.decoder["image"](maps.flatten(0, 1)).reshape(b, q, 32, *maps.shape[-2:])
        weights = self.decoder["token"](pooled)
        masks = torch.einsum("bqc,bqchw->bqhw", weights, maps) / 32**.5
        return masks, self.decoder["quality"](pooled)[..., 0], self.decoder["presence"](pooled)[..., 0]
