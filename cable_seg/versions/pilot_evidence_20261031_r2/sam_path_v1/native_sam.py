"""A native SAM3.1 text detector baseline: fixed 'cable', 200 native queries.

Only Q/V LoRA (r=8, alpha=16, no dropout) in the final eight ViT blocks is
trainable. The native text, fusion, detection and segmentation heads are frozen.
Tensor methods deliberately bypass inference_mode-decorated predictor APIs.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .backbone import file_sha256


def install_empty_point_shortcut(encoder):
    """Avoid CUDA grid_sample backward for the native zero-point text prompt.

    The official point encoder always samples even a [B, 0, 1, 2] grid.
    That empty sampling has identically zero feature/coordinate gradients but
    CUDA rejects its backward under strict deterministic mode. An empty view
    preserves those zero gradients without registering grid_sample backward.
    Every nonempty prompt is dispatched to the unchanged official method.
    """
    original = encoder._encode_points
    encoder.empty_point_shortcut_calls = 0
    def encode_points(points, points_mask, points_labels, img_feats):
        if points.shape[0] != 0:
            return original(points, points_mask, points_labels, img_feats)
        if img_feats.ndim != 4 or img_feats.shape[0] != points.shape[1] or img_feats.shape[1] != encoder.d_model:
            raise ValueError("Native empty-point feature shape changed")
        encoder.empty_point_shortcut_calls += 1
        empty = img_feats.flatten(2)[..., :0].permute(2, 0, 1)
        # Keep the empty coordinate derivative, too, should an empty coordinate
        # leaf require grad. No learned tensors or nonempty computation change.
        return empty + points.sum(-1, keepdim=True)*0, points_mask
    encoder._encode_points = encode_points


def install_empty_box_shortcut(encoder):
    """The same zero-token identity avoids the official empty ROIAlign kernel."""
    original = encoder._encode_boxes
    encoder.empty_box_shortcut_calls = 0
    def encode_boxes(boxes, boxes_mask, boxes_labels, img_feats):
        if boxes.shape[0] != 0:
            return original(boxes, boxes_mask, boxes_labels, img_feats)
        if img_feats.ndim != 4 or img_feats.shape[0] != boxes.shape[1] or img_feats.shape[1] != encoder.d_model:
            raise ValueError("Native empty-box feature shape changed")
        encoder.empty_box_shortcut_calls += 1
        empty = img_feats.flatten(2)[..., :0].permute(2, 0, 1)
        return empty + boxes.sum(-1, keepdim=True)*0, boxes_mask
    encoder._encode_boxes = encode_boxes


class QVLoRA(nn.Module):
    """Zero-initialized Q/V deltas; K and pretrained fused QKV remain unchanged."""
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        if linear.out_features != 3*linear.in_features:
            raise ValueError("QVLoRA requires a fused square QKV projection")
        self.base = linear.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        dim = linear.in_features
        self.q_a = nn.Parameter(torch.empty(rank, dim))
        self.q_b = nn.Parameter(torch.zeros(dim, rank))
        self.v_a = nn.Parameter(torch.empty(rank, dim))
        self.v_b = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        q_delta = F.linear(F.linear(x, self.q_a), self.q_b)
        v_delta = F.linear(F.linear(x, self.v_a), self.v_b)
        return original + torch.cat((q_delta, torch.zeros_like(q_delta), v_delta), -1)*(self.alpha/self.rank)


class CheckpointLoRABlock(nn.Module):
    """Recompute trainable blocks without enabling dropout/stochastic depth."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        if torch.is_grad_enabled() and any(p.requires_grad for p in self.block.parameters()):
            return activation_checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


class FrozenViTBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        with torch.no_grad():
            return self.block(x)


class DifferentiableViTMLP(nn.Module):
    """Same retained layers, replacing an inference-only fused addmm+GELU.

The upstream fast kernel explicitly rejects autograd. Standard PyTorch ops
provide the input gradient required by LoRA; BF16 rounding can differ slightly.
"""
    def __init__(self, original):
        super().__init__()
        for name in ("fc1", "act", "drop1", "norm", "fc2", "drop2"):
            setattr(self, name, getattr(original, name))

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


def _normalized_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    source = payload.get("model", payload)
    result = {}
    for key, value in source.items():
        if key.startswith("sam3_model."):
            key = "detector."+key[len("sam3_model."):]
        if key in result:
            raise ValueError(f"Duplicate normalized tensor: {key}")
        result[key] = value
    return result


class NativeSam31Base(nn.Module):
    def __init__(self, repo, checkpoint, canvas_size=1036, seed=42):
        super().__init__()
        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if canvas_size < 28 or canvas_size % 14:
            raise ValueError("Native SAM3.1 canvas must be a positive multiple of 14")
        if not (repo/"sam3/model_builder.py").is_file() or not checkpoint.is_file():
            raise FileNotFoundError("Native SAM3.1 requires the official repo and multiplex checkpoint")
        sys.path.insert(0, str(repo))
        from sam3 import model_builder as builder
        from sam3.model.vitdet import ViT
        from sam3.model.necks import Sam3DualViTDetNeck
        from sam3.model.vl_combiner import SAM3VLBackbone
        from sam3.model.sam3_image import Sam3Image
        if Path(builder.__file__).resolve() != repo/"sam3/model_builder.py":
            raise RuntimeError("Another sam3 installation is imported; use a fresh process")
        self.canvas_size = int(canvas_size)
        # The multiplex detector uses three SAM3 FPN scales. Retain precisely
        # these branches; its unused interactive/propagation necks are omitted.
        trunk = ViT(img_size=canvas_size, pretrain_img_size=336, patch_size=14,
                    embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625,
                    norm_layer="LayerNorm", drop_path_rate=.1, qkv_bias=True,
                    use_abs_pos=True, tile_abs_pos=True, global_att_blocks=(7, 15, 23, 31),
                    rel_pos_blocks=(), use_rope=True, use_interp_rope=True, window_size=24,
                    pretrain_use_cls_token=True, retain_cls_token=False, ln_pre=True,
                    ln_post=False, return_interm_layers=False, bias_patch_embed=False,
                    compile_mode=None, use_fa3=False, use_rope_real=False)
        neck = Sam3DualViTDetNeck(trunk, builder._create_position_encoding(None), 256,
                                  scale_factors=[4., 2., 1.], add_sam2_neck=False)
        text = builder._create_text_encoder(str(repo/"sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
        # Multiplex checkpoints omit the unused CLIP pooled projection. VEText
        # consumes only returned token embeddings; never admit a random tensor.
        text.encoder.text_projection = None
        self.native = Sam3Image(
            backbone=SAM3VLBackbone(neck, text, scalp=0),
            transformer=builder._create_sam3_transformer(),
            input_geometry_encoder=builder._create_geometry_encoder(),
            segmentation_head=builder._create_segmentation_head(),
            num_feature_levels=1, o2m_mask_predict=True,
            dot_prod_scoring=builder._create_dot_product_scoring(),
            use_instance_query=True, multimask_output=True,
            supervise_joint_box_scores=True, detach_presence_in_joint_score=False)
        if self.native.transformer.decoder.num_queries != 200:
            raise RuntimeError("Native SAM detector must retain its 200 pretrained queries")
        source = _normalized_checkpoint(checkpoint)
        selected, derived = {}, []
        for name, expected in self.native.state_dict().items():
            if name.endswith(".attn.freqs_cis") and expected.is_complex():
                selected[name] = expected
                derived.append(name)
                continue
            key = "detector."+name
            if key not in source or source[key].shape != expected.shape:
                raise RuntimeError(f"Required native tensor missing or wrong shape: {name}")
            selected[name] = source[key]
        self.native.load_state_dict(selected, strict=True)
        self.native.requires_grad_(False)
        install_empty_point_shortcut(self.native.geometry_encoder)
        install_empty_box_shortcut(self.native.geometry_encoder)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            for index in range(24):
                block = self.native.backbone.vision_backbone.trunk.blocks[index]
                self.native.backbone.vision_backbone.trunk.blocks[index] = FrozenViTBlock(block)
            for index in range(24, 32):
                block = self.native.backbone.vision_backbone.trunk.blocks[index]
                block.attn.qkv = QVLoRA(block.attn.qkv, rank=8, alpha=16)
                block.mlp = DifferentiableViTMLP(block.mlp)
                self.native.backbone.vision_backbone.trunk.blocks[index] = CheckpointLoRABlock(block)
        files = ("sam3/model_builder.py", "sam3/model/vitdet.py", "sam3/model/necks.py",
                 "sam3/model/vl_combiner.py", "sam3/model/sam3_image.py", "sam3/model/decoder.py",
                 "sam3/model/encoder.py", "sam3/model/maskformer_segmentation.py",
                 "sam3/model/geometry_encoders.py", "sam3/model/model_misc.py",
                 "sam3/model/text_encoder_ve.py", "sam3/model/tokenizer_ve.py",
                 "sam3/model/position_encoding.py", "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        self.receipt = dict(backend="sam31_native", repo=str(repo), checkpoint=str(checkpoint),
                            checkpoint_sha256=file_sha256(checkpoint), canvas_size=canvas_size,
                            source_sha256={name: file_sha256(repo/name) for name in files},
                            strict_retained_state=True, loaded_tensor_count=len(selected)-len(derived),
                            regenerated_rope_buffers=derived, native_rgb_resize=False,
                            native_detector_queries=200, fixed_text="cable",
                            lora=dict(blocks=list(range(24, 32)), projections=["Q", "V"],
                                      rank=8, alpha=16, dropout=0., parameters=262144),
                            native_joint_presence_score=True,
                            unused_text_pooled_projection_omitted=True,
                            empty_geometry_sampling="Empty native text-only point/box encoding uses empty feature views with exact zero feature/coordinate gradients, avoiding empty grid_sample/ROIAlign kernels; nonempty prompts retain the official methods. Strict deterministic mode stays enabled.",
                            autograd_adaptation="last8 frozen MLP fused inference kernel replaced by standard linear/GELU; BF16 rounding may differ",
                            unused_branches="video memory, interactive and propagation FPN omitted",
                            frozen_head_eval_mode=True)
        self.train(False)

    @property
    def encoder_module(self):
        return self.native.backbone.vision_backbone.trunk

    @property
    def decoder(self):
        return self.native.transformer.decoder

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # Eval disables native matching, DAC, stochastic depth and dropout, but
        # does not disable autograd. LoRA remains trainable under tensor forward.
        self.native.eval()
        return self


class NativeSam31LoRAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.num_queries != 200:
            raise ValueError("S must configure num_queries=200 (native SAM3.1 query count)")
        self.base = NativeSam31Base(config.sam_repo, config.sam_checkpoint,
                                    config.canvas_size, config.init_seed)
        self.last_calls = {}
        self._text_cache = None

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.base.train(mode)
        return self

    def _text(self, images):
        native = self.base.native
        signature = (str(images.device), self.config.encoder_amp)
        if self._text_cache is None or self._text_cache[0] != signature:
            # Constant frozen text can be cached, without retaining a graph.
            with torch.no_grad():
                values = native.backbone.forward_text(["cable"], device=images.device)
            self._text_cache = (signature, {k: v.detach() for k, v in values.items()})
        return dict(self._text_cache[1])

    def forward(self, images, valid=None):
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (self.config.canvas_size,)*2:
            raise ValueError("Expected natively padded RGB [B,3,canvas,canvas]")
        if valid is not None and (valid.shape != images[:, :1].shape or not valid.flatten(1).any(1).all()):
            raise ValueError("Invalid RGB-derived padding validity")
        from sam3.model.data_misc import FindStage
        native = self.base.native
        counts = {"encoder": 0, "decoder": 0}
        def count_encoder(*args): counts["encoder"] += 1
        def count_decoder(*args): counts["decoder"] += 1
        hooks = [self.base.encoder_module.register_forward_pre_hook(count_encoder),
                 self.base.decoder.register_forward_pre_hook(count_decoder)]
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if images.device.type == "cuda" and self.config.encoder_amp == "bf16" else nullcontext())
        try:
            with context:
                features = native.backbone.forward_image((images-.5)/.5)
                features.update(self._text(images))
                find_input = FindStage(img_ids=torch.arange(len(images), device=images.device),
                                       text_ids=torch.zeros(len(images), device=images.device, dtype=torch.long),
                                       input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                                       input_points=None, input_points_mask=None)
                outputs = native.forward_grounding(features, find_input, None,
                                                     native._get_dummy_prompt(len(images)))
            if counts != {"encoder": 1, "decoder": 1}:
                raise RuntimeError(f"Native single-forward contract violated: {counts}")
            logits = outputs["pred_logits"].squeeze(-1).float()
            masks = outputs["pred_masks"].float()
            if masks.shape[:2] != (len(images), 200) or logits.shape != (len(images), 200):
                raise RuntimeError("Native detector query/output shape changed")
            self.last_calls = dict(counts)
            return dict(mask_logits=masks, object_logits=logits,
                        quality_logits=torch.zeros_like(logits), calls=dict(counts),
                        aux={"native_detection": True,
                             "native_presence_logits": outputs.get("presence_logit_dec")})
        finally:
            for hook in hooks: hook.remove()

    def checkpoint_state_dict(self):
        return {key: value.detach().cpu().clone() for key, value in self.named_parameters()
                if value.requires_grad}

    def load_trainable_state_dict(self, state):
        expected = self.checkpoint_state_dict()
        if set(state) != set(expected):
            raise ValueError("Native LoRA checkpoint keys do not match")
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for key, value in state.items():
                if value.shape != expected[key].shape:
                    raise ValueError(f"Native LoRA checkpoint shape mismatch: {key}")
                parameters[key].copy_(value)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(total=total, trainable=trainable, base=total-trainable,
                    condition=0, tracker=0, descriptor=0,
                    decoder=sum(p.numel() for p in self.base.decoder.parameters()),
                    new_modules=trainable, lora=trainable, native_queries=200)
