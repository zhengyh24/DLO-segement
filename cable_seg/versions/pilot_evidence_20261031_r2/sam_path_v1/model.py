"""A/B/C/D ablations with automatic set queries and differentiable path memory.

No masks, positive/negative points, skeletons or GT instance IDs enter forward.
The path is a soft latent route over *shared* features, not a visible mask and
not a claim of recovered physical topology or labelled occlusion states.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Sam31Backbone, ToyBackbone


@dataclass
class ModelConfig:
    variant: str = "C"
    backend: str = "sam31"
    sam_repo: str = ""
    sam_checkpoint: str = ""
    sam2_repo: str = ""
    sam2_checkpoint: str = ""
    canvas_size: int = 1036
    num_queries: int = 24
    tracker_steps: int = 8
    decoder_query_chunk: int = 0
    hidden_dim: int = 256
    encoder_amp: str = "bf16"
    init_seed: int = 42

    def __post_init__(self):
        if self.variant not in VARIANT_SPECS:
            raise ValueError(f"Unknown variant: {self.variant}")
        if self.backend not in ("sam31", "sam21", "sam31_native", "toy") or self.encoder_amp not in ("bf16", "none"):
            raise ValueError("Unsupported backend/encoder precision")
        if (self.variant == "S") != (self.backend == "sam31_native"):
            raise ValueError("S requires backend=sam31_native, reserved for the native text detector")
        if self.variant == "S" and self.num_queries != 200:
            raise ValueError("Native SAM3 detector S requires num_queries=200")
        if self.backend == "sam21" and self.canvas_size != 1024:
            raise ValueError("SAM2.1 admission uses native 1024 RGB only")
        if self.hidden_dim != 256 or self.num_queries < 1 or self.tracker_steps < 1:
            raise ValueError("Require hidden_dim=256, positive query/step counts")
        if type(self.decoder_query_chunk) is not int or self.decoder_query_chunk < 0:
            raise ValueError("decoder_query_chunk must be nonnegative; 0 means one batched call")


# Public, immutable experiment identities. No hidden switches differ within an arm.
VARIANT_SPECS = {
    "A": ("none", "none", "coupled"),
    "B": ("recurrent", "none", "coupled"),
    "C": ("recurrent", "geometry_semantic", "coupled"),
    "D": ("recurrent", "geometry_semantic", "decoder_detach"),
    "MASK_ONLY": ("recurrent", "geometry_semantic", "mask_only"),
    "CNN": ("recurrent", "cnn", "coupled"),
    "NO_GEOMETRY": ("recurrent", "geometry_semantic", "coupled"),
    "NO_CONTRAST": ("recurrent", "geometry_semantic", "coupled"),
    "PARALLEL": ("parallel", "geometry_semantic", "coupled"),
    "S": ("none", "none", "native"),
}


def variant_loss_metadata(variant):
    tracker, descriptor, gradient = VARIANT_SPECS[variant]
    inactive = []
    if tracker == "none":
        inactive.append("path")
    if descriptor == "none":
        inactive.extend(("descriptor", "geometry"))
    if variant == "NO_GEOMETRY":
        inactive.append("geometry")
    if variant == "S":
        inactive.append("quality")
    return dict(inactive=inactive, gradient_policy=gradient,
                descriptor_contrast_active=descriptor != "none" and variant != "NO_CONTRAST",
                descriptor_compact_active=descriptor != "none")


def _grid(h, w, device, dtype):
    yy, xx = torch.meshgrid((torch.arange(h, device=device, dtype=dtype)+.5)/h,
                            (torch.arange(w, device=device, dtype=dtype)+.5)/w, indexing="ij")
    return torch.stack((xx, yy), -1).reshape(h*w, 2)


def _position(grid, channels=256):
    frequencies = 2 * math.pi * 2 ** torch.linspace(0, 6, channels//4, device=grid.device)
    angles = grid[..., None] * frequencies
    return torch.cat((angles.sin(), angles.cos()), -1).flatten(-2)


class AutomaticCondition(nn.Module):
    def __init__(self, queries):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, 256)*.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(
            256, 8, dim_feedforward=1024, dropout=0., activation="gelu",
            batch_first=True, norm_first=True) for _ in range(2)])
        self.norm = nn.LayerNorm(256)
        self.object_head = nn.Linear(256, 1)
        self.spatial_query = nn.Linear(256, 256)
        self.channel = nn.Linear(256, 256)

    def forward(self, low, valid):
        b, _, h, w = low.shape
        memory = low.flatten(2).transpose(1, 2)
        memory = memory + _position(_grid(h, w, low.device, low.dtype))[None]
        queries = self.queries[None].expand(b, -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory, memory_key_padding_mask=~valid.flatten(1))
        queries = self.norm(queries)
        logits = torch.einsum("bqc,bchw->bqhw", self.spatial_query(queries), low) / 16
        spatial = logits.sigmoid() * valid[:, None]
        dense = self.channel(queries).tanh()[..., None, None] * spatial[:, :, None]
        return queries, dense, self.object_head(queries)[..., 0]

    def forward_factors(self, low, valid):
        """Native expansion: avoid materializing full-image [B,Q,C,H,W]."""
        b, _, h, w = low.shape
        memory = low.flatten(2).transpose(1, 2)
        memory = memory + _position(_grid(h, w, low.device, low.dtype))[None]
        queries = self.queries[None].expand(b, -1, -1)
        for layer in self.layers:
            queries = layer(queries, memory, memory_key_padding_mask=~valid.flatten(1))
        queries = self.norm(queries)
        logits = torch.einsum("bqc,bchw->bqhw", self.spatial_query(queries), low) / 16
        spatial = logits.sigmoid() * valid[:, None]
        return queries, {"channels": self.channel(queries).tanh(), "spatial": spatial}, self.object_head(queries)[..., 0]


class GeometrySemanticDescriptor(nn.Module):
    """Small learned multiscale descriptor; identity/visible-geometry auxiliaries.

    The high features are SAM's interactive FPN outputs, not falsely described
    as early ViT layers. Shape convolutions and global context are learned.
    """
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(768, 256, 1)
        self.local = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, groups=256),
                                   nn.GroupNorm(16, 256), nn.GELU(), nn.Conv2d(256, 256, 1),
                                   nn.Conv2d(256, 256, 3, padding=2, dilation=2, groups=256),
                                   nn.GELU(), nn.Conv2d(256, 256, 1))
        self.context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 256, 1), nn.Sigmoid())
        self.norm = nn.GroupNorm(16, 256)
        self.geometry_head = nn.Conv2d(256, 3, 1)  # support + unsigned local tangent cos(2θ),sin(2θ)

    def forward(self, features):
        low = features["low"]
        parts = [low] + [F.adaptive_avg_pool2d(x, low.shape[-2:]) for x in features["high"]]
        fused = self.fuse(torch.cat(parts, 1))
        descriptor = self.norm(fused + self.local(fused) * self.context(fused))
        return descriptor, self.geometry_head(descriptor)


class PlainCNNDescriptor(nn.Module):
    """400,771 parameters versus C's 401,155; same inputs and auxiliaries."""
    def __init__(self):
        super().__init__()
        self.fuse = nn.Conv2d(768, 256, 1)
        self.local = nn.Sequential(nn.Conv2d(256, 124, 1), nn.GroupNorm(4, 124),
                                   nn.GELU(), nn.Conv2d(124, 124, 3, padding=1),
                                   nn.GELU(), nn.Conv2d(124, 256, 1))
        self.norm = nn.GroupNorm(16, 256)
        self.geometry_head = nn.Conv2d(256, 3, 1)

    def forward(self, features):
        low = features["low"]
        parts = [low] + [F.adaptive_avg_pool2d(x, low.shape[-2:]) for x in features["high"]]
        fused = self.fuse(torch.cat(parts, 1))
        descriptor = self.norm(fused + self.local(fused))
        return descriptor, self.geometry_head(descriptor)


class ParallelAttentionTracker(nn.Module):
    """Independent step queries; no recurrent motion, memory, or visit penalty.

    Eight steps have 988,800 parameters versus recurrent tracker's 989,703.
    The attention weights are real per-step spatial observations, not a separate
    visualization-only head. Mean token channels and coverage form guidance.
    """
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.step_embeddings = nn.Embedding(steps, 256)
        self.attention = nn.MultiheadAttention(256, 8, dropout=0., batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1408), nn.GELU(), nn.Linear(1408, 256))
        self.norm2 = nn.LayerNorm(256)

    def forward(self, descriptor, queries, valid, *, factorized=False):
        b, _, h, w = descriptor.shape
        q = queries.shape[1]
        grid = _grid(h, w, descriptor.device, descriptor.dtype)
        memory = descriptor.flatten(2).transpose(1, 2)
        query = (queries[:, :, None] + self.step_embeddings.weight[None, None]).reshape(b, q*self.steps, 256)
        observed, attention = self.attention(query, memory + _position(grid)[None], memory,
            key_padding_mask=~valid.flatten(1), need_weights=True, average_attn_weights=True)
        tokens = self.norm1(query + observed)
        tokens = self.norm2(tokens + self.ffn(tokens)).reshape(b, q, self.steps, 256)
        attention = attention.reshape(b, q, self.steps, h, w)
        coverage = attention.mean(2)
        coverage = coverage / coverage.amax((-2, -1), keepdim=True).clamp_min(1e-6)
        channels = tokens.mean(2).tanh()
        dense = ({"channels": channels, "spatial": coverage} if factorized else
                 channels[..., None, None] * coverage[:, :, None])
        return tokens, dense, {"path_attention": attention,
            "path_positions": attention.flatten(-2) @ grid, "path_coverage": coverage}


class SoftPathTracker(nn.Module):
    """Recurrent spatial attention, learned motion/scale, and continuous memory.

    T steps revisit cached features; neither the image encoder nor SAM decoder
    is called in this loop. Soft attention preserves the mask-loss gradient.
    The recurrent state has no hard-coded branch/merge/occlusion decision rules.
    """
    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        self.key = nn.Conv2d(256, 256, 1)
        self.query = nn.Linear(256, 256)
        self.initial_position = nn.Linear(256, 2)
        self.motion = nn.Sequential(nn.Linear(260, 256), nn.GELU(), nn.Linear(256, 4))
        self.cell = nn.GRUCell(512, 256)
        self.token = nn.Sequential(nn.Linear(260, 256), nn.GELU(), nn.Linear(256, 256))
        self.dense_channel = nn.Linear(256, 256)
        self.revisit_penalty = nn.Parameter(torch.tensor(-2.))

    def forward(self, descriptor, queries, valid, *, factorized=False):
        b, _, h, w = descriptor.shape
        q = queries.shape[1]
        grid = _grid(h, w, descriptor.device, descriptor.dtype)
        keys = self.key(descriptor).flatten(2)
        values = descriptor.flatten(2)
        state = queries
        position = self.initial_position(queries).sigmoid()
        direction = torch.zeros_like(position)
        visited = descriptor.new_zeros(b, q, h*w)
        paths, positions, tokens = [], [], []
        for _ in range(self.steps):
            motion = self.motion(torch.cat((state, position, direction), -1))
            # Learnable displacement and uncertainty; no pixel-length/gap thresholds.
            center = position + motion[..., :2].tanh() * .25
            scale = F.softplus(motion[..., 2:]) + 1 / max(h, w)
            relative = (grid[None, None] - center[:, :, None]) / scale[:, :, None]
            prior = -.5 * relative.square().sum(-1)
            content = torch.einsum("bqc,bcn->bqn", self.query(state), keys) / 16
            scores = content + prior - F.softplus(self.revisit_penalty) * visited
            scores = scores.masked_fill(~valid.flatten(1)[:, None], torch.finfo(scores.dtype).min)
            attention = scores.softmax(-1)
            observed = torch.einsum("bqn,bcn->bqc", attention, values)
            next_position = attention @ grid
            direction = next_position - position
            position = next_position
            state = self.cell(torch.cat((observed, queries), -1).reshape(b*q, 512),
                              state.reshape(b*q, 256)).reshape(b, q, 256)
            tokens.append(self.token(torch.cat((state, position, direction), -1)))
            paths.append(attention.reshape(b, q, h, w))
            positions.append(position)
            visited = visited + attention * (h*w)
        attention = torch.stack(paths, 2)
        coverage = attention.mean(2)
        coverage = coverage / coverage.amax((-2, -1), keepdim=True).clamp_min(1e-6)
        channels = self.dense_channel(state).tanh()
        dense = ({"channels": channels, "spatial": coverage} if factorized else
                 channels[..., None, None] * coverage[:, :, None])
        return torch.stack(tokens, 2), dense, {"path_attention": attention,
                "path_positions": torch.stack(positions, 2), "path_coverage": coverage}


class SamPathModel(nn.Module):
    def __new__(cls, config):
        parsed = ModelConfig(**config) if isinstance(config, dict) else config
        if parsed.variant == "S":
            from .native_sam import NativeSam31LoRAModel
            return NativeSam31LoRAModel(parsed)
        return super().__new__(cls)

    def __init__(self, config: ModelConfig | dict):
        super().__init__()
        self.config = ModelConfig(**config) if isinstance(config, dict) else config
        c = self.config
        # Isolated component seeds give A-D identical shared initial tensors;
        # removing one component never shifts the initialization of another.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.init_seed)
            if c.backend == "sam31":
                self.base = Sam31Backbone(c.sam_repo, c.sam_checkpoint, c.canvas_size, c.encoder_amp)
            elif c.backend == "sam21":
                from .backbone_sam2 import Sam21Backbone
                self.base = Sam21Backbone(c.sam2_repo, c.sam2_checkpoint, c.canvas_size, c.encoder_amp)
            else:
                self.base = ToyBackbone(c.canvas_size)
            torch.manual_seed(c.init_seed+1)
            self.condition = AutomaticCondition(c.num_queries)
            torch.manual_seed(c.init_seed+2)
            tracker_kind, descriptor_kind, _ = VARIANT_SPECS[c.variant]
            self.tracker = (None if tracker_kind == "none" else
                            ParallelAttentionTracker(c.tracker_steps) if tracker_kind == "parallel" else
                            SoftPathTracker(c.tracker_steps))
            torch.manual_seed(c.init_seed+3)
            self.descriptor = (GeometrySemanticDescriptor() if descriptor_kind == "geometry_semantic" else
                               PlainCNNDescriptor() if descriptor_kind == "cnn" else None)
        self.last_calls = {}

    def expected_calls(self):
        chunk = self.config.decoder_query_chunk or self.config.num_queries
        return {"encoder": 1, "decoder": math.ceil(self.config.num_queries/chunk)}

    def forward(self, images, valid=None):
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (self.config.canvas_size,)*2:
            raise ValueError("Expected RGB [B,3,canvas,canvas], already natively padded")
        if valid is None:
            valid = torch.ones_like(images[:, :1], dtype=torch.bool)
        if valid.shape != images[:, :1].shape:
            raise ValueError("valid must be RGB-derived padding validity [B,1,H,W], never GT overlap")
        # Count actual module entries in addition to the wrapper's one-call flow.
        counts = {"encoder": 0, "decoder": 0}
        encoder_module = (self.base.trunk if self.config.backend == "sam31" else
                          self.base.encoder_module if self.config.backend == "sam21" else self.base.encoder)
        decoder_module = self.base.decoder
        def count_encoder(*unused):
            counts["encoder"] += 1
        def count_decoder(*unused):
            counts["decoder"] += 1
        hooks = [encoder_module.register_forward_pre_hook(count_encoder)]
        # Toy ModuleDict is a composition, not a callable SAM decoder.
        if self.config.backend != "toy":
            hooks.append(decoder_module.register_forward_pre_hook(count_decoder))
        try:
            features = self.base.encode(images)
            low = features["low"]
            spatial_valid = F.adaptive_max_pool2d(valid.float(), low.shape[-2:])[:, 0].bool()
            if not spatial_valid.flatten(1).any(1).all():
                raise ValueError("Each input must contain at least one valid RGB pixel")
            queries, dense, object_logits = self.condition(low, spatial_valid)
            tokens = queries[:, :, None]
            aux = {}
            if self.tracker is not None:
                if self.descriptor is None:
                    descriptor = low
                else:
                    descriptor, geometry = self.descriptor(features)
                    aux.update(descriptor=descriptor, geometry_logits=geometry)
                path_tokens, path_dense, path_aux = self.tracker(descriptor, queries, spatial_valid)
                aux.update(path_aux)
                if self.config.variant == "D":
                    path_tokens, path_dense = path_tokens.detach(), path_dense.detach()
                elif self.config.variant == "MASK_ONLY":
                    # Exact tensors at the decoder boundary; the training helper
                    # temporarily suppresses only mask-loss gradients here.
                    aux["mask_gradient_boundaries"] = (path_tokens, path_dense)
                tokens = torch.cat((tokens, path_tokens), 2)
                dense = dense + path_dense
            chunk = self.config.decoder_query_chunk
            if not chunk:
                masks, quality, sam_presence = self.base.decode(features, tokens, dense)
                if self.config.backend == "toy":
                    counts["decoder"] += 1
            else:
                # Explicit capacity mode: multiple forward decodes, checkpointed
                # during training to bound retained activations. Not one decode.
                decoded = []
                for offset in range(0, self.config.num_queries, chunk):
                    piece = (tokens[:, offset:offset+chunk], dense[:, offset:offset+chunk])
                    if self.training and torch.is_grad_enabled():
                        from torch.utils.checkpoint import checkpoint
                        decoded.append(checkpoint(self.base.decode, features, *piece, use_reentrant=False))
                    else:
                        decoded.append(self.base.decode(features, *piece))
                    if self.config.backend == "toy":
                        counts["decoder"] += 1
                masks, quality, sam_presence = (torch.cat(parts, dim=1) for parts in zip(*decoded))
            aux["sam_object_logits"] = sam_presence
            if self.config.variant == "NO_GEOMETRY":
                aux["geometry_supervision"] = False
            if self.config.variant == "NO_CONTRAST":
                aux["contrast_supervision"] = False
            if counts != self.expected_calls():
                raise RuntimeError(f"Single-pass contract violated: {counts}")
            self.last_calls = dict(counts)
            return {"mask_logits": masks, "object_logits": object_logits,
                    "quality_logits": quality, "aux": aux, "calls": dict(counts)}
        finally:
            for hook in hooks:
                hook.remove()

    def checkpoint_state_dict(self):
        state = self.state_dict()
        if self.config.backend == "toy":
            return {key: value.detach().cpu().clone() for key, value in state.items()}
        trainable = {key for key, value in self.named_parameters() if value.requires_grad}
        return {key: value.detach().cpu().clone() for key, value in state.items()
                if key in trainable or not key.startswith("base.")}

    def load_trainable_state_dict(self, state):
        expected = set(self.checkpoint_state_dict())
        if set(state) != expected:
            raise ValueError(f"Checkpoint delta mismatch: missing={expected-set(state)}, extra={set(state)-expected}")
        merged = self.state_dict()
        for key, value in state.items():
            if value.shape != merged[key].shape:
                raise ValueError(f"Delta shape mismatch: {key}")
            merged[key] = value
        self.load_state_dict(merged, strict=True)

    def parameter_counts(self):
        counts = {"total": sum(p.numel() for p in self.parameters()),
                  "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
        for name in ("base", "condition", "tracker", "descriptor"):
            module = getattr(self, name)
            counts[name] = 0 if module is None else sum(p.numel() for p in module.parameters())
        counts["decoder"] = sum(p.numel() for p in self.base.decoder.parameters())
        counts["new_modules"] = counts["condition"] + counts["tracker"] + counts["descriptor"]
        return counts
