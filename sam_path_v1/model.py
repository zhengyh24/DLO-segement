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
    canvas_size: int = 1036
    num_queries: int = 24
    tracker_steps: int = 8
    hidden_dim: int = 256
    encoder_amp: str = "bf16"
    init_seed: int = 42

    def __post_init__(self):
        if self.variant not in "ABCD" or len(self.variant) != 1:
            raise ValueError("variant must be A, B, C, or D")
        if self.backend not in ("sam31", "toy") or self.encoder_amp not in ("bf16", "none"):
            raise ValueError("Unsupported backend/encoder precision")
        if self.hidden_dim != 256 or self.num_queries < 1 or self.tracker_steps < 1:
            raise ValueError("Require hidden_dim=256, positive query/step counts")


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

    def forward(self, descriptor, queries, valid):
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
        dense = self.dense_channel(state).tanh()[..., None, None] * coverage[:, :, None]
        return torch.stack(tokens, 2), dense, {"path_attention": attention,
                "path_positions": torch.stack(positions, 2), "path_coverage": coverage}


class SamPathModel(nn.Module):
    def __init__(self, config: ModelConfig | dict):
        super().__init__()
        self.config = ModelConfig(**config) if isinstance(config, dict) else config
        c = self.config
        # Isolated component seeds give A-D identical shared initial tensors;
        # removing one component never shifts the initialization of another.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.init_seed)
            self.base = (Sam31Backbone(c.sam_repo, c.sam_checkpoint, c.canvas_size, c.encoder_amp)
                         if c.backend == "sam31" else ToyBackbone(c.canvas_size))
            torch.manual_seed(c.init_seed+1)
            self.condition = AutomaticCondition(c.num_queries)
            torch.manual_seed(c.init_seed+2)
            self.tracker = SoftPathTracker(c.tracker_steps) if c.variant != "A" else None
            torch.manual_seed(c.init_seed+3)
            self.descriptor = GeometrySemanticDescriptor() if c.variant in ("C", "D") else None
        self.last_calls = {}

    def forward(self, images, valid=None):
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (self.config.canvas_size,)*2:
            raise ValueError("Expected RGB [B,3,canvas,canvas], already natively padded")
        if valid is None:
            valid = torch.ones_like(images[:, :1], dtype=torch.bool)
        if valid.shape != images[:, :1].shape:
            raise ValueError("valid must be RGB-derived padding validity [B,1,H,W], never GT overlap")
        # Count actual module entries in addition to the wrapper's one-call flow.
        counts = {"encoder": 0, "decoder": 0}
        encoder_module = self.base.trunk if self.config.backend == "sam31" else self.base.encoder
        decoder_module = self.base.decoder
        def count_encoder(*unused):
            counts["encoder"] += 1
        def count_decoder(*unused):
            counts["decoder"] += 1
        hooks = [encoder_module.register_forward_pre_hook(count_encoder)]
        # Toy ModuleDict is a composition, not a callable SAM decoder.
        if self.config.backend == "sam31":
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
                tokens = torch.cat((tokens, path_tokens), 2)
                dense = dense + path_dense
            masks, quality, sam_presence = self.base.decode(features, tokens, dense)
            if self.config.backend == "toy":
                counts["decoder"] += 1
            aux["sam_object_logits"] = sam_presence
            if counts != {"encoder": 1, "decoder": 1}:
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
