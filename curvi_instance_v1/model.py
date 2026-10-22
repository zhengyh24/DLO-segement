"""Native-crop, identity-conditioned instance tracking prototype.

This module is independent of ``curvi_identity`` and its fixed fragment graph.
``InstanceTracker`` predicts one visible instance mask for the current crop;
no ground-truth identity number is an input. A seed initializes a persistent
query from RGB features, an immutable anchor conditions every decoding step,
and a GRU updates the query from predicted-mask-pooled features.

The default widths are (32, 64, 128, 192), with a 128-dimensional query and
647,395 trainable parameters. The
encoder operates at native, 1/2, 1/4 and 1/8 resolution. Input images are never
resized; feature interpolation in the decoder explicitly targets skip sizes,
including odd image dimensions. Compute and activation memory scale linearly
with crop pixel count (quadratically with its side length). Full-resolution
skip features are retained, so this is a crop model, not a promise that an
arbitrarily large full image will fit in GPU memory. ``forward`` encodes its
crop again; ``encode`` exposes RGB-only features and seed logits for discovery.

Exit order is N, NE, E, SE, S, SW, W, NW. Exits describe remaining target
visible support at the crop frontier, not physical cable-end classifications.
All logits require sigmoid for probabilities. This untrained architecture is
an implementation prototype, not a measured segmentation result.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


EXIT_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _normalization(channels):
    # Keep at least two channels per group, including B=1, H=W=1 inputs.
    groups = max(g for g in range(1, min(8, channels // 2) + 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class _SeparableResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            _normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            _normalization(channels),
        )

    def forward(self, x):
        return F.silu(x + self.layers(x))


class _Downsample(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(source, source, 3, stride=2, padding=1, groups=source, bias=False),
            nn.Conv2d(source, target, 1, bias=False),
            _normalization(target),
            nn.SiLU(),
            _SeparableResidual(target),
        )

    def forward(self, x):
        return self.layers(x)


class _ConditionedBlock(nn.Module):
    def __init__(self, source, target, query_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(source + 2, target, 1, bias=False),
            _normalization(target),
            nn.SiLU(),
            _SeparableResidual(target),
        )
        self.film = nn.Linear(query_dim * 2, target * 2)
        # Small, nonzero modulation gives conditioning a gradient immediately.
        nn.init.normal_(self.film.weight, std=0.01)
        nn.init.zeros_(self.film.bias)

    def forward(self, features, seed, history, identity):
        size = features.shape[-2:]
        # Max pooling preserves a one-pixel seed at coarser feature scales.
        seed = F.adaptive_max_pool2d(seed, size)
        history = F.adaptive_avg_pool2d(history, size)
        x = self.fuse(torch.cat((features, seed, history), dim=1))
        scale, shift = self.film(identity).chunk(2, dim=1)
        return F.silu(x * (1 + scale.tanh()[:, :, None, None]) + shift[:, :, None, None])


def _weighted_pool(features, weights):
    # A 512x512 probability map already exceeds fp16's finite range when its
    # average is above 0.25. Accumulate numerator AND denominator in fp32;
    # casting only the division result would be too late under autocast.
    numerator = (features.float() * weights.float()).sum((-2, -1))
    denominator = weights.float().sum((-2, -1)).clamp_min(1e-6)
    return (numerator / denominator).to(features.dtype)


class InstanceTracker(nn.Module):
    """Lightweight U-Net with image-derived instance state.

    Args:
        query_dim: Persistent query/anchor width; defaults to 128.
        widths: Exactly four encoder channel counts, each at least four.

    ``encode(rgb)`` returns ``(features, seed_logits)``. The feature dictionary
    has keys ``x1``, ``x2``, ``x4``, ``x8`` and seed logits have shape Bx1xHxW.
    RGB must be floating point, normally normalized to [0, 1].

    ``forward(rgb, seed, history, query=None, anchor=None, spatial_valid=None)``
    accepts binary or soft Bx1xHxW seed/history masks. ``spatial_valid`` is an
    optional Bx1xHxW geometric image-support mask (default all ones). Derive it
    only from image dimensions and crop coordinates, NEVER GT overlap/labels.
    It excludes padding from seed/state/exit pooling, and is distinct from the
    supervision-only valid mask used by the loss. If both states are absent, seed
    neighborhood pooling initializes them. If only one state is supplied, it
    initializes the other. Pass BOTH returned state tensors on later steps to
    retain the original anchor. State shapes are Bxquery_dim. This method never
    modifies the supplied anchor. It does not detach state: rollout training
    may backpropagate across steps; inference should use inference_mode.
    """

    def __init__(self, query_dim=128, widths=(32, 64, 128, 192)):
        super().__init__()
        if not isinstance(query_dim, int) or query_dim < 1:
            raise ValueError("query_dim must be a positive integer")
        if len(widths) != 4 or any(not isinstance(c, int) or c < 4 for c in widths):
            raise ValueError("widths must contain four integers >= 4")
        self.query_dim = query_dim
        self.widths = tuple(widths)
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            _normalization(widths[0]),
            nn.SiLU(),
            _SeparableResidual(widths[0]),
        )
        self.down = nn.ModuleList(_Downsample(a, b) for a, b in zip(widths, widths[1:]))
        # Seed discovery depends only on RGB; it does not see instance prompts.
        self.seed_head = nn.Sequential(
            nn.Conv2d(widths[0] + widths[-1], widths[0], 1),
            nn.SiLU(),
            nn.Conv2d(widths[0], 1, 1),
        )
        self.initial_query = nn.Sequential(
            nn.Linear(widths[0], query_dim), nn.SiLU(), nn.Linear(query_dim, query_dim)
        )
        self.bottleneck = _ConditionedBlock(widths[-1], widths[-1], query_dim)
        self.up = nn.ModuleList(
            _ConditionedBlock(widths[i + 1] + widths[i], widths[i], query_dim)
            for i in (2, 1, 0)
        )
        self.mask_head = nn.Conv2d(widths[0], 1, 1)
        self.observation = nn.Sequential(nn.Linear(widths[0], query_dim), nn.SiLU())
        self.query_update = nn.GRUCell(query_dim, query_dim)
        # Shared directional head sees a local border sector, whole-crop target
        # features, q/anchor, and soft predicted unvisited support in that sector.
        self.exit_head = nn.Sequential(
            nn.Linear(widths[0] * 2 + query_dim * 2 + 1, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, 1),
        )

    @staticmethod
    def _check_rgb(rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[0], *rgb.shape[-2:]) < 1:
            raise ValueError("rgb must have shape Bx3xHxW with nonempty dimensions")
        if not rgb.is_floating_point():
            raise ValueError("rgb must be floating point, normally in [0, 1]")

    def encode(self, rgb):
        self._check_rgb(rgb)
        pyramid = [self.stem(rgb)]
        for block in self.down:
            pyramid.append(block(pyramid[-1]))
        coarse = F.interpolate(pyramid[-1], size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        seed_logits = self.seed_head(torch.cat((pyramid[0], coarse), dim=1))
        return dict(zip(("x1", "x2", "x4", "x8"), pyramid)), seed_logits

    @staticmethod
    def _border_sectors(height, width, device, dtype):
        # atan2(x, -y) has north at zero and increases clockwise. Normalized
        # coordinates give eight sectors even for rectangular crops.
        y = torch.linspace(-1, 1, height, device=device, dtype=torch.float32)
        x = torch.linspace(-1, 1, width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        angles = torch.remainder(torch.atan2(xx, -yy) + math.pi / 8, 2 * math.pi)
        sectors = torch.floor(angles / (math.pi / 4)).long()
        border = torch.maximum(xx.abs(), yy.abs()) >= 0.75
        return torch.stack([(border & (sectors == i)).to(dtype) for i in range(8)])

    def forward(self, rgb, seed, history, query=None, anchor=None, spatial_valid=None):
        self._check_rgb(rgb)
        expected = (rgb.shape[0], 1, *rgb.shape[-2:])
        if seed.shape != expected or history.shape != expected:
            raise ValueError("seed and history must have shape Bx1xHxW matching rgb")
        if seed.device != rgb.device or history.device != rgb.device:
            raise ValueError("rgb, seed and history must be on the same device")
        if spatial_valid is None:
            spatial_valid = torch.ones_like(seed, dtype=rgb.dtype)
        elif spatial_valid.shape != expected or spatial_valid.device != rgb.device:
            raise ValueError("spatial_valid must be Bx1xHxW on the RGB device")
        else:
            spatial_valid = (spatial_valid > 0).to(dtype=rgb.dtype)
        seed = seed.to(dtype=rgb.dtype)
        history = history.to(dtype=rgb.dtype)
        seed = seed * spatial_valid
        history = history * spatial_valid
        for name, state in (("query", query), ("anchor", anchor)):
            if state is not None and (state.shape != (rgb.shape[0], self.query_dim) or state.device != rgb.device):
                raise ValueError(f"{name} must have shape Bxquery_dim on the RGB device")
        features, seed_logits = self.encode(rgb)
        if query is None:
            if anchor is not None:
                query = anchor
            else:
                neighborhood = F.max_pool2d(seed, kernel_size=7, stride=1, padding=3)
                query = self.initial_query(_weighted_pool(features["x1"], neighborhood * spatial_valid))
        if anchor is None:
            anchor = query
        identity = torch.cat((query, anchor), dim=1)
        decoded = self.bottleneck(features["x8"], seed, history, identity)
        for block, key in zip(self.up, ("x4", "x2", "x1")):
            skip = features[key]
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            decoded = block(torch.cat((decoded, skip), dim=1), seed, history, identity)
        mask_logits = self.mask_head(decoded)
        probability = mask_logits.sigmoid()
        pooled = _weighted_pool(decoded, probability * spatial_valid)
        updated_query = self.query_update(self.observation(pooled), query)
        # A completely padded observation supplies no evidence to update state.
        has_observation = spatial_valid.flatten(1).any(1)[:, None]
        updated_query = torch.where(has_observation, updated_query, query)
        sectors = self._border_sectors(*rgb.shape[-2:], rgb.device, torch.float32)
        # einsum is autocast-eligible, so explicit float() alone is insufficient
        # to keep a spatial reduction in fp32. This also protects large crops.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            valid = spatial_valid.float()
            sector_area = torch.einsum("bchw,dhw->bdc", valid, sectors).clamp_min(1)
            sector_features = torch.einsum("bchw,dhw->bdc", decoded.float() * valid, sectors) / sector_area
            remaining = probability.float() * (1 - history.float().clamp(0, 1)) * valid
            support = torch.einsum("bchw,dhw->bdc", remaining, sectors) / sector_area
        sector_features = sector_features.to(decoded.dtype)
        support = support.to(decoded.dtype)
        exit_inputs = torch.cat((
            sector_features,
            pooled[:, None, :].expand(-1, 8, -1),
            identity[:, None, :].expand(-1, 8, -1),
            support,
        ), dim=-1)
        return {
            "mask_logits": mask_logits,
            "seed_logits": seed_logits,
            "exit_logits": self.exit_head(exit_inputs).squeeze(-1),
            "query": updated_query,
            "anchor": anchor,
        }
