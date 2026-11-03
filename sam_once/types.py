"""Native-coordinate contracts. No ground-truth or SAM imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import numpy as np


@dataclass(frozen=True)
class CropSpec:
    crop_id: str
    xyxy: tuple[int, int, int, int]

    @property
    def width(self):
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self):
        return self.xyxy[3] - self.xyxy[1]


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    crop_id: str
    positive: tuple[tuple[float, float], ...]
    negative: tuple[tuple[float, float], ...] = ()
    box: tuple[float, float, float, float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PromptPlan:
    crops: list[CropSpec]
    prompts: list[PromptSpec]
    ridge: np.ndarray
    audit: dict = field(default_factory=dict)

    def validate(self, rgb: np.ndarray, canvas_size: int = 1008):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected native uint8 RGB HxWx3")
        h, w = rgb.shape[:2]
        if self.ridge.shape != (h, w) or self.ridge.dtype.kind != "f" or not np.isfinite(self.ridge).all() or np.any((self.ridge < 0) | (self.ridge > 1)):
            raise ValueError("Invalid native ridge map")
        crops = {c.crop_id: c for c in self.crops}
        if len(crops) != len(self.crops):
            raise ValueError("Duplicate crop IDs")
        for c in self.crops:
            x0, y0, x1, y1 = c.xyxy
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("Crop outside native image")
            if max(c.width, c.height) > canvas_size:
                raise ValueError("Crop exceeds native SAM canvas; resizing is prohibited")
        if len({p.prompt_id for p in self.prompts}) != len(self.prompts):
            raise ValueError("Duplicate prompt IDs")
        for p in self.prompts:
            if p.crop_id not in crops or not 1 <= len(p.positive) <= 3:
                raise ValueError("Prompt needs a known crop and 1-3 positives")
            c = crops[p.crop_id]
            for x, y in (*p.positive, *p.negative):
                if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < c.width and 0 <= y < c.height):
                    raise ValueError("Prompt outside visible crop")
            if set(p.positive) & set(p.negative):
                raise ValueError("Conflicting positive and negative point")
            if p.box is not None:
                x0, y0, x1, y1 = p.box
                if not (0 <= x0 < x1 <= c.width and 0 <= y0 < y1 <= c.height):
                    raise ValueError("Invalid local XYXY box")
        return self

    def metadata(self):
        return {"crops": [asdict(c) for c in self.crops],
                "prompts": [asdict(p) for p in self.prompts], "audit": self.audit}


@dataclass
class ProposalRecord:
    prompt: PromptSpec
    crop: CropSpec
    initial_logits: np.ndarray
    current_logits: np.ndarray
    score: float
    metadata: dict = field(default_factory=dict)

    def validate(self):
        shape = (self.crop.height, self.crop.width)
        for a in (self.initial_logits, self.current_logits):
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("Proposal logits must be finite and in native crop coordinates")
        if self.prompt.crop_id != self.crop.crop_id or not np.isfinite(self.score):
            raise ValueError("Invalid proposal identity/score")
        return self
