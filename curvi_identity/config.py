from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class Config:
    small_size: int = 128
    large_size: int = 256
    min_arc: int = 12
    junction_radius: int = 2
    max_gap: float = 80.
    max_candidates: int = 1024
    threshold: float = .8
    margin: float = .1
    conflict_threshold: float = .2
    matching_seconds: float = 3.
    purity: float = .9
    training_enabled: bool = False
    dataset_root: str = 'data/WireSeg-36K'
    seed: int = 20261013
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = .001
    weight_decay: float = .01

    def __post_init__(self):
        if type(self.training_enabled) is not bool:
            raise ValueError('training_enabled must be a JSON boolean, never a string')
        if self.small_size < 32 or self.large_size < self.small_size:
            raise ValueError('Require 32 <= small_size <= large_size')
        if self.max_gap <= 0 or self.max_candidates <= 0 or self.min_arc < 2:
            raise ValueError('Invalid candidate limits')
        if self.junction_radius < 0 or self.matching_seconds <= 0:
            raise ValueError('Invalid graph configuration')
        if not 0 <= self.conflict_threshold < self.threshold <= 1 or not 0 <= self.margin <= 1:
            raise ValueError('Invalid association thresholds')
        if not .5 < self.purity <= 1 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError('Invalid supervision/training configuration')
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('Invalid optimizer configuration')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig')))
