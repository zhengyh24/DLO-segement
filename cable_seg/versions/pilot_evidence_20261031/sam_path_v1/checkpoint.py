"""Atomic experiment checkpoints. Only load checkpoints from a trusted source."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


# Paths are fixed here, never taken from a serialized checkpoint. Include
# deployment postprocessing as well as training mathematics in the identity.
ENGINE_SOURCE_FILES = (
    "train.py", "checkpoint.py", "model.py", "backbone.py", "losses.py", "data.py",
    "evaluate.py", "inference.py", "metrics.py", "predict.py",
    "native_sam.py", "backbone_sam2.py", "mosaic.py", "scalability.py", "identity.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_source_sha256():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in ENGINE_SOURCE_FILES}


def verify_engine_source(payload):
    """Reject stale mathematical code, while permitting deployment relocation.

    This checks file contents only. Training runtime/GPU equality is a separate
    exact-resume requirement, deliberately not imposed on inference.
    """
    recorded = payload.get("provenance", {}).get("engine_source_sha256")
    if not isinstance(recorded, dict):
        raise ValueError("Checkpoint has no engine_source_sha256 receipt; use the frozen source package "
                         "and its matching checkpoint. Unverified deployment is refused.")
    current = engine_source_sha256()
    missing = sorted(set(current) - set(recorded))
    extra = sorted(set(recorded) - set(current))
    changed = sorted(name for name in set(current) & set(recorded) if recorded[name] != current[name])
    if missing or extra or changed:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if extra:
            details.append("unexpected=" + ", ".join(extra))
        if changed:
            details.append("changed=" + ", ".join(changed))
        raise ValueError("Checkpoint engine source mismatch (" + "; ".join(details) + "). "
                         "Use the exact frozen package that trained this checkpoint; "
                         "do not edit its provenance to bypass verification.")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def atomic_json(path, payload):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(payload, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seed_everything(seed):
    # Must be set before the first CUDA BLAS call for deterministic CUDA math.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8; "
                         "set it before starting a fresh Python process")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return payload


def verify_base_receipt(model, payload):
    current, old = model.base.receipt, payload["base_receipt"]
    for key in ("checkpoint_sha256", "source_sha256", "canvas_size", "backend"):
        if old.get(key) != current.get(key):
            raise ValueError(f"Base model provenance mismatch: {key}")


def build_from_checkpoint(path, *, device="cpu", sam_repo=None, sam_checkpoint=None,
                          sam2_repo=None, sam2_checkpoint=None):
    payload = read_checkpoint(path)
    verify_engine_source(payload)
    from .model import ModelConfig, SamPathModel
    config = dict(payload["model_config"])
    if sam_repo is not None:
        config["sam_repo"] = str(sam_repo)
    if sam_checkpoint is not None:
        config["sam_checkpoint"] = str(sam_checkpoint)
    if sam2_repo is not None:
        config["sam2_repo"] = str(sam2_repo)
    if sam2_checkpoint is not None:
        config["sam2_checkpoint"] = str(sam2_checkpoint)
    seed_everything(config.get("init_seed", payload["train_config"]["seed"]))
    model = SamPathModel(ModelConfig(**config))
    verify_base_receipt(model, payload)
    model.load_trainable_state_dict(payload["model_state"])
    return model.to(device).eval(), payload


def checkpoint_payload(model, model_config, train_config, optimizer, scaler, progress, provenance):
    state = model.checkpoint_state_dict()
    return dict(format_version=1, model_state=state, model_config=asdict(model_config),
                train_config=train_config, base_receipt=model.base.receipt,
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                progress=dict(progress), rng=rng_state(), provenance=provenance,
                torch_version=str(torch.__version__),
                checkpoint_kind="full_toy" if model_config.backend == "toy" else "trainable_delta")
