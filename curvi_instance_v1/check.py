"""Functional contract checks; synthetic fixtures are NOT accuracy benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import TrackerDataset, exit_targets, native_crop
from .losses import tracking_loss
from .model import InstanceTracker
from .predict import InferenceConfig, load_checkpoint, predict_array
from .train import project_history, sequence_loss


class FixtureModel(torch.nn.Module):
    """Exact color masks to test the scheduler independently of learned quality."""
    def __init__(self):
        super().__init__()
        self.states = []

    def encode(self, rgb):
        return {}, (rgb[:, 2:3]-.5)*20

    def forward(self, rgb, seed, history, query=None, anchor=None, spatial_valid=None):
        if query is None:
            red = (rgb[:, 0:1]*seed).sum()
            green = (rgb[:, 1:2]*seed).sum()
            query = rgb.new_tensor([[float(green > red), 0.]])
            anchor = query.clone()
        else:
            assert seed.count_nonzero() == 0, "Continuation received a new seed"
        self.states.append((query.detach().cpu().clone(), anchor.detach().cpu().clone(), float(history.sum())))
        channel = int(query[0, 0].item())
        mask = (rgb[:, channel:channel+1]-.5)*20
        exits = rgb.new_full((1, 8), -10)
        exits[:, [0, 4] if channel else [2, 6]] = 10
        return {"mask_logits": mask, "seed_logits": self.encode(rgb)[1],
                "exit_logits": exits, "query": query+query.new_tensor([[0., 1.]]), "anchor": anchor}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(11)
    torch.set_num_threads(4)
    results = {}
    began = time.perf_counter()

    # Native negative origins and overlap mapping must preserve single pixels.
    full = np.arange(11*13).reshape(11, 13)
    crop = native_crop(full, (-3, 4), 8, fill=-1)
    assert crop[0, 3] == full[4, 0] and crop[6, 7] == full[10, 4]
    assert np.all(crop[:, :3] == -1) and np.all(crop[7] == -1)
    previous = torch.zeros(1, 1, 8, 8); previous[0, 0, 3, 6] = 1
    history = project_history([(previous, torch.tensor([[10, 20]]))], torch.tensor([[14, 21]]), 8)
    assert history.sum() == 1 and history[0, 0, 2, 2] == 1
    results["native_coordinate_and_history_projection"] = "pass"

    t = torch.zeros(8, 1, 32, 32)
    for b, (y, x) in enumerate(((0,16), (0,31), (16,31), (31,31), (31,16), (31,0), (16,0), (0,0))):
        t[b, 0, y, x] = 1
    assert torch.equal(exit_targets(t, torch.zeros_like(t), torch.ones_like(t)), torch.eye(8))
    assert exit_targets(t, t, torch.ones_like(t)).sum() == 0
    assert exit_targets(t, torch.zeros_like(t), torch.zeros_like(t)).sum() == 0
    results["eight_exits_and_history_suppression"] = "pass"

    ds = TrackerDataset(args.manifest, "train", tile=64, steps=3, seed=4,
                        allow_condition=True, negative_probability=0)
    item = ds[0]
    assert item["seed"][1:].count_nonzero() == 0
    assert ((item["target"] > 0) & (item["other"] > 0)).sum() == 0
    assert (item["valid"] <= item["spatial_valid"]).all()
    assert torch.all(item["origins"] >= 0)
    negative = TrackerDataset(args.manifest, "train", tile=64, steps=3, seed=4,
                              allow_condition=True, negative_probability=1)[0]
    assert negative["target"].sum() == 0 and negative["step_valid"].tolist() == [1., 0., 0.]
    val = TrackerDataset(args.manifest, "validation", tile=64, seed=4, allow_condition=True)
    a = val[0]; val.epoch = 9; b = val[0]
    assert torch.equal(a["rgb"], b["rgb"]) and torch.equal(a["origins"], b["origins"])
    try:
        TrackerDataset(args.manifest, "test", allow_condition=True)
        raise AssertionError("Test split should be rejected")
    except ValueError:
        pass
    results["data_split_masks_negative_seeds_validation_fixed"] = "pass"

    model = InstanceTracker()
    rgb = torch.rand(2, 3, 65, 79)
    seed = torch.zeros(2, 1, 65, 79); seed[:, :, 20, 30] = 1
    history = torch.zeros_like(seed)
    spatial = torch.ones_like(seed); spatial[:, :, -4:] = 0
    first = model(rgb, seed, history, spatial_valid=spatial)
    second = model(rgb, seed*0, history, query=first["query"], anchor=first["anchor"], spatial_valid=spatial)
    assert first["mask_logits"].shape == seed.shape
    assert second["anchor"] is first["anchor"]
    batch = {"target": (rgb[:, :1] > .5).float(), "valid": spatial,
             "other": (rgb[:, 1:2] > .5).float(), "seed_target": seed, "exits": torch.ones(2,8)}
    loss, _ = tracking_loss(second, batch)
    loss.backward()
    gru_gradient = sum(float(v.grad.abs().sum()) for v in model.query_update.parameters() if v.grad is not None)
    assert gru_gradient > 0
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    damaged = {k: v.detach().clone() for k, v in second.items()}
    damaged_batch = {k: v.clone() for k, v in batch.items()}
    for key in ("mask_logits", "seed_logits"):
        damaged[key][spatial == 0] = float("nan")
    for key in ("target", "other", "seed_target"):
        damaged_batch[key][spatial == 0] = float("nan")
    changed_loss, _ = tracking_loss(damaged, damaged_batch)
    assert torch.equal(loss.detach(), changed_loss)
    all_invalid = dict(batch, valid=spatial*0)
    empty_loss, _ = tracking_loss(second, all_invalid)
    assert empty_loss == 0
    results["odd_shape_recurrent_gradient_padding_loss"] = {"status": "pass", "gru_gradient_l1": gru_gradient}

    # Scheduler: long cable crosses several native windows with one persistent ID.
    rgb_np = np.zeros((37, 100, 3), np.uint8)
    rgb_np[18, :, 0] = 255; rgb_np[18, 4, 2] = 255
    fixture = FixtureModel()
    cfg = InferenceConfig(tile=32, stride=16, max_seeds=4, max_steps=16, max_calls=100, peak_radius=2)
    predicted = predict_array(fixture, rgb_np, cfg)
    assert np.all(predicted["instance_ids"][18] == 1)
    assert predicted["visible"].sum() == 100 and predicted["unknown"].sum() == 0
    assert predicted["audit"]["tracking_calls"] > 1 and not predicted["audit"]["truncated"]
    assert all(torch.equal(anchor, fixture.states[0][1]) for _, anchor, _ in fixture.states)
    assert any(hist > 0 for _, _, hist in fixture.states[1:])
    assert predicted["audit"]["input_pixels"] == predicted["audit"]["network_calls"]*32*32
    results["automatic_seed_native_rollout_persistent_id"] = predicted["audit"]["network_calls"]

    # Two touching objects cannot be merged just because of their crossing.
    cross = np.zeros((35, 47, 3), np.uint8)
    cross[18, :, 0] = 255; cross[:, 25, 1] = 255
    cross[18, 4, 2] = 255; cross[3, 25, 2] = 255
    predicted = predict_array(FixtureModel(), cross, InferenceConfig(tile=64, stride=32, max_seeds=4, max_calls=10, peak_radius=2))
    assert set(np.unique(predicted["instance_ids"])) == {0, 1, 2}
    assert predicted["unknown"][18, 25] and predicted["unknown"].sum() == 1
    assert predicted["visible"].sum() == 35+47-1
    limited = predict_array(FixtureModel(), rgb_np, InferenceConfig(tile=32, stride=16, max_calls=1))
    assert limited["audit"]["network_calls"] == 1 and limited["audit"]["truncated"]
    results["crossing_conflict_unknown_and_budget_enforcement"] = "pass"

    # No state-dict reload approximation, and smoke checkpoints cannot deploy silently.
    checkpoint = out/"fixture.pt"
    model.eval()
    torch.save({"schema":1, "kind":"smoke", "optimizer_steps":0, "model_config":{}, "model":model.state_dict()}, checkpoint)
    try:
        load_checkpoint(checkpoint, torch.device("cpu"))
        raise AssertionError("Untrained checkpoint was accepted")
    except ValueError:
        pass
    restored, _ = load_checkpoint(checkpoint, torch.device("cpu"), allow_smoke=True)
    with torch.no_grad():
        orig = model(rgb, seed, history, spatial_valid=spatial)
        copy = restored(rgb, seed, history, spatial_valid=spatial)
    assert all(torch.equal(orig[k], copy[k]) for k in orig)
    if args.checkpoint:
        actual, metadata = load_checkpoint(Path(args.checkpoint), torch.device("cpu"), allow_smoke=True)
        assert metadata["optimizer_steps"] > 0
        with torch.no_grad():
            assert torch.isfinite(actual(rgb[:1],seed[:1],history[:1])["mask_logits"]).all()
    results["checkpoint_exact_reload_and_smoke_guard"] = "pass"
    results.update(status="passed", elapsed_seconds=time.perf_counter()-began,
                   scientific_accuracy_result=False, model_parameters=sum(v.numel() for v in model.parameters()),
                   source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    (out/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
