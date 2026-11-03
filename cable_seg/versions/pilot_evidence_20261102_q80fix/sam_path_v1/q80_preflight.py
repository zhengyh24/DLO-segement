"""Exhaust all five epochs of the fixed Q80 sampler before admitting training."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from .checkpoint import atomic_json, engine_source_sha256, sha256
from .data import NativeInstanceDataset, load_record
from .mosaic import NativeMosaicDataset
from .train import TrainConfig


FAILED_SEED = 14223980488388098341


def run(config, output, threads=4):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    start = time.monotonic()
    data = NativeInstanceDataset(config.manifest, "train", config.model["canvas_size"],
                                config.seed, config.allow_condition, config.data_root)
    if len(data) != 2000 or config.epochs != 5 or config.seed != 42:
        raise ValueError("This recovery admission is specifically 2000 x 5, seed 42")
    cache, cache_bytes = {}, 0
    def status(**values):
        atomic_json(output/"status.json", dict(elapsed_seconds=time.monotonic()-start, **values))
    status(phase="loading_verified_train_records", records=0)
    # The cache is private to this CPU preflight subprocess and is discarded
    # before smoke/formal training. Hash and geometry validation are unchanged.
    def read_row(row):
        return str(row["id"]), load_record(data.root, row)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for ident, arrays in pool.map(read_row, data.rows):
            for array in arrays:
                array.flags.writeable = False
                cache_bytes += array.nbytes
            if cache_bytes > 32*1024**3:
                raise MemoryError("Preflight cache exceeds its explicit 32 GiB budget")
            if max(arrays[0].shape[:2]) > data.canvas_size:
                raise ValueError("Original train RGB exceeds the fixed native canvas")
            cache[ident] = arrays
            if len(cache) % 100 == 0:
                status(phase="loading_verified_train_records", records=len(cache), cache_bytes=cache_bytes)
    def cached(root, row):
        if Path(root) != data.root or row["split"] != "train":
            raise ValueError("Preflight cache must only serve frozen train donors")
        return cache[str(row["id"])]
    count, repaired, seen_failed, density = 0, [], False, Counter()
    manifest_hash = sha256(config.manifest)
    with patch("sam_path_v1.mosaic.load_record", new=cached):
        mosaic = NativeMosaicDataset(data, config.mosaic, config.model["num_queries"])
        with (output/"recipes.jsonl").open("x", encoding="utf-8") as stream:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                for epoch in range(config.epochs):
                    mosaic.epoch = epoch
                    for index, receipt in enumerate(pool.map(mosaic.inspect_recipe, range(len(data)))):
                        item = dict(epoch_index=epoch, index=index, id=data.rows[index]["id"], **receipt)
                        if receipt["kind"] == "mosaic":
                            n = receipt["instances"]
                            if not 25 <= n <= 64 or len({r["rgb_sha256"] for r in receipt["recipe"]}) != 16:
                                raise ValueError("Density/distinct-donor admission failed")
                            if any(r["split"] != "train" for r in receipt["recipe"]):
                                raise ValueError("Non-training donor in recipe")
                            density[n] += 1
                        if receipt.get("repair"):
                            repaired.append(item)
                            # Materialize every repaired case to test actual masks,
                            # padding, and deterministic correspondence to the plan.
                            sample = mosaic[index]
                            if sample["meta"]["mosaic"] != {k:v for k,v in receipt.items() if k != "kind"}:
                                raise ValueError("Plan/materialized repaired recipe mismatch")
                            if len(sample["masks"]) != receipt["instances"]:
                                raise ValueError("Repaired target cardinality mismatch")
                            if not bool((sample["masks"].sum(0) <= 1).all()):
                                raise ValueError("Repaired native targets overlap")
                            del sample
                        if receipt["seed"] == FAILED_SEED:
                            if not receipt.get("repair"):
                                raise ValueError("Known failed sample was not explicitly repaired")
                            seen_failed = True
                        stream.write(json.dumps(item, allow_nan=False)+"\n")
                        count += 1
                        if count % 100 == 0:
                            stream.flush()
                            status(phase="all_epoch_recipe_check", checked=count, total=10000,
                                   mosaics=sum(density.values()), repairs=len(repaired), cache_bytes=cache_bytes)
    if count != 10000 or not seen_failed:
        raise ValueError("Exhaustive preflight did not cover all positions and the known failure")
    result = dict(passed=True, checked=count, epochs=5, train_members=2000, seed=42,
        mosaic_samples=sum(density.values()), native_samples=count-sum(density.values()),
        density_histogram=dict(density), repairs=repaired, known_failed_seed_repaired=seen_failed,
        failures=0, manifest_sha256=manifest_hash, engine_source_sha256=engine_source_sha256(),
        recipes_sha256=sha256(output/"recipes.jsonl"), cache_bytes=cache_bytes,
        elapsed_seconds=time.monotonic()-start, no_test_or_validation_donors=True,
        interpretation="Exhaustive software/data admission; no training accuracy or general sampler guarantee")
    atomic_json(output/"completion.json", result)
    status(phase="completed", checked=count, repairs=len(repaired))
    print(json.dumps({k:v for k,v in result.items() if k not in ("repairs", "engine_source_sha256")}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--threads", type=int, default=4)
    args=p.parse_args()
    run(TrainConfig(**json.loads(Path(args.config).read_text())), args.output, args.threads)


if __name__ == "__main__":
    main()
