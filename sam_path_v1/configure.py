"""Write four paired pilot configurations using the server's own paths."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .model import ModelConfig
from .train import TrainConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--sam-repo", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New configuration directory")
    parser.add_argument("--allow-condition", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.output)
    configs = {}
    for variant in "ABCD":
        model = ModelConfig(variant=variant, sam_repo=str(Path(args.sam_repo).resolve()),
                            sam_checkpoint=str(Path(args.sam_checkpoint).resolve()),
                            num_queries=args.queries, tracker_steps=args.steps, init_seed=args.seed)
        train = TrainConfig(manifest=str(Path(args.manifest).resolve()), model=asdict(model),
                            data_root=str(Path(args.data_root).resolve()) if args.data_root else None,
                            allow_condition=args.allow_condition, device=args.device,
                            epochs=args.epochs, seed=args.seed)
        configs[variant] = asdict(train)
    root.mkdir(parents=True, exist_ok=False)
    for variant, config in configs.items():
        (root / f"{variant}.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    smoke = dict(configs["C"])
    smoke.update(max_batches=2, save_every=1)
    (root / "smoke_C.json").write_text(json.dumps(smoke, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"configs": str(root.resolve()), "variants": list(configs),
                      "smoke_budget": "2 update opportunities, separate fresh output"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
