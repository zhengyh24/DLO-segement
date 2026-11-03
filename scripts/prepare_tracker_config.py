"""Localize a published 2,000-image experiment config for a fresh training run."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    configs = Path(__file__).resolve().parents[1] / "results/sam_tracker/pilot/configs"
    parser.add_argument("--job", choices=sorted(f.stem for f in configs.glob("*.json")), required=True)
    parser.add_argument("--data", type=Path, required=True, help="Extracted pilot root containing native RGB/masks")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sam-repo", type=Path, required=True, help="Source of the selected SAM3 or SAM2 backend")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Base weights of that backend")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((configs / (args.job + ".json")).read_text(encoding="utf-8"))
    cfg["manifest"] = str(args.manifest.resolve(strict=True))
    cfg["data_root"] = str(args.data.resolve(strict=True))
    model = cfg["model"]
    sam = "sam2" if model["backend"] == "sam21" else "sam"
    for field in ("sam_repo", "sam_checkpoint", "sam2_repo", "sam2_checkpoint"):
        if field in model:
            model[field] = ""
    model[sam + "_repo"] = str(args.sam_repo.resolve(strict=True))
    model[sam + "_checkpoint"] = str(args.checkpoint.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
