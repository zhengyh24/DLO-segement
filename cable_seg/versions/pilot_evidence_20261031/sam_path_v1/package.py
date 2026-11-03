"""Create a small portable source archive; no datasets or weights are bundled."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    target = Path(args.output).resolve()
    if source == target or source in target.parents:
        raise ValueError("Archive must be outside the source package")
    if target.exists() or target.with_suffix(".sha256.json").exists():
        raise FileExistsError("Use a fresh archive path")
    files = sorted(p for p in source.rglob("*") if p.is_file()
                   and "__pycache__" not in p.parts and p.suffix in (".py", ".md", ".txt", ".json", ".sh"))
    target.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = "sam_path_v1/" + path.relative_to(source).as_posix()
            content = path.read_bytes()
            archive.writestr(name, content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        archive.writestr("SOURCE_SHA256.json", json.dumps(hashes, indent=2)+"\n")
    receipt = {"archive": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "bytes": target.stat().st_size, "source_files": len(files), "files": hashes,
               "requires_external": ["matching SAM3.1 source", "SAM3.1 multiplex checkpoint", "PNG/NPZ frozen dataset"]}
    target.with_suffix(".sha256.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
