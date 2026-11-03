"""Run the unchanged evidence reporter with disk-backed report handles."""
import argparse
from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
import sys


class ReportHandle(Mapping):
    def __init__(self, path, load):
        self.path = path
        self.load = load

    def __getitem__(self, key):
        return self.load(self.path)[key]

    def __iter__(self):
        return iter(self.load(self.path))

    def __len__(self):
        return len(self.load(self.path))


def build(engine, protocol_path, runs, output):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(engine).resolve()))
    from sam_path_v1 import evidence_report as reporter

    original = reporter.read
    protocol = original(protocol_path)
    runs = Path(runs).resolve()
    expected_path = runs / 'expected_evaluations.json'
    tasks = original(expected_path) if expected_path.is_file() else protocol.get('expected_evaluations', [])
    paths = set()
    for task in tasks:
        path = reporter.report_path(runs, task['report']).resolve()
        paths.add(path)
        if task.get('kind') == 'robustness':
            paths.update(path.parent / f'severity_{level}.json' for level in (0, 1, 2, 4, 8))

    @lru_cache(maxsize=2)
    def load(path):
        return original(path)

    def read(path):
        resolved = Path(path).resolve()
        if resolved in paths:
            # Parse inside the original reporter's try/except, so unreadable
            # reports are handled at exactly the same point as the eager reader.
            load(resolved)
            return ReportHandle(resolved, load)
        return original(path)

    reporter.read = read
    try:
        result = reporter.build_report(protocol, runs, output)
        return result, dict(report_handles=len(paths), cache_size=2, hits=load.cache_info().hits,
                            misses=load.cache_info().misses, reporter_file=reporter.__file__)
    finally:
        reporter.read = original
        load.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'protocol', 'runs', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result, memory = build(args.engine, args.protocol, args.runs, args.output)
    print(json.dumps(dict(complete=result['complete'], memory=memory), ensure_ascii=False))


if __name__ == '__main__':
    main()
