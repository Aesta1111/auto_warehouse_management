"""Compare exact per-seed TA-RWARE trace hashes from two platforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def hashes(summary: dict[str, Any]) -> dict[int, str]:
    return {int(item["seed"]): str(item["trace_sha256"]) for item in summary["seeds"]}


def main() -> None:
    args = parse_args()
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    left_hashes = hashes(left)
    right_hashes = hashes(right)
    seeds = sorted(set(left_hashes) | set(right_hashes))
    comparisons = [
        {
            "seed": seed,
            "left_sha256": left_hashes.get(seed),
            "right_sha256": right_hashes.get(seed),
            "exact": left_hashes.get(seed) == right_hashes.get(seed),
        }
        for seed in seeds
    ]
    summary = {
        "schema_version": 1,
        "left": str(args.left),
        "right": str(args.right),
        "all_exact": all(item["exact"] for item in comparisons),
        "seeds": comparisons,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
