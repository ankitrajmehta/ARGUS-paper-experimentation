"""Print a random sample of clean-set rows for a human to check.

Shows the image path, what the judge claimed to see, and the DriveLM label the
judge agreed with. Open the image, read the evidence line, decide whether the
two match. Disagreements are the point -- they are what the audit cannot
self-detect.

    python scripts/spotcheck.py            # 10 random rows
    python scripts/spotcheck.py -n 20 --seed 7
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

NAMES = {"a)": "PROCEED", "b)": "BRAKE"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", type=Path,
                   default=Path("results/clean_set/clean_200_balanced.json"))
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rows = json.loads(args.set.read_text())
    picked = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    for i, r in enumerate(picked, 1):
        print(f"\n[{i}/{len(picked)}]  label: {NAMES[r['action']]}  ({r['action']})")
        print(f"  image    : {r.get('front', '?')}")
        print(f"  claimed  : {r['visual_evidence']}")
        print(f"  drivelm  : {r['behavior']}")
    print(f"\nOpen each image and ask: does the evidence line describe what you see,")
    print(f"and does {', '.join(sorted(NAMES.values()))} match the road?")
    print("Note any row where it does not -- that is the number worth having.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
