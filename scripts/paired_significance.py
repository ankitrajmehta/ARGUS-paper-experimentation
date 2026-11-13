"""Paired significance for the DriveLM conditions.

Every condition is evaluated on the *same* scenes, so the conditions are
paired and the unpaired +/-10 point interval from n=100 is the wrong yardstick
-- it is far too conservative here. McNemar's test uses only the scenes that
disagree between two conditions, which is the actual evidence.

    python scripts/paired_significance.py results/kaggle_drivelm
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    ``b`` and ``c`` are the two discordant counts. Under the null each
    discordant scene is a fair coin, so the p-value is the two-sided binomial
    tail -- exact rather than the chi-square approximation, which is unreliable
    when b + c is small, as it is for the closer pairs here.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(directory: Path) -> dict[str, dict[int, bool]]:
    """condition -> {scene_index: chose the attacker's action}."""
    out: dict[str, dict[int, bool]] = {}
    for path in sorted(directory.rglob("*.json")):
        payload = json.loads(path.read_text())
        if "summary" not in payload:
            continue
        name = "no sign" if not payload["attacked"] else path.parent.name
        # The CHAI substring rule, to match the headline numbers.
        out[name] = {
            r["scene_index"]: (r["action"] == r["target"])
            for r in payload["results"]
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path("results/kaggle_drivelm"))
    args = parser.parse_args(argv)

    conds = load(args.directory)
    if "no sign" not in conds:
        raise SystemExit("no baseline run found")

    print("Each row: scenes where only the first condition hit the target (b), "
          "only the second (c),\nand the exact two-sided McNemar p-value.\n")
    head = f"{'comparison':28s} {'b':>4s} {'c':>4s} {'net':>6s} {'p':>9s}  verdict"
    print(head)
    print("-" * len(head))

    order = ["no sign", "plain", "authority", "urgency", "override"]
    names = [c for c in order if c in conds] + [c for c in conds if c not in order]

    def row(a: str, bname: str) -> None:
        shared = sorted(set(conds[a]) & set(conds[bname]))
        b = sum(conds[a][i] and not conds[bname][i] for i in shared)
        c = sum(conds[bname][i] and not conds[a][i] for i in shared)
        p = exact_mcnemar(b, c)
        net = (c - b) / len(shared)
        mark = "significant" if p < 0.05 else "not separable"
        print(f"{a + ' vs ' + bname:28s} {b:4d} {c:4d} {net:+6.1%} {p:9.4f}  {mark}")

    # Every attacked condition against the baseline: does the sign work at all?
    for name in names:
        if name != "no sign":
            row("no sign", name)
    print()
    # Then the styles against each other: is any wording really better?
    attacked = [n for n in names if n != "no sign"]
    for a, b in combinations(attacked, 2):
        row(a, b)

    print("\nb = scenes the first condition won, c = scenes the second won. "
          "\nOnly discordant scenes carry information; concordant ones are "
          "uninformative about a difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
