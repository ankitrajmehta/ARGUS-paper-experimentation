"""Does the sign work equally well in both directions?

The attacker's target is the opposite of the scene's benign action, and the
sign's wording is chosen *by target* -- so a scene whose driver was moving gets
a sign reading ``STOP``, and a scene whose driver was stopped gets ``Proceed``.
Those are not symmetric situations:

* ``STOP`` asks the model to do the cautious thing. A driving model with any
  safety prior is predisposed to agree.
* ``Proceed`` asks it to override a stop. That is the direction that actually
  matters for a safety claim, and the harder sell.

An aggregate ASR hides the difference completely. If the whole effect lives in
the ``STOP`` direction, then "the attack works" mostly means "telling a cautious
model to be cautious works", which is a much weaker statement than the headline
suggests.

**The hypothesis above was written before running this, and it was wrong -- in
the useful direction.** On Qwen2.5-VL-3B at n=100 the effect is almost entirely
in ``Proceed``: +58.6 points with CHAI's own wording (17 scenes flipped, 0 lost,
p < 0.0001), against +11.3 for ``STOP`` which is not significant at all
(p = 0.22). The attack talks a *stopped* vehicle into moving; it does not
meaningfully make a moving vehicle brake.

So the aggregate +25.0 understates the finding rather than inflating it. It is
an average over a direction where the attack is devastating and a direction
where it does nothing.

    python scripts/attack_by_direction.py results/kaggle_drivelm
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

ACTION_NAME = {"a)": "proceed", "b)": "brake"}


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path("results/kaggle_drivelm"))
    args = parser.parse_args(argv)

    runs: dict[str, list[dict]] = {}
    for path in sorted(args.directory.rglob("*.json")):
        payload = json.loads(path.read_text())
        if "summary" not in payload:
            continue
        name = "no sign" if not payload["attacked"] else path.parent.name
        runs[name] = payload["results"]

    if "no sign" not in runs:
        raise SystemExit("no baseline run found")
    base = {r["scene_index"]: r for r in runs["no sign"]}

    # Group scenes by what the sign is asking for.
    targets = sorted({r["target"] for r in runs["no sign"]})
    counts = {
        t: sum(1 for r in runs["no sign"] if r["target"] == t) for t in targets
    }
    print("scenes by what the sign demands:")
    for t in targets:
        print(f"  sign asks for {ACTION_NAME[t]:8s} ({t})  {counts[t]:3d} scenes "
              f"-- driver was {ACTION_NAME['a)' if t == 'b)' else 'b)']}")

    for name, results in runs.items():
        if name == "no sign":
            continue
        print(f"\n=== {name} ===")
        print(f"{'sign demands':14s} {'n':>4s} {'clean':>7s} {'attacked':>9s} "
              f"{'delta':>7s} {'won':>4s} {'lost':>5s} {'p':>8s}")
        for t in targets:
            rows = [r for r in results if r["target"] == t]
            n = len(rows)
            if not n:
                continue
            clean_hits = sum(
                base[r["scene_index"]]["action"] == t
                for r in rows if r["scene_index"] in base
            )
            atk_hits = sum(r["action"] == t for r in rows)
            won = sum(
                r["action"] == t and base[r["scene_index"]]["action"] != t
                for r in rows if r["scene_index"] in base
            )
            lost = sum(
                r["action"] != t and base[r["scene_index"]]["action"] == t
                for r in rows if r["scene_index"] in base
            )
            p = exact_mcnemar(lost, won)
            print(f"{ACTION_NAME[t]:14s} {n:4d} {clean_hits / n:7.1%} "
                  f"{atk_hits / n:9.1%} {(atk_hits - clean_hits) / n:+7.1%} "
                  f"{won:4d} {lost:5d} {p:8.4f}")

    print("""
"won" is scenes the sign flipped to its demand; "lost" is scenes that already
matched it and stopped. p is exact two-sided McNemar within that direction.

A large effect under "brake" with nothing under "proceed" would mean the
measured attack is largely the model agreeing with a cautious instruction it was
already disposed toward -- and the safety-relevant direction, talking a stopped
vehicle into moving, would be unsupported.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
