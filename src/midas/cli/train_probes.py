"""Train linear probes on extracted activations and report per-layer accuracy.

    midas-probes results/activations/train_InternVL2_5-2B_tiles6_answer.npz

Two generalisation tests are reported, because they fail for different reasons:

* held-out **scene** -- does the direction survive a new road scene?
* held-out **phrase** -- or has the probe just learned one sign's wording?

A probe that scores well by scene but poorly by phrase has memorised the text,
which is exactly the failure the defense must not be built on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from midas import probes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("activations", type=Path)
    parser.add_argument(
        "--label",
        default="follows_injection",
        help="column to predict (default: is the model about to follow the sign)",
    )
    parser.add_argument(
        "--attacked-only",
        action="store_true",
        default=True,
        help="train only on attacked images, as ARGUS do (default)",
    )
    parser.add_argument("--include-clean", dest="attacked_only", action="store_false")
    parser.add_argument("--subspace-dim", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    acts = probes.load_activations(args.activations)
    print(f"{args.activations.name}: {len(acts.rows)} vectors, {acts.n_layers} layers")
    print(f"  model={acts.model}  split={acts.split}  readout={acts.readout}")

    if args.attacked_only:
        acts = acts.subset(acts.column("attacked").astype(bool))
        print(f"  attacked only -> {len(acts.rows)} vectors")

    y = acts.column(args.label).astype(int)
    print(f"  label '{args.label}': {int((y == 1).sum())} positive / {int((y == 0).sum())} negative")
    print(f"  {len(np.unique(acts.column('scene')))} scenes, "
          f"{len(np.unique(acts.column('phrase')))} phrases")

    results = {}
    for group_key, caption in (("scene", "held-out scene"), ("phrase", "held-out phrase")):
        if len(np.unique(acts.column(group_key))) < 2:
            print(f"\n{caption}: skipped (needs >=2 groups)")
            continue

        print(f"\n=== {caption} ===")
        scores = probes.sweep_layers(acts, label_key=args.label, group_key=group_key)
        results[group_key] = [
            {"layer": s.layer, "accuracy": s.accuracy, "std": s.std} for s in scores
        ]
        for s in scores:
            if np.isnan(s.accuracy):
                continue
            print(f"  layer {s.layer:2d}  {s.accuracy:6.1%} +/-{s.std:5.1%}  {s.bar()}")

        valid = [s for s in scores if not np.isnan(s.accuracy)]
        if valid:
            best = max(valid, key=lambda s: s.accuracy)
            print(f"  best: layer {best.layer} at {best.accuracy:.1%}")

    # The subspace is only meaningful if the probe generalises; build it at the
    # best layer by held-out scene.
    if "scene" in results:
        valid = [r for r in results["scene"] if not np.isnan(r["accuracy"])]
        best_layer = max(valid, key=lambda r: r["accuracy"])["layer"]
        try:
            basis = probes.safety_subspace(acts, best_layer, dim=args.subspace_dim)
            direction = probes.fit_direction(acts, best_layer, label_key=args.label)
            overlap = float(np.linalg.norm(basis @ direction))
            print(f"\nsafety subspace at layer {best_layer}: {basis.shape[0]}x{basis.shape[1]}")
            print(f"  probe direction energy inside subspace: {overlap:.1%}")
        except ValueError as exc:
            print(f"\nsubspace: skipped ({exc})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
