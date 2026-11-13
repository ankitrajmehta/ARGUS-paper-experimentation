"""Train linear probes on extracted activations and report per-layer accuracy.

    midas-probes results/activations/train_InternVL2_5-2B_tiles6_answer.npz

Two honest splits are reported:

* held-out **scene** -- does the direction survive an unseen road scene?
* held-out **scene+phrase** -- stricter: unseen scene *and* unseen wording.

Holding out wording alone is deliberately omitted: it leaves every scene on both
sides of the split, and since the label is determined by the forced answer within
a scene, a probe memorises (scene, answer) and scores ~100% at layer 1. Compare
against the ~60% floor obtainable from the answer token alone.
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
    parser.add_argument("activations", type=Path, nargs="+")
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
    parser.add_argument(
        "--condition", nargs="+", default=None,
        help="restrict to these conditions: clean attack benign_text",
    )
    parser.add_argument("--subspace-dim", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    acts = probes.merge([probes.load_activations(p) for p in args.activations])
    print(f"{', '.join(p.name for p in args.activations)}")
    print(f"  {len(acts.rows)} vectors, {acts.n_layers} layers")
    print(f"  model={acts.model}  split={acts.split}  readout={acts.readout}")

    if args.condition:
        acts = acts.subset(np.isin(acts.column("condition"), args.condition))
        print(f"  conditions {args.condition} -> {len(acts.rows)} vectors")
    elif args.attacked_only:
        acts = acts.subset(acts.column("attacked").astype(bool))
        print(f"  attacked only -> {len(acts.rows)} vectors")

    y = acts.column(args.label).astype(int)
    print(f"  label '{args.label}': {int((y == 1).sum())} positive / {int((y == 0).sum())} negative")
    print(f"  {len(np.unique(acts.column('scene')))} scenes, "
          f"{len(np.unique(acts.column('phrase')))} phrases")

    results = {}
    for group_key, caption in (
        ("scene", "held-out scene"),
        ("scene+phrase", "held-out scene + phrase"),
    ):
        probe_col = "scene" if group_key == "scene+phrase" else group_key
        if len(np.unique(acts.column(probe_col))) < 2:
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
        else:
            # Printing an empty section under a heading reads as "the probe
            # learned nothing", which is a different claim from "this split
            # could not be evaluated". Say which.
            attack_phrases = np.unique(
                acts.column("phrase")[acts.column("condition") == "attack"]
            )
            print(
                f"  no fold had both classes in its test set, so this split could not "
                f"be scored -- nothing was measured here, which is not the same as a "
                f"probe that failed.\n"
                f"  the sweep has {len(attack_phrases)} attack wording(s); the strict "
                f"split needs a fold holding out an attack wording *and* a non-attack "
                f"one, so re-extract with --phrases 3 or more to get this number."
            )

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
