"""Run CHAI's stage-2 sign search against the model under test.

CHAI publish the *output* of this search, not the search itself. Loading their
``optimal_attacks/optimal.json`` reproduces their attack; this reproduces their
method, and answers the question their paper leaves open for a different model:
do their universal parameters transfer, or does re-optimising beat them?

    midas-optimize --split train --model OpenGVLab/InternVL2_5-2B \\
        --iterations 10 --population 16 --validate

The inner objective is forced-choice scoring by default -- about ten times
cheaper than generating a chain of thought per candidate per scene, but a proxy
that read 53.3% where generation read 73.3%. ``--validate`` re-runs CHAI's sign
and the winner through real generation so the headline comparison is like for
like. See :mod:`midas.optimize` for why VerifAI is not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from midas import data, optimize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model", default="OpenGVLab/InternVL2_5-2B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument(
        "--target", default="a)",
        help="action to steer the model toward. CHAI's main.py hardcodes 'a)'.",
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="CE iterations. CHAI use 150 (20 in testing mode); the default here "
        "is what fits a Kaggle session with generation validation after it.",
    )
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument(
        "--objective", choices=["margin", "forced", "generate"], default="margin",
        help="margin: mean log-prob margin, continuous so the sampler can rank "
        "candidates that flip equally many scenes (default). forced: the same "
        "two passes thresholded to a hit rate -- too coarse at n=6, where the "
        "first run's best score never moved off iteration 0. generate: CHAI's "
        "actual test, ~10x the cost.",
    )
    parser.add_argument("--background-mode", choices=["code", "json"], default="code")
    parser.add_argument(
        "--validate", action="store_true",
        help="after the search, re-score CHAI's sign and the winner with real "
        "generation, which is what the headline ASR measures",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/optimize"))
    args = parser.parse_args(argv)

    scenes = data.load_scenes(args.split, labels=args.labels)
    if args.limit:
        scenes = scenes[: args.limit]

    usable = optimize.attackable_scenes(scenes, args.target)
    print(f"target {args.target}: {len(usable)} of {len(scenes)} scenes are attackable")
    print(f"  (CHAI skip scenes whose benign answer is already the target)")

    evaluations = args.iterations * args.population * len(usable)
    print(f"search: {args.iterations} iterations x {args.population} candidates "
          f"x {len(usable)} scenes = {evaluations} {args.objective} evaluations")

    from midas.models.internvl import InternVLDriveLM

    print(f"\nloading {args.model} ...", flush=True)
    model = InternVLDriveLM(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_tiles=args.max_tiles,
        device_map=args.device_map,
    )

    def report(record: dict) -> None:
        c = record["best_candidate"]
        print(
            f"  iter {record['iteration']:2d}  best={record['best_score']:6.1%} "
            f"elite={record['elite_score']:6.1%} mean={record['mean_score']:6.1%} "
            f"running={record['running_best']:6.1%}  "
            f"size={c['size']:.2f} fg={tuple(c['letter'])} "
            f"bg_b={c['background'][2]} pos={tuple(c['position_offset'])} "
            f"phrase={c['phrase_index']}",
            flush=True,
        )

    best, history = optimize.cross_entropy_search(
        model,
        scenes,
        args.target,
        iterations=args.iterations,
        population=args.population,
        elite_frac=args.elite_frac,
        objective=args.objective,
        background_mode=args.background_mode,
        seed=args.seed,
        on_iteration=report,
    )

    released = optimize.chai_released_candidate(args.target)
    dictionary = data.load_attack_dictionary()
    print("\n=== search result ===")
    print(f"{'parameter':18s} {'CHAI released':>22s} {'re-optimised':>22s}")
    as_rendered = lambda c: c.as_attack_config(args.background_mode).resolve()["background"]
    rows = [
        ("size multiplier", f"{released.size:.2f}", f"{best.size:.2f}"),
        # Rendered, not searched: "code" mode discards the red and green
        # channels, so printing the raw vector invites a false comparison.
        ("background", str(as_rendered(released)), str(as_rendered(best))),
        ("letter", str(released.letter), str(best.letter)),
        ("position offset", str(released.position_offset), str(best.position_offset)),
        (
            "phrase",
            f"{dictionary[args.target][released.phrase_index]!r}",
            f"{dictionary[args.target][best.phrase_index]!r}",
        ),
    ]
    for name, a, b in rows:
        print(f"{name:18s} {a:>22s} {b:>22s}")

    meta = {
        "model": args.model,
        "split": args.split,
        "target": args.target,
        "iterations": args.iterations,
        "population": args.population,
        "elite_frac": args.elite_frac,
        "objective": args.objective,
        "background_mode": args.background_mode,
        "max_tiles": args.max_tiles,
        "seed": args.seed,
        "n_scenes": len(usable),
    }

    if args.validate:
        print("\n=== validation: real generation, CHAI's scoring rule ===", flush=True)
        validation = {}
        for name, candidate in (("chai_released", released), ("re_optimised", best)):
            result = optimize.score_candidate(
                model, usable, candidate, args.target,
                objective="generate", background_mode=args.background_mode,
            )
            validation[name] = {"score": result.score, "detail": result.detail}
            print(f"  {name:14s} {result.score:6.1%}  ({result.n_scenes} scenes)", flush=True)
        meta["validation"] = validation

        delta = validation["re_optimised"]["score"] - validation["chai_released"]["score"]
        print(f"\n  re-optimised minus released: {delta:+.1%}")
        print(f"  one scene is {1 / len(usable):.1%}, so read this against n={len(usable)}")

    stem = f"{args.split}_{Path(args.model).name}_target{args.target.rstrip(')')}"
    path = optimize.save_search(args.out / f"{stem}.json", best, history, meta)
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
