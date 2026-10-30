"""Re-derive the benign action for each scene from the model under test.

CHAI ship a ``correct_internvl`` label per scene, obtained by running *their*
model unattacked and taking the mode over eleven runs. We inherited those
labels, and they do not describe our model: InternVL2.5-8B agrees with them on
only 3 of 8 Transferability scenes.

That matters because the attack target is defined as the opposite of the benign
action. Where the stored label is wrong, the "target" is the action the model
already prefers, and the no-attack rate climbs to an implausible 62.5% -- which
is what we measured, and why that split was uninterpretable.

This reproduces CHAI's labelling procedure against whichever model is being
evaluated: run the unattacked scene ``--runs`` times with sampling, take the
mode. Results are written as a label file that ``midas-eval --labels`` consumes.

    midas-labels --split test --model OpenGVLab/InternVL2_5-8B --runs 11
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from midas import config, data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model", default=config.MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--runs", type=int, default=11,
        help="unattacked runs per scene; CHAI use 11 and take the mode",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/labels"))
    args = parser.parse_args(argv)

    scenes = data.load_scenes(args.split)
    if args.limit:
        scenes = scenes[: args.limit]

    from midas.models.internvl import InternVLDriveLM

    print(f"loading {args.model} ...")
    model = InternVLDriveLM(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        device_map=args.device_map,
        max_tiles=args.max_tiles,
        max_new_tokens=args.max_new_tokens,
        # Sampling, not greedy: with greedy decoding every run is identical and
        # the mode over eleven runs is just one run. CHAI sample here too.
        do_sample=True,
    )

    labels: dict[str, dict] = {}
    agree = 0
    for scene in scenes:
        images = scene.load_images()
        votes: Counter = Counter()
        for _ in range(args.runs):
            reply = model.ask(images)
            if reply.action is not None:
                votes[reply.action] += 1
            model.free_cache()

        if not votes:
            print(f"  scene {scene.index:2d}: no readable answer in {args.runs} runs; skipped")
            continue

        action, count = votes.most_common(1)[0]
        matches = action == scene.correct_internvl
        agree += matches
        labels[str(scene.index)] = {
            "action": action,
            "votes": dict(votes),
            "confidence": count / sum(votes.values()),
            "chai_label": scene.correct_internvl,
            "agrees_with_chai": matches,
        }
        mark = "same " if matches else "DIFFER"
        print(
            f"  {mark} scene {scene.index:2d}: ours={action} chai={scene.correct_internvl} "
            f"votes={dict(votes)}",
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.split}_{Path(args.model).name}_tiles{args.max_tiles}.json"
    path.write_text(
        json.dumps(
            {
                "split": args.split,
                "model": args.model,
                "max_tiles": args.max_tiles,
                "runs": args.runs,
                "labels": labels,
            },
            indent=2,
        )
    )
    n = len(labels)
    print(f"\n{agree}/{n} scenes agree with CHAI's stored label")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
