"""Sample a balanced, blind review set from DriveLM for manual label audit.

DriveLM's behaviour sentence says what the human driver *did*, not what was
*correct*. This samples frames so that judgement can be made from the image
alone and only afterwards compared with DriveLM's label.

Writes two files, deliberately separate:

* ``review_manifest.json`` -- image paths only, no labels. What the judge sees.
* ``review_key.json``      -- the hidden DriveLM labels, opened after judging.

    python scripts/build_review_set.py --per-class 30
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from midas.cli.build_scenes import action_from_behavior, AMBIGUOUS_BEHAVIORS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qa", type=Path, default=Path("data/drivelm/v1_1_train_nus.json"))
    p.add_argument("--root", type=Path, default=Path("data/drivelm"))
    p.add_argument("--out", type=Path, default=Path("results/clean_set"))
    p.add_argument("--per-class", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    qa = json.loads(args.qa.read_text())
    pool: dict[str, list[dict]] = {"a)": [], "b)": []}
    for scene_token, scene in qa.items():
        for frame_token, frame in scene["key_frames"].items():
            answers = (frame.get("QA") or {}).get("behavior", []) or []
            if not answers:
                continue
            behavior = answers[0]["A"]
            if any(x in behavior.lower() for x in AMBIGUOUS_BEHAVIORS):
                continue
            action = action_from_behavior(behavior)
            if action not in pool:
                continue
            front = (frame.get("image_paths") or {}).get("CAM_FRONT")
            if not front:
                continue
            path = args.root / front.replace("../nuscenes/", "nuscenes/")
            if not path.exists():
                continue
            pool[action].append({
                "scene_token": scene_token,
                "frame_token": frame_token,
                "front": str(path),
                "behavior": behavior,
            })

    print(f"pool  a)={len(pool['a)'])}  b)={len(pool['b)'])}")
    rng = random.Random(args.seed)
    manifest, key = [], []
    for action in ("a)", "b)"):
        picked = rng.sample(pool[action], min(args.per_class, len(pool[action])))
        for row in picked:
            manifest.append({"id": len(manifest), "front": row["front"]})
            key.append({
                "id": len(key),
                "drivelm_action": action,
                "behavior": row["behavior"],
                "scene_token": row["scene_token"],
                "frame_token": row["frame_token"],
            })

    order = list(range(len(manifest)))
    rng.shuffle(order)
    manifest = [{**manifest[i], "id": n} for n, i in enumerate(order)]
    key = [{**key[i], "id": n} for n, i in enumerate(order)]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "review_manifest.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "review_key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {len(manifest)} frames to {args.out} (labels shuffled and hidden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
