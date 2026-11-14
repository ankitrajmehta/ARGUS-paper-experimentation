"""Extend the blind review pool, excluding frames already judged.

Same protocol as build_review_set: manifest holds image paths only, the key
holds DriveLM's labels and is not opened until judging is finished.

    python scripts/extend_review_set.py --want-a 150 --want-b 120
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from midas.cli.build_scenes import action_from_behavior, AMBIGUOUS_BEHAVIORS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qa", type=Path, default=Path("data/drivelm/v1_1_train_nus.json"))
    p.add_argument("--root", type=Path, default=Path("data/drivelm"))
    p.add_argument("--out", type=Path, default=Path("results/clean_set"))
    p.add_argument("--want-a", type=int, default=150)
    p.add_argument("--want-b", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    seen = set()
    old_key = args.out / "review_key.json"
    if old_key.exists():
        seen = {r["frame_token"] for r in json.loads(old_key.read_text())}
    print(f"excluding {len(seen)} already-judged frames")

    qa = json.loads(args.qa.read_text())
    pool: dict[str, list[dict]] = {"a)": [], "b)": []}
    for scene_token, scene in qa.items():
        for frame_token, frame in scene["key_frames"].items():
            if frame_token in seen:
                continue
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
            pool[action].append({"scene_token": scene_token, "frame_token": frame_token,
                                 "front": str(path), "behavior": behavior})

    print(f"available  a)={len(pool['a)'])}  b)={len(pool['b)'])}")
    rng = random.Random(args.seed)
    manifest, key = [], []
    for action, want in (("a)", args.want_a), ("b)", args.want_b)):
        for row in rng.sample(pool[action], min(want, len(pool[action]))):
            manifest.append({"front": row["front"]})
            key.append({"drivelm_action": action, "behavior": row["behavior"],
                        "scene_token": row["scene_token"], "frame_token": row["frame_token"]})

    order = list(range(len(manifest)))
    rng.shuffle(order)
    manifest = [{"id": n, **manifest[i]} for n, i in enumerate(order)]
    key = [{"id": n, **key[i]} for n, i in enumerate(order)]

    (args.out / "review_manifest2.json").write_text(json.dumps(manifest, indent=1))
    (args.out / "review_key2.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {len(manifest)} frames, labels shuffled and held separately")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
