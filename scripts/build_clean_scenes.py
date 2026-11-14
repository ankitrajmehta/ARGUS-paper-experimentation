"""Turn the audited clean set into a scene file the renderer and evaluator read.

The clean set records frame tokens and the audited action. This joins them back
to the six camera paths from DriveLM and emits the same schema
``midas-build-scenes`` produces, so every downstream tool works unchanged.

    python scripts/build_clean_scenes.py --style descriptive
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from midas.cli.build_scenes import CAMERAS  # noqa: E402

POSITION = {"pos_x": 450, "pos_y": 1050}
SIZE = 2.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", type=Path,
                   default=Path("results/clean_set/clean_200_balanced.json"))
    p.add_argument("--qa", type=Path, default=Path("data/drivelm/v1_1_train_nus.json"))
    p.add_argument("--root", type=Path, default=Path("data/drivelm"))
    p.add_argument("--out", type=Path, default=Path("data/drivelm/scenes_clean200.json"))
    args = p.parse_args(argv)

    qa = json.loads(args.qa.read_text())
    frames = {}
    for scene_token, scene in qa.items():
        for frame_token, frame in scene["key_frames"].items():
            frames[frame_token] = frame

    records, missing = [], 0
    for row in json.loads(args.clean.read_text()):
        frame = frames.get(row["frame_token"])
        if frame is None:
            missing += 1
            continue
        paths = frame.get("image_paths") or {}
        images, ok = [], True
        for cam in CAMERAS:
            rel = paths.get(cam)
            if not rel:
                ok = False
                break
            images.append(rel.replace("../nuscenes/", "nuscenes/"))
        if not ok:
            missing += 1
            continue
        records.append({
            "image": images,
            "correct_internvl": row["action"],
            "correct_gpt": None,
            "target": None,
            "position": dict(POSITION),
            "size": SIZE,
            "drivelm": {
                "scene_token": row["scene_token"],
                "frame_token": row["frame_token"],
                "behavior": row["behavior"],
            },
            "audit": {"visual_evidence": row["visual_evidence"], "round": row["round"]},
        })

    args.out.write_text(json.dumps(records, indent=1))
    import collections
    print(f"wrote {len(records)} scenes to {args.out}  (missing {missing})")
    print("by action:", dict(collections.Counter(r["correct_internvl"] for r in records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
