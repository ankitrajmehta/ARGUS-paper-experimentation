"""Build a CHAI-format scene set from the full DriveLM release.

CHAI evaluate on 23 scenes. At that size one scene is 6.7 points and the 95%
interval is about +/-25, which is the binding limitation on every number in this
project. DriveLM ships thousands of annotated key frames with their nuScenes
images, so the same attack can be measured at a sample size that supports a
conclusion.

    python scripts/fetch_drivelm.py --out data/drivelm
    midas-build-scenes --limit 100

The output is written in CHAI's own record format, so ``midas-eval``,
``midas-labels``, ``midas-activations`` and the sign search all read it with no
changes:

    {"image": [six paths], "correct_internvl": null, "target": null,
     "position": {"pos_x": .., "pos_y": ..}, "size": ..}

**Labels are deliberately left null.** CHAI obtain the benign action by running
their own model unattacked and taking the mode over 11 runs, and this project
measured that the result is unstable -- 6 of their 15 scenes are near coin-flips
(PROJECT_RECORD section 10.1). Inventing labels here would bake one model's
opinion into the dataset. Derive them for the model under test instead::

    midas-labels --split drivelm --model <model> --runs 5
    midas-eval  --split drivelm --both --labels results/labels/drivelm_*.json

**Sign placement.** CHAI hand-placed the sign per scene. For scenes they never
saw there is no such annotation, so the default uses their median placement for
every scene -- a single universal sign, which is what CHAI claim their attack
is. ``--placement sample`` instead draws from the empirical spread of their 23
scenes, which tests whether the attack survives placement variation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from midas import config

# CHAI's camera order. drivelm.CAMERA_INDICES = (0, 3) selects CAM_FRONT and
# CAM_BACK from this list, so the order matters.
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# Median of CHAI's 23 hand-placed signs. Their field names are transposed
# relative to cv2: pos_y is the x coordinate and pos_x the y (see
# data.Scene.sign_origin), so this lands right of centre, vertically middle.
MEDIAN_PLACEMENT = {"pos_x": 450, "pos_y": 1050, "size": 2.0}


def chai_placements() -> list[dict]:
    """Every (position, size) CHAI chose, for --placement sample."""
    out = []
    for path in (config.CHAI_TRAIN_JSON, config.CHAI_TEST_JSON):
        if path.exists():
            for rec in json.loads(path.read_text()):
                out.append({**rec["position"], "size": float(rec["size"])})
    return out


def iter_key_frames(qa: dict):
    """Yield (scene_token, frame_token, frame) over the DriveLM QA file."""
    for scene_token, scene in qa.items():
        for frame_token, frame in (scene.get("key_frames") or {}).items():
            yield scene_token, frame_token, frame


def frame_images(frame: dict) -> list[str] | None:
    """The six camera paths in CHAI's order, or None if any is missing."""
    paths = frame.get("image_paths") or {}
    out = []
    for camera in CAMERAS:
        path = paths.get(camera)
        if not path:
            return None
        # DriveLM stores "../nuscenes/samples/CAM_FRONT/..."; normalise to a
        # path relative to the data root we extracted the zip into.
        out.append(path.replace("../", "").lstrip("/"))
    return out


def behavior_text(frame: dict) -> str:
    """DriveLM's own description of what the ego vehicle does.

    Recorded for inspection only. It is *not* used as a label: mapping free text
    like "the ego vehicle is driving fast" onto proceed/brake is a judgement
    call, and silently guessing it would repeat exactly the ground-truth
    problem section 10.1 documents.
    """
    for item in (frame.get("QA") or {}).get("behavior", []) or []:
        if item.get("A"):
            return str(item["A"]).strip()
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=config.DRIVELM_ROOT / "v1_1_train_nus.json",
        help="DriveLM QA json from scripts/fetch_drivelm.py",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=config.DRIVELM_ROOT,
        help="directory the image paths are relative to",
    )
    parser.add_argument("--limit", type=int, default=100, help="scenes to build")
    parser.add_argument(
        "--per-scene",
        type=int,
        default=1,
        help="key frames per nuScenes scene. 1 keeps the set visually diverse; "
        "frames from one scene are seconds apart and nearly duplicates.",
    )
    parser.add_argument(
        "--placement", choices=["fixed", "sample"], default="fixed",
        help="fixed: CHAI's median placement everywhere (a universal sign). "
        "sample: draw from their 23 hand-placed signs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-images",
        action="store_true",
        default=True,
        help="skip frames whose six images are not on disk (default)",
    )
    parser.add_argument("--no-require-images", dest="require_images", action="store_false")
    parser.add_argument("--out", type=Path, default=config.DRIVELM_SCENES)
    args = parser.parse_args(argv)

    if not args.qa.exists():
        raise SystemExit(
            f"{args.qa} not found. Run:  python scripts/fetch_drivelm.py --out {args.root}"
        )

    print(f"reading {args.qa} ...", flush=True)
    qa = json.loads(args.qa.read_text())
    print(f"  {len(qa)} nuScenes scenes")

    rng = random.Random(args.seed)
    placements = chai_placements() if args.placement == "sample" else []
    if args.placement == "sample" and not placements:
        raise SystemExit("--placement sample needs CHAI's scene files; run setup_vendor.sh")

    records: list[dict] = []
    seen_scenes: dict[str, int] = {}
    skipped_missing_paths = skipped_missing_files = 0

    for scene_token, frame_token, frame in iter_key_frames(qa):
        if len(records) >= args.limit:
            break
        if seen_scenes.get(scene_token, 0) >= args.per_scene:
            continue

        images = frame_images(frame)
        if images is None:
            skipped_missing_paths += 1
            continue
        if args.require_images and not all((args.root / p).exists() for p in images):
            skipped_missing_files += 1
            continue

        placement = rng.choice(placements) if placements else MEDIAN_PLACEMENT
        seen_scenes[scene_token] = seen_scenes.get(scene_token, 0) + 1
        records.append(
            {
                "image": images,
                # Filled in by midas-labels against the model under test.
                "correct_internvl": None,
                "correct_gpt": None,
                "target": None,
                "position": {"pos_x": placement["pos_x"], "pos_y": placement["pos_y"]},
                "size": float(placement["size"]),
                # Provenance, so a scene can be traced back to DriveLM.
                "drivelm": {
                    "scene_token": scene_token,
                    "frame_token": frame_token,
                    "behavior": behavior_text(frame),
                },
            }
        )

    if not records:
        raise SystemExit(
            "no usable frames found.\n"
            f"  frames without six camera paths: {skipped_missing_paths}\n"
            f"  frames whose images are not on disk: {skipped_missing_files}\n"
            "If the second number is large, the image zip was not extracted into "
            f"{args.root}."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1))

    print(f"\nbuilt {len(records)} scenes from {len(seen_scenes)} nuScenes scenes")
    print(f"  placement: {args.placement}")
    if skipped_missing_paths:
        print(f"  skipped {skipped_missing_paths} frames with incomplete camera paths")
    if skipped_missing_files:
        print(f"  skipped {skipped_missing_files} frames whose images are not on disk")
    print(f"saved -> {args.out}")
    print("\nnext, derive labels for the model under test:")
    print("  midas-labels --split drivelm --model Qwen/Qwen2.5-VL-3B-Instruct --runs 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
