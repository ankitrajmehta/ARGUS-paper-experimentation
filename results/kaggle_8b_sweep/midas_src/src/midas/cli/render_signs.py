"""Render CHAI's adversarial signs onto the DriveLM frames.

CPU-only, so the attack images can be produced and eyeballed before any GPU is
available. Checking legibility here is worth doing: if a sign is unreadable to a
human at this resolution, the model will not read it either.

    midas-render --split train --out results/signs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from midas import data, signs
from midas.config import AttackConfig


def side_by_side(clean: np.ndarray, attacked: np.ndarray) -> np.ndarray:
    """Stack clean and attacked frames with a divider, for quick visual diffing."""
    divider = np.full((clean.shape[0], 8, 3), 255, dtype=clean.dtype)
    return np.hstack([clean, divider, attacked])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--out", type=Path, default=Path("results/signs"))
    parser.add_argument(
        "--background-mode",
        choices=["code", "json"],
        default="code",
        help="'code' reproduces CHAI's executed renderer (255,0,B); 'json' uses optimal.json verbatim",
    )
    parser.add_argument("--size-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--target-mode",
        choices=["auto", "dataset"],
        default="auto",
        help="'auto' targets the opposite of the benign action; 'dataset' uses CHAI's stored target "
        "and skips scenes whose target is c)/d)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="force one target action for every scene, e.g. 'b)'. Overrides --target-mode.",
    )
    parser.add_argument("--compare", action="store_true", help="also write clean|attacked pairs")
    args = parser.parse_args(argv)

    cfg = AttackConfig(background_mode=args.background_mode, size_multiplier=args.size_multiplier)
    scenes = data.load_scenes(args.split)

    out_dir = args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for scene in scenes:
        target = args.target or data.resolve_target(scene, args.target_mode)
        if target is None:
            print(f"scene {scene.index:2d}  skipped: stored target {scene.target} not available to InternVL")
            continue

        images = scene.load_images()
        clean = images[scene.attacked_camera].copy()
        attacked, sign = signs.attack_scene(scene, images=images, target=target, cfg=cfg)
        front = attacked[scene.attacked_camera]
        rendered += 1

        stem = f"scene{scene.index:02d}_{sign.text.replace(' ', '-').lower()}"
        # cv2.imwrite expects BGR.
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
        if args.compare:
            pair = side_by_side(clean, front)
            cv2.imwrite(str(out_dir / f"{stem}_compare.jpg"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))

        print(
            f"scene {scene.index:2d}  benign={scene.correct_internvl}  target={target}  "
            f"sign={sign.text.upper()!r}  bg={sign.bg}  fg={sign.fg}  "
            f"scale={sign.scale}  origin={sign.origin}"
        )

    print(f"\nwrote {rendered}/{len(scenes)} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
