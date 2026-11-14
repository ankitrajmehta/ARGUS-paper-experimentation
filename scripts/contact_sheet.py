"""Tile the clean|attacked comparison frames into scannable contact sheets.

200 full-size pairs is 300 MB and no one scans that. This lays them out N per
sheet with the scene index, the audited benign action and the sign text burned
in, so a human can sweep a whole set quickly and then open only the frames that
look wrong.

    python scripts/contact_sheet.py --style descriptive --per-sheet 12
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import cv2
import numpy as np

NAMES = {"a)": "PROCEED", "b)": "BRAKE"}


def label(img: np.ndarray, text: str, height: int = 34) -> np.ndarray:
    bar = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--style", default="descriptive")
    p.add_argument("--images", type=Path, default=Path("results/review_images"))
    p.add_argument("--scenes", type=Path,
                   default=Path("data/drivelm/scenes_clean200.json"))
    p.add_argument("--per-sheet", type=int, default=12)
    p.add_argument("--width", type=int, default=760, help="width of each pair")
    args = p.parse_args(argv)

    scenes = json.loads(args.scenes.read_text())
    src = args.images / args.style
    # Sort numerically: plain sorted() puts scene100 before scene009.
    pairs = sorted(src.glob("*_compare.jpg"),
                   key=lambda p: int(re.match(r"scene(\d+)_", p.name).group(1)))
    if not pairs:
        print(f"no comparison frames in {src}")
        return 1

    out_dir = args.images / f"{args.style}_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()

    tiles = []
    for path in pairs:
        idx = int(re.match(r"scene(\d+)_", path.name).group(1))
        sign = path.stem.split("_", 1)[1].rsplit("_compare", 1)[0].replace("-", " ").upper()
        benign = scenes[idx]["correct_internvl"]
        img = cv2.imread(str(path))
        h = int(img.shape[0] * args.width / img.shape[1])
        img = cv2.resize(img, (args.width, h), interpolation=cv2.INTER_AREA)
        tiles.append(label(img, f"#{idx:03d}  clean|attacked   benign={NAMES[benign]}   sign={sign}"))

    cols = 2
    written = 0
    for start in range(0, len(tiles), args.per_sheet):
        chunk = tiles[start:start + args.per_sheet]
        rows = []
        for i in range(0, len(chunk), cols):
            row = chunk[i:i + cols]
            while len(row) < cols:
                row.append(np.zeros_like(row[0]))
            rows.append(np.hstack(row))
        sheet = np.vstack(rows)
        name = out_dir / f"sheet_{start // args.per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        written += 1

    print(f"wrote {written} sheets ({len(tiles)} frames, {args.per_sheet}/sheet) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
