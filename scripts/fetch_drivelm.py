"""Download the DriveLM-nuScenes release from Hugging Face.

CHAI ship 23 scenes. DriveLM ships thousands, and critically it ships the
nuScenes *images* alongside the QA, so no nuscenes.org account is needed:

    v1_1_train_nus.json        193 MB   QA keyed by scene token
    drivelm_nus_imgs_train.zip 3.5 GB   the six camera frames per key frame

**The dataset is gated.** Accept the terms once, signed in, at
https://huggingface.co/datasets/OpenDriveLab/DriveLM then either run
``hf auth login`` or export ``HF_TOKEN``. Without that this script fails with a
401 and no amount of retrying helps.

Licence is CC BY-NC-SA 4.0 -- academic use, attribution, share-alike.

    python scripts/fetch_drivelm.py --out data/drivelm
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO = "OpenDriveLab/DriveLM"
QA_FILE = "v1_1_train_nus.json"
IMG_ZIP = "drivelm_nus_imgs_train.zip"


def have_credentials() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    return (Path.home() / ".cache" / "huggingface" / "token").exists()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/drivelm"))
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="fetch just the 193 MB QA json, not the 3.5 GB image zip. Enough to "
        "inspect the data and decide how many scenes to build.",
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="do not delete the zip after extracting"
    )
    args = parser.parse_args(argv)

    if not have_credentials():
        sys.exit(
            "No Hugging Face credentials found.\n\n"
            "  1. Sign in at https://huggingface.co and open\n"
            "     https://huggingface.co/datasets/OpenDriveLab/DriveLM\n"
            "  2. Accept the dataset terms (one click, CC BY-NC-SA 4.0).\n"
            "  3. Run:  hf auth login     (or export HF_TOKEN=hf_...)\n\n"
            "Then re-run this script."
        )

    from huggingface_hub import hf_hub_download

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = [QA_FILE] if args.qa_only else [QA_FILE, IMG_ZIP]

    for name in wanted:
        print(f"downloading {name} ...", flush=True)
        path = hf_hub_download(
            repo_id=REPO, filename=name, repo_type="dataset", local_dir=str(args.out)
        )
        print(f"  -> {path}")

        if name.endswith(".zip"):
            print("  extracting (this takes a few minutes) ...", flush=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(args.out)
            if not args.keep_zip:
                Path(path).unlink()
                print("  removed the zip")

    print(f"\ndone -> {args.out}")
    print("next:  midas-build-scenes --limit 100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
