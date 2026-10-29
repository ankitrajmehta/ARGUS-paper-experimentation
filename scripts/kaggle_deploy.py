"""Deploy MIDAS to Kaggle using only the Kaggle API — no GitHub involved.

The code travels as a private Kaggle Dataset rather than a git clone, which
removes the only reason this project needed a GitHub remote. CHAI's released
artefacts ride along in the same dataset, so the notebook does not depend on
that repository staying reachable either.

    python scripts/kaggle_deploy.py upload    # push code+data as a dataset
    python scripts/kaggle_deploy.py run       # push the notebook and start it
    python scripts/kaggle_deploy.py status    # poll the running kernel
    python scripts/kaggle_deploy.py fetch     # download results when complete

Requires ~/.kaggle/kaggle.json. Internet must be enabled on the Kaggle account
(phone verification), because the notebook still installs pinned packages and
downloads the model from HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "kaggle" / "_staging"
KERNEL_DIR = ROOT / "kaggle" / "_kernel"

DATASET_SLUG = "midas-chai-source"
KERNEL_SLUG = "midas-chai-8b"

# Everything the notebook needs at import time. vendor/chai is included so the
# run does not depend on CHAI's GitHub repository being reachable.
INCLUDE = ["src", "tests", "scripts", "pyproject.toml", "requirements.txt",
           "requirements-gpu.txt", "README.md"]
VENDOR = ROOT / "vendor" / "chai"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the Kaggle CLI from this project's virtualenv if present."""
    exe = ROOT / ".venv" / "bin" / "kaggle"
    cmd = [str(exe) if exe.exists() else "kaggle", *args]
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def require_credentials() -> str:
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        sys.exit(
            "No ~/.kaggle/kaggle.json found.\n"
            "Get one from kaggle.com -> Settings -> API -> Create New Token,\n"
            "then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ "
            "&& chmod 600 ~/.kaggle/kaggle.json"
        )
    # The API refuses to run if the token is world-readable.
    if path.stat().st_mode & 0o077:
        path.chmod(0o600)
    return json.loads(path.read_text())["username"]


def cmd_upload(args: argparse.Namespace) -> None:
    user = require_credentials()

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    for item in INCLUDE:
        source = ROOT / item
        if not source.exists():
            print(f"  skip (absent): {item}")
            continue
        target = STAGING / item
        if source.is_dir():
            shutil.copytree(
                source, target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
        else:
            shutil.copy2(source, target)
        print(f"  staged: {item}")

    if not VENDOR.exists():
        sys.exit("vendor/chai is missing. Run scripts/setup_vendor.sh first.")
    shutil.copytree(
        VENDOR, STAGING / "vendor" / "chai",
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    print("  staged: vendor/chai (CHAI's frames, dictionary and renderer)")

    (STAGING / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "MIDAS CHAI source",
                "id": f"{user}/{DATASET_SLUG}",
                "licenses": [{"name": "unknown"}],
            },
            indent=2,
        )
    )

    size = sum(f.stat().st_size for f in STAGING.rglob("*") if f.is_file())
    print(f"\nstaged {size / 1e6:.1f} MB at {STAGING}")

    existing = kaggle("datasets", "list", "-m", "-s", DATASET_SLUG, check=False)
    is_new = DATASET_SLUG not in existing.stdout

    if is_new:
        print(f"creating private dataset {user}/{DATASET_SLUG} ...")
        result = kaggle("datasets", "create", "-p", str(STAGING), "-r", "zip", check=False)
    else:
        print(f"updating dataset {user}/{DATASET_SLUG} ...")
        result = kaggle("datasets", "version", "-p", str(STAGING),
                        "-m", args.message, "-r", "zip", check=False)

    print(result.stdout or result.stderr)
    if result.returncode != 0:
        sys.exit("upload failed")
    print("\nDataset processing takes a minute or two. Check with:\n"
          f"  kaggle datasets status {user}/{DATASET_SLUG}")


def cmd_run(args: argparse.Namespace) -> None:
    user = require_credentials()

    notebook = ROOT / "kaggle" / "midas_kaggle.ipynb"
    if not notebook.exists():
        sys.exit("Notebook missing. Run: python scripts/make_kaggle_notebook.py")

    if KERNEL_DIR.exists():
        shutil.rmtree(KERNEL_DIR)
    KERNEL_DIR.mkdir(parents=True)
    shutil.copy2(notebook, KERNEL_DIR / "midas_kaggle.ipynb")

    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{user}/{KERNEL_SLUG}",
                "title": "MIDAS: CHAI attack on InternVL2.5-8B",
                "code_file": "midas_kaggle.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                # T4 x2: the 8B model needs ~16 GB and is sharded across both.
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{user}/{DATASET_SLUG}"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
    )

    print(f"pushing kernel {user}/{KERNEL_SLUG} (accelerator: T4 x2) ...")
    push = ["kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", "NvidiaTeslaT4"]
    if args.no_run:
        push.append("--no-run")
    result = kaggle(*push, check=False)
    print(result.stdout or result.stderr)
    if result.returncode != 0:
        sys.exit("push failed")

    print("\nThe kernel runs on Kaggle's infrastructure, so it continues after this\n"
          "session ends. Poll it with:\n"
          f"  python scripts/kaggle_deploy.py status")


def cmd_status(args: argparse.Namespace) -> None:
    user = require_credentials()
    ref = f"{user}/{KERNEL_SLUG}"
    print(kaggle("kernels", "status", ref, check=False).stdout.strip())
    if args.logs:
        print("\n--- logs (tail) ---")
        logs = kaggle("kernels", "output", ref, "-p", "/tmp", check=False)
        print(logs.stdout or logs.stderr)


def cmd_fetch(args: argparse.Namespace) -> None:
    user = require_credentials()
    out = ROOT / "results" / "kaggle_8b"
    out.mkdir(parents=True, exist_ok=True)
    result = kaggle("kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(out), check=False)
    print(result.stdout or result.stderr)
    files = sorted(p.name for p in out.rglob("*") if p.is_file())
    print(f"\n{len(files)} files in {out}")
    for name in files[:20]:
        print(f"  {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="push code and CHAI artefacts as a private dataset")
    up.add_argument("-m", "--message", default="update source")
    up.set_defaults(func=cmd_upload)

    run = sub.add_parser("run", help="push the notebook and start it on T4 x2")
    run.add_argument("--no-run", action="store_true", help="save without executing")
    run.set_defaults(func=cmd_run)

    st = sub.add_parser("status", help="poll the kernel")
    st.add_argument("--logs", action="store_true")
    st.set_defaults(func=cmd_status)

    fe = sub.add_parser("fetch", help="download kernel output")
    fe.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
