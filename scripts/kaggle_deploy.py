"""Deploy MIDAS to Kaggle using only the Kaggle API — no GitHub involved.

The code travels as a private Kaggle Dataset rather than a git clone, which
removes the only reason this project needed a GitHub remote. CHAI's released
artefacts ride along in the same dataset, so the notebook does not depend on
that repository staying reachable either.

    python scripts/kaggle_deploy.py upload    # push code+data as a dataset
    python scripts/kaggle_deploy.py run       # push the notebook and start it
    python scripts/kaggle_deploy.py status    # poll the running kernel
    python scripts/kaggle_deploy.py fetch     # download results when complete

Requires Kaggle credentials (~/.kaggle/kaggle.json or ~/.kaggle/access_token).
Internet must be enabled on the Kaggle account
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
# Kaggle derives the kernel slug from the *title*, not from the id in the
# metadata, and then serves the kernel at the title-derived URL. The two must
# agree or `status` and `output` fail with a confusing permission error, so this
# is the slugified title rather than a shorter name of our choosing.
KERNEL_TITLE = "MIDAS: CHAI attack on InternVL2.5-8B"
KERNEL_SLUG = "midas-chai-attack-on-internvl2-5-8b"

# A second, sweep-only kernel. It inherits the first kernel's output rather than
# recomputing activations, which take about an hour of GPU time.
SWEEP_TITLE = "MIDAS: defense sweep on InternVL2.5-8B"
SWEEP_SLUG = "midas-defense-sweep-on-internvl2-5-8b"

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
    """Return the Kaggle username, accepting either credential format.

    Kaggle has two: the older ``kaggle.json`` carrying username and key, and the
    newer ``access_token`` holding a bare ``KGAT_...`` string. The newer file has
    no username in it, so that is read back from the CLI's own config.
    """
    home = Path.home() / ".kaggle"
    legacy, modern = home / "kaggle.json", home / "access_token"

    if legacy.exists():
        if legacy.stat().st_mode & 0o077:
            legacy.chmod(0o600)
        return json.loads(legacy.read_text())["username"]

    if not modern.exists():
        sys.exit(
            "No Kaggle credentials found.\n"
            "kaggle.com -> Settings -> API -> Create New Token, then either:\n"
            "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "or, for the newer token format:\n"
            "  mkdir -p ~/.kaggle && echo KGAT_... > ~/.kaggle/access_token\n"
            "and chmod 600 the file either way."
        )
    if modern.stat().st_mode & 0o077:
        modern.chmod(0o600)

    config = kaggle("config", "view", check=False).stdout
    for line in config.splitlines():
        if "username:" in line:
            return line.split("username:")[1].strip()
    sys.exit("Could not determine the Kaggle username from `kaggle config view`.")


def find_activations(requested, roots: list[Path] | None = None) -> Path | None:
    """Locate a DriveLM activation sweep to ship with the source dataset.

    ``requested`` is a directory, or ``True`` to search the usual fetch
    destinations. A directory with no ``.npz`` in it is treated as absent: an
    empty fetch directory is the normal state before a run has completed, and
    silently staging nothing is worse than saying so.
    """
    if roots is None:
        roots = [
            ROOT / "results" / "kaggle_drivelm" / "results" / "drivelm" / "activations",
            ROOT / "results" / "kaggle_8b" / "results" / "drivelm" / "activations",
        ]
    candidates = ([Path(requested)] if requested is not True else []) + roots
    for c in candidates:
        if c.is_dir() and any(c.glob("*.npz")):
            return c
    return None


def stage_activations(requested, dest: Path, roots: list[Path] | None = None) -> Path | None:
    """Copy an activation sweep into the staging tree.

    The steering stage fits its probes from these, and re-extracting them costs
    a model load plus a forward pass per sample -- so a steering-only kernel
    inherits them rather than recomputing.
    """
    source = find_activations(requested, roots)
    if source is None:
        print("  skip (absent): no DriveLM activations found. Fetch a completed run:")
        print("    python scripts/kaggle_deploy.py fetch --out results/kaggle_drivelm")
        return None

    dest.mkdir(parents=True, exist_ok=True)
    # The .json sidecar carries the row metadata -- scene, phrase, condition,
    # follows_injection. Without it the .npz is an unlabelled block of floats
    # and no probe can be fitted from it, so both are copied and a missing
    # sidecar is called out rather than discovered on the GPU.
    staged = 0
    for pattern in ("*.npz", "*.json"):
        for f in sorted(source.glob(pattern)):
            shutil.copy2(f, dest / f.name)
            staged += 1
    orphans = sorted(
        f.name for f in dest.glob("*.npz") if not f.with_suffix(".json").exists()
    )
    size = sum(f.stat().st_size for f in dest.iterdir()) / 1e6
    print(f"  staged: drivelm/activations ({staged} files, {size:.0f} MB) from {source}")
    if orphans:
        print(f"    WARNING: no metadata sidecar for {orphans}; "
              "these cannot be used to fit a probe")
    return source


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

    # Benign labels re-derived on the 8B. Shipping them (a few KB) lets a rerun
    # evaluate against our own labels without repeating the derivation, which is
    # 11 unattacked runs per scene -- 4.6 GPU-hours, and sign-independent, so
    # there is never a reason to redo it.
    # DriveLM scenes, if built. Only the images those scenes reference are
    # staged -- the full release is 3.5 GB, while 100 scenes is about 600
    # frames, which fits a Kaggle dataset comfortably.
    # Every scene file in data/drivelm, not just scenes.json: the
    # single-direction replication set is a second one, and staging only the
    # default would leave its stage unable to find anything.
    drivelm_root = ROOT / "data" / "drivelm"
    scene_files = sorted(drivelm_root.glob("scenes*.json")) if args.drivelm else []
    if args.drivelm and scene_files:
        import json as _json

        dest = STAGING / "drivelm"
        dest.mkdir(parents=True, exist_ok=True)
        # Frames are shared between scene files, so copy each one once.
        copied: set[str] = set()
        missing = 0
        for scenes_json in scene_files:
            records = _json.loads(scenes_json.read_text())
            shutil.copy2(scenes_json, dest / scenes_json.name)
            for rec in records:
                for rel in rec["image"]:
                    if rel in copied:
                        continue
                    src = drivelm_root / rel
                    if not src.exists():
                        missing += 1
                        continue
                    out = dest / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, out)
                    copied.add(rel)
            print(f"  staged: drivelm/{scenes_json.name} ({len(records)} scenes)")
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print(f"  staged: drivelm frames ({len(copied)} unique, {size / 1e6:.0f} MB)")
        if missing:
            print(f"    WARNING: {missing} frames missing on disk")
    elif args.drivelm:
        print(f"  skip (absent): no scenes*.json in {drivelm_root} "
              "-- run midas-build-scenes first")

    if args.activations:
        stage_activations(args.activations, STAGING / "drivelm" / "activations")

    labels = ROOT / "results" / "kaggle_8b" / "results" / "labels"
    if labels.exists():
        shutil.copytree(labels, STAGING / "labels")
        print(f"  staged: labels ({len(list(labels.glob('*.json')))} files)")
    else:
        print("  skip (absent): derived labels")

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
    # Recorded only on success, so a failed upload cannot make `run` believe
    # the dataset is current.
    record_uploaded_source()
    print("\nDataset processing takes a minute or two. Check with:\n"
          f"  kaggle datasets status {user}/{DATASET_SLUG}")


UPLOAD_STAMP = ROOT / "kaggle" / ".uploaded_source_hash"


def source_hash() -> str:
    """Content hash of everything the kernel imports.

    Content, not modification time: `git checkout` restores a file's bytes but
    not its mtime, so an mtime check refuses to launch after an ordinary
    revert. Hashing the bytes asks the question that actually matters -- is the
    code on Kaggle the code in front of me.
    """
    import hashlib

    digest = hashlib.sha256()
    for f in sorted((ROOT / "src").rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        digest.update(f.relative_to(ROOT).as_posix().encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()


def record_uploaded_source() -> None:
    """Remember what was uploaded, so `run` can tell if it has drifted."""
    UPLOAD_STAMP.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_STAMP.write_text(source_hash())


def check_source_is_uploaded(force: bool = False) -> None:
    """Refuse to launch a kernel against source older than the working tree.

    The kernel installs ``src`` from the *dataset*, not from git, so editing a
    module and pushing a notebook that uses it launches the old code. That cost
    a GPU session: v19 died on
    ``AttributeError: module 'midas.styles' has no attribute 'DISJOINT_STYLES'``
    because the dataset predated the commit that added it by two hours.
    """
    if not UPLOAD_STAMP.exists():
        print("  note: no record of what source was uploaded; skipping the check")
        return
    if UPLOAD_STAMP.read_text().strip() == source_hash():
        print("  source check: the uploaded dataset matches src/, good")
        return

    message = (
        "src/ has changed since the dataset was last uploaded.\n"
        "The kernel installs src from the dataset, so it would run the OLD code.\n\n"
        "  python scripts/kaggle_deploy.py upload --drivelm --activations\n\n"
        "Or pass --skip-source-check if this is deliberate."
    )
    if force:
        print(f"  WARNING: {message}")
    else:
        sys.exit(f"refusing to launch: {message}")


def cmd_run(args: argparse.Namespace) -> None:
    user = require_credentials()
    check_source_is_uploaded(force=args.skip_source_check)

    name = "midas_stages.ipynb" if args.stages else "midas_kaggle.ipynb"
    notebook = ROOT / "kaggle" / name
    if not notebook.exists():
        hint = " --only <stages>" if args.stages else ""
        sys.exit(f"{name} missing. Run: python scripts/make_kaggle_notebook.py{hint}")

    if KERNEL_DIR.exists():
        shutil.rmtree(KERNEL_DIR)
    KERNEL_DIR.mkdir(parents=True)
    # Kaggle keys a kernel by its slug, not its filename, so pushing under the
    # same name replaces the running version rather than starting a second one.
    shutil.copy2(notebook, KERNEL_DIR / "midas_kaggle.ipynb")
    if args.stages:
        import json as _json

        kept = sorted({
            c.get("metadata", {}).get("midas_stage")
            for c in _json.loads(notebook.read_text())["cells"]
        } - {None})
        print(f"pushing a stage-limited notebook: {', '.join(kept)}")

    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{user}/{KERNEL_SLUG}",
                "title": KERNEL_TITLE,
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


def cmd_sweep(args: argparse.Namespace) -> None:
    user = require_credentials()

    notebook = ROOT / "kaggle" / "midas_sweep.ipynb"
    if not notebook.exists():
        sys.exit("Sweep notebook missing. Run: python scripts/make_kaggle_notebook.py")

    folder = ROOT / "kaggle" / "_kernel_sweep"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    shutil.copy2(notebook, folder / "midas_sweep.ipynb")

    (folder / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{user}/{SWEEP_SLUG}",
                "title": SWEEP_TITLE,
                "code_file": "midas_sweep.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{user}/{DATASET_SLUG}"],
                # Inherit the main run's activations and fitted defense.
                "kernel_sources": [f"{user}/{KERNEL_SLUG}"],
                "competition_sources": [],
            },
            indent=2,
        )
    )

    print(f"pushing sweep kernel {user}/{SWEEP_SLUG} ...")
    result = kaggle("kernels", "push", "-p", str(folder),
                    "--accelerator", "NvidiaTeslaT4", check=False)
    print(result.stdout or result.stderr)
    if result.returncode != 0:
        sys.exit("push failed")
    print(f"\nPoll with: python scripts/kaggle_deploy.py status --sweep")


def cmd_status(args: argparse.Namespace) -> None:
    user = require_credentials()
    ref = f"{user}/{SWEEP_SLUG if args.sweep else KERNEL_SLUG}"
    print(kaggle("kernels", "status", ref, check=False).stdout.strip())
    if args.logs:
        print("\n--- logs (tail) ---")
        logs = kaggle("kernels", "output", ref, "-p", "/tmp", check=False)
        print(logs.stdout or logs.stderr)


def cmd_fetch(args: argparse.Namespace) -> None:
    user = require_credentials()
    slug = SWEEP_SLUG if args.sweep else KERNEL_SLUG
    default = "kaggle_8b_sweep" if args.sweep else "kaggle_8b"
    out = Path(args.out) if args.out else ROOT / "results" / default

    # Kaggle's output includes a full copy of the attached source dataset --
    # several hundred MB of vendored nuScenes frames -- and overwrites the
    # results archive of whatever run was fetched here last. Refuse to write
    # into a directory that already holds a run, so each one keeps its own.
    if out.exists() and any(out.iterdir()) and not args.force:
        sys.exit(
            f"{out} already holds a fetched run. Pass --out <dir> for this one, "
            f"or --force to overwrite it."
        )

    out.mkdir(parents=True, exist_ok=True)
    result = kaggle("kernels", "output", f"{user}/{slug}", "-p", str(out), check=False)
    print(result.stdout or result.stderr)

    # The source copy is the attached dataset, already in version control.
    source_copy = out / "midas_src"
    if source_copy.exists():
        shutil.rmtree(source_copy)
        print("removed midas_src/ (a copy of the attached dataset)")

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
    up.add_argument(
        "--drivelm", action="store_true",
        help="also stage data/drivelm/scenes.json and the frames it references",
    )
    up.add_argument(
        "--activations", nargs="?", const=True, default=None, metavar="DIR",
        help="also stage DriveLM activation sweeps, so a steering-only kernel can "
        "fit its probes without re-extracting them (a model load plus a forward "
        "pass per sample). With no argument, looks in results/kaggle_drivelm then "
        "results/kaggle_8b.",
    )
    up.set_defaults(func=cmd_upload)

    run = sub.add_parser("run", help="push the notebook and start it on T4 x2")
    run.add_argument("--no-run", action="store_true", help="save without executing")
    run.add_argument(
        "--skip-source-check", action="store_true",
        help="launch even if src/ is newer than the uploaded dataset. The kernel "
        "installs src from the dataset, so by default this is refused.",
    )
    run.add_argument(
        "--stages",
        action="store_true",
        help="push kaggle/midas_stages.ipynb (a stage-limited notebook built with "
        "make_kaggle_notebook.py --only) instead of the full pipeline",
    )
    run.set_defaults(func=cmd_run)

    sw = sub.add_parser("sweep", help="push the sweep-only kernel (inherits the main run)")
    sw.set_defaults(func=cmd_sweep)

    st = sub.add_parser("status", help="poll the kernel")
    st.add_argument("--logs", action="store_true")
    st.add_argument("--sweep", action="store_true", help="poll the sweep kernel instead")
    st.set_defaults(func=cmd_status)

    fe = sub.add_parser("fetch", help="download kernel output")
    fe.add_argument("--sweep", action="store_true", help="fetch the sweep kernel instead")
    fe.add_argument("--out", default=None, help="where to write it (default: results/kaggle_8b)")
    fe.add_argument("--force", action="store_true", help="overwrite a run already fetched there")
    fe.set_defaults(func=cmd_fetch)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
