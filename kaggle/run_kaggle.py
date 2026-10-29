"""Kaggle kernel: set up ARGUS from the argus-code dataset, fetch pinned models/data, run E1 or E2.

E2 runs as its own kernel that mounts the E1 kernel's output (for the transfer_e1 probe); E2b-only
runs mount the E2 kernel's output too, reusing its activations; E2c also mounts E2b's.

Everything heavy (code copy, data, models, HF cache) lives in /tmp/argus; only results and logs
go to /kaggle/working, which Kaggle saves as the kernel output.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

# ---- run settings (edit, then `kaggle kernels push`) ----
EXPERIMENT = "e1"
RUN_NAME = "e1_3b_fp16_n100"
CONFIG = "configs/kaggle_e1_3b.yaml"
MAX_SAMPLES = 100
BEHAVIOR_N = 50

# Pinned so a Kaggle run sees exactly the data the laptop used.
MODEL_REPO, MODEL_REV = "Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"
VTQA_REPO, VTQA_REV = "CalfKing/vtqa2023", "720e9e8053154c6e9fa80e6bb5784d41c365eb94"
ALPACA_REPO, ALPACA_REV = "tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017"
ALPACA_FILE = "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
THOR_REPO, THOR_REV = "weikaih/ai2thor-random-views-20k", "895f146d5a7c006fc6db3cff2e361e4663b8b8b2"
# The laptop E2 split was built from exactly these two shards.
THOR_FILES = ["data/train-00000-of-00043.parquet", "data/train-00021-of-00043.parquet"]
# sha256[:16] of the laptop e2_{split}.json with image paths dropped; a rebuild should match.
E2_FINGERPRINTS = {"train": "97875269dcb104e2", "test": "0c5b71aac458cbca"}
PACKAGES =["transformers==4.51.3", "datasets==3.6.0", "huggingface_hub==0.36.2", "qwen-vl-utils==0.0.14"]

ROOT = Path("/tmp/argus")
CODE = ROOT / "ARGUS"
DATA = ROOT / "datasets"
MODELS = ROOT / "models"
WORK = Path("/kaggle/working")


def sh(cmd: list[str], **kwargs) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def step(name: str) -> None:
    print(f"\n######## {name}  [{time.strftime('%H:%M:%S')}] ########", flush=True)


def extract_code() -> None:
    archives = sorted(Path("/kaggle/input").rglob("argus_code.tar.gz"))
    if archives:
        CODE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archives[0]) as tar:
            tar.extractall(CODE)
        source = archives[0].parent
    else:
        # Kaggle sometimes unpacks uploaded archives; fall back to the unpacked tree.
        trees = [p.parent.parent for p in Path("/kaggle/input").rglob("src/argus") if p.is_dir()]
        if not trees:
            raise FileNotFoundError("ARGUS code not found under /kaggle/input; attach the argus-code dataset")
        shutil.copytree(trees[0], CODE)
        source = trees[0]
    commits = sorted(source.parent.rglob("COMMIT")) or sorted(Path("/kaggle/input").rglob("COMMIT"))
    print(f"code: {source}  commit: {commits[0].read_text().strip() if commits else 'unknown'}")


def fetch_data() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(MODEL_REPO, revision=MODEL_REV, local_dir=MODELS / MODEL_REPO.split("/")[1])
    if EXPERIMENT in ("e2", "e2b", "e2c", "e3a", "e3b"):
        for name in THOR_FILES:
            hf_hub_download(THOR_REPO, name, repo_type="dataset", revision=THOR_REV, local_dir=DATA / "ai2thor")
        return
    vtqa = DATA / "vtqa"
    for name in ["annotations", "image"]:
        if (vtqa / name).exists():
            continue
        archive = hf_hub_download(VTQA_REPO, f"data/{name}.zip", repo_type="dataset", revision=VTQA_REV, cache_dir=ROOT / "dl")
        # annotations.zip holds bare json files, image.zip holds split folders; both unpack into vtqa/<name>.
        with zipfile.ZipFile(archive) as z:
            z.extractall(vtqa / name)
        os.remove(archive)
    hf_hub_download(ALPACA_REPO, ALPACA_FILE, repo_type="dataset", revision=ALPACA_REV, local_dir=DATA / "alpaca")


def run_logged(command: list[str], env: dict, log_path: Path) -> None:
    print("$", " ".join(command), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def check_e2_fingerprints(out: Path) -> None:
    import hashlib
    import json

    for split, expected in E2_FINGERPRINTS.items():
        rows = json.loads((out / "data" / f"e2_{split}.json").read_text(encoding="utf-8"))
        keys = sorted(k for k in rows[0] if k != "images")
        payload = json.dumps([{k: r[k] for k in keys} for r in rows], sort_keys=True, ensure_ascii=False).encode()
        got = hashlib.sha256(payload).hexdigest()[:16]
        print(f"E2 {split} fingerprint {got} ({'matches laptop' if got == expected else 'DIFFERS from laptop ' + expected})")


def e1_output_dir() -> Path | None:
    hits = sorted(p.parent.parent for p in Path("/kaggle/input").rglob("probes/detection") if p.is_dir())
    return hits[0] if hits else None


def main() -> None:
    log_dir = WORK / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(ROOT / "hf_cache"), PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1")

    step("gpu")
    sh(["nvidia-smi"])
    step("packages")
    sh([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    step("code")
    extract_code()
    step("models + data")
    fetch_data()

    env = {**os.environ, "PYTHONPATH": str(CODE / "src")}
    step("fit check")
    print(f"python: {sys.executable}")
    sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"], env=env)

    out = WORK / "outputs" / RUN_NAME
    log_path = log_dir / f"{RUN_NAME}.log"
    step(f"{EXPERIMENT} {RUN_NAME}")
    if EXPERIMENT in ("e3a", "e3b"):
        run_e3(out, env, log_path)
        step("done")
        return
    if EXPERIMENT == "e1":
        run_logged([
            sys.executable, str(CODE / "experiments/e1/run_e1.py"),
            "--config", str(CODE / CONFIG),
            "--output-dir", str(out),
            "--max-samples", str(MAX_SAMPLES),
            "--behavior-n", str(BEHAVIOR_N),
        ], env, log_path)
    else:
        e1_dir = e1_output_dir()
        print(f"E1 probes for transfer_e1: {e1_dir}")
        e1_args = ["--e1-dir", str(e1_dir)] if e1_dir else []
        if EXPERIMENT == "e2":
            e2_dir, e2b = out, WORK / "outputs" / f"{RUN_NAME}_e2b"
            build_e2_frames(e2_dir, env, log_path)
            scripts = CODE / "experiments/e2"
            run_logged([sys.executable, str(scripts / "collect_e2.py"), "--config", str(CODE / CONFIG), "--output-dir", str(e2_dir), "--behavior-n", str(BEHAVIOR_N)], env, log_path)
            run_logged([sys.executable, str(scripts / "analyze_e2.py"), "--output-dir", str(e2_dir), *e1_args], env, log_path)
        elif EXPERIMENT == "e2b":
            # E2b only: reuse a finished argus-e2 run's activations; rebuild its frames (cheap, seeded).
            e2_dir, e2b = ROOT / "e2", out
            build_e2_frames(e2_dir, env, log_path)
            copy_mounted("e2_train.pt", e2_dir)
        else:
            # E2c: reuse finished E2 and E2b runs; only the new varied-phrasing signs need the GPU.
            e2_dir, e2b_dir, scripts = ROOT / "e2", ROOT / "e2b", CODE / "experiments/e2c"
            build_e2_frames(e2_dir, env, log_path)
            copy_mounted("e2_train.pt", e2_dir)
            copy_mounted("e2b_test.pt", e2b_dir)
            run_logged([sys.executable, str(scripts / "build_e2c.py"), "--e2-dir", str(e2_dir), "--output-dir", str(out)], env, log_path)
            run_logged([sys.executable, str(scripts / "collect_e2c.py"), "--config", str(CODE / CONFIG), "--output-dir", str(out)], env, log_path)
            run_logged([sys.executable, str(scripts / "analyze_e2c.py"), "--e2-dir", str(e2_dir), "--e2b-dir", str(e2b_dir), "--output-dir", str(out)], env, log_path)
            shutil.rmtree(out / "data" / "train", ignore_errors=True)
            shutil.rmtree(out / "data" / "test", ignore_errors=True)
            step("done")
            return
        run_e2b(e2_dir, e2b, e1_args, env, log_path)
        # Rendered frames are reproducible from the seed; keep the kernel output small.
        for folder in [out / "data" / "train", out / "data" / "test", e2b / "data" / "test"]:
            shutil.rmtree(folder, ignore_errors=True)
    step("done")


def copy_mounted(marker: str, dest: Path) -> None:
    # A mounted kernel output holds <run>/activations/<marker> and <run>/results/; copy both.
    hits = sorted(Path("/kaggle/input").rglob(f"activations/{marker}"))
    if not hits:
        raise FileNotFoundError(f"no activations/{marker} under /kaggle/input; attach the kernel output that produced it")
    run_dir = hits[0].parent.parent
    print(f"reusing {run_dir}")
    for name in ["activations", "results"]:
        if (run_dir / name).exists():
            shutil.copytree(run_dir / name, dest / name, dirs_exist_ok=True)


def build_e2_frames(e2_dir: Path, env: dict, log_path: Path) -> None:
    script = CODE / "experiments/e2/build_e2.py"
    run_logged([sys.executable, str(script), "--frames-dir", str(DATA / "ai2thor/data"), "--output-dir", str(e2_dir)], env, log_path)
    check_e2_fingerprints(e2_dir)


def check_rebuild(rebuilt: Path) -> None:
    # Rebuilt sign frames must carry exactly the texts/actions of the run whose responses E3 reuses.
    import json

    def strip(path: Path) -> list[dict]:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [{k: v for k, v in r.items() if k not in ("images", "clean_image")} for r in rows]

    hits = sorted(Path("/kaggle/input").rglob(f"data/{rebuilt.name}"))
    if not hits:
        raise FileNotFoundError(f"no mounted data/{rebuilt.name} to check the rebuild against")
    for hit in hits:
        if strip(hit) != strip(rebuilt):
            raise RuntimeError(f"rebuilt {rebuilt} differs from mounted {hit}")
    print(f"rebuilt {rebuilt.name} matches {len(hits)} mounted copies")


def run_e3(out: Path, env: dict, log_path: Path) -> None:
    # E3 reuses E2/E2b/E2c activations and responses; frames are rebuilt (seeded) and checked against the mounts.
    e2_dir, e2b_dir, e2c_dir = ROOT / "e2", ROOT / "e2b", ROOT / "e2c"
    build_e2_frames(e2_dir, env, log_path)
    copy_mounted("e2_train.pt", e2_dir)
    run_logged([sys.executable, str(CODE / "experiments/e2b/build_e2b.py"), "--e2-dir", str(e2_dir), "--output-dir", str(e2b_dir)], env, log_path)
    copy_mounted("e2b_test.pt", e2b_dir)
    run_logged([sys.executable, str(CODE / "experiments/e2c/build_e2c.py"), "--e2-dir", str(e2_dir), "--output-dir", str(e2c_dir)], env, log_path)
    copy_mounted("e2c_train.pt", e2c_dir)
    for path in [e2b_dir / "data" / "e2b_test.json", e2c_dir / "data" / "e2c_train.json", e2c_dir / "data" / "e2c_test.json"]:
        check_rebuild(path)

    scripts = CODE / "experiments/e3"
    sources = ["--e2-dir", str(e2_dir), "--e2b-dir", str(e2b_dir), "--e2c-dir", str(e2c_dir)]
    frames = ["--max-frames", str(MAX_SAMPLES)] if MAX_SAMPLES < 94 else []
    if EXPERIMENT == "e3a":
        e1_dir = e1_output_dir()
        e1_args = ["--e1-dir", str(e1_dir)] if e1_dir else []
        run_logged([sys.executable, str(scripts / "collect_e3a.py"), "--config", str(CODE / CONFIG), *sources, "--output-dir", str(out), *frames], env, log_path)
        run_logged([sys.executable, str(scripts / "analyze_e3a.py"), *sources, "--output-dir", str(out), *e1_args], env, log_path)
        return
    hits = sorted(Path("/kaggle/input").rglob("results/e3a_report.json"))
    if not hits:
        raise FileNotFoundError("no results/e3a_report.json under /kaggle/input; attach the argus-e3a output")
    e3a_dir = hits[0].parent.parent
    print(f"E3a probes from {e3a_dir}")
    e3a_args = ["--e3a-dir", str(e3a_dir), "--output-dir", str(out)]
    run_logged([sys.executable, str(scripts / "diagnose_e3a.py"), *sources, *e3a_args], env, log_path)
    run_logged([sys.executable, str(scripts / "respond_e3b.py"), "--config", str(CODE / CONFIG), *sources, *e3a_args, *frames], env, log_path)
    run_logged([sys.executable, str(scripts / "analyze_e3b.py"), *sources, *e3a_args, *frames], env, log_path)
    # Masked frames are reproducible; keep a few for the report.
    for path in sorted((out / "data" / "masked").glob("*.png"))[8:]:
        path.unlink()


def run_e2b(e2_dir: Path, e2b: Path, e1_args: list[str], env: dict, log_path: Path) -> None:
    # Same test frames with plain attack wording and legitimate imperative signs.
    scripts = CODE / "experiments/e2b"
    run_logged([sys.executable, str(scripts / "build_e2b.py"), "--e2-dir", str(e2_dir), "--output-dir", str(e2b)], env, log_path)
    run_logged([sys.executable, str(scripts / "collect_e2b.py"), "--config", str(CODE / CONFIG), "--output-dir", str(e2b), "--behavior-n", str(BEHAVIOR_N)], env, log_path)
    run_logged([sys.executable, str(scripts / "analyze_e2b.py"), "--e2-dir", str(e2_dir), "--output-dir", str(e2b), *e1_args], env, log_path)


if __name__ == "__main__":
    main()
