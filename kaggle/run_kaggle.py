"""Kaggle kernel: set up ARGUS from the argus-code dataset, fetch pinned models/data, run E1.

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
RUN_NAME = "e1_3b_fp16_n100"
CONFIG = "configs/kaggle_e1_3b.yaml"
MAX_SAMPLES = 100
BEHAVIOR_N = 50

# Pinned so a Kaggle run sees exactly the data the laptop used.
MODEL_REPO, MODEL_REV = "Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"
VTQA_REPO, VTQA_REV = "CalfKing/vtqa2023", "720e9e8053154c6e9fa80e6bb5784d41c365eb94"
ALPACA_REPO, ALPACA_REV = "tatsu-lab/alpaca", "dce01c9b08f87459cf36a430d809084718273017"
ALPACA_FILE = "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
PACKAGES = ["transformers==4.51.3", "datasets==3.6.0", "huggingface_hub==0.36.2", "qwen-vl-utils==0.0.14"]

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

    step(f"E1 {RUN_NAME}")
    command = [
        sys.executable, str(CODE / "experiments/e1/run_e1.py"),
        "--config", str(CODE / CONFIG),
        "--output-dir", str(WORK / "outputs" / RUN_NAME),
        "--max-samples", str(MAX_SAMPLES),
        "--behavior-n", str(BEHAVIOR_N),
    ]
    with open(log_dir / f"{RUN_NAME}.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    step(f"done, exit code {code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
