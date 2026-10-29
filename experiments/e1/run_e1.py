"""E1: are "follow user" vs "follow injection" linearly separable on this model?

Chains the existing ARGUS stages (data -> augmentation -> activations -> probes ->
orthogonal probes) and then runs analyze_e1.py for controls and the report.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
PATH_KEYS = ["alpaca_path", "glue_dir", "trigger_path", "font_path"]


def stage_path(name: str) -> str:
    return str(SRC_DIR / "argus" / "stages" / f"{name}.py")


def write_run_config(base_config: Path, output_dir: Path, max_samples: int | None) -> Path:
    with base_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # load_config resolves relative paths against the config's grandparent, so absolutize before moving it.
    for key in PATH_KEYS:
        value = config["data"].get(key)
        if value and not Path(value).is_absolute():
            config["data"][key] = str((ROOT_DIR / value).resolve())
    config["run"]["output_dir"] = str(output_dir)
    if max_samples is not None:
        config.setdefault("debug", {})["max_samples"] = max_samples
    run_config = output_dir / "configs" / "run_config.yaml"
    run_config.parent.mkdir(parents=True, exist_ok=True)
    with run_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return run_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "e1_qwen2vl_2b.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--behavior-n", type=int, default=50, help="Val samples for the natural attack-success check.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    config = str(write_run_config(Path(args.config).resolve(), output_dir, args.max_samples))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    common = ["--config", config] + (["--force"] if args.force else [])
    python = sys.executable

    stages = [
        ("create_data", [stage_path("create_data"), *common]),
        ("validate_data", [stage_path("validate_data"), "--config", config]),
        ("augment_answers", [stage_path("generate_augmented_answers"), *common]),
        ("collect_detection_train", [stage_path("collect_activations"), *common, "--mode", "detection", "--split", "train"]),
        ("collect_detection_val", [stage_path("collect_activations"), *common, "--mode", "detection", "--split", "val"]),
        ("collect_edit_train", [stage_path("collect_activations"), *common, "--mode", "edit", "--split", "train"]),
        ("collect_edit_val", [stage_path("collect_activations"), *common, "--mode", "edit", "--split", "val"]),
        ("train_detection_probe", [stage_path("train_probe"), *common, "--kind", "detection"]),
        ("train_edit_probe", [stage_path("train_probe"), *common, "--kind", "edit"]),
        ("train_auxiliary_probes", [stage_path("train_auxiliary_probes"), *common]),
        ("analyze", [str(Path(__file__).with_name("analyze_e1.py")), "--config", config, "--behavior-n", str(args.behavior_n)]),
    ]
    for name, command in stages:
        print(f"\n==== {name} ====", flush=True)
        subprocess.run([python, *command], check=True, env=env)


if __name__ == "__main__":
    main()
