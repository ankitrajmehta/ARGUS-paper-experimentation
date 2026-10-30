"""E1-S: the full ARGUS pipeline on the paper's own data (VTQA), to test whether ARGUS steering works on this model.

Stage order (ARGUS validate_config.py): E1 stages (create_data .. train_auxiliary_probes, via run_e1.py) ->
preliminary_experiments (layer/strength search) -> train_vector -> train_final_steering_probe -> estimate_epsilon ->
run_defense_eval no_defense -> defense with steering only (post-filter off) -> full ARGUS (steering + post-filter).
Then analyze_e1s.py writes the report. Every stage skips outputs that already exist (resumable).
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


def stage(name: str) -> str:
    return str(SRC_DIR / "argus" / "stages" / f"{name}.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT_DIR / "configs" / "kaggle_e1s_3b.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--behavior-n", type=int, default=50)
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    py = sys.executable

    def run(name: str, command: list[str]) -> None:
        print(f"\n==== {name} ====", flush=True)
        subprocess.run([py, *command], check=True, env=env)

    e1 = [str(ROOT_DIR / "experiments" / "e1" / "run_e1.py"), "--config", args.config, "--output-dir", str(out),
          "--behavior-n", str(args.behavior_n)] + (["--max-samples", str(args.max_samples)] if args.max_samples else [])
    run("E1 stages (data, augmentation, activations, probes, auxiliary probes)", e1)

    config = out / "configs" / "run_config.yaml"
    # Steering-only variant of the same run: identical paths, post-filter off.
    with config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("defense", {})["enable_post_filtering"] = False
    steer_config = out / "configs" / "run_config_steer_only.yaml"
    with steer_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    c = ["--config", str(config)]
    run("preliminary_experiments", [stage("preliminary_experiments"), *c])
    run("train_vector", [stage("train_vector"), *c])
    run("train_final_steering_probe", [stage("train_post_filter_probe"), *c])
    run("estimate_epsilon", [stage("estimate_epsilon"), *c])
    run("eval no_defense", [stage("run_defense_eval"), *c, "--mode", "no_defense"])
    run("eval steering only", [stage("run_defense_eval"), "--config", str(steer_config), "--mode", "defense", "--tag", "steer_only"])
    run("eval full ARGUS", [stage("run_defense_eval"), *c, "--mode", "defense"])
    run("analyze", [str(Path(__file__).with_name("analyze_e1s.py")), "--config", str(config)])


if __name__ == "__main__":
    main()
