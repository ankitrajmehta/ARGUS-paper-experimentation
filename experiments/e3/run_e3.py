"""E3: the ARGUS pipeline exactly as released (scripts/run_all.py), then a steering-only eval, then the report.

The steering-only eval is the paper's "ARGUS w/o PF" row: the same trained vector, probes and epsilon, with
defense.enable_post_filtering false. Upstream writes both defense modes to results/image_argus.json, so the
full-ARGUS file is moved aside while the steering-only eval runs and put back afterwards. No src change.
Every stage skips outputs that already exist, so rerunning resumes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = Path(args.config).resolve()
    out = Path(args.output_dir).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    py = sys.executable

    def run(name: str, command: list[str]) -> None:
        print(f"\n==== {name} ====", flush=True)
        subprocess.run([py, *command], check=True, env=env, cwd=ROOT_DIR)

    run("ARGUS run_all.py (as released)", [str(ROOT_DIR / "scripts" / "run_all.py"), "--config", str(config), "--output-dir", str(out)])

    results = out / "results"
    full, parked, steer = results / "image_argus.json", results / "image_argus.full.json", results / "image_argus_steer_only.json"
    if parked.exists() and not full.exists():
        parked.rename(full)  # an earlier steering-only eval was interrupted
    if not steer.exists():
        with config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("defense", {})["enable_post_filtering"] = False
        # Same folder as the main config: ARGUS resolves relative paths from the config's grandparent.
        steer_config = config.with_name(config.stem + "_steer_only.yaml")
        with steer_config.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        full.rename(parked)
        try:
            run("eval steering only (post-filter off)", [str(ROOT_DIR / "src" / "argus" / "stages" / "run_defense_eval.py"),
                                                          "--config", str(steer_config), "--output-dir", str(out), "--mode", "defense"])
            full.rename(steer)
        finally:
            if full.exists() and not steer.exists():
                full.unlink()  # partial output of a failed steering-only eval
            parked.rename(full)
    run("analyze", [str(Path(__file__).with_name("analyze_e3.py")), "--config", str(config), "--output-dir", str(out)])


if __name__ == "__main__":
    main()
