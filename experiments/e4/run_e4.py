"""E4 GPU step: the text-only task-consistency checker, run once per distinct (task, action) pair.

Pairs come from every E3 arm's final answers (or only the undefended answers if E3b is not available),
plus every attacker action paired with every test task, so the checker's verdicts do not depend on which
arm happened to produce an action. Answers are cached in results/e4_checker.json (resumable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e3"))
from analyze_e3b import load_arms  # noqa: E402
from benchmark import benchmark  # noqa: E402
from common import TEST_VARIANTS, baseline_responses, load_frames  # noqa: E402
from guard import canonical, checker_prompt  # noqa: E402

from argus.config import load_config  # noqa: E402
from argus.io_utils import load_json, save_json  # noqa: E402
from argus.modeling import vision_runner  # noqa: E402


@torch.no_grad()
def text_generate(runner, text: str, max_new_tokens: int = 4) -> str:
    conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    prompt = runner.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = runner.processor(text=[prompt], return_tensors="pt").to(runner.device)
    output = runner.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return runner.processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def all_answers(dirs: dict[str, Path], e3a_dir: Path | None, e3b_dir: Path | None) -> dict[str, dict[tuple[str, int], str]]:
    """Final answer per arm -> (variant, frame id). Only 'none' when E3 outputs are missing."""
    if e3a_dir and e3b_dir and (e3b_dir / "results").exists():
        answer, arm_names, *_ = load_arms(dirs, e3a_dir, e3b_dir / "results")
        frames = load_frames(dirs, "test")
        return {arm: {(v, i): answer(arm, v, i) for v in TEST_VARIANTS for i in frames} for arm in arm_names}
    base = baseline_responses(dirs)
    return {"none": {(v, i): base[v][i] for v in TEST_VARIANTS for i in base[v]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--e2-dir", required=True)
    parser.add_argument("--e2b-dir", required=True)
    parser.add_argument("--e2c-dir", required=True)
    parser.add_argument("--e3a-dir")
    parser.add_argument("--e3b-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dirs = {"e2": Path(args.e2_dir), "e2b": Path(args.e2b_dir), "e2c": Path(args.e2c_dir)}
    e3a, e3b = (Path(p) if p else None for p in (args.e3a_dir, args.e3b_dir))
    out = Path(args.output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(dirs, "test")
    tasks = {f["task"] for f in frames.values()}
    attacker_actions = {a for f in frames.values() for a in f["actions"].values()}
    pairs = {(frames[i]["task"], canonical(a)) for arm in all_answers(dirs, e3a, e3b).values() for (v, i), a in arm.items() if canonical(a)}
    pairs |= {(t, canonical(a)) for t in tasks for a in attacker_actions}
    pairs |= {(r["task"], canonical(r["action"])) for r in benchmark()}
    pairs = {p for p in pairs if not p[1].startswith(("move_forward", "turn_", "stop"))}

    cache_path = out / "e4_checker.json"
    cache = {(r["task"], r["action"]): r for r in load_json(cache_path)["rows"]} if cache_path.exists() else {}
    todo = sorted(pairs - set(cache))
    print(f"{len(pairs)} (task, object-action) pairs, {len(todo)} new")
    if todo:
        runner = vision_runner(load_config(args.config))
        for task, action in tqdm(todo, desc="checker"):
            cache[(task, action)] = {"task": task, "action": action, "answer": text_generate(runner, checker_prompt(task, action))}
        save_json({"prompt": checker_prompt("{task}", "{action}"), "rows": sorted(cache.values(), key=lambda r: (r["task"], r["action"]))}, cache_path)
    print(f"checker says yes to {sum(r['answer'].lower().startswith('yes') for r in cache.values())}/{len(cache)}")


if __name__ == "__main__":
    main()
