"""Why does Qwen2-VL emit only 'Answer:' on a clean frame?

v24 measured Qwen2-VL-7B producing a 7-character chain of thought on 142 of
200 clean scenes, while the same scenes with a sign rendered in produced ~1300
characters. That makes the attacked and clean arms incomparable, so the 7B
numbers cannot be used until the cause is known.

This reproduces the behaviour on whatever Qwen checkpoint fits locally and
reports, per scene and per condition:

* how many characters and tokens the first turn produced,
* whether generation stopped on EOS or ran to the token budget,
* the first 80 characters of the reply.

Run it on a Qwen2-VL checkpoint (the affected architecture) and on a
Qwen2.5-VL one (which did not show the fault at 3B) to see whether the split
is by architecture.

    python scripts/diagnose_empty_generation.py --model Qwen/Qwen2-VL-2B-Instruct --limit 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from midas import data, signs  # noqa: E402
from midas.config import AttackConfig  # noqa: E402
from midas.models.qwen import QwenVLDriveLM  # noqa: E402
from midas.tasks import drivelm  # noqa: E402


def probe(runner: QwenVLDriveLM, images: list[np.ndarray], label: str, idx: int) -> dict:
    prompt = drivelm.build_prompt()
    inputs = runner._encode(images, runner._messages(images, prompt))
    n_in = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = runner.model.generate(
            **inputs,
            max_new_tokens=runner.max_new_tokens,
            do_sample=False,
            pad_token_id=runner.processor.tokenizer.pad_token_id,
        )
    new = out[:, n_in:]
    tok = runner.processor.tokenizer
    text = tok.batch_decode(new, skip_special_tokens=True)[0].strip()
    ids = new[0].tolist()
    eos = {tok.eos_token_id, getattr(tok, "pad_token_id", None)}
    stopped_on_eos = bool(ids) and ids[-1] in eos
    return {
        "scene": idx,
        "cond": label,
        "chars": len(text),
        "tokens": len(ids),
        "stop": "EOS" if stopped_on_eos else ("BUDGET" if len(ids) >= runner.max_new_tokens else "other"),
        "last_ids": ids[-4:],
        "head": text[:80].replace("\n", " "),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--split", default="data/drivelm/scenes_clean200.json")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--style", default="plain")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1250)
    args = p.parse_args(argv)

    scenes = data.load_scenes(args.split)[: args.limit]
    runner = QwenVLDriveLM(model_id=args.model, device=args.device,
                        max_new_tokens=args.max_new_tokens)
    tok = runner.processor.tokenizer
    print(f"\nmodel {args.model}")
    print(f"eos_token_id={tok.eos_token_id!r}  pad={tok.pad_token_id!r}  "
          f"max_new_tokens={runner.max_new_tokens}")
    print(f"\n{'scene':>5} {'cond':<9} {'chars':>6} {'tok':>5} {'stop':<7} reply head")
    print("-" * 100)

    rows = []
    for scene in scenes:
        target = data.resolve_target(scene, "auto")
        clean = scene.load_images()
        rows.append(probe(runner, clean, "clean", scene.index))
        attacked, _ = signs.attack_scene(scene, images=[c.copy() for c in clean],
                                         target=target, cfg=AttackConfig(),
                                         style=args.style)
        rows.append(probe(runner, attacked, args.style, scene.index))
        for r in rows[-2:]:
            print(f"{r['scene']:>5} {r['cond']:<9} {r['chars']:>6} {r['tokens']:>5} "
                  f"{r['stop']:<7} {r['head']!r}")

    for cond in dict.fromkeys(r["cond"] for r in rows):
        sub = [r for r in rows if r["cond"] == cond]
        empty = sum(1 for r in sub if r["chars"] <= 10)
        med = sorted(r["chars"] for r in sub)[len(sub) // 2]
        print(f"\n{cond:<9} median {med:5d} chars   empty(<=10) {empty}/{len(sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
