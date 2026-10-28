"""Run the CHAI/DriveLM evaluation: baseline, attack, or both.

Needs a GPU. Typical use on the college box::

    midas-eval --split train --both          # baseline and attack in one pass
    midas-eval --split test  --attack        # transferability, attacked only
    midas-eval --split train --both --load-4bit   # on a <24 GB card
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from midas import config, data, evaluate, signs
from midas.models.base import Reply
from midas.config import AttackConfig
from midas.evaluate import RunRecord, SceneResult


class _StubModel:
    """Fake model for --dry-run: exercises the pipeline with no GPU.

    Answers the benign action on clean frames and flips to the target on roughly
    two thirds of attacked frames, so the plumbing, scoring and output files can
    be checked on a laptop before touching the real model.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.pending: tuple[str, str, bool] | None = None

    def ask(self, images, retry_on_unreadable: bool = True) -> Reply:
        benign, target, attacked = self.pending or ("b)", "a)", False)
        action = target if attacked and self._rng.random() < 0.67 else benign
        word = "proceed" if action == "a)" else "brake"
        return Reply(long=f"[dry-run] ... {word}", short=word, action=action, retried=False)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _run_condition(
    model,
    scenes: list[data.Scene],
    *,
    attacked: bool,
    cfg: AttackConfig,
    target_mode: str,
    repeats: int,
    verbose: bool,
    journal: Path | None = None,
    steerer=None,
) -> list[SceneResult]:
    results: list[SceneResult] = []
    for repeat in range(repeats):
        for scene in scenes:
            target = data.resolve_target(scene, target_mode)
            if target is None:
                continue

            images = scene.load_images()
            sign_text = None
            if attacked:
                images, sign = signs.attack_scene(scene, images=images, target=target, cfg=cfg)
                sign_text = sign.text.upper()

            if isinstance(model, _StubModel):
                model.pending = (scene.correct_internvl, target, attacked)
            reply = (
                model.ask(images, steerer=steerer)
                if steerer is not None
                else model.ask(images)
            )
            result = SceneResult(
                scene_index=scene.index,
                split=scene.split,
                attacked=attacked,
                target=target,
                benign_label=scene.correct_internvl,
                action=reply.action,
                short=reply.short.strip(),
                long=reply.long.strip() if verbose else "",
                sign_text=sign_text,
                retried=reply.retried,
                repeat=repeat,
                gate_score=reply.gate_score,
                steered=reply.steered,
            )
            results.append(result)

            # Append before anything else can fail, so an interrupted sweep
            # keeps every scene it finished.
            if journal is not None:
                with journal.open("a") as fh:
                    fh.write(json.dumps(asdict(result)) + "\n")

            # Release cached allocator blocks between scenes: on unified memory
            # the cache otherwise grows until the process swaps.
            mem = ""
            if hasattr(model, "free_cache"):
                model.free_cache()
                mem = f"  [{model.memory_gb():.1f}GB]"

            flag = "HIT " if result.hit_target else "    "
            # flush: these runs are long and usually redirected to a log, where
            # block buffering would hide progress for tens of minutes.
            print(
                f"  {flag}scene {scene.index:2d} rep{repeat} "
                f"target={target} benign={scene.correct_internvl} "
                f"-> {result.action or 'UNREADABLE'}  {result.short[:50]!r}{mem}",
                flush=True,
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--attack", action="store_true", help="attacked frames only")
    mode.add_argument("--baseline", action="store_true", help="clean frames only (default)")
    mode.add_argument("--both", action="store_true", help="clean then attacked, and compare")

    parser.add_argument("--model", default=config.MODEL_ID)
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="auto picks bfloat16 on capable CUDA, float16 on MPS, float32 on CPU",
    )
    parser.add_argument(
        "--load-4bit", action="store_true", help="quantise to fit a <24 GB card"
    )
    parser.add_argument(
        "--no-flash-attn", action="store_true", help="disable FlashAttention (older GPUs)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1250)
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=12,
        help="image tiles per frame (CHAI use 12). Lower is faster but less faithful.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="sample instead of greedy decoding. CHAI sample; we default to greedy so the "
        "attack effect is not buried in decoding noise.",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="runs per scene; >1 is only useful with --sample"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="first N scenes, for smoke tests")
    parser.add_argument("--target-mode", choices=["auto", "dataset"], default="auto")
    parser.add_argument("--background-mode", choices=["code", "json"], default="code")
    parser.add_argument("--out", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--save-long", action="store_true", help="store full chain-of-thought in the JSON"
    )
    parser.add_argument(
        "--defend",
        type=Path,
        default=None,
        help="path to a fitted defense artifact; steers generation when the gate fires",
    )
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--no-subspace", action="store_true", help="ablate: raw direction")
    parser.add_argument("--no-adaptive", action="store_true", help="ablate: fixed strength")
    parser.add_argument("--no-gate", action="store_true", help="ablate: always steer")
    parser.add_argument(
        "--top-layers", type=int, default=None,
        help="steer only this many layers, best probe first (default: all fitted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use a stub model: checks data, rendering, scoring and outputs with no GPU",
    )
    args = parser.parse_args(argv)

    _seed_everything(args.seed)

    scenes = data.load_scenes(args.split)
    if args.limit:
        scenes = scenes[: args.limit]

    missing = data.missing_frames(scenes)
    if missing:
        raise SystemExit(f"{len(missing)} frames missing, first: {missing[0]}")

    cfg = AttackConfig(background_mode=args.background_mode)

    if args.dry_run:
        print("dry run: stub model, no GPU")
        model = _StubModel(seed=args.seed)
    else:
        # Imported here so --help, --dry-run and the data checks above work
        # without torch installed.
        from midas.models.internvl import InternVLDriveLM

        print(f"loading {args.model} ...")
        model = InternVLDriveLM(
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            load_4bit=args.load_4bit,
            use_flash_attn=not args.no_flash_attn,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            max_tiles=args.max_tiles,
        )

    steerer = None
    if args.defend:
        from midas.steering import DefenseArtifact, SteerConfig, Steerer

        artifact = DefenseArtifact.load(args.defend)
        steerer = Steerer(
            model,
            artifact,
            SteerConfig(
                strength=args.strength,
                use_subspace=not args.no_subspace,
                adaptive=not args.no_adaptive,
                use_gate=not args.no_gate,
                top_layers=args.top_layers,
            ),
        )
        print(
            f"defense: layers {steerer.selected() and [d.layer for d in steerer.selected()]}, "
            f"gate layer {artifact.gate_layer}, "
            f"subspace={not args.no_subspace} adaptive={not args.no_adaptive} "
            f"gate={not args.no_gate} strength={args.strength}"
        )

    conditions = []
    if args.both:
        conditions = [("baseline", False), ("attack", True)]
    elif args.attack:
        conditions = [("attack", True)]
    else:
        conditions = [("baseline", False)]

    collected: dict[str, list[SceneResult]] = {}
    for label, attacked in conditions:
        print(f"\n=== {label} ({args.split}, {len(scenes)} scenes) ===", flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        journal = args.out / f"{label}_{args.split}.jsonl"
        journal.unlink(missing_ok=True)
        results = _run_condition(
            model,
            scenes,
            attacked=attacked,
            cfg=cfg,
            target_mode=args.target_mode,
            repeats=args.repeats,
            verbose=args.save_long,
            journal=journal,
            steerer=steerer,
        )
        collected[label] = results

        record = RunRecord(
            label=label,
            split=args.split,
            attacked=attacked,
            model_id=args.model,
            do_sample=args.sample,
            seed=args.seed,
            repeats=args.repeats,
            background_mode=args.background_mode if attacked else None,
            target_mode=args.target_mode,
            results=results,
        )
        path = record.save(args.out)
        print(f"\n{evaluate.summarise(results).as_row(label)}")
        print(f"saved -> {path}")

    if args.both:
        print("\n=== comparison ===")
        print(evaluate.compare(collected["baseline"], collected["attack"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
