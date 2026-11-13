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

    It also accepts a ``steerer`` and runs its gate. There are no real
    activations to gate on, so the decision is meaningless -- but every other
    part of the defense path (artifact loading, layer selection, the gate call,
    recording ``gate_score`` and ``steered`` in the output) is real, and those
    are what used to fail minutes into a GPU session.
    """

    # The stub has no decoder blocks, so a Steerer must never try to hook it.
    ACTIVATION_LAYER_OFFSET = 0

    def __init__(self, seed: int = 0, hidden: int = 8) -> None:
        self._rng = random.Random(seed)
        self.pending: tuple[str, str, bool] | None = None
        self.hidden = hidden

    def _layers(self):
        return []

    def ask(self, images, retry_on_unreadable: bool = True, steerer=None) -> Reply:
        gate_score = None
        steered = False
        if steerer is not None:
            activation = None
            if steerer.config.use_gate:
                # Shape, not content: a mismatch against the fitted gate is a
                # real error worth catching here rather than on the GPU.
                activation = np.zeros_like(steerer.artifact.gate_mean)
            steered = steerer.prepare(activation)
            gate_score = steerer.last_gate_score

        benign, target, attacked = self.pending or ("b)", "a)", False)
        action = target if attacked and self._rng.random() < 0.67 else benign
        word = "proceed" if action == "a)" else "brake"
        return Reply(
            long=f"[dry-run] ... {word}",
            short=word,
            action=action,
            retried=False,
            gate_score=gate_score,
            steered=steered,
        )


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
    style: str | None = None,
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
                images, sign = signs.attack_scene(
                    scene, images=images, target=target, cfg=cfg, style=style
                )
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
    parser.add_argument(
        "--split",
        default="train",
        help="train | test (CHAI's 23 scenes) | drivelm (built by "
        "midas-build-scenes) | a path to a scene JSON",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--attack", action="store_true", help="attacked frames only")
    mode.add_argument("--baseline", action="store_true", help="clean frames only (default)")
    mode.add_argument("--both", action="store_true", help="clean then attacked, and compare")

    parser.add_argument("--model", default=config.MODEL_ID)
    parser.add_argument(
        "--backend", choices=["auto", "internvl", "qwen"], default="auto",
        help="model family; inferred from the checkpoint name by default",
    )
    parser.add_argument(
        "--vision-dtype", default="auto",
        help="Qwen only. The vision tower emits NaN in float16 on frames "
        "carrying rendered text, so float32 is chosen automatically under a "
        "float16 language model.",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=None,
        help="Qwen only; the equivalent of --max-tiles. Default 409600.",
    )
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
        "--device-map",
        default=None,
        help="shard across GPUs with accelerate, e.g. 'auto' for an 8B model on two T4s",
    )
    parser.add_argument(
        "--no-flash-attn", action="store_true", help="disable FlashAttention (older GPUs)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=1250)
    parser.add_argument(
        "--swap-summarise",
        action="store_true",
        help="offer the two actions in the reverse order on the second turn, as a "
        "control for positional bias",
    )
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
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="benign-action labels re-derived for this model (see midas-labels). "
        "Without it, CHAI's stored labels are used, which describe their model "
        "and not necessarily ours.",
    )
    parser.add_argument("--target-mode", choices=["auto", "dataset"], default="auto")
    parser.add_argument("--background-mode", choices=["code", "json"], default="code")
    parser.add_argument(
        "--param-source",
        choices=["chai_internvl", "optimal_json", "custom"],
        default="chai_internvl",
        help="where the sign's appearance comes from. chai_internvl (default) is "
        "what CHAI's own InternVL path renders; optimal_json is their "
        "optimal.json, which that path never reads. See config.AttackConfig.",
    )
    parser.add_argument(
        "--attack-params",
        type=Path,
        default=None,
        help="a search result from midas-optimize; implies --param-source custom",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="injection style instead of CHAI's optimised phrase "
        "(plain | authority | override | urgency). Same colours, scale and "
        "position; only the wording changes.",
    )
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
    parser.add_argument(
        "--schedule", choices=["uniform", "argus"], default="uniform",
        help="uniform: the same adaptive capped step every pass (what section 12 "
        "measured, and it was inert). argus: the released schedule -- one uncapped "
        "full-magnitude step at the first generated token, then Eq. 7 steps, last "
        "position only.",
    )
    parser.add_argument(
        "--first-step-norm", type=float, default=None,
        help="with --schedule argus, the activation-space norm of the step-0 vector. "
        "The sibling project's run measured 9.9. Defaults to --strength.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.0,
        help="with --schedule argus, push past the probe boundary rather than onto it",
    )
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

    scenes = data.load_scenes(args.split, labels=args.labels)
    if args.limit:
        scenes = scenes[: args.limit]

    missing = data.missing_frames(scenes)
    if missing:
        raise SystemExit(f"{len(missing)} frames missing, first: {missing[0]}")

    if args.attack_params:
        from midas import optimize

        candidate = optimize.load_candidate(args.attack_params)
        cfg = candidate.as_attack_config(args.background_mode)
        print(f"sign from {args.attack_params}: {candidate.to_json()}")
    else:
        cfg = AttackConfig(
            param_source=args.param_source, background_mode=args.background_mode
        )

    if args.dry_run:
        print("dry run: stub model, no GPU")
        model = _StubModel(seed=args.seed)
    else:
        # Imported here so --help, --dry-run and the data checks above work
        # without torch installed.
        from midas.models import load_model

        print(f"loading {args.model} (backend {args.backend}) ...")
        model = load_model(
            args.model,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            vision_dtype=args.vision_dtype,
            load_4bit=args.load_4bit,
            use_flash_attn=not args.no_flash_attn,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            max_tiles=args.max_tiles,
            max_pixels=args.max_pixels,
            swap_summarise_options=args.swap_summarise,
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
                schedule=args.schedule,
                first_step_norm=args.first_step_norm,
                epsilon=args.epsilon,
            ),
        )
        print(
            f"defense: layers {steerer.selected() and [d.layer for d in steerer.selected()]}, "
            f"gate layer {artifact.gate_layer}, "
            f"schedule={args.schedule} "
            f"subspace={not args.no_subspace} adaptive={not args.no_adaptive} "
            f"gate={not args.no_gate} strength={args.strength}"
        )
        if args.schedule == "argus":
            print(
                f"         step 0 norm {args.first_step_norm or args.strength} uncapped, "
                f"then relu((||w||s+b+{args.epsilon})/||w||^2); "
                f"||w|| per layer: "
                + " ".join(f"L{d.layer}={d.probe_norm:.1f}" for d in steerer.selected())
            )
            if all(d.probe_norm == 1.0 for d in steerer.selected()):
                print(
                    "         WARNING: every ||w|| is exactly 1.0, so this artifact predates "
                    "probe-norm recording and the Eq. 7 steps will be ~||w|| times too large. "
                    "Refit with midas-fit-defense."
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
        journal = args.out / f"{label}_{data.split_slug(args.split)}.jsonl"
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
            style=args.style,
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
        record.label_source = str(args.labels) if args.labels else "chai"
        record.param_source = (
            str(args.attack_params) if args.attack_params else args.param_source
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
