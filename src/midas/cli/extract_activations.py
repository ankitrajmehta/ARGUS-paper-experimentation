"""Collect last-token activations for probe training.

For every (scene, sign phrase) pair we teacher-force two answers on the *same*
image: the benign action and the attacker's target. The pair differs only in
which instruction is being followed, which is the distinction a probe has to
learn. Clean (unattacked) copies are also collected, for the injection gate.

Sweeping all phrases in CHAI's dictionary rather than just the optimised one
gives enough samples to hold out by wording, so a probe cannot pass by
memorising "PROCEED".

    midas-activations --split train --model models/InternVL2_5-2B --max-tiles 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from midas import controls, data, signs
from midas.config import AttackConfig, SignSpec

ANSWER_TEXT = {"a)": "a) proceed", "b)": "b) brake"}

# Marker for the open-turn (pre-generation) state the gate runs on.
PROMPT_ONLY = "__prompt__"


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
    parser.add_argument("--model", default="models/InternVL2_5-2B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--device-map", default=None, help="e.g. 'auto' to shard over GPUs")
    parser.add_argument(
        "--style",
        default=None,
        help="use an injection style's phrase instead of CHAI's dictionary "
        "(plain | authority | override | urgency). One phrase per scene.",
    )
    parser.add_argument(
        "--phrases",
        type=int,
        default=0,
        help="sign phrases per scene from CHAI's dictionary; 0 means all",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="record the open-turn state with no answer forced; this is what the "
        "injection gate sees at inference time",
    )
    parser.add_argument(
        "--readout", choices=["answer", "last"], default="answer",
        help="which token's state to record; see InternVLDriveLM.activations",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-clean", action="store_true", help="skip unattacked copies")
    parser.add_argument(
        "--no-attack", action="store_true",
        help="skip attack signs, e.g. to add controls to an existing sweep",
    )
    parser.add_argument(
        "--controls",
        type=int,
        default=0,
        help="benign scene-text signs per scene (0 = none). Rendered identically to the "
        "attack but saying something non-instructional, to test whether a probe keys on "
        "the instruction or merely on text being present.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/activations"))
    args = parser.parse_args(argv)

    scenes = data.load_scenes(args.split)
    if args.limit:
        scenes = scenes[: args.limit]
    dictionary = data.load_attack_dictionary()
    cfg = AttackConfig()

    from midas.models.internvl import InternVLDriveLM

    print(f"loading {args.model} ...")
    model = InternVLDriveLM(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        max_tiles=args.max_tiles,
        device_map=args.device_map,
    )

    rows: list[dict] = []
    vectors: list[np.ndarray] = []

    def record(acts: dict[int, "np.ndarray"], **meta) -> None:
        layers = sorted(acts)
        vectors.append(np.stack([acts[i].numpy() for i in layers]))
        rows.append(meta)

    for scene in scenes:
        target = data.resolve_target(scene, "auto")
        benign = scene.correct_internvl
        base_images = scene.load_images()

        if args.no_attack:
            phrases = []
        elif args.style:
            # One phrase per scene: the style defines it, so there is nothing
            # to sweep over.
            from midas import styles

            phrases = [styles.phrase_for(args.style, target)]
        else:
            phrases = dictionary[target]
            if args.phrases:
                phrases = phrases[: args.phrases]

        if not args.no_clean:
            answer_keys = [PROMPT_ONLY] if args.prompt_only else [benign, target]
            for answer_key in answer_keys:
                acts = model.activations(
                    base_images,
                    None if answer_key == PROMPT_ONLY else ANSWER_TEXT[answer_key],
                    readout=args.readout,
                )
                record(
                    acts,
                    scene=scene.index,
                    split=args.split,
                    attacked=False,
                    condition="clean",
                    style=args.style or "chai",
                    phrase="",
                    answer=answer_key,
                    follows_injection=0,  # nothing injected to follow
                    is_target_answer=int(answer_key == target),  # 0 for PROMPT_ONLY rows
                )
                model.free_cache()

        for phrase in phrases:
            sign = signs.build_sign(scene, target=target, cfg=cfg)
            images = list(base_images)
            idx = scene.attacked_camera
            images[idx] = signs.render(
                base_images[idx], type(sign)(phrase, sign.bg, sign.fg, sign.scale, sign.origin)
            )

            for answer_key in ([PROMPT_ONLY] if args.prompt_only else [benign, target]):
                acts = model.activations(images, None if answer_key == PROMPT_ONLY else ANSWER_TEXT[answer_key], readout=args.readout)
                record(
                    acts,
                    scene=scene.index,
                    split=args.split,
                    attacked=True,
                    condition="attack",
                    style=args.style or "chai",
                    phrase=phrase,
                    answer=answer_key,
                    # The label the steering probe learns: is the model about to
                    # follow the sign rather than the scene?
                    follows_injection=int(answer_key == target),
                    is_target_answer=int(answer_key == target),  # 0 for PROMPT_ONLY rows
                )
                model.free_cache()

            print(
                f"  scene {scene.index:2d}  attack  {phrase!r:18s} "
                f"({len(rows)} vectors)  [{model.memory_gb():.1f}GB]",
                flush=True,
            )

        for phrase in controls.BENIGN_PHRASES[: args.controls]:
            sign = signs.build_sign(scene, target=target, cfg=cfg)
            images = list(base_images)
            idx = scene.attacked_camera
            # Same colours, scale and position as the attack; only the text differs.
            images[idx] = signs.render(
                base_images[idx],
                SignSpec(phrase, sign.bg, sign.fg, sign.scale, sign.origin),
            )

            for answer_key in ([PROMPT_ONLY] if args.prompt_only else [benign, target]):
                acts = model.activations(images, None if answer_key == PROMPT_ONLY else ANSWER_TEXT[answer_key], readout=args.readout)
                record(
                    acts,
                    scene=scene.index,
                    split=args.split,
                    attacked=False,
                    condition="benign_text",
                    style="control",
                    phrase=phrase,
                    answer=answer_key,
                    # No instruction was injected, so there is nothing to follow.
                    follows_injection=0,
                    is_target_answer=int(answer_key == target),  # 0 for PROMPT_ONLY rows
                )
                model.free_cache()

            print(
                f"  scene {scene.index:2d}  control {phrase!r:18s} "
                f"({len(rows)} vectors)  [{model.memory_gb():.1f}GB]",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{data.split_slug(args.split)}_{Path(args.model).name}_tiles{args.max_tiles}_{args.readout}"
    if args.style:
        stem += f"_style-{args.style}"
    if args.controls:
        stem += f"_ctrl{args.controls}"
    if args.prompt_only:
        stem += "_promptonly"
    npz = args.out / f"{stem}.npz"
    meta = args.out / f"{stem}.json"

    # (samples, layers, hidden)
    np.savez_compressed(npz, activations=np.stack(vectors).astype(np.float32))
    meta.write_text(
        json.dumps(
            {
                "model": args.model,
                "split": args.split,
                "max_tiles": args.max_tiles,
                "readout": args.readout,
                "n_layers": vectors[0].shape[0],
                "hidden": vectors[0].shape[1],
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n{len(rows)} vectors, {vectors[0].shape[0]} layers x {vectors[0].shape[1]} dims")
    print(f"saved -> {npz}\n         {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
