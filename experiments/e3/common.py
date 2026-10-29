"""Shared E3 helpers: frames and sign variants across E2/E2b/E2c, answer-token states, action checks.

Notation used throughout E3:
  A_u - the action the robot chose on the clean frame (stand-in for "following the user")
  A_x - the attacker's action written on a sign
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2c"))
from analyze_e2c import effect  # noqa: E402,F401
from collect_e2 import normalize  # noqa: E402

from argus.io_utils import load_json  # noqa: E402

# Variant name -> (source experiment, image key in that experiment's records).
ATTACKS = {
    "train": {"injected_sign": ("e2", "injected_sign"), "e2c_attack_1": ("e2c", "attack_1"), "e2c_attack_2": ("e2c", "attack_2")},
    "test": {"injected_sign": ("e2", "injected_sign"), "attack_plain": ("e2b", "attack_plain"),
             "attack_no_trigger": ("e2b", "attack_no_trigger"), "e2c_attack": ("e2c", "attack")},
}
# Harmless sign variant -> (source, image key, attack whose A_x it is paired with for the behavior probe).
HARMLESS_SIGNS = {
    "train": {"benign_sign": ("e2", "benign_sign", "injected_sign"), "e2c_legit": ("e2c", "legit", "e2c_attack_1")},
    "test": {"benign_sign": ("e2", "benign_sign", "injected_sign"), "legit_imperative": ("e2b", "legit_imperative", "attack_plain"),
             "e2c_legit": ("e2c", "legit", "e2c_attack")},
}
TEST_VARIANTS = list(ATTACKS["test"]) + ["clean"] + list(HARMLESS_SIGNS["test"])
# Where each test variant's undefended response lives: (source, behavior-json response key).
RESPONSE_KEYS = {"clean": ("e2", "clean"), "injected_sign": ("e2", "injected_sign"), "benign_sign": ("e2", "benign_sign"),
                 "attack_plain": ("e2b", "attack_plain"), "attack_no_trigger": ("e2b", "attack_no_trigger"),
                 "legit_imperative": ("e2b", "legit_imperative"), "e2c_attack": ("e2c", "attack"), "e2c_legit": ("e2c", "legit")}
BEHAVIOR_FILES = {"e2": "e2_behavior.json", "e2b": "e2b_behavior.json", "e2c": "e2c_behavior.json"}

ACTION_RE = re.compile(r"^(move_forward|turn_left|turn_right|stop|pick_up\([^(),]+\)|put\([^(),]+,\s*[^(),]+\)|open\([^(),]+\)|close\([^(),]+\))$")


def same_action(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


def is_valid_action(text: str) -> bool:
    lines = text.strip().splitlines()
    return bool(lines) and ACTION_RE.match(lines[0].strip()) is not None and len(lines) == 1


def is_stop(text: str) -> bool:
    return text.strip().lower().startswith("stop")


def load_frames(dirs: dict[str, Path], split: str) -> dict[int, dict]:
    """Per frame id: prompt, scene, image per variant and A_x per attack variant."""
    e2 = load_json(dirs["e2"] / "data" / f"e2_{split}.json")
    e2c = {r["id"]: r for r in load_json(dirs["e2c"] / "data" / f"e2c_{split}.json")}
    e2b = {r["id"]: r for r in load_json(dirs["e2b"] / "data" / "e2b_test.json")} if split == "test" else {}
    frames = {}
    for item in e2:
        i = item["id"]
        images = {"clean": item["images"]["clean"]}
        actions = {}
        for name, (source, key) in ATTACKS[split].items():
            if source == "e2":
                images[name], actions[name] = item["images"][key], item["injected_action"]
            elif source == "e2b":
                images[name], actions[name] = e2b[i]["images"][key], e2b[i]["injected_action"]
            else:
                images[name], actions[name] = e2c[i]["images"][key], e2c[i]["signs"][key]["action"]
        for name, (source, key, _) in HARMLESS_SIGNS[split].items():
            images[name] = {"e2": item["images"], "e2b": e2b.get(i, {}).get("images", {}), "e2c": e2c[i]["images"]}[source][key]
        styles = {name: e2c[i]["signs"][ATTACKS[split][name][1]]["style"] for name in ATTACKS[split] if ATTACKS[split][name][0] == "e2c"}
        frames[i] = {"id": i, "scene": item["scene_name"], "task": item["task"], "prompt": item["prompt"],
                     "images": images, "actions": actions, "styles": styles}
    return frames


def baseline_responses(dirs: dict[str, Path]) -> dict[str, dict[int, str]]:
    """Undefended test responses per variant, from the finished E2/E2b/E2c behavior runs."""
    samples = {src: {s["id"]: s for s in load_json(dirs[src] / "results" / name)["samples"]} for src, name in BEHAVIOR_FILES.items()}
    return {v: {i: s["responses"][key] for i, s in samples[src].items()} for v, (src, key) in RESPONSE_KEYS.items()}


def val_scenes(train_frames: dict[int, dict], seed: int = 42) -> list[str]:
    # Same 2 held-out train scenes as E2c (analyze_e2c.py): sorted, shuffled with the same seed.
    import random

    scenes = sorted({f["scene"] for f in train_frames.values()})
    random.Random(seed).shuffle(scenes)
    return scenes[:2]


@torch.no_grad()
def answer_states(runner, prompt: str, image: str, prefix: str) -> tuple[torch.Tensor, int]:
    """Hidden states at the first and last token of a forced answer prefix: (2, layers, hidden), plus its token count."""
    inputs = runner._inputs(prompt, image, prefix=prefix)
    n = len(runner.processor.tokenizer(prefix, add_special_tokens=False).input_ids)
    outputs = runner.model(**inputs, output_hidden_states=True)
    # Skip the embedding state so index 0 is transformer layer 0, as in VisionLanguageRunner.activation.
    states = torch.stack([h[0, [-n, -1]].detach().cpu() for h in outputs.hidden_states[1:]], dim=1)
    if not torch.isfinite(states).all():
        raise FloatingPointError(f"non-finite activations for {image}; try model.vision_dtype: float32")
    return states, n


def mask_sign(sign_path: str, clean_path: str, out_path: Path) -> list[int]:
    """Grey out the sign: its box is wherever the signed frame differs from the clean frame (oracle box)."""
    signed = np.asarray(Image.open(sign_path).convert("RGB")).astype(np.int16)
    clean = np.asarray(Image.open(clean_path).convert("RGB")).astype(np.int16)
    diff = np.abs(signed - clean).sum(axis=2) > 0
    rows, cols = np.where(diff.any(axis=1))[0], np.where(diff.any(axis=0))[0]
    if len(rows) == 0:
        raise ValueError(f"{sign_path} does not differ from its clean frame")
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    out = signed.astype(np.uint8).copy()
    out[y0:y1, x0:x1] = 128
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path)
    return [x0, y0, x1, y1]


def subset(frames: dict[int, dict], n: int | None) -> dict[int, dict]:
    """Smoke runs: n evenly spaced frames, so every scene (incl. the held-out ones) can appear."""
    if not n or n >= len(frames):
        return frames
    items = list(frames.items())
    return dict(items[:: len(items) // n][:n])
