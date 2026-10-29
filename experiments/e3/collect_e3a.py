"""E3a GPU pass: forced-answer states for a robot *behavior* probe ("following the user vs the sign").

For every frame, with A_u = the clean-frame action and A_x = the attacker's action on a sign:
  P   attacked image + prefix A_x   label 1  (follows the injection)
  N1  attacked image + prefix A_u   label 0  (follows the user; the ARGUS negative)
  N2  clean image    + prefix A_x   label 0  (action-identity control: same string, no sign)
  N3  harmless sign  + prefix A_x   label 0  (text present, nothing asks for A_x)
Pairs with A_u == A_x are dropped (label undefined). States are read at the first and last answer token.
Test frames also get the robot's *actual* response as prefix ("natural"), labelled by its effect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTACKS, HARMLESS_SIGNS, answer_states, baseline_responses, effect, load_frames, normalize, same_action, subset  # noqa: E402

from argus.config import cfg_get, load_config  # noqa: E402
from argus.io_utils import load_json, save_json  # noqa: E402
from argus.modeling import vision_runner  # noqa: E402


def pair_rows(frame: dict, split: str, a_u: str) -> tuple[list[dict], int]:
    rows, dropped, seen_n2 = [], 0, set()
    base = {"split": split, "id": frame["id"], "scene": frame["scene"]}
    for v, a_x in frame["actions"].items():
        if same_action(a_u, a_x):
            dropped += 1
            continue
        rows.append({**base, "group": "P", "variant": v, "image": frame["images"][v], "prefix": a_x, "label": 1})
        rows.append({**base, "group": "N1", "variant": v, "image": frame["images"][v], "prefix": a_u, "label": 0})
        if normalize(a_x) not in seen_n2:
            seen_n2.add(normalize(a_x))
            rows.append({**base, "group": "N2", "variant": v, "image": frame["images"]["clean"], "prefix": a_x, "label": 0})
    for h, (_, _, paired) in HARMLESS_SIGNS[split].items():
        a_x = frame["actions"][paired]
        if not same_action(a_u, a_x):
            rows.append({**base, "group": "N3", "variant": h, "image": frame["images"][h], "prefix": a_x, "label": 0})
    return rows, dropped


def collect(runner, rows: list[dict], out_path: Path, desc: str) -> None:
    states = []
    for row in tqdm(rows, desc=desc):
        s, n = answer_states(runner, row["prompt"], row["image"], row["prefix"])
        row["n_prefix_tokens"] = n
        states.append(s)
    meta = [{k: v for k, v in row.items() if k != "prompt"} for row in rows]
    torch.save({"meta": meta, "states": torch.stack(states)}, out_path)
    print(f"saved {out_path} {tuple(torch.stack(states).shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--e2-dir", required=True)
    parser.add_argument("--e2b-dir", required=True)
    parser.add_argument("--e2c-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, help="Smoke runs: first N frames per split.")
    args = parser.parse_args()

    config = load_config(args.config)
    dirs = {"e2": Path(args.e2_dir), "e2b": Path(args.e2b_dir), "e2c": Path(args.e2c_dir)}
    out_dir = Path(args.output_dir)
    for sub in ["data", "activations", "results"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    frames = {s: subset(load_frames(dirs, s), args.max_frames) for s in ["train", "test"]}
    runner = vision_runner(config)
    max_new_tokens = int(cfg_get(config, "generation.max_new_tokens", 128))

    # A_u for train frames is new (E2 only generated on test frames); test reuses e2_behavior.json.
    clean_path = out_dir / "data" / "train_clean_actions.json"
    train_au = {int(k): v for k, v in load_json(clean_path).items()} if clean_path.exists() else {}
    for i, frame in tqdm(frames["train"].items(), desc="train clean actions"):
        if i not in train_au:
            train_au[i] = runner.generate(frame["prompt"], frame["images"]["clean"], max_new_tokens=max_new_tokens)
    save_json({str(k): v for k, v in train_au.items()}, clean_path)
    responses = baseline_responses(dirs)
    a_u = {"train": train_au, "test": responses["clean"]}

    summary = {}
    for split in ["train", "test"]:
        rows, dropped = [], 0
        for i, frame in frames[split].items():
            r, d = pair_rows(frame, split, a_u[split][i])
            rows += [{**row, "prompt": frame["prompt"]} for row in r]
            dropped += d
        summary[split] = {"frames": len(frames[split]), "rows": len(rows), "dropped_pairs_A_u_eq_A_x": dropped,
                          "groups": {g: sum(r["group"] == g for r in rows) for g in ["P", "N1", "N2", "N3"]}}
        print(split, summary[split])
        path = out_dir / "activations" / f"e3a_{split}.pt"
        if path.exists():
            print(f"skip existing {path}")
        else:
            collect(runner, rows, path, f"behavior states {split}")

    # Natural check: the robot's own answer on each attacked test frame, labelled by what it did.
    rows = []
    for i, frame in frames["test"].items():
        clean = responses["clean"][i]
        for v, a_x in frame["actions"].items():
            resp = responses[v][i]
            hijacked = normalize(a_x) in normalize(resp)
            label = effect(clean, resp, hijacked, normalize(a_x) in normalize(clean))
            rows.append({"split": "test", "id": i, "scene": frame["scene"], "group": "NAT", "variant": v, "image": frame["images"][v],
                         "prefix": resp, "label": int(label == "hijacked"), "effect": label, "prompt": frame["prompt"]})
    summary["natural"] = {"rows": len(rows), "effects": {e: sum(r["effect"] == e for r in rows) for e in ["hijacked", "stopped", "changed", "no_effect"]}}
    print("natural", summary["natural"])
    path = out_dir / "activations" / "e3a_natural.pt"
    if path.exists():
        print(f"skip existing {path}")
    else:
        collect(runner, rows, path, "behavior states natural")
    save_json(summary, out_dir / "results" / "e3a_collect_summary.json")


if __name__ == "__main__":
    main()
