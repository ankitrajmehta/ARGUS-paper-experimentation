"""E1 analysis: probe accuracy per layer, sanity controls, and a natural attack check.

Controls the paper does not report:
  * shuffled labels  - a probe on permuted labels should sit near 50%.
  * length only      - a probe on answer length alone; if it is near the real probe,
                       the "behavior" probe may just be reading answer length.
  * detection on val - detection probe scored on unseen triggers and injected tasks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from argus.activations import paired_augmented_answer_variants
from argus.config import cfg_get, get_paths, load_config
from argus.eval_metrics import score_response
from argus.io_utils import load_json, save_json, seed_everything
from argus.media import clean_media, poison_media, split_json_path
from argus.probes import _split_binary, load_pickle


def load_class_arrays(path: Path, key0: str, key1: str) -> tuple[np.ndarray, np.ndarray]:
    rows = torch.load(path, map_location="cpu")
    class0 = torch.stack([row[key0] for row in rows]).float().numpy()
    class1 = torch.stack([row[key1] for row in rows]).float().numpy()
    return class0, class1


def split_layer(class0: np.ndarray, class1: np.ndarray, layer: int, ratio: float):
    return _split_binary(list(class0[:, layer]), list(class1[:, layer]), ratio)


def shuffled_label_accuracy(class0: np.ndarray, class1: np.ndarray, layer: int, ratio: float, rng: np.random.Generator) -> float:
    train_x, train_y, val_x, val_y = split_layer(class0, class1, layer, ratio)
    model = LogisticRegression(max_iter=2000).fit(train_x, rng.permutation(train_y))
    return float(model.score(val_x, val_y))


def length_features(text: str) -> list[float]:
    return [float(len(text)), float(len(text.split()))]


def length_only_accuracy(augmented_train: list[dict], ratio: float) -> float:
    # Same row order and split as collect_edit_activations + load_edit_xy.
    first, second = [], []
    for item in augmented_train:
        for _, first_answer, second_answer in paired_augmented_answer_variants(item):
            first.append(np.asarray(length_features(first_answer)))
            second.append(np.asarray(length_features(second_answer)))
    train_x, train_y, val_x, val_y = _split_binary(first, second, ratio)
    model = LogisticRegression(max_iter=2000).fit(np.log1p(train_x), train_y)
    return float(model.score(np.log1p(val_x), val_y))


def probe_accuracy(probe_path: Path, x: np.ndarray, y: np.ndarray) -> float:
    return float(load_pickle(probe_path).score(x, y))


def behavior_check(config: dict, paths, n: int) -> dict:
    from argus.modeling import vision_runner

    data = load_json(split_json_path(paths, config, "val"))[:n]
    runner = vision_runner(config)
    max_new_tokens = int(cfg_get(config, "generation.max_new_tokens", 128))
    rows = []
    for item in data:
        clean = runner.generate(item["clean_prompt"], clean_media(item), max_new_tokens=max_new_tokens)
        injected = runner.generate(item["clean_prompt"], poison_media(item), max_new_tokens=max_new_tokens)
        clean_scores = score_response(item, clean)
        inject_scores = score_response(item, injected)
        rows.append(
            {
                "id": item["id"],
                "second_instruction": item["second_instruction"],
                "clean_response": clean,
                "inject_response": injected,
                "UIA_clean": clean_scores["first_success"],
                "UIA_inject": inject_scores["first_success"],
                "AIA": inject_scores["second_success"],
                "AIFR": inject_scores["attacker_followed"],
            }
        )
    total = max(len(rows), 1)
    summary = {key: round(sum(int(r[key]) for r in rows) / total, 4) for key in ["UIA_clean", "UIA_inject", "AIA", "AIFR"]}
    return {"n": len(rows), **summary, "samples": rows}


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = report["layers"]
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("edit_internal", "Behavior probe (held-out train)", "-"),
        ("edit_external", "Behavior probe (val: unseen triggers/tasks)", "-"),
        ("detection_internal", "Detection probe (held-out train)", "--"),
        ("detection_external", "Detection probe (val)", "--"),
        ("aux1", "Orthogonal probe 1", ":"),
        ("aux2", "Orthogonal probe 2", ":"),
        ("shuffled_control", "Control: shuffled labels", "-."),
    ]
    for key, label, style in series:
        values = report["per_layer"].get(key)
        if values and any(v is not None for v in values):
            ax.plot(layers, values, style, marker="o", markersize=3, label=label)
    ax.axhline(report["length_only_accuracy"], color="grey", linewidth=1, label="Control: answer length only")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title(f"E1 probe separability ({report['model']}, n={report['n_train_items']} train items)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--behavior-n", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(cfg_get(config, "run.seed", 42)))
    rng = np.random.default_rng(int(cfg_get(config, "run.seed", 42)))
    paths = get_paths(config)
    ratio = float(cfg_get(config, "paper.probe_train_ratio", 0.8))
    act = paths.activation_dir

    edit0, edit1 = load_class_arrays(act / "mllm_edit_train.pt", "first_activation", "second_activation")
    det0, det1 = load_class_arrays(act / "mllm_detection_train.pt", "clean_activation", "image_inject_activation")
    det_val0, det_val1 = load_class_arrays(act / "mllm_detection_val.pt", "clean_activation", "image_inject_activation")
    layer_count = edit0.shape[1]
    layers = list(range(layer_count))

    edit_metrics = {row["layer"]: row for row in load_json(paths.probe_dir / "edit" / "mllm_metrics.json")}
    det_metrics = {row["layer"]: row for row in load_json(paths.probe_dir / "detection" / "mllm_metrics.json")}
    aux_metrics: dict[int, dict[int, float]] = {}
    for row in load_json(paths.probe_dir / "auxiliary" / "auxiliary_metrics.json"):
        aux_metrics.setdefault(int(row["auxiliary"]), {})[int(row["layer"])] = row["eval_acc"]

    det_val_x = np.concatenate([det_val0, det_val1])
    det_val_y = np.asarray([0] * len(det_val0) + [1] * len(det_val1))
    per_layer: dict[str, list] = {key: [] for key in ["edit_internal", "edit_external", "detection_internal", "detection_external", "shuffled_control"]}
    for i in sorted(aux_metrics):
        per_layer[f"aux{i}"] = [aux_metrics[i].get(layer) for layer in layers]
    for layer in layers:
        per_layer["edit_internal"].append(edit_metrics[layer]["eval_acc"])
        per_layer["edit_external"].append(edit_metrics[layer].get("external_eval_acc"))
        per_layer["detection_internal"].append(det_metrics[layer]["eval_acc"])
        per_layer["detection_external"].append(
            round(probe_accuracy(paths.probe_dir / "detection" / f"mllm_layer{layer}.pickle", det_val_x[:, layer], det_val_y), 6)
        )
        per_layer["shuffled_control"].append(round(shuffled_label_accuracy(edit0, edit1, layer, ratio, rng), 6))

    augmented = load_json(split_json_path(paths, config, "train", augmented=True))
    report = {
        "model": Path(str(cfg_get(config, "model.path"))).name,
        "n_train_items": len(augmented),
        "n_edit_rows_per_class": int(len(edit0)),
        "n_detection_val_per_class": int(len(det_val0)),
        "image_max_pixels": cfg_get(config, "data.image_max_pixels"),
        "layers": layers,
        "per_layer": per_layer,
        "length_only_accuracy": round(length_only_accuracy(augmented, ratio), 6),
    }
    for key in ["edit_internal", "edit_external", "detection_internal", "detection_external"]:
        values = [v for v in per_layer[key] if v is not None]
        if values:
            best = int(np.argmax(values))
            report[f"best_{key}"] = {"layer": layers[best], "acc": values[best]}
    if args.behavior_n > 0:
        report["behavior_check"] = behavior_check(config, paths, args.behavior_n)

    out_dir = paths.result_dir / "e1"
    save_json(report, out_dir / "e1_report.json")
    plot(report, out_dir / "e1_probe_accuracy.png")
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_layer", "behavior_check"}}, indent=2))
    if "behavior_check" in report:
        print(json.dumps({k: v for k, v in report["behavior_check"].items() if k != "samples"}, indent=2))
    print(out_dir)


if __name__ == "__main__":
    main()
