"""E2 analysis: does injection detection survive embodied scenes and benign scene text?

Per layer, reports the rate each test variant is flagged as "injected":
  TPR_sign / TPR_strip  - injected variants caught (higher is better)
  FPR_benign / FPR_clean - harmless variants wrongly flagged (lower is better)
Probes compared:
  transfer_e1      - E1 detection probe (VTQA, text strip) applied as-is
  indomain_clean   - trained on embodied train: clean vs injected_sign
  indomain_hardneg - same, plus benign_sign as negatives
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from argus.io_utils import save_json
from argus.probes import load_pickle

VARIANTS = ["clean", "injected_strip", "injected_sign", "benign_sign"]
RATE_KEYS = {"TPR_sign": "injected_sign", "TPR_strip": "injected_strip", "FPR_benign": "benign_sign", "FPR_clean": "clean"}


def load_variants(path: Path) -> dict[str, np.ndarray]:
    rows = torch.load(path, map_location="cpu")
    return {v: torch.stack([row[v] for row in rows]).float().numpy() for v in VARIANTS}


def flag_rates(predict, test: dict[str, np.ndarray], layer: int) -> dict[str, float]:
    return {key: round(float(np.mean(predict(test[v][:, layer]))), 4) for key, v in RATE_KEYS.items()}


def balanced_score(rates: dict[str, float]) -> float:
    # Equal weight on catching in-scene injections and not flagging benign text or clean frames.
    return round((rates["TPR_sign"] + (1 - rates["FPR_benign"]) + (1 - rates["FPR_clean"])) / 3, 4)


def train_probe(train: dict[str, np.ndarray], layer: int, negatives: list[str]) -> LogisticRegression:
    x = np.concatenate([train[v][:, layer] for v in negatives] + [train["injected_sign"][:, layer]])
    y = np.asarray([0] * sum(len(train[v]) for v in negatives) + [1] * len(train["injected_sign"]))
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(x, y)


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probes = [p for p in ["transfer_e1", "indomain_clean", "indomain_hardneg"] if p in report["probes"]]
    fig, axes = plt.subplots(1, len(probes), figsize=(5 * len(probes), 4.2), sharey=True, squeeze=False)
    styles = {"TPR_sign": ("tab:red", "-"), "TPR_strip": ("tab:orange", "--"), "FPR_benign": ("tab:blue", "-"), "FPR_clean": ("tab:green", "--")}
    for ax, name in zip(axes[0], probes):
        per_layer = report["probes"][name]["per_layer"]
        layers = [row["layer"] for row in per_layer]
        for key, (color, style) in styles.items():
            ax.plot(layers, [row[key] for row in per_layer], style, color=color, marker="o", markersize=3, label=key)
        ax.set_title(name)
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Fraction flagged as injected")
    axes[0][-1].legend(fontsize=8, loc="center right")
    fig.suptitle(f"E2 injection detection on AI2-THOR frames (test: {report['n_test']} frames, unseen scenes/commands)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--e1-dir", help="E1 output dir; enables the transfer_e1 probe.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    train = load_variants(out_dir / "activations" / "e2_train.pt")
    test = load_variants(out_dir / "activations" / "e2_test.pt")
    layer_count = train["clean"].shape[1]

    probes: dict[str, dict] = {}
    if args.e1_dir:
        det_dir = Path(args.e1_dir) / "probes" / "detection"
        probes["transfer_e1"] = {"per_layer": [
            {"layer": l, **flag_rates(load_pickle(det_dir / f"mllm_layer{l}.pickle").predict, test, l)} for l in range(layer_count)
        ]}
    for name, negatives in [("indomain_clean", ["clean"]), ("indomain_hardneg", ["clean", "benign_sign"])]:
        probes[name] = {"negatives": negatives, "per_layer": [
            {"layer": l, **flag_rates(train_probe(train, l, negatives).predict, test, l)} for l in range(layer_count)
        ]}
    for probe in probes.values():
        for row in probe["per_layer"]:
            row["balanced_score"] = balanced_score(row)
        probe["best_layer"] = max(probe["per_layer"], key=lambda row: row["balanced_score"])

    report = {"n_train": int(len(train["clean"])), "n_test": int(len(test["clean"])), "probes": probes}
    result_dir = out_dir / "results"
    save_json(report, result_dir / "e2_report.json")
    plot(report, result_dir / "e2_detection.png")
    for name, probe in probes.items():
        print(name, "best:", probe["best_layer"])
    print(result_dir)


if __name__ == "__main__":
    main()
