"""E2b analysis: flag rate of each sign variant under the E2 probes.

Probes are trained on E2 train exactly as in analyze_e2.py. The layer reported per probe is picked
with E2's balanced score (E2 test only), so E2b variants never influence the choice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2"))
from analyze_e2 import VARIANTS as E2_VARIANTS  # noqa: E402
from analyze_e2 import balanced_score, load_variants, train_probe  # noqa: E402

from argus.io_utils import save_json  # noqa: E402
from argus.probes import load_pickle  # noqa: E402
from build_e2b import VARIANTS as E2B_VARIANTS  # noqa: E402

SHOULD_FLAG = {"injected_sign": True, "injected_strip": True, "attack_plain": True, "attack_no_trigger": True,
               "clean": False, "benign_sign": False, "legit_imperative": False}


def load_e2b(path: Path, ids: list[int]) -> dict[str, np.ndarray]:
    rows = {row["id"]: row for row in torch.load(path, map_location="cpu")}
    return {v: torch.stack([rows[i][v] for i in ids]).float().numpy() for v in E2B_VARIANTS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--e1-dir", help="E1 output dir; adds the transfer_e1 probe.")
    args = parser.parse_args()

    e2_dir, out_dir = Path(args.e2_dir), Path(args.output_dir)
    train = load_variants(e2_dir / "activations" / "e2_train.pt")
    test_rows = torch.load(e2_dir / "activations" / "e2_test.pt", map_location="cpu")
    test = {v: torch.stack([row[v] for row in test_rows]).float().numpy() for v in E2_VARIANTS}
    test.update(load_e2b(out_dir / "activations" / "e2b_test.pt", [row["id"] for row in test_rows]))
    layer_count = train["clean"].shape[1]

    predictors = {
        "indomain_clean": lambda l: train_probe(train, l, ["clean"]).predict,
        "indomain_hardneg": lambda l: train_probe(train, l, ["clean", "benign_sign"]).predict,
    }
    if args.e1_dir:
        det_dir = Path(args.e1_dir) / "probes" / "detection"
        predictors["transfer_e1"] = lambda l: load_pickle(det_dir / f"mllm_layer{l}.pickle").predict

    probes = {}
    for name, make in predictors.items():
        per_layer = []
        for layer in range(layer_count):
            predict = make(layer)
            rates = {v: round(float(np.mean(predict(x[:, layer]))), 4) for v, x in test.items()}
            e2_rates = {"TPR_sign": rates["injected_sign"], "FPR_benign": rates["benign_sign"], "FPR_clean": rates["clean"]}
            per_layer.append({"layer": layer, "flag_rate": rates, "e2_balanced_score": balanced_score(e2_rates)})
        best = max(per_layer, key=lambda row: row["e2_balanced_score"])
        probes[name] = {"per_layer": per_layer, "e2_best_layer": best["layer"], "at_e2_best_layer": best["flag_rate"]}
        print(f"{name} @ layer {best['layer']} (picked on E2 only):")
        for variant, rate in best["flag_rate"].items():
            print(f"  {variant:18s} flagged {rate:5.1%}   ({'should flag' if SHOULD_FLAG[variant] else 'should pass'})")

    report = {"n_train": int(len(train["clean"])), "n_test": int(len(test["clean"])), "should_flag": SHOULD_FLAG, "probes": probes}
    save_json(report, out_dir / "results" / "e2b_report.json")
    plot(report, out_dir / "results" / "e2b_flag_rates.png")


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["probes"])
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        per_layer = report["probes"][name]["per_layer"]
        layers = [row["layer"] for row in per_layer]
        for variant, flag in report["should_flag"].items():
            ax.plot(layers, [row["flag_rate"][variant] for row in per_layer], "-" if flag else "--", marker="o", markersize=2, label=variant)
        ax.axvline(report["probes"][name]["e2_best_layer"], color="grey", linewidth=0.8)
        ax.set_title(name)
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Fraction flagged (solid: should flag, dashed: should pass)")
    axes[0][-1].legend(fontsize=7, loc="center right")
    fig.suptitle(f"E2b: detector vs. attack wording and legitimate imperative signs ({report['n_test']} test frames)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
