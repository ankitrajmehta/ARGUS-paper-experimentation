"""E1-S report: does ARGUS steering work on VTQA with this model?

Reads the ARGUS outputs of one run_e1s.py run and summarises:
  preliminary   fixed-strength steering per layer, attack vs defense direction (AIA gap), strength sensitivity,
                multi-layer search, chosen layers/strength (ARGUS selection rule)
  vector        trained step-0 vector: layers, strength, mixture weights, early-stop history
  eval          test split, three modes: no defense / steering only (post-filter off) / full ARGUS
                UIA_clean, UIA_inject, AIA, AIFR (paper metrics) with 95% bootstrap CIs over items,
                detector firing rate on clean and injected inputs, post-filter "Sorry!" rates, and a paired view:
                of the items the attack wins undefended, how many does each mode fix
Metrics (ARGUS eval_metrics.py): UIA = user's answer appears in the response; AIA = attacker's answer appears;
AIFR = response follows the attacker's task (label or long common substring).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
from argus.config import get_paths, load_config  # noqa: E402
from argus.io_utils import load_json, save_json  # noqa: E402

MODES = {"no_defense": "image_no_defense.json", "steer_only": "image_argus_steer_only.json", "argus_full": "image_argus.json"}


def ci(values: list[bool], reps: int = 2000, seed: int = 0) -> list[float]:
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = get_paths(config)
    res = paths.result_dir
    report: dict = {}

    pre = load_json(res / "preliminary_experiments.json")
    report["preliminary"] = {k: pre[k] for k in ["no_steering", "fixed_alpha", "layer_range", "aia_gaps", "ranked_layers", "best_single_layer",
                                                 "best_layers", "best_strength", "best_strength_metrics"]}
    report["preliminary"]["fixed_layer"] = [{k: r[k] for k in ["layer", "direction", "UIA", "AIA", "AIFR"]} for r in pre["fixed_layer_results"]]
    report["preliminary"]["sensitivity"] = [{k: r[k] for k in ["alpha", "UIA", "AIA", "AIFR"]} for r in pre["single_sensitivity_results"]]
    report["preliminary"]["multi"] = [{k: r[k] for k in ["top_n", "layers", "alpha", "UIA", "AIA", "AIFR"]} for r in pre["multi_layer_results"]]

    state = torch.load(paths.vector_dir / "activation_editor.pth", map_location="cpu")
    early = load_json(paths.vector_dir / "vector_early_stop.json")
    report["vector"] = {
        "layers": state["layers"], "strength": state["strength"],
        "mixture_weights": {str(l): [round(float(w), 4) for w in torch.softmax(state[f"editors.{i}.a"].float(), 0)] for i, l in enumerate(state["layers"])},
        "early_stop_samples": early["sample_count"], "best_objective_UIA_minus_AIA": early["best_objective"],
        "history": [{k: h[k] for k in ["step", "objective", "UIA", "AIA", "AIFR", "improved"] if k in h} for h in early["history"]],
    }
    report["epsilon"] = load_json(paths.epsilon_dir / "epsilon_image.json")

    evals = {m: load_json(res / f) for m, f in MODES.items() if (res / f).exists()}
    report["eval"] = {}
    for mode, data in evals.items():
        rows = data["results"]
        report["eval"][mode] = {
            "n": len(rows),
            "UIA_clean": {"mean": data["UIA_clean"], "ci95": ci([r["clean_first_success"] for r in rows])},
            "UIA_inject": {"mean": data["UIA_inject"], "ci95": ci([r["first_success"] for r in rows])},
            "AIA": {"mean": data["AIA"], "ci95": ci([r["second_success"] for r in rows])},
            "AIFR": {"mean": data["AIFR"], "ci95": ci([r["attacker_followed"] for r in rows])},
            "detector_fires_inject": round(float(np.mean([r["attack_detected"] for r in rows])), 4),
            "detector_fires_clean": round(float(np.mean([r["clean_attack_detected"] for r in rows])), 4),
            "post_filter_sorry_inject": round(float(np.mean([r["post_filter_rejected"] for r in rows])), 4),
            "post_filter_sorry_clean": round(float(np.mean([r["clean_post_filter_rejected"] for r in rows])), 4),
        }
    if "no_defense" in evals:
        base = {r["id"]: r for r in evals["no_defense"]["results"]}
        won = [i for i, r in base.items() if r["second_success"] or r["attacker_followed"]]
        report["paired"] = {"attack_wins_undefended": len(won)}
        for mode, data in evals.items():
            if mode == "no_defense":
                continue
            by = {r["id"]: r for r in data["results"]}
            report["paired"][mode] = {
                "attack_blocked": sum(not (by[i]["second_success"] or by[i]["attacker_followed"]) for i in won),
                "blocked_and_user_answer": sum(not (by[i]["second_success"] or by[i]["attacker_followed"]) and by[i]["first_success"] for i in won),
                "user_answer_lost_on_clean": sum(base[i]["clean_first_success"] and not by[i]["clean_first_success"] for i in by),
                "user_answer_gained_on_inject": sum(not base[i]["first_success"] and by[i]["first_success"] for i in by),
            }
        report["examples"] = [{"id": i, "no_defense": base[i]["inject_response"][:200],
                               **{m: {r["id"]: r for r in d["results"]}[i]["inject_response"][:200] for m, d in evals.items() if m != "no_defense"}}
                              for i in won[:8]]

    print("preliminary: no steering", pre["no_steering"], "| best", pre["best_layers"], pre["best_strength"], pre["best_strength_metrics"])
    print("AIA gap (attack - defense direction) per layer:", {g["layer"]: g["aia_gap"] for g in pre["aia_gaps"]})
    for mode, r in report["eval"].items():
        print(f"{mode:11s} n={r['n']} UIA_clean {r['UIA_clean']['mean']:.3f} UIA_inject {r['UIA_inject']['mean']:.3f} "
              f"AIA {r['AIA']['mean']:.3f} {r['AIA']['ci95']} AIFR {r['AIFR']['mean']:.3f} det(inj/clean) "
              f"{r['detector_fires_inject']:.2f}/{r['detector_fires_clean']:.2f} sorry(inj/clean) {r['post_filter_sorry_inject']:.2f}/{r['post_filter_sorry_clean']:.2f}")
    print("paired:", report.get("paired"))
    out = res / "e1s"
    out.mkdir(parents=True, exist_ok=True)
    save_json(report, out / "e1s_report.json")
    plot(report, out / "e1s_steering.png")


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    fixed = report["preliminary"]["fixed_layer"]
    for direction, style in [("attack", "r-o"), ("defense", "g-o")]:
        rows = [r for r in fixed if r["direction"] == direction]
        axes[0].plot([r["layer"] for r in rows], [r["AIA"] for r in rows], style, label=f"{direction} direction AIA")
        axes[0].plot([r["layer"] for r in rows], [r["UIA"] for r in rows], style.replace("-", "--"), alpha=0.6, label=f"{direction} direction UIA")
    axes[0].axhline(report["preliminary"]["no_steering"]["AIA"], color="k", ls=":", label="no steering AIA")
    axes[0].set_title(f"Fixed steering (alpha={report['preliminary']['fixed_alpha']}) per layer, val")
    axes[0].set_xlabel("Layer")
    axes[0].legend(fontsize=7)
    sens = report["preliminary"]["sensitivity"]
    axes[1].plot([r["alpha"] for r in sens], [r["AIA"] for r in sens], "g-o", label="AIA")
    axes[1].plot([r["alpha"] for r in sens], [r["UIA"] for r in sens], "b-o", label="UIA")
    axes[1].set_title(f"Strength sensitivity at L{report['preliminary']['best_single_layer']} (defense), val")
    axes[1].set_xlabel("alpha")
    axes[1].legend(fontsize=7)
    modes = list(report["eval"])
    keys = ["UIA_clean", "UIA_inject", "AIA", "AIFR"]
    width = 0.8 / len(keys)
    for k, key in enumerate(keys):
        means = [report["eval"][m][key]["mean"] for m in modes]
        lo = [m - report["eval"][mo][key]["ci95"][0] for m, mo in zip(means, modes)]
        hi = [report["eval"][mo][key]["ci95"][1] - m for m, mo in zip(means, modes)]
        axes[2].bar(np.arange(len(modes)) + (k - 1.5) * width, means, width, yerr=[lo, hi], capsize=2, label=key)
    axes[2].set_xticks(np.arange(len(modes)))
    axes[2].set_xticklabels(modes)
    axes[2].set_title(f"Test (n={report['eval'][modes[0]]['n']}): no defense vs ARGUS")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)
    fig.suptitle("E1-S: ARGUS steering on VTQA, Qwen2.5-VL-3B")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
