"""E3a analysis: is there a robot *behavior* direction, or does the probe just read the action string?

Probes (logistic regression on one layer's state at the first or last answer token):
  naive       P vs N1                 (ARGUS-style: same attacked image, injected vs user answer)
  controlled  P vs N1 + N2 + N3       (adds clean-frame and harmless-sign negatives with the same A_x)
Layer and read position are chosen on 2 held-out *train* scenes (as in E2c), then refit on all train scenes.
Key control: N2 (clean frame forced to say A_x). A probe that flags N2 reads the action, not obedience.
Also saves the probes E3b steers along (with ARGUS epsilon) and the E2c detection flags used for gating.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e2c"))
from analyze_e2c import PROBES as DETECT_PROBES  # noqa: E402
from analyze_e2c import fit as fit_detect  # noqa: E402
from analyze_e2c import load_variants, stack  # noqa: E402
from common import load_frames, val_scenes  # noqa: E402

from argus.io_utils import load_json, save_json  # noqa: E402
from argus.probes import load_pickle, save_pickle  # noqa: E402

SPECS = {"naive": ["P", "N1"], "controlled": ["P", "N1", "N2", "N3"]}
POSITIONS = ["first", "last"]
# Pre-registered thresholds for calling the controlled probe a real behavior signal.
VERDICT = {"p_vs_n1_acc_min": 0.90, "n2_flag_max": 0.10, "n3_flag_max": 0.10, "shuffled_max": 0.60, "natural_auc_min": 0.80}


class Rows:
    def __init__(self, path: Path):
        data = torch.load(path, map_location="cpu")
        self.meta, self.states = data["meta"], data["states"]
        self.group = np.array([m["group"] for m in self.meta])
        self.variant = np.array([m["variant"] for m in self.meta])
        self.label = np.array([m["label"] for m in self.meta])
        self.scene = np.array([m["scene"] for m in self.meta])
        self.length = np.array([m["n_prefix_tokens"] for m in self.meta], dtype=float)

    def x(self, mask: np.ndarray, pos: int, layer: int) -> np.ndarray:
        return self.states[torch.from_numpy(np.where(mask)[0]), pos, layer].float().numpy()


def fit(rows: Rows, mask: np.ndarray, pos: int, layer: int, groups: list[str], shuffle_seed: int | None = None) -> LogisticRegression:
    m = mask & np.isin(rows.group, groups)
    y = rows.label[m]
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(rows.x(m, pos, layer), y)


def flag_rate(probe, rows: Rows, mask: np.ndarray, pos: int, layer: int) -> float | None:
    return float(probe.predict(rows.x(mask, pos, layer)).mean()) if mask.any() else None


def mean_correct(flags: dict[str, float | None]) -> float:
    # Each group counts equally: P should be flagged, every negative group should pass.
    return float(np.mean([f if g == "P" else 1 - f for g, f in flags.items() if f is not None]))


def group_flags(probe, rows: Rows, mask: np.ndarray, pos: int, layer: int, groups: list[str]) -> dict[str, float | None]:
    return {g: flag_rate(probe, rows, mask & (rows.group == g), pos, layer) for g in groups}


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    return round(float(roc_auc_score(labels, scores)), 4) if 0 < labels.sum() < len(labels) else None


def evaluate(probe, test: Rows, nat: Rows, pos: int, layer: int) -> dict:
    allm = np.ones(len(test.meta), bool)
    flags = group_flags(probe, test, allm, pos, layer, ["P", "N1", "N2", "N3"])
    pn = np.isin(test.group, ["P", "N1"])
    per_variant = {}
    for v in sorted(set(test.variant)):
        vm = test.variant == v
        per_variant[v] = {g: (None if f is None else round(f, 4)) for g, f in group_flags(probe, test, vm, pos, layer, ["P", "N1", "N2", "N3"]).items()}
    nat_scores = probe.decision_function(nat.x(np.ones(len(nat.meta), bool), pos, layer))
    effects = np.array([m["effect"] for m in nat.meta])
    nat_flags = nat_scores > 0
    return {
        "flag_rate": {g: (None if f is None else round(f, 4)) for g, f in flags.items()},
        "p_vs_n1_acc": round((flags["P"] + 1 - flags["N1"]) / 2, 4),
        "p_vs_n1_auc": auc(probe.decision_function(test.x(pn, pos, layer)), test.label[pn]),
        "per_variant": per_variant,
        "natural": {
            "auc_hijacked_vs_rest": auc(nat_scores, (effects == "hijacked").astype(int)),
            "auc_stopped_vs_rest": auc(nat_scores, (effects == "stopped").astype(int)),
            "flag_rate_by_effect": {e: f"{int(nat_flags[effects == e].sum())}/{int((effects == e).sum())}" for e in ["hijacked", "stopped", "changed", "no_effect"]},
        },
    }


def epsilon(probe, x: np.ndarray, y: np.ndarray) -> float:
    # ARGUS (estimate_epsilon.py): mean |score| of correctly classified user-following (label 0) samples.
    scores = x @ probe.coef_.squeeze() + probe.intercept_[0]
    neg = scores[(y == 0) & (scores < 0)]
    return float(-neg.mean())


def steering_probe(probe, x: np.ndarray, y: np.ndarray, layer: int, pos: str | None, name: str) -> dict:
    w = probe.coef_.squeeze().astype(np.float32)
    return {"name": name, "w": w, "b": float(probe.intercept_[0]), "eps": epsilon(probe, x, y), "layer": layer, "position": pos,
            "norm": float(np.linalg.norm(w))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-dir", required=True)
    parser.add_argument("--e2b-dir", required=True)
    parser.add_argument("--e2c-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--e1-dir", help="E1 output dir; adds cosine with the VTQA behavior (edit) probe.")
    parser.add_argument("--smoke", action="store_true", help="Tolerate missing E2c test activations (laptop smoke run).")
    args = parser.parse_args()

    dirs = {"e2": Path(args.e2_dir), "e2b": Path(args.e2b_dir), "e2c": Path(args.e2c_dir)}
    out_dir = Path(args.output_dir)
    train, test, nat = (Rows(out_dir / "activations" / f"e3a_{s}.pt") for s in ["train", "test", "natural"])
    layers = train.states.shape[2]
    held = val_scenes(load_frames(dirs, "train"))
    val = np.isin(train.scene, held)
    fit_mask, all_train = ~val, np.ones(len(train.meta), bool)
    print(f"layer/position selection on held-out train scenes {held} ({int(val.sum())} rows)")

    report = {"val_scenes": held, "n_rows": {"train": len(train.meta), "test": len(test.meta), "natural": len(nat.meta)},
              "verdict_thresholds": VERDICT, "probes": {}}
    chosen, saved = {}, {}
    for name, groups in SPECS.items():
        val_scores = {p: [] for p in POSITIONS}
        for pi, pos in enumerate(POSITIONS):
            for layer in range(layers):
                held_probe = fit(train, fit_mask, pi, layer, groups)
                val_scores[pos].append(round(mean_correct(group_flags(held_probe, train, val, pi, layer, groups)), 4))
        pos = max(POSITIONS, key=lambda p: max(val_scores[p]))
        pi, layer = POSITIONS.index(pos), int(np.argmax(val_scores[pos]))
        chosen[name] = (pi, layer)
        per_layer = []
        for l in range(layers):
            probe = fit(train, all_train, pi, l, groups)
            ev = evaluate(probe, test, nat, pi, l)
            per_layer.append({"layer": l, "p_vs_n1_acc": ev["p_vs_n1_acc"], "flag_rate": ev["flag_rate"],
                              "natural_auc_hijacked": ev["natural"]["auc_hijacked_vs_rest"]})
        probe = fit(train, all_train, pi, layer, groups)
        result = evaluate(probe, test, nat, pi, layer)
        test_flags = group_flags(probe, test, np.ones(len(test.meta), bool), pi, layer, groups)
        shuffled = fit(train, all_train, pi, layer, groups, shuffle_seed=0)
        m = np.isin(train.group, groups)
        length_probe = LogisticRegression(class_weight="balanced").fit(train.length[m, None], train.label[m])
        length_flags = {g: float(length_probe.predict(test.length[test.group == g, None]).mean()) for g in groups if (test.group == g).any()}
        result.update({
            "chosen_layer": layer, "chosen_position": pos, "val_scores": val_scores,
            "test_mean_correct": round(mean_correct(test_flags), 4),
            "shuffled_mean_correct": round(mean_correct(group_flags(shuffled, test, np.ones(len(test.meta), bool), pi, layer, groups)), 4),
            "length_only_mean_correct": round(mean_correct(length_flags), 4), "length_only_flag_rate": length_flags,
            "per_layer": per_layer,
        })
        report["probes"][name] = result
        xm = all_train & np.isin(train.group, groups)
        saved[name] = steering_probe(probe, train.x(xm, pi, layer), train.label[xm], layer, pos, f"behavior_{name}")
        if name == "controlled":
            vm = fit_mask & np.isin(train.group, groups)
            saved[f"{name}_val"] = steering_probe(fit(train, fit_mask, pi, layer, groups), train.x(vm, pi, layer), train.label[vm], layer, pos,
                                                  f"behavior_{name}_val")
        print(f"\n{name}: layer {layer}, {pos} answer token")
        print(f"  test flag rates {result['flag_rate']}  P-vs-N1 acc {result['p_vs_n1_acc']} AUC {result['p_vs_n1_auc']}")
        print(f"  shuffled {result['shuffled_mean_correct']}  length-only {result['length_only_mean_correct']}  natural {result['natural']}")

    c = report["probes"]["controlled"]
    checks = {
        "p_vs_n1_acc": c["p_vs_n1_acc"] >= VERDICT["p_vs_n1_acc_min"],
        "n2_flag": c["flag_rate"]["N2"] <= VERDICT["n2_flag_max"],
        "n3_flag": c["flag_rate"]["N3"] <= VERDICT["n3_flag_max"],
        "shuffled": c["shuffled_mean_correct"] <= VERDICT["shuffled_max"],
        "natural_auc": (c["natural"]["auc_hijacked_vs_rest"] or 0) >= VERDICT["natural_auc_min"],
    }
    report["verdict"] = {"checks": checks, "behavior_probe_real": all(checks.values()),
                         "naive_reads_action_string": report["probes"]["naive"]["flag_rate"]["N2"] >= 0.5}
    print(f"\nverdict: {report['verdict']}")

    # E2c detection probe (varied phrasings + legit negatives), refit from the E2/E2c activations.
    e2_dir, e2c_dir = dirs["e2"], dirs["e2c"]
    train_ids = [r["id"] for r in torch.load(e2_dir / "activations" / "e2_train.pt", map_location="cpu")]
    test_ids = [r["id"] for r in torch.load(e2_dir / "activations" / "e2_test.pt", map_location="cpu")]
    det_train = {**load_variants(e2_dir / "activations" / "e2_train.pt"), **stack(e2c_dir / "activations" / "e2c_train.pt", train_ids)}
    det_test = load_variants(e2_dir / "activations" / "e2_test.pt")
    det_test.update(stack(dirs["e2b"] / "activations" / "e2b_test.pt", test_ids))
    if (e2c_dir / "activations" / "e2c_test.pt").exists():
        det_test.update({f"e2c_{v}": x for v, x in stack(e2c_dir / "activations" / "e2c_test.pt", test_ids).items()})
    elif not args.smoke:
        raise FileNotFoundError(e2c_dir / "activations" / "e2c_test.pt")
    e2c_report = e2c_dir / "results" / "e2c_report.json"
    det_layer = load_json(e2c_report)["probes"]["varied"]["chosen_layer"] if e2c_report.exists() else 21
    spec = DETECT_PROBES["varied"]
    scene_of = {r["id"]: r["scene_name"] for r in load_json(e2_dir / "data" / "e2_train.json")}
    det_val = np.array([scene_of[i] in set(held) for i in train_ids])
    for suffix, mask in [("", np.ones(len(train_ids), bool)), ("_val", ~det_val)]:
        probe = fit_detect(det_train, mask, det_layer, spec)
        x = np.concatenate([det_train[v][mask, det_layer] for v in spec["negatives"] + spec["positives"]])
        y = np.concatenate([np.zeros(mask.sum() * len(spec["negatives"])), np.ones(mask.sum() * len(spec["positives"]))])
        saved[f"detect{suffix}"] = steering_probe(probe, x, y, det_layer, None, f"detect_e2c{suffix}")
        if not suffix:
            flags = {v: [{"id": i, "flag": bool(s > 0), "score": round(float(s), 4)} for i, s in zip(test_ids, probe.decision_function(x_[:, det_layer]))]
                     for v, x_ in det_test.items()}
            save_json({"layer": det_layer, "flags": flags}, out_dir / "results" / "e3a_detect_flags.json")
            report["detect_flag_rate"] = {v: round(float(np.mean([f["flag"] for f in fl])), 4) for v, fl in flags.items()}
            print(f"E2c detection probe L{det_layer} flag rates: {report['detect_flag_rate']}")

    # Directions: behavior vs detection (same layer), vs the E1 VTQA behavior probe.
    cos = {}
    for name in SPECS:
        pi, layer = chosen[name]
        w = saved[name]["w"] / saved[name]["norm"]
        wd = fit_detect(det_train, np.ones(len(train_ids), bool), layer, spec).coef_.squeeze()
        cos[name] = {"layer": layer, "vs_detect_same_layer": round(float(w @ wd / np.linalg.norm(wd)), 4)}
        e1_probe = Path(args.e1_dir) / "probes" / "edit" / f"mllm_layer{layer}.pickle" if args.e1_dir else None
        if e1_probe and e1_probe.exists():
            we = np.asarray(load_pickle(e1_probe).coef_).squeeze()
            cos[name]["vs_e1_vtqa_behavior"] = round(float(w @ we / np.linalg.norm(we)), 4)
    report["cosine"] = cos
    report["steering_probes"] = {k: {kk: vv for kk, vv in v.items() if kk != "w"} for k, v in saved.items()}
    print(f"cosines: {cos}")

    for name, probe in saved.items():
        save_pickle(probe, out_dir / "probes" / f"{probe['name']}.pickle")
    save_json(report, out_dir / "results" / "e3a_report.json")
    plot(report, out_dir / "results" / "e3a_behavior_probe.png")


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, name in zip(axes, ["naive", "controlled"]):
        probe = report["probes"][name]
        rows = probe["per_layer"]
        layers = [r["layer"] for r in rows]
        ax.plot(layers, [r["p_vs_n1_acc"] for r in rows], "-", marker="o", markersize=2, label="P vs N1 accuracy (want high)")
        ax.plot(layers, [r["flag_rate"]["N2"] for r in rows], "--", marker="o", markersize=2, label="N2 clean+A_x flagged (want low)")
        ax.plot(layers, [r["flag_rate"]["N3"] for r in rows], "--", marker="o", markersize=2, label="N3 harmless sign+A_x flagged (want low)")
        ax.plot(layers, [r["natural_auc_hijacked"] or np.nan for r in rows], ":", label="natural AUC hijacked vs rest")
        ax.axvline(probe["chosen_layer"], color="grey", linewidth=0.8)
        ax.set_title(f"{name} ({probe['chosen_position']} answer token)")
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Rate (test houses)")
    axes[1].legend(fontsize=7, loc="center right")
    fig.suptitle("E3a: robot behavior probe vs the action-identity control")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
