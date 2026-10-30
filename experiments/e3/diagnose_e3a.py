"""E3a post-hoc diagnostic (written after the v1 verdict): behavior-probe flags per attacker action.

v1 failed only the natural-AUC check (0.68). Train attackers never use a navigation action, test has `turn_left`;
this prints P and natural flag rates per A_x to test whether the misses are the unseen action type.
Does not change the pre-registered verdict.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e3a import Rows, auc  # noqa: E402
from common import load_frames  # noqa: E402

from argus.io_utils import save_json  # noqa: E402
from argus.probes import load_pickle  # noqa: E402

POS = {"first": 0, "last": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-dir", required=True)
    parser.add_argument("--e2b-dir", required=True)
    parser.add_argument("--e2c-dir", required=True)
    parser.add_argument("--e3a-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dirs = {"e2": Path(args.e2_dir), "e2b": Path(args.e2b_dir), "e2c": Path(args.e2c_dir)}
    frames = load_frames(dirs, "test")
    e3a = Path(args.e3a_dir)
    test, nat = (Rows(e3a / "activations" / f"e3a_{s}.pt") for s in ["test", "natural"])
    report = {}
    for name in ["behavior_controlled", "behavior_naive"]:
        p = load_pickle(e3a / "probes" / f"{name}.pickle")
        score = lambda rows: rows.states[:, POS[p["position"]], p["layer"]].float().numpy() @ p["w"] + p["b"]
        st, sn = score(test), score(nat)
        by_ax = defaultdict(lambda: defaultdict(list))
        for m, s in zip(test.meta, st):
            if m["group"] in ("P", "N2", "N3"):
                by_ax[m["prefix"]][m["group"]].append(s > 0)
        nat_ax = np.array([frames[m["id"]]["actions"][m["variant"]] for m in nat.meta])
        effects = np.array([m["effect"] for m in nat.meta])
        per_action = {}
        for a in sorted(set(nat_ax)):
            mask = nat_ax == a
            per_action[a] = {
                **{f"{g}_flag": round(float(np.mean(v)), 4) for g, v in by_ax[a].items()},
                "natural_flagged_by_effect": {e: f"{int((sn[mask & (effects == e)] > 0).sum())}/{int((mask & (effects == e)).sum())}"
                                              for e in ["hijacked", "stopped", "changed", "no_effect"]},
            }
        nav = nat_ax == "turn_left"
        report[name] = {
            "layer": p["layer"], "position": p["position"], "per_attacker_action": per_action,
            "natural_auc_hijacked_without_turn_left": auc(sn[~nav], (effects[~nav] == "hijacked").astype(int)),
            "natural_auc_hijacked_turn_left_only": auc(sn[nav], (effects[nav] == "hijacked").astype(int)),
        }
        print(name, {k: v for k, v in report[name].items() if k != "per_attacker_action"})
        for a, r in per_action.items():
            print(f"  {a:24s} {r}")
    out = Path(args.output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    save_json(report, out / "e3a_diagnose.json")


if __name__ == "__main__":
    main()
