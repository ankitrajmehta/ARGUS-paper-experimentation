"""Fit the steering direction, safety subspace and injection gate.

    midas-fit-defense results/activations/*.npz --out results/defense/train_2b

Layers are chosen by held-out-scene probe accuracy rather than fixed in advance,
since the best layer differs by model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from midas import probes
from midas.steering import DefenseArtifact, LayerDefense


def _fit_probe(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, random_state=0).fit(scaler.transform(X), y)
    return clf, scaler


def _ranked_layers(acts: probes.ActivationSet, label_key: str, group_key: str = "scene"):
    scores = probes.sweep_layers(acts, label_key=label_key, group_key=group_key)
    valid = [s for s in scores if not np.isnan(s.accuracy)]
    return sorted(valid, key=lambda s: -s.accuracy)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("activations", type=Path, nargs="+")
    parser.add_argument(
        "--gate-activations",
        type=Path,
        nargs="+",
        default=None,
        help="separate sweep for the gate, normally --prompt-only. The gate must be "
        "fitted on the open-turn state it sees at inference; reusing answer-token "
        "activations flatters it with a signal that does not exist yet.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/defense/artifact"))
    parser.add_argument("--subspace-dim", type=int, default=5)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None, help="override steering layers"
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="fit this many layers, best held-out probe accuracy first. ARGUS steer "
        "several layers at once; steering one left behaviour unchanged here.",
    )
    parser.add_argument("--gate-layer", type=int, default=None)
    args = parser.parse_args(argv)

    acts = probes.merge([probes.load_activations(p) for p in args.activations])
    cond = acts.column("condition")
    counts = {c: int((cond == c).sum()) for c in np.unique(cond)}
    print(f"{len(acts.rows)} vectors: {counts}")

    # --- steering directions: on attacked frames, which instruction is followed?
    print("\nsteering probes (attacked frames only):")
    attacked = acts.subset(cond == "attack")

    if args.layers:
        chosen = [(layer, float("nan")) for layer in args.layers]
    else:
        ranked = _ranked_layers(attacked, "follows_injection")
        chosen = [(s.layer, s.accuracy) for s in ranked[: args.top_n]]

    layer_defenses = []
    for layer, accuracy in chosen:
        X = attacked.vectors[:, layer, :]
        y = attacked.column("follows_injection").astype(int)
        clf, scaler = _fit_probe(X, y)
        probe_norm = float(np.linalg.norm(clf.coef_[0]))
        direction = clf.coef_[0] / probe_norm

        # Distance of the "following the scene" class from the boundary, in the
        # standardised space the probe lives in; adaptive steering aims for it.
        z = scaler.transform(X)
        margin = float(np.abs(z[y == 0] @ direction).mean())

        subspace = probes.safety_subspace(attacked, layer, dim=args.subspace_dim)
        energy = float(np.linalg.norm(subspace @ direction))
        print(f"    layer {layer:2d}  probe {accuracy:6.1%}  margin {margin:5.2f}  "
              f"subspace energy {energy:5.1%}  ||w|| {probe_norm:5.1f}")

        layer_defenses.append(
            LayerDefense(
                layer=layer,
                direction=direction,
                subspace=subspace,
                mean=scaler.mean_,
                scale=scaler.scale_,
                margin=margin,
                accuracy=accuracy,
                probe_norm=probe_norm,
                bias=float(clf.intercept_[0]),
            )
        )

    # --- gate: an injected instruction, or just ordinary signage?
    print("\ninjection gate (attack vs clean + benign text):")
    if args.gate_activations:
        gate_source = probes.merge(
            [probes.load_activations(p) for p in args.gate_activations]
        )
        gcond = gate_source.column("condition")
        print(f"    using {len(gate_source.rows)} prompt-only vectors: "
              f"{ {c: int((gcond == c).sum()) for c in np.unique(gcond)} }")
    else:
        gate_source = acts
        gcond = cond
        print("    WARNING: fitting the gate on answer-token activations; at inference "
              "no answer exists yet. Pass --gate-activations from a --prompt-only sweep.")

    gate_set = gate_source.subset(np.isin(gcond, ["attack", "benign_text", "clean"]))
    gate_y = (gate_set.column("condition") == "attack").astype(int)
    for row, label in zip(gate_set.rows, gate_y):
        row["is_attack"] = int(label)

    if (gate_y == 0).sum() == 0:
        raise SystemExit(
            "no clean or benign_text vectors: the gate cannot be fitted, and without it "
            "the defense would steer every frame. Re-run extraction with --controls."
        )

    if args.gate_layer:
        gate_layer = args.gate_layer
    else:
        gate_ranked = _ranked_layers(gate_set, "is_attack", "scene+phrase")
        gate_layer = gate_ranked[0].layer
        print(f"    best gate layer {gate_layer} at {gate_ranked[0].accuracy:.1%} "
              f"(held-out scene+phrase)")
    gate_X = gate_set.vectors[:, gate_layer, :]
    gate_clf, gate_scaler = _fit_probe(gate_X, gate_y)

    artifact = DefenseArtifact(
        layers=layer_defenses,
        gate_layer=gate_layer,
        gate_w=gate_clf.coef_[0],
        gate_b=float(gate_clf.intercept_[0]),
        gate_mean=gate_scaler.mean_,
        gate_scale=gate_scaler.scale_,
        model=acts.model,
        notes=f"fitted on {len(acts.rows)} vectors: {counts}",
    )
    path = artifact.save(args.out)
    print(f"\nsaved -> {path}\n         {path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
