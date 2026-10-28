"""Linear probes over the model's activations.

The question a probe answers is ARGUS's: in the model's internal state, is
"about to follow the sign" linearly separable from "about to follow the scene"?
If it is, that direction is what the defense steers along.

The validation split matters more than the probe. With 15 scenes and a handful
of sign phrases, a probe evaluated on random splits can score near-perfectly by
memorising a scene's background or a phrase's wording. Held-out *groups* --
unseen scenes, or unseen wordings -- are the only honest estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class ActivationSet:
    """Activations plus the metadata needed to group them."""

    vectors: np.ndarray  # (samples, layers, hidden)
    rows: list[dict]
    model: str
    split: str
    readout: str

    @property
    def n_layers(self) -> int:
        return self.vectors.shape[1]

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.rows])

    def subset(self, mask: np.ndarray) -> "ActivationSet":
        rows = [r for r, keep in zip(self.rows, mask) if keep]
        return ActivationSet(self.vectors[mask], rows, self.model, self.split, self.readout)


def load_activations(npz_path: Path) -> ActivationSet:
    meta_path = npz_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    vectors = np.load(npz_path)["activations"]
    return ActivationSet(
        vectors=vectors,
        rows=meta["rows"],
        model=meta["model"],
        split=meta["split"],
        readout=meta["readout"],
    )


@dataclass
class LayerScore:
    layer: int
    accuracy: float
    std: float
    n: int

    def bar(self, width: int = 28) -> str:
        filled = int(round(self.accuracy * width))
        return "#" * filled + "." * (width - filled)


def probe_layer(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Grouped cross-validated accuracy of a logistic probe on one layer.

    Groups never straddle the train/test boundary, so the score reflects
    generalisation to unseen scenes (or unseen wordings), not recall.
    """
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return float("nan"), float("nan")
    splits = min(n_splits, n_groups)

    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=splits).split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        scores.append(clf.score(scaler.transform(X[test_idx]), y[test_idx]))

    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def sweep_layers(
    acts: ActivationSet,
    label_key: str = "follows_injection",
    group_key: str = "scene",
    n_splits: int = 5,
) -> list[LayerScore]:
    """Probe accuracy at every layer."""
    y = acts.column(label_key).astype(int)
    groups = acts.column(group_key)
    scores = []
    for layer in range(acts.n_layers):
        acc, std = probe_layer(acts.vectors[:, layer, :], y, groups, n_splits=n_splits)
        scores.append(LayerScore(layer, acc, std, len(y)))
    return scores


def fit_direction(
    acts: ActivationSet,
    layer: int,
    label_key: str = "follows_injection",
) -> np.ndarray:
    """Unit-norm probe weight at one layer.

    This is the candidate steering direction: it points from "following the
    scene" toward "following the sign", so the defense moves against it.
    """
    X = acts.vectors[:, layer, :]
    y = acts.column(label_key).astype(int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, random_state=0).fit(scaler.transform(X), y)
    w = clf.coef_[0]
    return w / np.linalg.norm(w)


def safety_subspace(
    acts: ActivationSet,
    layer: int,
    dim: int = 5,
    label_key: str = "follows_injection",
) -> np.ndarray:
    """SafeSteer-style subspace: top singular directions of the paired difference.

    Each attacked image contributes one difference vector (target answer minus
    benign answer), so the SVD sees only the follow-the-sign axis, with scene
    content differenced away.

    Returns ``(dim, hidden)`` orthonormal rows.
    """
    y = acts.column(label_key).astype(int)
    scenes = acts.column("scene")
    phrases = acts.column("phrase")

    diffs = []
    for s in np.unique(scenes):
        for p in np.unique(phrases[scenes == s]):
            sel = (scenes == s) & (phrases == p)
            pos = acts.vectors[sel & (y == 1), layer, :]
            neg = acts.vectors[sel & (y == 0), layer, :]
            if len(pos) and len(neg):
                diffs.append(pos.mean(0) - neg.mean(0))

    if not diffs:
        raise ValueError("no paired samples; need both answers for the same scene and phrase")

    A = np.stack(diffs)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:dim]
