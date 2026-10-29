"""Activation steering: the defense itself.

Three pieces, following ARGUS and SafeSteer:

* a **direction** at one or more mid-network layers, pointing from "following
  the scene" toward "following the sign". Steering moves against it.
* a **subspace** (SafeSteer): the top singular directions of the paired
  activation differences. Projecting the raw direction into it drops components
  that happen to correlate in this small sample but carry no safety signal.
* a **gate** (ARGUS): a second probe deciding whether an injected instruction is
  present at all, so clean frames and ordinary signage are left untouched.

Steering several layers at once is the point of the per-layer structure below.
A single-layer intervention left behaviour unchanged even though it moved 92% of
activations across the probe boundary offline, and ARGUS steer the top-N layers
rather than one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LayerDefense:
    """The steering geometry fitted at one layer.

    The probe lives in standardised space, so ``mean``/``scale`` must travel
    with the weights or the decision boundary is meaningless at runtime.
    """

    layer: int
    direction: np.ndarray  # (hidden,) unit norm, toward following the sign
    subspace: np.ndarray  # (dim, hidden) orthonormal rows
    mean: np.ndarray  # (hidden,)
    scale: np.ndarray  # (hidden,)
    # Mean distance of "following the scene" samples from the hyperplane;
    # adaptive steering aims to land there rather than just over the boundary.
    margin: float
    accuracy: float = float("nan")  # held-out probe accuracy, for ranking layers

    @property
    def refined_direction(self) -> np.ndarray:
        """Direction projected into the safety subspace, re-normalised."""
        v = self.subspace.T @ (self.subspace @ self.direction)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else self.direction

    def follow_score(self, activation: np.ndarray) -> float:
        """Signed distance from the probe; > 0 means following the sign."""
        z = (activation - self.mean) / self.scale
        return float(z @ self.direction)


@dataclass
class DefenseArtifact:
    """A fitted defense: steering geometry per layer, plus one gate."""

    layers: list[LayerDefense]
    gate_layer: int
    gate_w: np.ndarray
    gate_b: float
    gate_mean: np.ndarray
    gate_scale: np.ndarray
    model: str = ""
    notes: str = ""

    @property
    def layer_indices(self) -> list[int]:
        return [d.layer for d in self.layers]

    def gate_score(self, activation: np.ndarray) -> float:
        """Signed distance from the gate hyperplane; > 0 means injection."""
        z = (activation - self.gate_mean) / self.gate_scale
        return float(z @ self.gate_w + self.gate_b)

    def save(self, path: Path) -> Path:
        # np.savez_compressed appends .npz, so normalise here or load() cannot
        # find what save() reported writing.
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            direction=np.stack([d.direction for d in self.layers]),
            subspace=np.stack([d.subspace for d in self.layers]),
            mean=np.stack([d.mean for d in self.layers]),
            scale=np.stack([d.scale for d in self.layers]),
            gate_w=self.gate_w,
            gate_mean=self.gate_mean,
            gate_scale=self.gate_scale,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "layers": [
                        {"layer": d.layer, "margin": d.margin, "accuracy": d.accuracy}
                        for d in self.layers
                    ],
                    "gate_layer": self.gate_layer,
                    "gate_b": self.gate_b,
                    "model": self.model,
                    "notes": self.notes,
                    "subspace_dim": int(self.layers[0].subspace.shape[0]),
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "DefenseArtifact":
        # Accept the stem, the .npz or the .json; they name one artifact.
        path = Path(path).with_suffix("")
        arrays = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        layers = [
            LayerDefense(
                layer=entry["layer"],
                direction=arrays["direction"][i],
                subspace=arrays["subspace"][i],
                mean=arrays["mean"][i],
                scale=arrays["scale"][i],
                margin=entry["margin"],
                accuracy=entry.get("accuracy", float("nan")),
            )
            for i, entry in enumerate(meta["layers"])
        ]
        return cls(
            layers=layers,
            gate_layer=meta["gate_layer"],
            gate_w=arrays["gate_w"],
            gate_b=meta["gate_b"],
            gate_mean=arrays["gate_mean"],
            gate_scale=arrays["gate_scale"],
            model=meta.get("model", ""),
            notes=meta.get("notes", ""),
        )


@dataclass
class SteerConfig:
    """How hard to steer, and whether to decide per sample."""

    strength: float = 1.0
    use_subspace: bool = True
    adaptive: bool = True
    use_gate: bool = True
    # Gate threshold on the signed distance. 0 is the fitted boundary; raising it
    # makes the defense more conservative about intervening.
    gate_threshold: float = 0.0
    # Cap on a single step, as a fraction of the activation's own norm. Without
    # it a mis-scaled probe pushes hidden states outside the distribution the
    # layers above were trained on and the model emits word salad.
    max_step_ratio: float = 0.25
    # Steer every position rather than only the last. During generation the last
    # position *is* the current token, so "last" is right there. But a single
    # prefill pass over a complete sequence -- as forced-choice scoring uses --
    # computes the answer tokens' logits from earlier positions, so steering only
    # the final position would change nothing measurable.
    all_positions: bool = False
    # Steer only this many layers, best probe first. None means all fitted layers.
    top_layers: int | None = None


class Steerer:
    """Applies a DefenseArtifact to a live InternVL model via forward hooks."""

    def __init__(self, model, artifact: DefenseArtifact, config: SteerConfig | None = None):
        self.model = model
        self.artifact = artifact
        self.config = config or SteerConfig()
        self._handles: list = []
        self._tensors: dict[int, tuple] = {}
        self._active = False
        self.last_gate_score: float | None = None
        self.n_steps_steered = 0
        self.n_steps_clipped = 0

    # -- layer access ----------------------------------------------------

    def _decoder_layers(self):
        """InternVL wraps an InternLM2/Qwen2 causal LM; its blocks live here."""
        return self.model.model.language_model.model.layers

    def selected(self) -> list[LayerDefense]:
        """Layers to steer, best probe first."""
        ranked = sorted(
            self.artifact.layers,
            key=lambda d: (-d.accuracy if not np.isnan(d.accuracy) else 0.0, d.layer),
        )
        return ranked[: self.config.top_layers] if self.config.top_layers else ranked

    # -- gating ----------------------------------------------------------

    def prepare(self, activation: np.ndarray | None) -> bool:
        """Decide whether to steer this sample, from a gate-layer activation."""
        self.n_steps_steered = 0
        self.n_steps_clipped = 0
        if not self.config.use_gate or activation is None:
            self._active = True
            self.last_gate_score = None
            return True
        score = self.artifact.gate_score(activation)
        self.last_gate_score = score
        self._active = score > self.config.gate_threshold
        return self._active

    # -- the intervention ------------------------------------------------

    def _cached(self, defense: LayerDefense, device):
        """Probe geometry as on-device tensors.

        Cached because the step is computed inside a forward hook for every
        token: doing it in NumPy meant a host round-trip per position, which is
        what made a full-sequence steered pass several times slower than an
        unsteered one.
        """
        key = defense.layer
        if key not in self._tensors:
            import torch

            direction = (
                defense.refined_direction if self.config.use_subspace else defense.direction
            )

            def to(array):
                return torch.as_tensor(
                    np.ascontiguousarray(array, dtype=np.float32), device=device
                )

            self._tensors[key] = (
                to(defense.mean),
                to(defense.scale),
                to(direction),
                to(defense.scale * direction),
            )
        return self._tensors[key]

    def _steer(self, hidden, defense: LayerDefense) -> None:
        """Move the selected positions of one layer's hidden state, in place.

        A shift of ``t`` along the probe direction in standardised space is
        ``t * scale * direction`` in activation space. Adaptive strength follows
        ARGUS: choose ``t`` so the steered activation lands at the centre of the
        "following the scene" class rather than just over the boundary.
        """
        import torch

        n = hidden.shape[1]
        if self.config.all_positions and n > 1:
            index = torch.arange(n, device=hidden.device)
        else:
            index = torch.tensor([n - 1], device=hidden.device)

        rows = hidden[0, index, :].to(torch.float32)
        mean, scale, direction, step_dir = self._cached(defense, hidden.device)

        if self.config.adaptive:
            scores = ((rows - mean) / scale) @ direction
            t = torch.clamp(scores + defense.margin, min=0.0)
        else:
            t = torch.ones(rows.shape[0], device=hidden.device, dtype=torch.float32)
        t = t * self.config.strength

        delta = -t.unsqueeze(1) * step_dir.unsqueeze(0)

        # Backstop against a mis-scaled probe.
        norms = delta.norm(dim=1)
        ceilings = rows.norm(dim=1) * self.config.max_step_ratio
        over = norms > ceilings
        if bool(over.any()):
            factor = torch.where(over, ceilings / norms.clamp(min=1e-8), torch.ones_like(norms))
            delta = delta * factor.unsqueeze(1)
            self.n_steps_clipped += int(over.sum())

        hidden[0, index, :] = (rows + delta).to(hidden.dtype)
        self.n_steps_steered += int(index.numel())

    def _make_hook(self, defense: LayerDefense):
        def hook(module, args, output):
            if not self._active:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            self._steer(hidden, defense)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        return hook

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "Steerer":
        blocks = self._decoder_layers()
        for defense in self.selected():
            # Activations are indexed with layer 0 = embeddings, so
            # hidden_states[i] is the output of decoder block i-1.
            idx = defense.layer - 1
            if not 0 <= idx < len(blocks):
                raise IndexError(
                    f"layer {defense.layer} out of range for {len(blocks)} decoder blocks"
                )
            self._handles.append(blocks[idx].register_forward_hook(self._make_hook(defense)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
