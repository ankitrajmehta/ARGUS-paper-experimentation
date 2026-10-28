"""Activation steering: the defense itself.

Three pieces, following ARGUS and SafeSteer:

* a **direction** at a mid-network layer, pointing from "following the scene"
  toward "following the sign". Steering moves against it.
* a **subspace** (SafeSteer): the top singular directions of the paired
  activation differences. Projecting the raw direction into it drops components
  that happen to correlate in this small sample but carry no safety signal.
* a **gate** (ARGUS): a second probe deciding whether an injected instruction is
  present at all, so clean frames and ordinary signage are left untouched.

Steering is applied to the last token's hidden state at the chosen layer, during
both prefill and each generation step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DefenseArtifact:
    """Everything needed to steer, fitted offline and reloaded at inference.

    Probes are fitted on standardised features, so the scaler statistics have to
    travel with the weights or the decision boundary is meaningless at runtime.
    """

    layer: int
    direction: np.ndarray  # (hidden,) unit norm, points toward following the sign
    subspace: np.ndarray  # (dim, hidden) orthonormal rows
    mean: np.ndarray  # (hidden,) standardiser used when fitting
    scale: np.ndarray  # (hidden,)

    gate_layer: int
    gate_w: np.ndarray  # (hidden,)
    gate_b: float
    gate_mean: np.ndarray
    gate_scale: np.ndarray

    # Average signed distance of "following the scene" samples from the probe
    # hyperplane. Adaptive steering aims to land a steered activation here
    # rather than merely on the far side of the boundary.
    margin: float
    model: str = ""
    notes: str = ""

    @property
    def refined_direction(self) -> np.ndarray:
        """Direction after projection into the safety subspace, re-normalised."""
        v = self.subspace.T @ (self.subspace @ self.direction)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else self.direction

    def gate_score(self, activation: np.ndarray) -> float:
        """Signed distance from the gate hyperplane; > 0 means injection."""
        z = (activation - self.gate_mean) / self.gate_scale
        return float(z @ self.gate_w + self.gate_b)

    def follow_score(self, activation: np.ndarray) -> float:
        """Signed distance from the steering probe; > 0 means following the sign."""
        z = (activation - self.mean) / self.scale
        return float(z @ self.direction)

    def save(self, path: Path) -> Path:
        # np.savez_compressed appends .npz, so normalise here or load() cannot
        # find what save() reported writing.
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            direction=self.direction,
            subspace=self.subspace,
            mean=self.mean,
            scale=self.scale,
            gate_w=self.gate_w,
            gate_mean=self.gate_mean,
            gate_scale=self.gate_scale,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "layer": self.layer,
                    "gate_layer": self.gate_layer,
                    "gate_b": self.gate_b,
                    "margin": self.margin,
                    "model": self.model,
                    "notes": self.notes,
                    "subspace_dim": int(self.subspace.shape[0]),
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
        return cls(
            layer=meta["layer"],
            direction=arrays["direction"],
            subspace=arrays["subspace"],
            mean=arrays["mean"],
            scale=arrays["scale"],
            gate_layer=meta["gate_layer"],
            gate_w=arrays["gate_w"],
            gate_b=meta["gate_b"],
            gate_mean=arrays["gate_mean"],
            gate_scale=arrays["gate_scale"],
            margin=meta["margin"],
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
    # Cap on a single step, as a fraction of the activation's own norm.
    max_step_ratio: float = 0.25
    # Gate threshold on the signed distance. 0 is the fitted boundary; raising it
    # makes the defense more conservative about intervening.
    gate_threshold: float = 0.0


class Steerer:
    """Applies a DefenseArtifact to a live InternVL model via forward hooks."""

    def __init__(self, model, artifact: DefenseArtifact, config: SteerConfig | None = None):
        self.model = model
        self.artifact = artifact
        self.config = config or SteerConfig()
        self._handles: list = []
        # Set per sample by prepare(); None means "decide nothing, do not steer".
        self._active = False
        self.last_gate_score: float | None = None
        self.n_steps_steered = 0
        self.n_steps_clipped = 0

    # -- layer access ----------------------------------------------------

    def _decoder_layers(self):
        """InternVL wraps an InternLM2/Qwen2 causal LM; its blocks live here."""
        return self.model.model.language_model.model.layers

    # -- gating ----------------------------------------------------------

    def prepare(self, activation: np.ndarray | None) -> bool:
        """Decide whether to steer this sample, from a gate-layer activation."""
        self.n_steps_steered = 0
        if not self.config.use_gate or activation is None:
            self._active = True
            self.last_gate_score = None
            return True
        score = self.artifact.gate_score(activation)
        self.last_gate_score = score
        self._active = score > self.config.gate_threshold
        return self._active

    # -- the intervention ------------------------------------------------

    def _delta(self, hidden_row: np.ndarray) -> np.ndarray:
        """How far to move one activation, in raw model space.

        The probe lives in standardised space: ``z = (a - mean) / scale`` and
        ``score = z . d``. To shift the score by a known amount the step has to
        be mapped back through the scaler, so a move of ``t`` along ``d`` in
        standardised space is ``t * scale * d`` in activation space.

        Adaptive strength follows ARGUS: choose ``t`` so the steered activation
        lands at the centre of the "following the scene" class rather than just
        over the boundary, i.e. ``t = score + margin``.

        An earlier version multiplied this by ``||a||`` as SafeSteer do. That is
        only valid when the coefficient is O(1); here it is a standardised
        distance of 5-10, so the step came out several times the activation norm
        and the model emitted word salad. The two scalings are not composable --
        this uses the probe geometry alone, with a cap as a backstop.
        """
        art = self.artifact
        direction = art.refined_direction if self.config.use_subspace else art.direction

        if self.config.adaptive:
            t = max(0.0, art.follow_score(hidden_row) + art.margin)
        else:
            t = 1.0
        t *= self.config.strength

        if t <= 0:
            return np.zeros_like(hidden_row)

        delta = -t * art.scale * direction

        # Backstop: never move an activation further than a fixed fraction of its
        # own norm. Without it a mis-scaled probe can push the hidden state far
        # outside the distribution the layers above were trained on.
        norm = np.linalg.norm(delta)
        ceiling = self.config.max_step_ratio * np.linalg.norm(hidden_row)
        if norm > ceiling > 0:
            delta *= ceiling / norm
            self.n_steps_clipped += 1
        return delta

    def _hook(self, module, args, output):
        if not self._active:
            return output

        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        row = hidden[0, -1, :].detach().float().cpu().numpy()
        delta = self._delta(row)
        if np.allclose(delta, 0):
            return output

        step = torch.from_numpy(delta).to(hidden.device, hidden.dtype)
        hidden[0, -1, :] = hidden[0, -1, :] + step
        self.n_steps_steered += 1
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "Steerer":
        layers = self._decoder_layers()
        if not 0 <= self.artifact.layer - 1 < len(layers):
            raise IndexError(
                f"layer {self.artifact.layer} out of range for {len(layers)} decoder blocks"
            )
        # Activations are indexed with layer 0 = embeddings, so hidden_states[i]
        # is the output of decoder block i-1.
        target = layers[self.artifact.layer - 1]
        self._handles.append(target.register_forward_hook(self._hook, with_kwargs=False))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._active = False
