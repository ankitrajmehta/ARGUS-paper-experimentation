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
    # Norm and intercept of the *un-normalised* probe weight, in the same
    # standardised space. ``direction`` has these divided out, which is all the
    # uniform schedule needs -- but ARGUS's step size is
    # ``relu((w.h + b)/||w||^2)``, so reproducing it needs ||w|| and b back.
    probe_norm: float = 1.0
    bias: float = 0.0

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

    def argus_alpha(self, score: float, epsilon: float = 0.0) -> float:
        """ARGUS's Eq. 7 step size, from a signed distance along ``direction``.

        Theirs is ``relu((w.h + b + eps)/||w||^2)`` on a raw activation. Here
        ``score`` is ``z.direction``, so ``w.h + b`` is ``probe_norm * score +
        bias``. The remaining ``1/||w||^2`` is why their measured steps were
        ~1.1 against a ||w|| of ~46: each one closes about 2% of the score,
        which is what makes the uncapped first step do nearly all the work.
        """
        raw = self.probe_norm * score + self.bias + epsilon
        return max(0.0, raw / max(self.probe_norm**2, 1e-12))


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
                        {
                            "layer": d.layer,
                            "margin": d.margin,
                            "accuracy": d.accuracy,
                            "probe_norm": d.probe_norm,
                            "bias": d.bias,
                        }
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
                # Artifacts fitted before the ARGUS schedule existed carry no
                # probe norm. 1.0 leaves the uniform schedule identical and
                # makes the ARGUS schedule fall back to a plain signed
                # distance, which is at least not silently wrong.
                probe_norm=entry.get("probe_norm", 1.0),
                bias=entry.get("bias", 0.0),
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

    # Which step schedule to use.
    #
    # ``uniform``  every forward pass gets the same adaptive, capped step. This
    #              is what section 12 measured, and it was inert: 136,725
    #              activations moved, 1.6 nats of margin shift, attack rate
    #              unchanged at exactly 40.0%.
    # ``argus``    the released ARGUS schedule. One **uncapped full-magnitude**
    #              step at the first generated token, then tiny Eq. 7 steps.
    #              Their own note is that attacks look "all-or-nothing from the
    #              first token", so the first step is the intervention and the
    #              rest only hold the line.
    schedule: str = "uniform"
    # ARGUS step-0 magnitude, in activation-space norm. Their run measured 9.9.
    # None falls back to ``strength``.
    first_step_norm: float | None = None
    # ARGUS's epsilon: pushes past the boundary rather than onto it.
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.schedule not in ("uniform", "argus"):
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.schedule == "argus" and self.all_positions:
            # Their hook indexes [:, -1, :] unconditionally. Steering every
            # position is a different intervention, and conflating the two is
            # exactly how section 12's result became uninterpretable.
            raise ValueError("schedule='argus' steers the last position only")


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
        # Forward-pass index per layer, so the ARGUS schedule can tell the
        # first generated token from the rest.
        self._step: dict[int, int] = {}
        self.first_step_norms: list[float] = []

    # -- layer access ----------------------------------------------------

    def _decoder_layers(self):
        """The decoder block list of whichever runner we were handed.

        The Qwen runner already locates these (its nesting differs by
        transformers version), so prefer its own accessor and keep the
        InternVL path as the fallback.
        """
        getter = getattr(self.model, "_layers", None)
        if callable(getter):
            return getter()
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
        self._step.clear()
        self.first_step_norms = []
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

            # The step in activation space. A shift of t along the probe in
            # standardised space is t * scale * direction here; the unit
            # version lets the ARGUS schedule set an exact activation-space
            # norm, which is how their strength is reported.
            step = defense.scale * direction
            unit = step / max(float(np.linalg.norm(step)), 1e-12)

            self._tensors[key] = (
                to(defense.mean),
                to(defense.scale),
                to(direction),
                to(step),
                to(unit),
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
        mean, scale, direction, step_dir, _unit = self._cached(defense, hidden.device)

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

    def _steer_argus(self, hidden, defense: LayerDefense, step: int) -> None:
        """The released ARGUS intervention: one big step, then small ones.

        Phase one, at the first generated token, adds the steering vector at
        **full magnitude with no cap and no adaptive scaling**. Phase two, every
        token after, takes Eq. 7 steps that each close only a few percent of the
        probe score -- they hold the state on the safe side rather than moving
        it there.

        Always the last position only. During decoding the last position *is*
        the current token, so this is the whole generated sequence, one token at
        a time; during the prefill it is the final prompt token, which is where
        the first generated token's logits come from.
        """
        import torch

        n = hidden.shape[1]
        index = torch.tensor([n - 1], device=hidden.device)
        rows = hidden[0, index, :].to(torch.float32)
        mean, scale, direction, _step_dir, unit = self._cached(defense, hidden.device)

        if step == 0:
            magnitude = self.config.first_step_norm
            if magnitude is None:
                magnitude = self.config.strength
            t = torch.full((rows.shape[0],), float(magnitude), device=hidden.device)
            self.first_step_norms.append(float(magnitude))
        else:
            scores = ((rows - mean) / scale) @ direction
            raw = defense.probe_norm * scores + defense.bias + self.config.epsilon
            t = torch.clamp(raw / max(defense.probe_norm**2, 1e-12), min=0.0)
            t = t * self.config.strength

        # ``unit`` has activation-space norm 1, so ||delta|| == t exactly, which
        # is the quantity ARGUS report. No clipping: the uncapped first step is
        # the part of their design our section 12 attempt did not have.
        hidden[0, index, :] = (rows - t.unsqueeze(1) * unit.unsqueeze(0)).to(hidden.dtype)
        self.n_steps_steered += int(index.numel())

    def _make_hook(self, defense: LayerDefense):
        def hook(module, args, output):
            if not self._active:
                return output
            hidden = output[0] if isinstance(output, tuple) else output

            if self.config.schedule == "argus":
                # A pass over more than one position is a prefill, which starts
                # a fresh generation -- and ``ask`` runs three of them per
                # scene. Resetting here keeps "step 0" meaning the first
                # generated token of each turn rather than of the whole scene.
                if hidden.shape[1] > 1:
                    self._step[defense.layer] = 0
                step = self._step.get(defense.layer, 0)
                self._step[defense.layer] = step + 1
                self._steer_argus(hidden, defense, step)
            else:
                self._steer(hidden, defense)

            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        return hook

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "Steerer":
        blocks = self._decoder_layers()
        # How a fitted layer index maps to a decoder block depends on how the
        # runner read activations in the first place: 1 for InternVL, whose
        # hidden_states[0] is the embedding output, 0 for Qwen, which hooks the
        # blocks directly. Defaulting to 1 keeps every InternVL artifact
        # behaving exactly as before.
        offset = getattr(self.model, "ACTIVATION_LAYER_OFFSET", 1)
        for defense in self.selected():
            idx = defense.layer - offset
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
