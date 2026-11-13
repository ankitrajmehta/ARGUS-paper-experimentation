"""Qwen2-VL / Qwen2.5-VL runner for the CHAI DriveLM exchange.

A second model family, so the attack is not measured on one architecture. The
public surface matches :class:`midas.models.internvl.InternVLDriveLM` -- ``ask``,
``score_answers``, ``activations``, ``free_cache``, ``memory_gb`` -- so every CLI
and the optimiser work unchanged.

Three things here come from a sibling project that ran these models on the same
Kaggle T4s, and each one costs a session to rediscover:

**The vision tower must run in float32 on a T4.** With the language model in
float16 the vision tower overflows to NaN on text-heavy frames. Every frame we
evaluate has text rendered onto it, so this is not an edge case for us -- it is
the normal path. ``vision_dtype`` defaults to float32 whenever the language
model is float16.

**Pixel budget, not tile count.** InternVL splits an image into 448px tiles and
we cap the count; Qwen resizes to a pixel budget instead. ``max_pixels`` is the
equivalent knob, and the default matches the sibling project's 409600 (~520
visual tokens per frame).

**Transformers version.** Qwen2-VL needs >= 4.45, Qwen2.5-VL needs >= 4.49, and
InternVL2.5's remote code breaks on 4.50+. 4.49.x is the only window where all
three work, which is why requirements-gpu.txt pins it there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from midas.models.base import Reply
from midas.models.internvl import resolve_device, resolve_dtype
from midas.tasks import drivelm

# Matches the sibling project; ~520 visual tokens per frame at 28x28 patches.
DEFAULT_MAX_PIXELS = 409_600
DEFAULT_MIN_PIXELS = 50_176  # 256 * 28 * 28, the processor's own floor


def model_class_for(model_id: str):
    """Pick the architecture from the checkpoint's own config.

    Qwen2-VL and Qwen2.5-VL share this runner but are different classes, and
    guessing from the name fails on local paths and forks.
    """
    import transformers

    model_type = None
    local = Path(model_id) / "config.json"
    if local.exists():
        model_type = json.loads(local.read_text()).get("model_type")
    else:
        from transformers import AutoConfig

        model_type = AutoConfig.from_pretrained(model_id, trust_remote_code=True).model_type

    if model_type == "qwen2_5_vl":
        cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
        if cls is None:
            raise ImportError(
                "Qwen2.5-VL needs transformers>=4.49; "
                f"installed {transformers.__version__}"
            )
        return cls
    if model_type == "qwen2_vl":
        return transformers.Qwen2VLForConditionalGeneration
    raise ValueError(f"{model_id} has model_type {model_type!r}, not a Qwen-VL checkpoint")


class QwenVLDriveLM:
    """CHAI's two-turn DriveLM exchange on a Qwen-VL model."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        vision_dtype: str = "auto",
        device_map: str | None = None,
        max_new_tokens: int = 1250,
        do_sample: bool = False,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        swap_summarise_options: bool = False,
    ) -> None:
        from transformers import AutoProcessor

        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.swap_summarise_options = swap_summarise_options

        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True
        )
        # Left padding keeps the last position meaningful, which the probes read.
        self.processor.tokenizer.padding_side = "left"

        cls = model_class_for(model_id)
        kwargs: dict = {"torch_dtype": self.dtype, "trust_remote_code": True}
        if device_map:
            kwargs["device_map"] = device_map
        self.model = cls.from_pretrained(model_id, **kwargs).eval()
        if not device_map:
            self.model = self.model.to(self.device)

        self.vision_dtype = self._resolve_vision_dtype(vision_dtype)
        if self.vision_dtype != self.dtype:
            self._vision_tower().to(self.vision_dtype)

        print(
            f"  device={self.device} dtype={self.dtype} "
            f"vision_dtype={self.vision_dtype} max_pixels={self.max_pixels} "
            f"layers={self.n_layers}",
            flush=True,
        )

    # ------------------------------------------------------------------ setup
    def _resolve_vision_dtype(self, requested: str) -> torch.dtype:
        """float32 vision under a float16 language model.

        The fp16 vision tower produces NaN on frames carrying rendered text,
        which is every attacked frame we evaluate.
        """
        if requested != "auto":
            return getattr(torch, requested)
        return torch.float32 if self.dtype == torch.float16 else self.dtype

    def _vision_tower(self):
        for attr in ("visual", "vision_tower"):
            tower = getattr(self.model, attr, None)
            if tower is None:
                tower = getattr(getattr(self.model, "model", None), attr, None)
            if tower is not None:
                return tower
        raise AttributeError(f"no vision tower found on {type(self.model).__name__}")

    def _layers(self):
        """The decoder layer list, where activations are read and steered."""
        for path in (("model", "language_model", "layers"), ("model", "layers"), ("layers",)):
            node = self.model
            for name in path:
                node = getattr(node, name, None)
                if node is None:
                    break
            if node is not None:
                return node
        raise AttributeError(f"no decoder layers found on {type(self.model).__name__}")

    @property
    def n_layers(self) -> int:
        return len(self._layers())

    def free_cache(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def memory_gb(self) -> float:
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    # ------------------------------------------------------------- prompting
    def _messages(self, images: list[np.ndarray], prompt: str) -> list[dict]:
        """CHAI shows the model CAM_FRONT and CAM_BACK, in that order."""
        content = [{"type": "image"} for _ in drivelm.CAMERA_INDICES]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _pil_frames(self, images: list[np.ndarray]) -> list[Image.Image]:
        return [Image.fromarray(images[i]) for i in drivelm.CAMERA_INDICES]

    def _encode(self, images: list[np.ndarray], messages: list[dict]):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(
            text=[text], images=self._pil_frames(images), return_tensors="pt"
        )
        return {k: v.to(self.model.device) for k, v in batch.items()}

    def _generate(self, inputs: dict) -> str:
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=None if not self.do_sample else 1.0,
                top_p=None if not self.do_sample else 1.0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------- api
    def ask(
        self,
        images: list[np.ndarray],
        retry_on_unreadable: bool = True,
        steerer=None,
    ) -> Reply:
        """The two-turn exchange: chain of thought, then the one-word summary."""
        if steerer is not None:
            raise NotImplementedError(
                "steering hooks are fitted for InternVL; see PROJECT_RECORD section 12"
            )

        inputs = self._encode(images, self._messages(images, drivelm.build_prompt()))
        long = self._generate(inputs)

        retried = False
        if retry_on_unreadable and drivelm.chai_needs_retry(long):
            # CHAI re-query once when the reply names no action.
            retried = True
            long = self._generate(inputs)

        # Second turn: a fresh exchange carrying the first answer's tail, which
        # is what CHAI do rather than continuing the conversation.
        summarise = drivelm.build_summarise_prompt(long, self.swap_summarise_options)
        short = self._generate(self._encode(images, self._messages(images, summarise)))

        return Reply(
            long=long,
            short=short,
            action=drivelm.parse_action(short),
            retried=retried,
        )

    def score_answers(
        self,
        images: list[np.ndarray],
        answers: dict[str, str],
        steerer=None,
    ) -> dict[str, float]:
        """Mean log-probability of each candidate answer.

        The forced-choice proxy the sign search uses: two forward passes rather
        than generating a full chain of thought per candidate.
        """
        if steerer is not None:
            raise NotImplementedError("steering is InternVL-only here")

        messages = self._messages(images, drivelm.build_prompt())
        prefix = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        frames = self._pil_frames(images)

        scores: dict[str, float] = {}
        for key, answer in answers.items():
            batch = self.processor(
                text=[prefix + answer], images=frames, return_tensors="pt"
            )
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            # Score only the answer tokens, not the prompt.
            n_prompt = self.processor(
                text=[prefix], images=frames, return_tensors="pt"
            )["input_ids"].shape[1]

            labels = batch["input_ids"].clone()
            labels[:, :n_prompt] = -100
            with torch.inference_mode():
                out = self.model(**batch, labels=labels)
            # loss is the mean NLL over scored tokens.
            scores[key] = float(-out.loss.item())
        return scores

    def activations(
        self,
        images: list[np.ndarray],
        answer: str | None = None,
        readout: str = "answer",
        layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Hidden state at one token position, per layer.

        ``answer=None`` reads the open turn -- the state that exists at
        inference, before any token is generated. With an answer it is
        teacher-forced and read at the final answer token.
        """
        messages = self._messages(images, drivelm.build_prompt())
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if answer is not None:
            text = text + answer

        batch = self.processor(
            text=[text], images=self._pil_frames(images), return_tensors="pt"
        )
        batch = {k: v.to(self.model.device) for k, v in batch.items()}

        wanted = set(range(self.n_layers)) if layers is None else set(layers)
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = hidden[0, -1, :].detach().float().cpu()
            return hook

        decoder = self._layers()
        for idx in sorted(wanted):
            handles.append(decoder[idx].register_forward_hook(make_hook(idx)))
        try:
            with torch.inference_mode():
                self.model(**batch)
        finally:
            for handle in handles:
                handle.remove()
        return captured
