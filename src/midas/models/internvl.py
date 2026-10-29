"""InternVL2.5-8B wrapper for the DriveLM task.

Reproduces CHAI's inference path (``gpt/gpt_call.py``) with their hardcoded
machine-specific paths removed and greedy decoding available.

The image preprocessing below is InternVL's standard dynamic tiling, copied from
the model card rather than imported, because the version that ships inside the
remote-code bundle is not importable without loading the whole model.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from midas import config
from midas.models.base import Reply
from midas.tasks import drivelm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def supports_flash_attention(device: str) -> bool:
    """Whether FlashAttention-2 can run on this device.

    The FA2 kernels need compute capability 8.0 (Ampere) or newer. A Turing card
    such as Kaggle's T4 is 7.5, where the kernel aborts with "FlashAttention only
    supports Ampere GPUs or newer". Checking the capability is safer than
    checking whether the package imports, because the package installs happily
    on Turing and only fails at the first forward pass.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_device(requested: str = "auto") -> str:
    """Pick a device, preferring CUDA then Apple Silicon then CPU."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """Pick a dtype appropriate to the device.

    bfloat16 needs compute capability >= 8.0 on NVIDIA, and its op coverage on
    Apple's MPS backend is still incomplete, so anything but a capable CUDA card
    falls back to float16.
    """
    if requested != "auto":
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if requested not in dtype:
            raise ValueError(f"dtype must be auto/bfloat16/float16/float32, got {requested!r}")
        return dtype[requested]

    # Gate bfloat16 on compute capability, not torch.cuda.is_bf16_supported().
    # That helper returns True on a Turing T4 (sm_75) because recent PyTorch
    # counts emulated support, but Turing has no bf16 tensor cores, so the
    # emulated path is far slower than float16 and buys nothing.
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    if device == "cpu":
        # float16 matmuls are not implemented on most CPUs.
        return torch.float32
    return torch.float16


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target)
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Split an image into up to ``max_num`` 448x448 tiles, preserving aspect ratio."""
    width, height = image.size
    aspect_ratio = width / height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        },
        key=lambda x: x[0] * x[1],
    )

    ratio = _closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]

    resized = image.resize((target_width, target_height))
    tiles = []
    cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def preprocess(image: np.ndarray | Image.Image, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    """RGB array or PIL image -> (tiles, 3, 448, 448) pixel values."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


class InternVLDriveLM:
    """CHAI's DriveLM agent backed by InternVL2.5-8B."""

    def __init__(
        self,
        model_id: str = config.MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        load_4bit: bool = False,
        use_flash_attn: bool = True,
        device_map: str | None = None,
        max_new_tokens: int = 1250,
        do_sample: bool = False,
        max_tiles: int = 12,
    ) -> None:
        # CHAI use max_num=12 with a thumbnail (util/internvl_image_loader.py).
        # On a 1600x900 frame that is 9 tiles per image, 18 across the two
        # cameras, ~4600 image tokens. Lowering it trades fidelity for speed.
        self.max_tiles = max_tiles
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        # FlashAttention-2 is CUDA-only and needs Ampere or newer.
        if use_flash_attn and not supports_flash_attention(self.device):
            use_flash_attn = False

        kwargs: dict = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_flash_attn": use_flash_attn,
        }
        if load_4bit:
            if self.device != "cuda":
                raise ValueError("--load-4bit needs bitsandbytes, which is CUDA-only")
            # Fits an 8B model on a 12 GB card at some quality cost; record in results.
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = device_map or "auto"
        elif device_map:
            # Shard across several GPUs, e.g. an 8B model over two 16 GB T4s.
            # accelerate places the layers, so the .to(device) below must not run.
            kwargs["device_map"] = device_map

        print(
            f"  device={self.device} dtype={self.dtype} "
            f"flash_attn={use_flash_attn} device_map={device_map or 'none'}"
        )
        self.sharded = bool(load_4bit or device_map)
        self.model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        if not self.sharded:
            self.model = self.model.to(self.device)
        else:
            # With accelerate placing layers, inputs go to wherever the embedding
            # lives rather than to a single nominated device.
            self.device = str(next(self.model.parameters()).device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

    def free_cache(self) -> None:
        """Release cached allocator blocks.

        Both the CUDA and MPS allocators hold freed blocks for reuse. Across a
        long sweep that cache grows steadily, and on Apple Silicon -- where GPU
        and CPU share one pool -- it eventually pushes the process into swap and
        the run slows to a crawl. Call this between scenes.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def memory_gb(self) -> float:
        """Currently allocated GPU memory, for progress logging."""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1e9
        if self.device == "mps":
            return torch.mps.current_allocated_memory() / 1e9
        return 0.0

    def _generation_config(self) -> dict:
        cfg = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            cfg["temperature"] = 1.0
        return cfg

    def _conv_template(self):
        """The conversation template helper from InternVL's remote code."""
        import sys

        module = sys.modules[type(self.model).__module__]
        return module.get_conv_template(self.model.template)

    def _encode_images(self, images: list[np.ndarray]):
        """Tile and stack the two camera frames CHAI show the model."""
        tiles, counts = [], []
        for idx in drivelm.CAMERA_INDICES:
            t = preprocess(images[idx], max_num=self.max_tiles).to(self.dtype).to(self.device)
            tiles.append(t)
            counts.append(t.size(0))
        return torch.cat(tiles, dim=0), counts

    @torch.no_grad()
    def activations(
        self,
        images: list[np.ndarray],
        answer: str | None,
        layers: list[int] | None = None,
        question: str | None = None,
        readout: str = "answer",
    ) -> dict[int, torch.Tensor]:
        """Last-token hidden states when the model is *given* ``answer``.

        Teacher forcing, following ARGUS: rather than generating and labelling
        whatever came out, we append each candidate answer to the same prompt and
        read the activations directly. Both classes then differ only in the
        answer, not in sampling luck or chain-of-thought length.

        ``answer=None`` leaves the assistant turn open, exactly as generation
        sees it. That is the state the injection gate must run on at inference,
        since no answer exists yet; training the gate on answer-token states
        instead would fit a position the model never occupies in deployment.

        ``readout`` selects which token's state to take. The chat template
        closes the assistant turn with ``<|im_end|>\n``, so the literal final
        token is a newline that is identical for every answer -- at layer 0 the
        two classes are then exactly equal, and all separation has to arrive via
        attention. ``"answer"`` (default) instead reads the last token before
        that suffix, i.e. the action word itself. ``"last"`` keeps the literal
        final token, which is what ARGUS describe.

        Returns ``{layer_index: (hidden_size,) float32 cpu tensor}``, where layer
        0 is the embedding output.
        """
        if readout not in {"answer", "last"}:
            raise ValueError(f"readout must be 'answer' or 'last', got {readout!r}")
        pixel_values, num_patches_list = self._encode_images(images)

        # Mirrors chat(), except the assistant turn carries the answer instead
        # of being left open for generation.
        IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CTX)

        template = self._conv_template()
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question or drivelm.build_prompt())
        template.append_message(template.roles[1], answer)
        query = template.get_prompt()
        if answer is None:
            # An open turn ends at the assistant header, which is where the
            # model would begin generating.
            readout = "last"

        for num_patches in num_patches_list:
            image_tokens = IMG_START + IMG_CTX * self.model.num_image_token * num_patches + IMG_END
            query = query.replace("<image>", image_tokens, 1)

        encoded = self.tokenizer(query, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)

        position = -1
        if readout == "answer":
            # Last token before the closing <|im_end|> of the assistant turn.
            end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            matches = (encoded["input_ids"][0] == end_id).nonzero().flatten()
            if len(matches):
                position = int(matches[-1]) - 1
        attention_mask = encoded["attention_mask"].to(self.device)
        # forward() squeezes this unconditionally; one flag per tile.
        image_flags = torch.ones(pixel_values.size(0), 1, dtype=torch.long, device=self.device)

        out = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = out.hidden_states
        wanted = layers if layers is not None else range(len(hidden))
        # float32 on CPU: probes are fit in numpy, and fp16 loses precision there.
        acts = {i: hidden[i][0, position, :].float().cpu() for i in wanted}

        del pixel_values, input_ids, attention_mask, image_flags, out, hidden
        return acts

    @torch.no_grad()
    def score_answers(
        self,
        images: list[np.ndarray],
        answers: dict[str, str],
        steerer=None,
    ) -> dict[str, float]:
        """Mean log-probability the model assigns to each candidate answer.

        A forced-choice proxy for ``ask()``: two forward passes instead of
        generating ~768 tokens twice, so roughly ten times cheaper. That matters
        because tuning steering on a handful of generated scenes is tuning on
        noise -- one scene flipping moves ASR by 33 points at n=3.

        It measures which answer the model prefers, not what it would actually
        emit after a long chain of thought, so it is a tuning signal rather than
        a substitute for the headline ASR.
        """
        import contextlib

        pixel_values, num_patches_list = self._encode_images(images)
        IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CTX)
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        def build(answer: str | None) -> str:
            template = self._conv_template()
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], drivelm.build_prompt())
            template.append_message(template.roles[1], answer)
            query = template.get_prompt()
            for num_patches in num_patches_list:
                tokens = IMG_START + IMG_CTX * self.model.num_image_token * num_patches + IMG_END
                query = query.replace("<image>", tokens, 1)
            return query

        # The open turn ends where the answer would begin, which locates the
        # answer tokens without assuming a fixed template suffix.
        prefix_len = self.tokenizer(build(None), return_tensors="pt")["input_ids"].shape[1]

        if steerer is not None:
            gate_activation = None
            if steerer.config.use_gate:
                layer = steerer.artifact.gate_layer
                gate_activation = self.activations(images, None, layers=[layer])[layer].numpy()
            steerer.prepare(gate_activation)

        scores: dict[str, float] = {}
        for key, answer in answers.items():
            encoded = self.tokenizer(build(answer), return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            image_flags = torch.ones(
                pixel_values.size(0), 1, dtype=torch.long, device=self.device
            )

            context = steerer if steerer is not None else contextlib.nullcontext()
            with context:
                out = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    return_dict=True,
                )

            logits = out.logits[0].float()
            ids = input_ids[0]
            # Answer tokens run from the end of the open turn to <|im_end|>.
            matches = (ids == end_id).nonzero().flatten()
            stop = int(matches[-1]) if len(matches) else len(ids)
            span = range(prefix_len, stop)
            if not len(span):
                scores[key] = float("nan")
                continue

            logprobs = torch.log_softmax(logits, dim=-1)
            total = sum(float(logprobs[i - 1, ids[i]]) for i in span)
            scores[key] = total / len(span)
            del out, logits, logprobs

        del pixel_values
        return scores

    @property
    def n_layers(self) -> int:
        return self.model.config.llm_config.num_hidden_layers

    @torch.no_grad()
    def ask(
        self,
        images: list[np.ndarray],
        retry_on_unreadable: bool = True,
        steerer=None,
    ) -> Reply:
        """Run CHAI's two-turn DriveLM exchange on one scene.

        ``images`` is the full six-camera list; only CAM_FRONT and CAM_BACK are
        shown to the model, matching upstream.

        With a ``steerer``, the gate is evaluated first on the open-turn state --
        the same position it was fitted on -- and steering is then applied to
        every generated token.
        """
        import contextlib
        gate_score = None
        steered = False
        if steerer is not None:
            gate_activation = None
            if steerer.config.use_gate:
                layer = steerer.artifact.gate_layer
                gate_activation = self.activations(images, None, layers=[layer])[layer].numpy()
            steered = steerer.prepare(gate_activation)
            gate_score = steerer.last_gate_score
            self.free_cache()

        pixel_values = []
        num_patches_list = []
        for idx in drivelm.CAMERA_INDICES:
            tiles = preprocess(images[idx], max_num=self.max_tiles).to(self.dtype).to(self.device)
            pixel_values.append(tiles)
            num_patches_list.append(tiles.size(0))
        pixel_values = torch.cat(pixel_values, dim=0)

        gen_cfg = self._generation_config()

        # Hooks stay installed for the whole exchange, so both turns are steered.
        context = steerer if (steerer is not None and steered) else contextlib.nullcontext()
        with context:
            long, history = self.model.chat(
                self.tokenizer,
                pixel_values,
                drivelm.build_prompt(),
                gen_cfg,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )

            retried = False
            if retry_on_unreadable and drivelm.chai_needs_retry(long):
                # CHAI re-queries once when the reply names no action.
                retried = True
                long, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    drivelm.build_prompt(),
                    gen_cfg,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=True,
                )

            short, _ = self.model.chat(
                self.tokenizer,
                pixel_values,
                drivelm.build_summarise_prompt(long),
                gen_cfg,
                num_patches_list=num_patches_list,
                history=history,
                return_history=True,
            )

        reply = Reply(
            long=long,
            short=short,
            action=drivelm.parse_action(short),
            retried=retried,
            gate_score=gate_score,
            steered=steered,
        )
        del pixel_values, history
        return reply
