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
        device: str = "cuda",
        dtype: str = "bfloat16",
        load_4bit: bool = False,
        use_flash_attn: bool = True,
        max_new_tokens: int = 1250,
        do_sample: bool = False,
    ) -> None:
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"dtype must be bfloat16 or float16, got {dtype!r}")

        kwargs: dict = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_flash_attn": use_flash_attn,
        }
        if load_4bit:
            # Fits an 8B model on a 12 GB card at some quality cost; record in results.
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        if not load_4bit:
            self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

    def _generation_config(self) -> dict:
        cfg = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            cfg["temperature"] = 1.0
        return cfg

    @torch.no_grad()
    def ask(self, images: list[np.ndarray], retry_on_unreadable: bool = True) -> Reply:
        """Run CHAI's two-turn DriveLM exchange on one scene.

        ``images`` is the full six-camera list; only CAM_FRONT and CAM_BACK are
        shown to the model, matching upstream.
        """
        pixel_values = []
        num_patches_list = []
        for idx in drivelm.CAMERA_INDICES:
            tiles = preprocess(images[idx]).to(self.model.dtype).to(self.device)
            pixel_values.append(tiles)
            num_patches_list.append(tiles.size(0))
        pixel_values = torch.cat(pixel_values, dim=0)

        gen_cfg = self._generation_config()

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

        return Reply(
            long=long,
            short=short,
            action=drivelm.parse_action(short),
            retried=retried,
        )
