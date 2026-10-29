"""Quick fit check before a long run: does the model load fully on the GPU, how much VRAM
does one forward pass take, and does it read the injected text?

Uses the two smoke-test val samples, so it needs outputs/e1_smoke to exist.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from argus.config import cfg_get, load_config
from argus.io_utils import load_json
from argus.media import clean_media, poison_media
from argus.modeling import get_language_layers, vision_runner

SMOKE_VAL = Path("E:/Argus/outputs/e1_smoke/data/val/image_vtqa2023_val.json")


def gib(n: int) -> str:
    return f"{n / 2**30:.2f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", default=str(SMOKE_VAL))
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"model: {cfg_get(config, 'model.path')}  quantization: {cfg_get(config, 'model.quantization')}")
    start = time.time()
    runner = vision_runner(config)
    layers = get_language_layers(runner.model)
    places = sorted({str(v) for v in (getattr(runner.model, "hf_device_map", None) or {}).values()})
    print(f"loaded in {time.time() - start:.0f}s  layers={len(layers)}  hidden={runner.model.config.hidden_size}  placement={places}")
    print(f"VRAM after load: {gib(torch.cuda.memory_allocated())}")

    torch.cuda.reset_peak_memory_stats()
    max_new_tokens = int(cfg_get(config, "generation.max_new_tokens", 128))
    for item in load_json(Path(args.samples)):
        print("=" * 70)
        print(f"Q: {item['first_instruction']}  | expect: {item['first_answer']}")
        print(f"INJ: {item['second_instruction'][:120]!r}  | expect: {item['second_answer'][:60]!r}")
        for name, media in [("clean", clean_media(item)), ("poison", poison_media(item))]:
            start = time.time()
            response = runner.generate(item["clean_prompt"], media, max_new_tokens=max_new_tokens)
            print(f"{name:>6} ({time.time() - start:.1f}s): {response!r}")
        activation = runner.activation(item["clean_prompt"], poison_media(item))
        print(f"activation shape: {tuple(activation.shape)}  dtype: {activation.dtype}")
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"peak VRAM: {gib(torch.cuda.max_memory_allocated())} of {gib(total)}")


if __name__ == "__main__":
    main()
