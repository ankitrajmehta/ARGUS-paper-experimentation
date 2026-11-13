"""Model backends.

Two families are supported. They answer the same DriveLM exchange and expose
the same methods, but they tile images differently and need different
transformers versions, so the knobs are not interchangeable:

================  ====================  ==========================
                  InternVL2.5           Qwen2-VL / Qwen2.5-VL
================  ====================  ==========================
image budget      ``--max-tiles``       ``--max-pixels``
precision         dtype only            dtype + ``--vision-dtype``
steering          fitted (section 12)   not fitted
================  ====================  ==========================

``load_model`` picks the family from the checkpoint name so a run only has to
name the model. Pass ``backend=`` to override for a local path that does not
carry the family in its name.
"""

from __future__ import annotations


def detect_backend(model_id: str) -> str:
    name = str(model_id).lower()
    if "internvl" in name:
        return "internvl"
    if "qwen" in name:
        return "qwen"
    raise ValueError(
        f"cannot tell which family {model_id!r} belongs to; pass --backend internvl|qwen"
    )


def load_model(model_id: str, backend: str = "auto", **kwargs):
    """Build a DriveLM runner, dropping options the backend does not take.

    The CLIs share one argument parser across both families, so each backend is
    handed only the keys it understands rather than every flag the parser
    defines.
    """
    if backend == "auto":
        backend = detect_backend(model_id)

    if backend == "internvl":
        from midas.models.internvl import InternVLDriveLM

        allowed = {
            "device", "dtype", "load_4bit", "use_flash_attn", "device_map",
            "max_new_tokens", "do_sample", "max_tiles", "swap_summarise_options",
        }
        return InternVLDriveLM(model_id=model_id, **_keep(kwargs, allowed))

    if backend == "qwen":
        from midas.models.qwen import QwenVLDriveLM

        allowed = {
            "device", "dtype", "vision_dtype", "device_map", "max_new_tokens",
            "do_sample", "max_pixels", "min_pixels", "swap_summarise_options",
        }
        return QwenVLDriveLM(model_id=model_id, **_keep(kwargs, allowed))

    raise ValueError(f"unknown backend {backend!r}; expected internvl or qwen")


def _keep(kwargs: dict, allowed: set[str]) -> dict:
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
