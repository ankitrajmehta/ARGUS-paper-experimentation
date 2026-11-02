"""Rendering CHAI's adversarial signs onto camera frames.

The drawing itself is delegated to CHAI's own ``Attacker`` class so our images
are pixel-identical to theirs. We load it by file path rather than adding
``vendor/chai`` to ``sys.path``, because that directory contains packages named
``attack``, ``util`` and ``gpt`` that would shadow other imports.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

import numpy as np

from midas import config, data
from midas.config import AttackConfig, SignSpec


@lru_cache(maxsize=1)
def _attacker():
    """Instantiate CHAI's Attacker from its source file."""
    path = config.CHAI_ATTACKER_PY
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/setup_vendor.sh to clone CHAI's drivelm branch."
        )
    spec = importlib.util.spec_from_file_location("chai_create_attack", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered so dataclasses/pickle inside the module resolve, but under a
    # namespaced key that cannot collide with a real package.
    sys.modules["chai_create_attack"] = module
    spec.loader.exec_module(module)
    return module.Attacker()


def build_sign(
    scene: data.Scene,
    target: str | None = None,
    cfg: AttackConfig | None = None,
    style: str | None = None,
) -> SignSpec:
    """Build CHAI's sign for one scene.

    Sign appearance comes from ``cfg.param_source`` -- by default the values
    CHAI's own InternVL path uses (see ``config.CHAI_UNIVERSAL_INTERNVL``). The
    wording comes from ``attack_dicts_internvl.json``, or from an injection
    style when ``style`` is given.
    """
    cfg = cfg or AttackConfig()
    # Defaults to the opposite of the benign action rather than the stored
    # target field; see Scene.internvl_target.
    target = target or scene.internvl_target

    params = cfg.resolve()
    bg = tuple(int(c) for c in params["background"])
    fg = tuple(int(c) for c in params["letter"])
    x, y = scene.sign_origin
    dx, dy = params["position_offset"]
    scale = scene.size * params["size_multiplier"]

    if style is not None:
        # Same colours, scale and position as CHAI's sign; only the wording
        # changes, so any difference is attributable to the phrasing.
        from midas import styles

        return SignSpec(
            text=styles.phrase_for(style, target),
            bg=bg, fg=fg, scale=scale, origin=(x + dx, y + dy),
        )

    dictionary = data.load_attack_dictionary()
    if target not in dictionary:
        raise KeyError(f"no phrases for target {target!r}; have {sorted(dictionary)}")
    phrases = dictionary[target]

    phrase_index = params["phrase_index"]
    if phrase_index is None:
        # optimal.json indexes the dictionary per target: "a)" -> optimal_a.
        optimal = data.load_optimal_attack()
        phrase_index = int(optimal.get(f"optimal_{target.rstrip(')')}", 0))
    phrase_index = int(phrase_index)
    if not 0 <= phrase_index < len(phrases):
        raise IndexError(
            f"phrase index {phrase_index} is out of range for {len(phrases)} "
            f"phrases on target {target!r}"
        )

    return SignSpec(
        text=phrases[phrase_index],
        bg=bg, fg=fg, scale=scale, origin=(x + dx, y + dy),
    )


# CHAI's put_text draws text_sp[0] and text_sp[1] and nothing else, so a third
# word is dropped without any error. "IGNORE SIGNS PROCEED" renders identically
# to "IGNORE SIGNS": the sign loses the action entirely while still looking
# plausible.
MAX_CHAI_WORDS = 2


class PhraseTooLong(ValueError):
    """A phrase CHAI's renderer would silently truncate."""


def check_renderable(text: str, allow_multiline: bool = False) -> None:
    """Reject phrases CHAI's renderer cannot draw in full."""
    words = text.split()
    if allow_multiline or len(words) <= MAX_CHAI_WORDS:
        return
    raise PhraseTooLong(
        f"{text!r} has {len(words)} words; CHAI's renderer draws only the first "
        f"{MAX_CHAI_WORDS} and discards the rest, so the sign would read "
        f"{' '.join(words[:MAX_CHAI_WORDS])!r}. Shorten it, or pass "
        "allow_multiline=True to use the extended renderer."
    )


def _render_multiline(image: np.ndarray, sign: SignSpec) -> np.ndarray:
    """Draw a phrase of any length, one word per line.

    A documented extension, not CHAI's code: it is only reached for phrases of
    three or more words, which CHAI cannot express at all. Shorter phrases go
    through their renderer untouched so every existing result is unaffected.

    Lines are centred on the widest, matching the spirit of CHAI's two-word
    layout, which centres the first word against the width of the second.
    """
    attacker = _attacker()
    words = sign.text.split()
    canvas = image.copy()

    # Measure first: the per-line offset depends on the widest word.
    widths = []
    for word in words:
        probe = image.copy()
        _, _, width, height = attacker._put_text_en(
            probe, word, sign.bg, sign.fg, sign.scale, sign.origin
        )
        widths.append((width, height))
    widest = max(w for w, _ in widths)

    x, y = sign.origin
    for word, (width, height) in zip(words, widths):
        attacker._put_text_en(
            canvas, word, sign.bg, sign.fg, sign.scale, (x, y), widest
        )
        y += height + 10  # matches the padding CHAI use between their two lines
    return canvas


def render(
    image: np.ndarray,
    sign: SignSpec,
    language: str = "English",
    allow_multiline: bool = False,
) -> np.ndarray:
    """Draw a sign onto one RGB frame and return the result.

    CHAI's put_text draws in place via cv2, so we pass a copy.
    """
    check_renderable(sign.text, allow_multiline=allow_multiline)
    if allow_multiline and len(sign.text.split()) > MAX_CHAI_WORDS:
        upper = SignSpec(sign.text.upper(), sign.bg, sign.fg, sign.scale, sign.origin)
        return _render_multiline(image, upper)

    canvas = image.copy()
    # CHAI upper-cases the phrase at render time (main.py: ``text.upper()``).
    return _attacker().put_text(
        canvas,
        sign.text.upper(),
        sign.bg,
        sign.fg,
        sign.scale,
        sign.origin,
        language=language,
    )


def attack_scene(
    scene: data.Scene,
    images: list[np.ndarray] | None = None,
    target: str | None = None,
    cfg: AttackConfig | None = None,
    style: str | None = None,
) -> tuple[list[np.ndarray], SignSpec]:
    """Apply the sign to a scene's frames.

    Only CAM_FRONT is modified, matching CHAI's threat model: the attacker places
    one physical sign ahead of the vehicle.
    """
    cfg = cfg or AttackConfig()
    images = list(images) if images is not None else scene.load_images()
    sign = build_sign(scene, target=target, cfg=cfg, style=style)
    idx = scene.attacked_camera
    images[idx] = render(images[idx], sign, language=cfg.language)
    return images, sign
