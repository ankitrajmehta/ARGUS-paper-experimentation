"""Paths and attack constants for the CHAI/DriveLM task.

Everything that could differ between machines lives here, so the college GPU box
only needs MIDAS_ROOT and MIDAS_MODEL set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("MIDAS_ROOT", Path(__file__).resolve().parents[2]))

VENDOR_CHAI = ROOT / "vendor" / "chai"
RESULTS = ROOT / "results"

# CHAI's released artefacts, all relative to the cloned drivelm branch.
CHAI_TRAIN_JSON = VENDOR_CHAI / "training.json"
CHAI_TEST_JSON = VENDOR_CHAI / "testing.json"
CHAI_ATTACK_DICT = VENDOR_CHAI / "attack_dicts_internvl.json"
CHAI_OPTIMAL = VENDOR_CHAI / "optimal_attacks" / "optimal.json"
CHAI_ATTACKER_PY = VENDOR_CHAI / "attack" / "create_attack.py"

# InternVL2.5-8B is CHAI's open-weight target; its released signs are tuned for it.
MODEL_ID = os.environ.get("MIDAS_MODEL", "OpenGVLab/InternVL2_5-8B")

# DriveLM answer options. CHAI gives InternVL a two-way choice, GPT a four-way one;
# we target InternVL, so two.
ANSWER_NAMES_INTERNVL = {"a)": "proceed", "b)": "brake"}
ANSWER_NAMES_GPT = {"a)": "proceed", "b)": "turn left", "c)": "turn right", "d)": "brake"}


@dataclass(frozen=True)
class SignSpec:
    """One rendered sign: what it says, how it looks, where it goes.

    Colours are RGB. CHAI loads frames with cv2.imread then converts BGR->RGB, so
    the colour tuples handed to cv2.putText land in RGB order.
    """

    text: str
    bg: tuple[int, int, int]
    fg: tuple[int, int, int]
    scale: float
    # cv2 pixel coordinates (x, y). CHAI stores these per scene as
    # position.pos_y -> x and position.pos_x -> y; see data.Scene.sign_origin.
    origin: tuple[int, int]


# CHAI's main.py sets ``testing = True``, so its InternVL evaluation does not
# read optimal_attacks/optimal.json at all -- load_universal("internvl") returns
# these hardcoded values instead. Transcribed from vendor/chai/main.py:
#
#     r_letter = g_letter = b_letter = 0
#     r_background = g_background = b_background = 0
#     optimal_text_select = [0, 0, 0, 0]
#     size = 0.93
#
# get_score then rewrites the background to (255, 0, b_background), giving pure
# red. So the sign CHAI actually evaluated is black text on red at 0.93x scale,
# with no positional offset.
CHAI_UNIVERSAL_INTERNVL = {
    "background": (255, 0, 0),
    "letter": (0, 0, 0),
    "size_multiplier": 0.93,
    "position_offset": (0, 0),
    "phrase_index": 0,
}


@dataclass(frozen=True)
class AttackConfig:
    """Which sign to render, and where its parameters come from.

    param_source
        ``"chai_internvl"`` (default) uses CHAI_UNIVERSAL_INTERNVL -- the values
        their code actually runs for InternVL. ``"optimal_json"`` uses
        optimal_attacks/optimal.json, which records a *different* sign
        (light-pink text on red at 1.0x) and which their InternVL path never
        reads. We used optimal.json up to 2026-11-02; see PROJECT_RECORD error
        10. ``"custom"`` takes the parameters from ``custom`` below, which is how
        a re-optimised sign is evaluated.

    background_mode
        optimal.json records an optimised background of (77, 169, 15) -- green --
        but main.py.get_score overwrites red and green just before rendering::

            rgb_background = (255, 0, int(rgb_background[2]))

        so the evaluated sign is red. "code" reproduces what their pipeline
        executed; "json" uses the colours their file records. Only consulted when
        param_source is "optimal_json".

    size_multiplier / position_offset
        Their optimiser searched a size multiplier in [0.8, 1.2] and pixel
        offsets in [-25, 25]. optimal.json records neither, so under
        "optimal_json" we fall back to each scene's own size and position.
    """

    param_source: str = "chai_internvl"  # "chai_internvl" | "optimal_json" | "custom"
    background_mode: str = "code"  # "code" | "json"
    size_multiplier: float = 1.0
    position_offset: tuple[int, int] = (0, 0)
    language: str = "English"
    # Set when param_source == "custom": keys as in CHAI_UNIVERSAL_INTERNVL.
    custom: dict | None = None

    def __post_init__(self) -> None:
        if self.background_mode not in {"code", "json"}:
            raise ValueError(f"background_mode must be 'code' or 'json', got {self.background_mode!r}")
        sources = {"chai_internvl", "optimal_json", "custom"}
        if self.param_source not in sources:
            raise ValueError(f"param_source must be one of {sorted(sources)}, got {self.param_source!r}")
        if self.param_source == "custom" and not self.custom:
            raise ValueError("param_source='custom' needs custom= parameters")

    def resolve(self) -> dict:
        """The sign parameters this config selects.

        Returns keys ``background``, ``letter``, ``size_multiplier``,
        ``position_offset`` and ``phrase_index``.
        """
        if self.param_source == "chai_internvl":
            return dict(CHAI_UNIVERSAL_INTERNVL)
        if self.param_source == "custom":
            merged = dict(CHAI_UNIVERSAL_INTERNVL)
            merged.update(self.custom or {})
            return merged

        from midas import data

        optimal = data.load_optimal_attack()
        background = (
            (255, 0, int(optimal["b_background"]))
            if self.background_mode == "code"
            else (
                int(optimal["r_background"]),
                int(optimal["g_background"]),
                int(optimal["b_background"]),
            )
        )
        return {
            "background": background,
            "letter": (
                int(optimal["r_letter"]),
                int(optimal["g_letter"]),
                int(optimal["b_letter"]),
            ),
            "size_multiplier": self.size_multiplier,
            "position_offset": self.position_offset,
            # Indexed per target by the caller; optimal.json stores optimal_a/_b.
            "phrase_index": None,
        }


@dataclass
class RunConfig:
    """One evaluation run."""

    split: str = "train"  # "train" (CHAI Known) | "test" (CHAI Transferability)
    attack: bool = False
    model_id: str = MODEL_ID
    max_new_tokens: int = 256
    # Greedy decoding: the no-attack baseline is noisy (CHAI report 29-33% of
    # frames already yielding the target action), so sampling would bury the effect.
    do_sample: bool = False
    seed: int = 0
    repeats: int = 1
    out_dir: Path = field(default_factory=lambda: RESULTS)
