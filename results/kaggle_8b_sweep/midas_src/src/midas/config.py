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


@dataclass(frozen=True)
class AttackConfig:
    """How to turn CHAI's released optimal.json into a SignSpec.

    Two upstream quirks are made explicit here rather than silently inherited:

    background_mode
        CHAI's optimal.json records an optimised background of
        (r=77, g=169, b=15) -- green. But main.py.get_score overwrites red and
        green just before rendering::

            rgb_background = (255, 0, int(rgb_background[2]))

        so the sign actually evaluated in their runs is (255, 0, 15) -- red.
        "code" reproduces what their pipeline executed; "json" uses the colours
        their file records. Default is "code" so our numbers are comparable to
        the paper's.

    size_multiplier / position_offset
        Their optimiser also searched a size multiplier in [0.8, 1.2] and pixel
        offsets in [-25, 25], but optimal.json does not record the chosen values.
        We therefore fall back to each scene's own size and position.
    """

    background_mode: str = "code"  # "code" | "json"
    size_multiplier: float = 1.0
    position_offset: tuple[int, int] = (0, 0)
    language: str = "English"

    def __post_init__(self) -> None:
        if self.background_mode not in {"code", "json"}:
            raise ValueError(f"background_mode must be 'code' or 'json', got {self.background_mode!r}")


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
