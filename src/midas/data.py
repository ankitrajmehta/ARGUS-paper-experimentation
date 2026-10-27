"""Loading CHAI's released DriveLM scenes.

A scene is one nuScenes frame set (6 cameras) plus the labels CHAI recorded for
it: the action the model picks without an attack, and the action the attacker
wants instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from midas import config


@dataclass
class Scene:
    """One CHAI DriveLM sample."""

    index: int
    split: str
    image_paths: list[Path]
    # Benign action, established by CHAI by running the model unattacked and
    # taking the mode over 11 runs. Stored per model since GPT sees four options
    # and InternVL two.
    correct_internvl: str
    correct_gpt: str
    target: str  # the action the attacker wants, e.g. "a)"
    _position: dict[str, int]
    size: float

    @property
    def sign_origin(self) -> tuple[int, int]:
        """Sign anchor as cv2 (x, y).

        CHAI's field names are transposed relative to cv2: main.py builds
        ``initial_position = (y, x)`` from ``pos_x``/``pos_y`` and hands that
        straight to cv2, so ``pos_y`` is the horizontal coordinate. nuScenes
        frames are 1600x900, and the stored values (e.g. pos_y=1050, pos_x=450)
        only land inside the frame under this reading.
        """
        return (self._position["pos_y"], self._position["pos_x"])

    @property
    def attacked_camera(self) -> int:
        """Index of the frame the sign goes on. CHAI only ever attacks CAM_FRONT."""
        return 0

    @property
    def internvl_target(self) -> str:
        """The attacker's target under InternVL's two-option prompt.

        The stored ``target`` field is GPT-oriented: it is spread over all four
        options (a/b/c/d), but InternVL is only offered a) proceed and b) brake,
        and CHAI's released phrase dictionary only covers those two. Worse, for
        4 of the 23 scenes the stored target already equals the benign InternVL
        answer, which would score as a success with no attack applied.

        So for InternVL we take the attacker's goal to be the *opposite* of the
        benign action, which is CHAI's own framing of the threat: make the car
        "accelerate toward a pedestrian when the benign output would have been
        brake". Every scene becomes usable and none is degenerate.
        """
        return "a)" if self.correct_internvl == "b)" else "b)"

    def load_images(self) -> list[np.ndarray]:
        """Load all six camera frames as RGB arrays.

        Matches CHAI's DatabaseLoadCustom: cv2.imread then BGR->RGB.
        """
        images = []
        for path in self.image_paths:
            raw = cv2.imread(str(path))
            if raw is None:
                raise FileNotFoundError(f"could not read frame: {path}")
            images.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        return images


def load_scenes(split: str = "train") -> list[Scene]:
    """Load the Known (train) or Transferability (test) split."""
    if split == "train":
        json_path = config.CHAI_TRAIN_JSON
    elif split == "test":
        json_path = config.CHAI_TEST_JSON
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run scripts/setup_vendor.sh to clone CHAI's drivelm branch."
        )

    records = json.loads(json_path.read_text())
    scenes = []
    for i, rec in enumerate(records):
        # Paths in the JSON are relative to the CHAI repo root.
        paths = [config.VENDOR_CHAI / p for p in rec["image"]]
        scenes.append(
            Scene(
                index=i,
                split=split,
                image_paths=paths,
                correct_internvl=rec["correct_internvl"],
                correct_gpt=rec["correct_gpt"],
                target=rec["target"],
                _position=rec["position"],
                size=float(rec["size"]),
            )
        )
    return scenes


def resolve_target(scene: Scene, mode: str = "auto") -> str | None:
    """Pick the attacker's target action for a scene.

    ``auto``    opposite of the benign InternVL action (see Scene.internvl_target).
    ``dataset`` CHAI's stored target field; returns None when it is c)/d), which
                InternVL's two-option prompt cannot express.
    """
    if mode == "auto":
        return scene.internvl_target
    if mode == "dataset":
        return scene.target if scene.target in ANSWER_OPTIONS_INTERNVL else None
    raise ValueError(f"mode must be 'auto' or 'dataset', got {mode!r}")


ANSWER_OPTIONS_INTERNVL = ("a)", "b)")


def load_attack_dictionary() -> dict[str, list[str]]:
    """CHAI's released phrase dictionary, keyed by target action.

    Returns e.g. ``{"a)": ["Proceed", "Move Forward", ...], "b)": ["STOP", ...]}``.
    These are the phrases their dictionary-generation stage produced for InternVL,
    so we do not need to re-run that stage or call a hosted API.
    """
    records = json.loads(config.CHAI_ATTACK_DICT.read_text())
    return {rec["target"]: rec["attacks"]["English"] for rec in records}


def load_optimal_attack() -> dict:
    """CHAI's released optimiser output: sign colours and per-target phrase indices."""
    records = json.loads(config.CHAI_OPTIMAL.read_text())
    if not records:
        raise ValueError(f"{config.CHAI_OPTIMAL} is empty")
    return records[0]


def missing_frames(scenes: list[Scene]) -> list[Path]:
    """Frames referenced by the JSON but absent on disk."""
    return [p for s in scenes for p in s.image_paths if not p.exists()]
