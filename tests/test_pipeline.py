"""Sanity checks that run without a GPU.

Catch the things that would silently corrupt results: a drifted prompt, a sign
rendered off-frame, a scene whose attack target equals its benign answer.

    .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import numpy as np
import pytest

from midas import data, signs
from midas.config import AttackConfig
from midas.evaluate import SceneResult, summarise
from midas.tasks import drivelm

SPLITS = ["train", "test"]


@pytest.fixture(scope="module", params=SPLITS)
def scenes(request):
    return data.load_scenes(request.param)


def test_frames_present(scenes):
    assert data.missing_frames(scenes) == []


def test_prompt_matches_upstream():
    assert drivelm.verify_matches_upstream() == []


def test_question_chain_shape():
    # 10, not 11: upstream omits a comma so two questions share one string.
    assert len(drivelm.QUESTIONS) == 10
    assert "a) proceed, b) brake" in drivelm.QUESTIONS[-1]


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("a) proceed", "a)"),
        ("brake", "b)"),
        ("The ego vehicle should proceed.", "a)"),
        ("proceed or brake", None),  # ambiguous, not an attacker win
        ("no idea", None),
    ],
)
def test_parse_action(answer, expected):
    assert drivelm.parse_action(answer) == expected


def test_target_is_never_the_benign_action(scenes):
    """An attack that asks for the action the model already picks proves nothing."""
    for scene in scenes:
        assert data.resolve_target(scene, "auto") != scene.correct_internvl


def test_sign_renders_inside_frame(scenes):
    """Every sign must be fully visible, else the attack is weaker than reported."""
    for scene in scenes:
        images = scene.load_images()
        clean = images[0].copy()
        attacked, _ = signs.attack_scene(scene, images=images)
        changed = np.any(attacked[0] != clean, axis=2)

        assert changed.any(), f"scene {scene.index}: sign changed no pixels"
        ys, xs = np.where(changed)
        h, w = changed.shape
        assert xs.min() > 0 and xs.max() < w - 1, f"scene {scene.index}: sign clipped horizontally"
        assert ys.min() > 0 and ys.max() < h - 1, f"scene {scene.index}: sign clipped vertically"


def test_only_front_camera_is_modified():
    scene = data.load_scenes("train")[0]
    images = scene.load_images()
    before = [im.copy() for im in images]
    attacked, _ = signs.attack_scene(scene, images=images)
    for i, (a, b) in enumerate(zip(attacked, before)):
        if i == scene.attacked_camera:
            assert not np.array_equal(a, b)
        else:
            assert np.array_equal(a, b), f"camera {i} should be untouched"


def test_background_mode_differs():
    """The two modes must not silently collapse; 'code' is red, 'json' is green."""
    scene = data.load_scenes("train")[0]
    code = signs.build_sign(scene, cfg=AttackConfig(background_mode="code"))
    js = signs.build_sign(scene, cfg=AttackConfig(background_mode="json"))
    assert code.bg != js.bg
    assert code.bg == (255, 0, 15)


def test_unreadable_counts_against_asr():
    results = [
        SceneResult(0, "train", True, "a)", "b)", "a)", "proceed"),
        SceneResult(1, "train", True, "a)", "b)", None, "???"),
    ]
    s = summarise(results)
    assert s.asr == 0.5
    assert s.unreadable_rate == 0.5
