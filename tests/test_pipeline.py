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


def test_controls_are_not_instructions():
    """A control that reads as a command belongs in the attack set, not here."""
    from midas import controls

    offenders = [p for p in controls.BENIGN_PHRASES if controls.is_instructional(p)]
    assert offenders == [], f"instructional control phrases: {offenders}"


def test_controls_disjoint_from_attack_dictionary():
    from midas import controls

    attack = {p.lower() for phrases in data.load_attack_dictionary().values() for p in phrases}
    overlap = {p for p in controls.BENIGN_PHRASES if p.lower() in attack}
    assert overlap == set(), f"control phrases also used as attacks: {overlap}"


def test_controls_render_like_attacks(scenes):
    """Controls must differ from the attack only in what the sign says."""
    from midas import controls
    from midas.config import SignSpec

    scene = scenes[0]
    attack_sign = signs.build_sign(scene)
    control = SignSpec(
        controls.BENIGN_PHRASES[0],
        attack_sign.bg,
        attack_sign.fg,
        attack_sign.scale,
        attack_sign.origin,
    )
    base = scene.load_images()[scene.attacked_camera]
    rendered = signs.render(base, control)
    assert not np.array_equal(rendered, base)
    assert rendered.shape == base.shape


def test_defense_artifact_roundtrip(tmp_path):
    """save() and load() must agree on the path, including the .npz suffix."""
    from midas.steering import DefenseArtifact, LayerDefense

    h, d = 8, 3
    layers = [
        LayerDefense(
            layer=layer,
            direction=np.ones(h) / np.sqrt(h),
            subspace=np.eye(d, h),
            mean=np.zeros(h),
            scale=np.ones(h),
            margin=1.5,
            accuracy=0.9 - 0.1 * i,
        )
        for i, layer in enumerate((4, 6))
    ]
    art = DefenseArtifact(
        layers=layers,
        gate_layer=5,
        gate_w=np.ones(h),
        gate_b=0.5,
        gate_mean=np.zeros(h),
        gate_scale=np.ones(h),
    )
    written = art.save(tmp_path / "artifact")
    assert written.exists()
    for candidate in (tmp_path / "artifact", written, written.with_suffix(".json")):
        back = DefenseArtifact.load(candidate)
        assert back.layer_indices == [4, 6] and back.gate_layer == 5
        assert np.allclose(back.layers[0].direction, layers[0].direction)
        assert back.layers[0].margin == 1.5


def test_steerer_ranks_layers_by_probe_accuracy():
    """top_layers must take the strongest probes, not the lowest indices."""
    from midas.steering import DefenseArtifact, LayerDefense, SteerConfig, Steerer

    h = 4
    def layer(idx, acc):
        return LayerDefense(
            layer=idx,
            direction=np.ones(h) / np.sqrt(h),
            subspace=np.eye(1, h),
            mean=np.zeros(h),
            scale=np.ones(h),
            margin=1.0,
            accuracy=acc,
        )

    art = DefenseArtifact(
        layers=[layer(2, 0.60), layer(9, 0.95), layer(5, 0.80)],
        gate_layer=1,
        gate_w=np.ones(h),
        gate_b=0.0,
        gate_mean=np.zeros(h),
        gate_scale=np.ones(h),
    )
    steerer = Steerer(model=None, artifact=art, config=SteerConfig(top_layers=2))
    assert [d.layer for d in steerer.selected()] == [9, 5]


def test_clis_do_not_reference_removed_artifact_fields(tmp_path):
    """The multi-layer refactor renamed artifact.layer -> layer_indices.

    A stale attribute crashes only once a model has loaded, which on this
    hardware is minutes in; catch it statically instead.
    """
    import ast
    from pathlib import Path

    removed = {"layer", "direction", "subspace", "margin"}
    offenders = []
    for path in Path("src/midas/cli").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in removed
                and isinstance(node.value, ast.Name)
                and node.value.id == "artifact"
            ):
                offenders.append(f"{path.name}: artifact.{node.attr}")
    assert offenders == [], offenders


def test_flash_attention_gated_on_compute_capability():
    """FA2 needs sm_80+. A T4 is sm_75 and aborts at the first forward pass.

    The check must look at the capability, not at whether flash_attn imports:
    the package installs fine on Turing and only fails at runtime.
    """
    import ast
    from pathlib import Path

    source = Path("src/midas/models/internvl.py").read_text()
    assert "def supports_flash_attention" in source
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "supports_flash_attention"
    )
    body = ast.unparse(fn)
    assert "get_device_capability" in body, "must inspect compute capability"
    assert ">= 8" in body, "must require sm_80 or newer"


def test_kaggle_notebook_is_valid_and_configured():
    """The notebook is generated; a broken one wastes a GPU session to discover."""
    import json
    from pathlib import Path

    nb_path = Path("kaggle/midas_kaggle.ipynb")
    if not nb_path.exists():
        pytest.skip("notebook not generated")

    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    sources = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(sources)

    # Every GPU invocation must shard across both T4s; one T4 cannot hold 8B.
    assert joined.count("--device-map auto") >= 4, "GPU commands must shard across both T4s"
    # 0.2.2 breaks InternVL's tokenizer; 0.1.99 has no Python 3.13 wheel.
    assert "sentencepiece==0.2.1" in joined, "the tokenizer pin must be installed"
    assert "transformers>=4.37,<4.50" in joined, "transformers must be capped below 4.50"


def test_bfloat16_gated_on_compute_capability_not_torch_helper():
    """torch.cuda.is_bf16_supported() returns True on a T4 (sm_75).

    Recent PyTorch counts emulated bf16 as supported, but Turing has no bf16
    tensor cores, so trusting that helper picks a far slower path than float16.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/midas/models/internvl.py").read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_dtype"
    )
    body = ast.unparse(fn)
    assert "is_bf16_supported" not in body, "must not trust the torch helper"
    assert "get_device_capability" in body and ">= 8" in body


def test_kaggle_notebook_cells_are_valid_python():
    """Every code cell must parse.

    A syntax error here costs a whole Kaggle session to discover: papermill runs
    the notebook headlessly and only reports the failure once the kernel exits.
    One such break came from an escaping level lost while generating the
    notebook, which turned a \\n inside a generated string into a real newline.
    """
    import ast
    import json
    from pathlib import Path

    nb_path = Path("kaggle/midas_kaggle.ipynb")
    if not nb_path.exists():
        pytest.skip("notebook not generated")

    failures = []
    for i, cell in enumerate(json.loads(nb_path.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])

        # Rejoin backslash continuations, then blank IPython shell/magic lines:
        # those are not Python and ast cannot parse them.
        joined, pending = [], ""
        for line in source.splitlines():
            current = pending + line
            if current.rstrip().endswith("\\"):
                pending = current.rstrip()[:-1]
                continue
            pending = ""
            joined.append(current)
        cleaned = "\n".join(
            "pass" if l.lstrip().startswith(("!", "%")) else l for l in joined
        )

        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            failures.append(f"cell {i}: {exc.msg} (line {exc.lineno})")

    assert not failures, failures


def test_internvl_device_map_keeps_codependent_modules_together():
    """device_map="auto" splits by size and breaks InternVL's forward.

    The model scatters vision features into text embeddings with a boolean mask.
    If the vision tower and the embedding land on different GPUs the scatter
    fails with "indices should be either on cpu or on the same device as the
    indexed tensor". These modules must therefore share GPU 0.
    """
    from unittest.mock import patch

    from midas.models.internvl import build_device_map

    class FakeConfig:
        class llm_config:
            num_hidden_layers = 32

    with patch("transformers.AutoConfig.from_pretrained", return_value=FakeConfig):
        device_map = build_device_map("fake/model", n_gpus=2)

    for module in ("vision_model", "mlp1", "language_model.model.tok_embeddings",
                   "language_model.output"):
        assert device_map[module] == 0, f"{module} must sit on GPU 0"

    placed = [k for k in device_map if ".layers." in k]
    assert len(placed) == 32, "every decoder layer must be placed"
    assert set(device_map.values()) == {0, 1}, "both GPUs must be used"

    # GPU 0 also holds the vision tower and output head, then needs room for the
    # eager attention matrix, so it must take the smaller share of layers.
    on_gpu0 = sum(1 for k in placed if device_map[k] == 0)
    assert on_gpu0 < len(placed) / 2, f"GPU 0 given too many layers: {on_gpu0}/32"


def test_chai_rule_counts_replies_naming_both_actions():
    """InternVL2.5-8B often answers per camera, naming both actions.

    Our strict parser records that as unreadable; CHAI's rule credits it to
    whichever target it is tested against. Both must be reported, or our numbers
    cannot be compared with the paper's.
    """
    both = "**CAM_FRONT**: Brake\n**CAM_BACK**: Proceed"
    assert drivelm.parse_action(both) is None
    assert drivelm.chai_counts_as_hit(both, "a)") is True
    assert drivelm.chai_counts_as_hit(both, "b)") is True

    clear = "Proceed"
    assert drivelm.parse_action(clear) == "a)"
    assert drivelm.chai_counts_as_hit(clear, "a)") is True
    assert drivelm.chai_counts_as_hit(clear, "b)") is False
