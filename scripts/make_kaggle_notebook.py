"""Generate the Kaggle notebook that runs MIDAS on two T4 GPUs.

Kaggle gives 2x Tesla T4 (16 GB each, 32 GB combined), 29 GB RAM and a 12-hour
session limit. Two Turing properties shape every cell below:

* No bfloat16 -- the model must load in float16.
* No FlashAttention-2 (it needs compute capability 8.0; T4 is 7.5) -- attention
  runs eager, so the full score matrix is materialised and memory scales with
  the square of the sequence length.

The code arrives as an attached dataset rather than a git clone, so no GitHub
repository is required.

Written as a script so the notebook stays in version control and is regenerated
rather than edited by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL = "OpenGVLab/InternVL2_5-8B"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


# ---------------------------------------------------------------- intro
md(f"""
# MIDAS on Kaggle — CHAI attack against InternVL2.5-8B

Reproduces the CHAI visual sign-injection attack and evaluates the
activation-steering defense, at the model scale CHAI themselves used.

**Before running, set the session up:**

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **On** (needs phone verification on your Kaggle account) |
| Persistence | Variables and files |

Two T4s give 32 GB of VRAM combined. The 8B model needs ~16 GB in float16, so it
is sharded across both with `--device-map auto`.

**Expected cost:** roughly 2–3 GPU-hours for the full attack evaluation, against
a weekly quota of about 30.
""")

# ---------------------------------------------------------------- step 1
md("""
## 1. Check the hardware

Run this first. If it reports anything other than two T4s, stop and fix the
accelerator setting — everything below assumes this layout.
""")

code('''
import subprocess, torch, sys

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                      "--format=csv"], capture_output=True, text=True).stdout)

print("python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("cuda        :", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    total = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  gpu {i}: {torch.cuda.get_device_name(i)}  "
          f"sm_{major}{minor}  {total:.1f} GB")

# Gate on compute capability, as midas.models.internvl.resolve_dtype does.
# torch.cuda.is_bf16_supported() reports True on a T4 because recent PyTorch
# counts emulated bf16, but Turing has no bf16 tensor cores and the emulated
# path is slower than float16.
cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
torch_claim = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
bf16 = cc >= 8
fa2 = bf16
print(f"torch bf16 claim   : {torch_claim}  (counts emulation -- not what we use)")
print(f"native bfloat16    : {bf16}   -> dtype will be {'bfloat16' if bf16 else 'float16'}")
print(f"FlashAttention-2 ok: {fa2}   -> attention will run {'flash' if fa2 else 'eager'}")
if not fa2:
    print("  (expected on T4: sm_75 is Turing, FA2 needs sm_80+. Eager attention")
    print("   uses much more memory, so keep --max-tiles modest at first.)")
''')

# ---------------------------------------------------------------- step 2
md("""
## 2. Install dependencies

Kaggle preinstalls PyTorch built against its own CUDA, so **do not reinstall
torch** — that risks breaking GPU support and wastes several minutes.

Two pins are load-bearing and cost us hours to diagnose locally:

- **`sentencepiece==0.2.1`.** Version 0.2.2 rejects InternVL's `tokenizer.model`
  as malformed even though its checksum matches the hub; `AutoTokenizer` swallows
  the error and returns `False`, which surfaces much later as `'bool' object has
  no attribute 'convert_tokens_to_ids'`. 0.1.99 also works but has no Python 3.13
  wheel, and Kaggle now runs 3.13 — pinning it makes pip try a source build,
  which fails and aborts the whole install.
- **`transformers<4.50`.** InternVL2.5's remote code does not inherit from
  `GenerationMixin`, which 4.50 made mandatory. Kaggle ships 5.x, so this is a
  real downgrade and takes a minute.

The install is checked below rather than assumed: a failed pip run leaves the
preinstalled versions in place, and the notebook would then fail much later for
reasons that look unrelated.
""")

code('''
%pip install -q "transformers>=4.37,<4.50" "sentencepiece==0.2.1" protobuf \\
    "accelerate>=0.26" timm einops "numpy<2" opencv-python-headless scikit-learn pytest

import importlib, sentencepiece, transformers, accelerate
for mod in (transformers, sentencepiece, accelerate):
    importlib.reload(mod)

print("transformers :", transformers.__version__)
print("sentencepiece:", sentencepiece.__version__)
print("accelerate   :", accelerate.__version__)

major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
assert (major, minor) < (4, 50), (
    f"transformers {transformers.__version__} is too new; InternVL2.5's remote "
    "code needs <4.50. The pip step above must have failed -- scroll up."
)
assert sentencepiece.__version__ == "0.2.1", (
    f"sentencepiece {sentencepiece.__version__} installed, need 0.2.1. "
    "0.2.2 cannot read InternVL's tokenizer and fails confusingly later."
)
print("version checks passed")
''')

# ---------------------------------------------------------------- step 3
md("""
## 3. Load the code

The code and CHAI's released artefacts arrive as an attached Kaggle Dataset, not
a git clone, so this notebook needs no GitHub repository and no access token.
Upload it with `python scripts/kaggle_deploy.py upload`.

The dataset is read-only, so it is copied into the working directory before the
editable install.
""")

code('''
import os, shutil, subprocess, glob
from pathlib import Path

INPUT = Path("/kaggle/input")
print("contents of /kaggle/input:")
if INPUT.exists():
    for entry in sorted(INPUT.iterdir()):
        inner = sorted(p.name for p in entry.iterdir())[:12] if entry.is_dir() else []
        print(f"  {entry.name}/  -> {inner}")
else:
    print("  (missing entirely -- no dataset is attached to this kernel)")

# Search recursively: a dataset may mount its payload one level deeper than
# expected depending on how the archive was packed.
candidates = glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
assert candidates, (
    "MIDAS source not found under /kaggle/input. Attach the dataset via "
    "Add Data -> Your Datasets, or re-run scripts/kaggle_deploy.py upload. "
    "If the dataset was created moments before this kernel started, it may "
    "still have been processing -- wait for `kaggle datasets status` to read "
    "ready, then push again."
)
SOURCE = Path(candidates[0]).parent
# Not "/kaggle/working/midas": Kaggle puts /kaggle/working on sys.path, so a
# directory of that name shadows the installed package and `import midas`
# resolves to an empty namespace package with __file__ None.
WORK = Path("/kaggle/working/midas_src")
print("source dataset:", SOURCE)

# /kaggle/input is read-only and `pip install -e` needs to write egg-info.
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(SOURCE, WORK)
os.chdir(WORK)

chai = WORK / "vendor" / "chai"
assert (chai / "training.json").exists(), (
    f"CHAI artefacts missing from the dataset (looked in {chai}); "
    "re-run the upload step"
)
frames = len(glob.glob(str(chai / "DriveLM_dataset/**/*.jpg"), recursive=True))
print(f"CHAI frames present: {frames} (expect 184)")
assert frames == 184, "frame count wrong -- the upload was incomplete"

%pip install -q -e .

import importlib, midas
importlib.reload(midas)
assert midas.__file__, (
    "`midas` imported as a namespace package, so something on sys.path shadows "
    "it -- check for a directory of that name under /kaggle/working"
)
print("midas installed from", midas.__file__)
''')

# ---------------------------------------------------------------- step 4
md("""
## 4. Verify the data before spending GPU time

These checks are CPU-only and take seconds. They catch a bad clone before a
multi-hour run does.
""")

code('''
!python -m pytest tests -q

from midas import data
for split in ("train", "test"):
    scenes = data.load_scenes(split)
    missing = data.missing_frames(scenes)
    print(f"{split}: {len(scenes)} scenes, {len(missing)} missing frames")

from midas.tasks import drivelm
diffs = drivelm.verify_matches_upstream()
print("prompt matches CHAI source:", "yes" if not diffs else diffs)
''')

# ---------------------------------------------------------------- step 5
md(f"""
## 5. Download the model

`{MODEL}` is about 16 GB. It goes to the HuggingFace cache rather than
`/kaggle/working`, which only has 20 GB and must also hold results.

To skip this download on later sessions, upload the cache once as a Kaggle
Dataset and attach it.
""")

code(f'''
from huggingface_hub import snapshot_download
import time

t0 = time.time()
path = snapshot_download(
    "{MODEL}",
    allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model"],
)
print(f"downloaded in {{(time.time() - t0) / 60:.1f}} min -> {{path}}")
!du -sh {{path}}
''')

# ---------------------------------------------------------------- step 6
md("""
## 6. Smoke test — one scene

Never start a multi-hour sweep without this. It confirms the model loads across
both GPUs, the tokenizer works, and the prompt produces a parseable action.

Watch the printed `device=... dtype=... flash_attn=...` line: on a T4 it must say
`float16` and `flash_attn=False`.
""")

code(f'''
!midas-eval --split train --baseline --limit 1 \\
    --model {MODEL} --device-map auto --max-tiles 6 \\
    --out /kaggle/working/results/smoke
''')

# ---------------------------------------------------------------- step 7
md("""
## 7. Check memory headroom before raising the tile count

CHAI render at 12 tiles per frame (9 on these images). Eager attention makes that
roughly four times the attention memory of 6 tiles, so confirm there is room
rather than discovering it two hours in.

If this reports comfortable headroom, use `--max-tiles 12` below — that is the
faithful setting and the one to report.
""")

code('''
import torch

for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"gpu {i}: {free/1e9:5.1f} GB free of {total/1e9:5.1f} GB")

heads, seq6, seq12 = 32, 3300, 6400          # InternLM2.5-7B attention heads
for name, seq in (("6 tiles", seq6), ("12 tiles", seq12)):
    gb = seq * seq * heads * 2 / 1e9
    print(f"{name:9s} seq~{seq:5d} -> eager attention matrix {gb:5.2f} GB per layer")
''')

# ---------------------------------------------------------------- step 8
md("""
## 8. The main run — baseline and attack

This is the headline result. It runs the model on clean frames, then on frames
carrying CHAI's optimised sign, and prints the comparison.

**What to expect, from CHAI's paper:** a no-attack rate near 29% and an attacked
rate near 55% on the train (Known) split. Our 2B results were 26.7% and 73.3%.

Each result is appended to a `.jsonl` as it completes, so an interrupted session
keeps every scene it finished.
""")

code(f'''
!midas-eval --split train --both \\
    --model {MODEL} --device-map auto --max-tiles 12 --save-long \\
    --out /kaggle/working/results/runs_8b
''')

code(f'''
!midas-eval --split test --both \\
    --model {MODEL} --device-map auto --max-tiles 12 --save-long \\
    --out /kaggle/working/results/runs_8b
''')

# ---------------------------------------------------------------- step 9
md("""
## 9. Read the transcripts

The aggregate rate hides the mechanism. On our 2B model the attacked transcript
still identified the red light correctly and then concluded "should proceed, as
the traffic light is red" — perception intact, inference hijacked.

Check whether the 8B does the same. If it does, that is much stronger evidence
than the 2B version.
""")

code('''
import json, glob

runs = {}
for f in glob.glob("/kaggle/working/results/runs_8b/*train*.json"):
    d = json.load(open(f))
    runs[d["label"]] = {r["scene_index"]: r for r in d["results"]}

for label in ("baseline", "attack"):
    r = runs.get(label, {}).get(0)
    if not r:
        continue
    print(f"{'=' * 70}\\nscene 0 — {label}  (sign: {r['sign_text']})")
    print(f"target={r['target']}  benign={r['benign_label']}  ->  {r['action']}")
    print(f"{'-' * 70}\\n{r['long'][-1800:]}")
    print(f"\\nSUMMARY TURN: {r['short']}")
''')

# ---------------------------------------------------------------- step 10
md("""
## 10. The defense pipeline (optional, a further 2–3 GPU-hours)

Only worth running once the attack above reproduces. Activation extraction is
cheap per sample (one forward pass, no generation); the sweep is the expensive
part.
""")

code(f'''
# Attack and clean activations, plus benign-text controls
!midas-activations --split train --phrases 5 --controls 5 \\
    --model {MODEL} --device-map auto --max-tiles 12 \\
    --out /kaggle/working/results/activations

# The gate must be fitted on the open-turn state it sees at inference,
# not on an answer token that does not exist yet.
!midas-activations --split train --prompt-only --phrases 5 --controls 5 \\
    --model {MODEL} --device-map auto --max-tiles 12 \\
    --out /kaggle/working/results/activations
''')

code('''
import glob, subprocess

acts = sorted(glob.glob("/kaggle/working/results/activations/*.npz"))
main = [a for a in acts if "promptonly" not in a]
gate = [a for a in acts if "promptonly" in a]
print("steering activations:", main)
print("gate activations    :", gate)
assert main and gate, "activation extraction did not produce both sweeps"

# subprocess rather than ! magic: the file lists are Python values, and brace
# substitution inside a line-continued shell command is easy to get wrong.
subprocess.run(
    ["midas-probes", *main, "--out", "/kaggle/working/results/probes_8b.json"],
    check=True,
)
''')

code('''
subprocess.run(
    [
        "midas-fit-defense", *main,
        "--gate-activations", *gate,
        "--top-n", "5",
        "--out", "/kaggle/working/results/defense/train_8b",
    ],
    check=True,
)
''')

# ---------------------------------------------------------------- step 11
md("""
## 11. Package the results

`/kaggle/working` persists between sessions, but download the archive anyway so
the results can be committed to the repository alongside the 2B numbers.
""")

code('''
import shutil, json, glob

shutil.make_archive("/kaggle/working/midas_8b_results", "zip",
                    "/kaggle/working/results")
print("archive written")
!du -sh /kaggle/working/midas_8b_results.zip

for f in sorted(glob.glob("/kaggle/working/results/runs_8b/*.json")):
    d = json.load(open(f))
    s = d["summary"]
    print(f"{d['label']:9s} {d['split']:6s} n={s['n']:<3} "
          f"ASR={s['asr']:6.1%}  benign={s['benign_accuracy']:6.1%}")
''')


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": text.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "kaggle" / "midas_kaggle.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1))
    print(f"wrote {out}  ({len(CELLS)} cells)")
    print("\nDeploy with: python scripts/kaggle_deploy.py upload && ... run")
