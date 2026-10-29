# Running MIDAS on Kaggle

`midas_kaggle.ipynb` runs the CHAI attack and the steering defense against
**InternVL2.5-8B** — the model CHAI themselves target — on Kaggle's free GPUs.
Regenerate it with `python scripts/make_kaggle_notebook.py` rather than editing
it by hand.

## What you need to do first

Only two things, and neither involves GitHub.

1. **Get a Kaggle API token.** kaggle.com → Settings → API → *Create New Token*.
   Then:

   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Verify your phone number on Kaggle.** The Internet toggle stays disabled
   until you do, and the notebook needs internet to install the pinned packages
   and download the model.

## Deploying without GitHub

The code travels as a private Kaggle Dataset. CHAI's released artefacts — the
184 frames, the phrase dictionary, the sign renderer — ride along in the same
dataset, so the run does not depend on their GitHub repository either.

```bash
python scripts/kaggle_deploy.py upload    # code + CHAI artefacts -> private dataset
python scripts/kaggle_deploy.py run       # push the notebook and start it on T4 x2
python scripts/kaggle_deploy.py status    # poll
python scripts/kaggle_deploy.py fetch     # download results
```

`run` submits the notebook in save-and-run mode, so it executes on Kaggle's
infrastructure and **keeps going after you close your laptop**. That suits a
2–3 hour evaluation better than an interactive session.

Re-running `upload` after a code change creates a new dataset version; the
notebook always picks up the latest.

## Session settings

| Setting | Value | Why |
|---|---|---|
| Accelerator | **GPU T4 x2** | 16 GB each; the 8B model needs ~16 GB and is sharded across both |
| Internet | **On** | pip and the HuggingFace download |
| Persistence | Variables and files | keeps `/kaggle/working` between sessions |

`kaggle_deploy.py run` sets all three automatically through the kernel metadata;
they only need setting by hand if you open the notebook in the web editor.

Kaggle allows roughly 30 GPU-hours per week and 12 hours per session. Budget
2–3 hours for the attack evaluation and a further 2–3 for the defense pipeline.

## Two things about the T4 that shape the run

The T4 is a Turing card (compute capability 7.5), and that has two consequences
the code handles automatically:

- **No bfloat16.** Turing has no native bf16, so `--dtype auto` resolves to
  float16. Forcing bfloat16 would be slow or wrong.
- **No FlashAttention-2.** FA2 kernels require compute capability 8.0 or newer.
  On a T4 they abort with *"FlashAttention only supports Ampere GPUs or newer"*.
  `supports_flash_attention()` checks the capability rather than whether the
  package imports, because the package installs fine on Turing and only fails at
  the first forward pass.

The second point has a real cost. Without FA2, attention runs eager and
materialises the full score matrix, so memory grows with the square of the
sequence length. At CHAI's 12-tile setting the sequence is ~6400 tokens, which is
~2.6 GB per layer for this model. Cell 7 measures the headroom before you commit
to a long run; this exact issue made the 12-tile setting impossible on the
development laptop.

## Order of operations

| Cell | Step | Cost |
|---|---|---|
| 1 | Check the GPUs | seconds |
| 2 | Install dependencies | ~2 min |
| 3 | Clone this repo and CHAI's artefacts | ~1 min |
| 4 | Run the test suite and verify the data | seconds |
| 5 | Download InternVL2.5-8B (~16 GB) | 5–15 min |
| 6 | **Smoke test on one scene** | ~2 min |
| 7 | Measure memory headroom | seconds |
| 8 | Baseline and attack, both splits | 2–3 h |
| 9 | Print model transcripts | seconds |
| 10 | Defense pipeline (optional) | 2–3 h |
| 11 | Package results | seconds |

Do not skip cells 6 and 7. A failure there costs two minutes; the same failure
discovered inside cell 8 costs hours.

## What to compare against

CHAI report, for InternVL2.5-8B on the DriveLM task:

| | Known images | Transferability images |
|---|---|---|
| No attack | 29.4% | 33.1% |
| CHAI sign | 54.7% | 51.3% |

Our 2B results were 26.7% → 73.3% on the train split. The test split was
inconclusive at n=8 because the 2B disagreed with CHAI's recorded labels on 5 of
8 scenes; the 8B should resolve that.

## Avoiding the 16 GB download every session

After the first successful run, save the HuggingFace cache as a Kaggle Dataset
and attach it to later sessions. Point `HF_HOME` at the attached path and
`snapshot_download` becomes a no-op.

## Known failure modes

| Symptom | Cause |
|---|---|
| `MIDAS source dataset not attached` | run the `upload` step, and wait for Kaggle to finish processing it |
| `'bool' object has no attribute 'convert_tokens_to_ids'` | `sentencepiece` is not 0.1.99; `AutoTokenizer` returned `False` instead of raising |
| `FlashAttention only supports Ampere GPUs or newer` | something forced `use_flash_attn=True`; the capability check should prevent this |
| `PreTrainedModel will NOT inherit from GenerationMixin` | `transformers` is 4.50 or newer |
| CUDA out of memory in cell 8 | lower `--max-tiles` to 6 and note the deviation in results |
| Model loads onto one GPU only | `--device-map auto` was omitted |
