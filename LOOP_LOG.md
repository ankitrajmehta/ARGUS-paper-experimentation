# Autonomous run log

Running unattended from 2026-11-13. Every iteration appends here: what was
tried, what the numbers were, and what was decided. `PROJECT_RECORD.md` stays
the curated record; this is the raw journal, including dead ends.

**Brief.** Run the DriveLM attack on a second model family at 100 scenes, check
it continuously, and if the result holds, implement ARGUS-style steering the way
the sibling project did (`vendor/argus-ref/repo`, branch `argus/exp/e3`). Every
check written down. Kaggle for all compute.

---

## Plan

| # | Step | State |
|---|---|---|
| 1 | Smoke test: Qwen2.5-VL-3B on 10 DriveLM scenes | in progress |
| 2 | Full run: 100 scenes, baseline + 4 framings | pending |
| 3 | Label agreement: DriveLM behaviour vs model vote | pending |
| 4 | ARGUS steering on this data, as released | pending |

**Decision rule for step 4.** Steering is only worth building if the attack
actually works here. The bar: attacked rate at least 10 points above the no-sign
baseline on 100 scenes, with the clean baseline in a believable range (not a
degenerate model answering one word, as InternVL2.5-4B did). If the attack does
not clear that, steering has nothing to defend and the finding is that the
attack does not transfer to Qwen.

---

## Iteration 1 — 2026-11-13 ~18:00

**State at start.** DriveLM downloaded (3.5 GB, 24,432 frames, 696 nuScenes
scenes). 100-scene set built and committed, 71 proceed / 29 brake, zero missing
frames. Qwen backend written and unit-tested but **never run against real
weights**. 68 tests passing.

**Decision: smoke test before the full run.** The Qwen path is new code. Every
previous Kaggle session lost time to a config that only failed once weights
loaded — the fp16 vision NaN, the device_map split, the tokenizer pin. Ten
scenes costs ~15 minutes and protects a two-hour run.

**Change made.** The notebook stage originally ran `midas-labels` to derive the
benign action. That is now unnecessary: the scenes carry labels from DriveLM's
recorded driver behaviour. Dropping it saves about an hour per run. The
label-agreement comparison is kept as its own step, because it is a result in
its own right rather than a prerequisite.

**Kaggle account.** The token in `~/.kaggle/access_token` had been replaced on
12 Nov with one for a different account (`ankitpokhrel101`, used by an unrelated
project). The first upload went there and *created* a new empty dataset rather
than versioning the existing one. Restored the original `ankitpokhrel` token,
backed the other one up to `access_token.spyder-backup` rather than discarding
it, and confirmed both prior kernels and the dataset are intact.

**Result.** Smoke kernel pushed 12:22 UTC: Qwen2.5-VL-3B, 10 DriveLM scenes,
baseline plus four framings. Dataset uploaded at 126.8 MB, of which 84 MB is
600 DriveLM frames.

**Two efficiency changes while waiting**, both baked in at notebook-generation
time:

* `MIDAS_NO_INTERNVL=1` skips the 16 GB InternVL download. A Qwen-only run was
  fetching weights it never loads.
* `MIDAS_DRIVELM_LIMIT` caps the scene count, so the same stage serves a smoke
  run and the full run.

**GitHub preparation.** A push would have been rejected: three blobs exceed
GitHub's 100 MB per-file hard limit (a 150 MB results archive and two activation
tensors at 100 MB and 50 MB). Checking what they actually contain turned up
something worth fixing on its own merits -- the 123 per-scene result JSONs total
**4.6 MB** and are the evidence behind every number in the report, yet they were
gitignored and survived only inside that zip. They are now tracked; the archives
and tensors are not, and stay on disk. The remaining obstacle is that the
oversized blobs are still in *history*, so a rewrite is needed before any push.
Deferred until the experiment result is in, rather than interleaving a history
rewrite with experiment monitoring.
