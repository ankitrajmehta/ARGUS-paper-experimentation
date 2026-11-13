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

**Pushed to the team repo.** `ankitrajmehta/ARGUS-paper-experimentation`,
branch **`chai/main`** -- mirroring their `argus/main` convention so nothing of
theirs is touched. (First pushed as `midas/main`; renamed on request, since the
work is a CHAI reproduction and MIDAS was only the proposal's name.) Their eight `argus/*` branches and `main` are unchanged, and
`main` here now tracks `team/midas/main`.

Three checks before pushing, since the repo is **public**:

1. *Write access.* The API reports `push: true` for this account, so it is a
   real collaborator permission rather than a fork-and-PR situation.
2. *Credential scan* over the working tree **and all 55 commits** of history:
   no Kaggle, Hugging Face, GitHub or AWS tokens. The only `KGAT_` matches are
   the literal placeholder `KGAT_...` in help text.
3. *Size.* GitHub rejects any push containing a blob over 100 MB anywhere in
   history, and three exceeded it. `git filter-repo` dropped every `*.zip` and
   `*.npz`, taking `.git` from **476 MB to 62 MB**. A full copy was taken to
   `/tmp/midas_git_backup` first. Verified afterwards: 55 commits intact, every
   key file present, 122 result JSONs tracked, 68 tests passing.

The archives and tensors remain on local disk; only git stopped tracking them.

## ARGUS steering — design, read off the sibling repo's source

Recorded before building, because the sign convention is the one detail that
silently inverts the result: their own layer sweep measured AIA 53% steering
toward the attack against 6% steering away, from the same vector.

**Label convention** (`ARGUS/src/argus/probes.py:32-45`). The edit probe is
trained on pairs from one scene: `first_activation` is the **user's** answer and
carries label 0; `second_activation` is the **attacker's** answer and carries
label 1. So the probe weight `w` points *toward* attacker-following, and
`normalized_probe_vector` returns `-coef/‖coef‖` with the comment "Defense
direction points from attacker-following back to user-following."

Our `follows_injection = 1` means following the sign, i.e. the attacker. **Same
convention**, so `probes.fit_direction` is directly reusable and the defense
direction is `−w/‖w‖`.

**Two probes, not one** (`defense.py`):

| Probe | Read at | Used for |
|---|---|---|
| detection | last prompt token, step 0 | gate: steer at all or not |
| edit | forced answer token | basis for the step-0 vector |
| final_steering | forced answer token | the Eq. 7 adaptive steps |

**The intervention is two-phase**, which is where our earlier attempt differed:

* **step 0** — `h[:, -1, :] += vector`, at full magnitude, **no adaptive scaling
  and no cap**. Their run measured ‖vector‖ ≈ 9.9.
* **steps ≥ 1** — `alpha = relu((w·h + b + ε)/‖w‖²)`, step `alpha · (−w/‖w‖)`.
  Their run measured ≈ 1.1, and with ‖w‖ ≈ 46 each step closes only ~2% of the
  score.
* **last token position only**, never the whole sequence.

So the first token does nearly all the work, and their own note says attacks
look "all-or-nothing from the first token". Our section 12 result -- 136,725
activations modified, 1.6 nats of margin shift, attack rate unchanged at exactly
40.0% -- applied *moderate capped* steps at *every* position and token. That is
close to the opposite of this design, and is the most likely reason it was
inert.

**The trained vector is a mixture** of the edit probe and three auxiliary
probes, `strength · softmax(a) · basis` with four learned weights per layer. The
sibling project measured it ending ≈99% on the edit probe, within 0.65° of it,
so for a first implementation `strength · (−w_edit/‖w_edit‖)` is faithful to
what the released code actually produces. `strength` is whatever the preliminary
layer/alpha search selects; theirs chose layers [13, 12] at α = 10.

---

## Iteration 2 — 2026-11-13 13:01 UTC — smoke test passed

**Qwen2.5-VL-3B runs clean.** 26.6 min for 10 scenes x 5 conditions. No
traceback, and **no NaN** -- the float32 vision tower did its job on frames
carrying rendered text, which was the failure mode the sibling project warned
about.

| Condition | ASR | vs clean | unreadable |
|---|---|---|---|
| no sign | 30.0% | -- | 0.0% |
| plain (CHAI's wording) | **90.0%** | **+60.0** | 0.0% |
| authority | 70.0% | +40.0 | 0.0% |
| override | 60.0% | +30.0 | 0.0% |
| urgency | 50.0% | +20.0 | 0.0% |

**The attack transfers, and far more strongly than on InternVL** -- +60 points
against +13.3 for the same CHAI sign on InternVL2.5-8B. It is not an InternVL
quirk.

**0% unreadable in every condition**, against 13-27% on the InternVL 8B. Qwen
answers the two-option question cleanly, so the strict and CHAI scoring rules
agree exactly. That removes the scoring-rule ambiguity that cost up to 13 points
of spread in the earlier work.

**The style ordering is reversed from InternVL** -- plain best, urgency worst,
where on InternVL2.5-2B it was urgency best, plain worst. **At n = 10 one scene
is 10 points and the interval is +/-31, so this ordering is not evidence of
anything yet.** It is a question for the 100-scene run, and if it holds it is
interesting: it would mean the best phrasing is model-specific, which weakens
the idea of a single universal wording.

**Decision: proceed to the full run.** The bar was 10 points over baseline; this
is 60.

**Launched 13:03 UTC** -- 100 scenes, baseline plus four framings, plus
activation collection for the steering work. Estimated ~3.5 h from the smoke
timing (~24 s per scene-evaluation, 500 evaluations, plus setup).

---

## Iteration 3 — 2026-11-13 14:12 UTC — steering built while the run is in flight

Kernel v16 still `RUNNING` at 1 h 09 m. Nothing to decide until it lands, so the
hour went on the work that was going to block the next step anyway: the steering
path did not exist for Qwen at all (`QwenVLDriveLM.ask` raised
`NotImplementedError`). Now built, with the ARGUS schedule from the design above.

**What was added.**

* `SteerConfig.schedule` -- `"uniform"` (what section 12 measured) or
  `"argus"`. The argus path refuses `all_positions`, because their hook indexes
  `[:, -1, :]` unconditionally and conflating the two is how the section 12
  result became uninterpretable.
* `Steerer._steer_argus` -- step 0 adds the vector at **full magnitude with no
  cap**, steps >= 1 take Eq. 7 steps. The step counter resets on every prefill,
  so each of the three generations in `ask()` gets its own step 0 rather than
  the chain-of-thought turn taking the big step and the summarise turn -- the
  one the action is parsed from -- getting nothing.
* `LayerDefense.probe_norm` and `.bias`. Eq. 7 is
  `relu((w.h + b + eps)/||w||^2)` and `direction` has `||w||` divided out, so
  without these the step size cannot be reconstructed. `midas-fit-defense` now
  records them; artifacts fitted earlier default to 1.0 and `midas-eval` prints
  a warning rather than silently taking steps ~`||w||` times too large.
* `midas-eval --schedule/--first-step-norm/--epsilon`.

**Two defects found while building, both of the silent kind.**

1. **The layer offset was wrong for Qwen.** `Steerer.__enter__` hooked decoder
   block `layer - 1`, correct for InternVL because its `activations()` reads
   `out.hidden_states`, whose index 0 is the embedding output. The Qwen runner
   hooks `decoder[i]` directly, so its layer *i* **is** block *i*. Reusing the
   InternVL offset would have steered a neighbouring layer and produced a
   plausible null result with no error anywhere. Each runner now declares
   `ACTIVATION_LAYER_OFFSET` and the Steerer reads it.

2. **`--dry-run --defend` crashed** -- `_StubModel.ask()` took no `steerer`.
   The dry run exists to catch wiring faults on a laptop instead of ten minutes
   into a GPU session, and the defense was the one path it could not check. The
   stub now runs the gate and records the telemetry.

**And the test that should have caught the first one was weaker than its own
docstring.** `test_each_stage_can_run_without_the_others` claimed "no stage may
depend on a name another one assigns" but compared against a hardcoded set of
four names -- so it only caught leaks someone had already thought of. The new
steering stage leaked `DRIVELM_SCENES` and `DRIVELM_MODEL` straight past it.
Rewritten to flag any load of a name the notebook never binds; it then failed on
exactly those two and nothing else, which is also evidence every other stage is
clean. Both names moved to the shared setup cell.

**Forced-choice scoring now refuses the argus schedule.** That path is a single
prefill over a complete sequence, so the answer tokens' logits come from
positions *before* the one being steered: the log-probs would come back
byte-identical and read as "the defense does nothing" -- the same shape as the
section 12 result. It raises instead.

78 -> 80 tests. The schedule difference is pinned by assertion, not by
inspection: step 0 has the stated norm with a 0.25 cap configured, step 1 moves
exactly `1/||w||` of the score, and `relu` leaves an already-safe state alone.

**Notebook stage `argus` added.** Fits the probes and gate from the activations
the running kernel is collecting, then compares six configurations at 40 scenes
each: no defense, argus at first-step norms 5/10/20, **uniform at the same
strength** as the ablation that isolates the schedule, and argus ungated to show
how much of the effect is the gate. Both baseline and attacked conditions run
under each, because a defense that simply brakes everywhere scores a perfect
attack rate and wrecks clean accuracy -- the pair has to be read together. It
prints its own verdict, including the case where the argus schedule is *also*
inert, which would mean the suspect is the direction or the layer choice rather
than the step shape.

---

## Iteration 4 — 2026-11-13 15:13 UTC — quiet hold, one gap closed

Kernel v16 `RUNNING` at 2 h 10 m of an estimated ~3.5 h. On track; nothing to
decide.

**Gap found in the handover path.** The steering stage looks for activations in
`/kaggle/input/**/drivelm/activations`, but `kaggle_deploy.py upload --drivelm`
only staged `scenes.json` and the frames. A steering-only kernel would therefore
have found nothing and re-extracted the sweeps -- a model load plus a forward
pass per sample, for data the running kernel is producing right now.

Added `upload --activations [DIR]`, defaulting to a search of
`results/kaggle_drivelm` then `results/kaggle_8b`. Two details worth recording:

* **The `.json` sidecar travels with the `.npz`.** It carries scene, phrase,
  condition and `follows_injection`; without it the archive is an unlabelled
  block of floats and no probe can be fitted. A `.npz` with no sidecar is now
  named in a warning rather than discovered after the model has loaded.
* **An empty directory reports absent rather than staging nothing.** That is the
  normal state before a run completes, and a silent no-op there would produce a
  kernel that fails at the probe-fitting cell for no visible reason.

Pulled out as `find_activations` / `stage_activations` so it is testable without
a live upload, and tested. 80 -> 81.

---

## Iteration 5 — 2026-11-13 16:02 UTC — pre-flight found a crash on the real data

Kernel v16 `RUNNING` at 2 h 59 m. Rather than wait idle, I ran the whole
steering chain on **synthetic activations shaped exactly like the real ones**
(36 layers, 2048 hidden, one attack wording, two benign controls -- what
`--phrases 1 --controls 2` produces). Two findings, one of them a blocker.

### `midas-fit-defense` crashed on this extraction

```
gate_layer = gate_ranked[0].layer
IndexError: list index out of range
```

The gate layer is chosen by the strict `scene+phrase` split, which needs a fold
whose held-out phrases contain **both** an attack wording and a non-attack one.
With `--phrases 1` the sweep has four phrase values in total -- one attack, one
empty (clean), two controls -- spread over five folds, so every fold holds out at
most one and **no test fold ever has both classes**. Every layer scores `nan`,
the ranked list comes back empty, and indexing it raises.

This is not a synthetic-data artefact: it is the exact extraction the running
kernel is performing. The crash would have happened *after* the sweeps were
computed, in the next session, with nothing to show for the GPU time.

Fixed by falling back to held-out *scenes* and **saying so**:

```
WARNING: no scene+phrase fold had both classes (1 attack wording(s) in this
sweep), so the gate's generalisation to unseen wording cannot be measured here.
Falling back to held-out scenes only; re-extract with --phrases 3 or more.
```

The weaker validation is written into the artifact's own notes
(`gate validation: scene only (phrase holdout not possible)`), so a gate whose
wording generalisation was never tested cannot later be reported as though it
had been. A genuinely degenerate case -- nan at every layer under *both* splits
-- now exits with the class counts rather than an IndexError.

**The honest limitation to carry forward:** with one attack wording, the gate can
only ever be tested on the wording it trained on. It may be detecting *that
phrase's pixels* rather than "an injected instruction". Measuring the difference
needs `--phrases 3`, which is a separate extraction.

### The hook path works, but our ||w|| is nothing like theirs

Ran the fitted artifact through the real `Steerer` on a toy torch model:

```
gate score attack-like  +8.98 -> steer True
gate score clean-like  -14.90 -> steer False
prefill: last-position move = 9.900   (asked for 9.9)
         position 0 move      = 0.000   (must be 0)
decode steps: 0.0000, 0.1008, 0.0000
```

Gate discriminates, step 0 lands on its stated norm, other positions untouched,
later steps tiny -- the schedule behaves as designed.

**But the fitted `||w||` is ~1.9, where ARGUS measured ~46.** Eq. 7 divides by
`||w||^2` and the step is `alpha/||w||`, so each of our tail steps closes about
**50%** of the probe score where theirs closed **2%**. We reproduce their
*formula* faithfully and still get a tail that is ~25x more aggressive, because
sklearn's `LogisticRegression(C=1.0)` on 2048 standardised dimensions produces a
much smaller-norm weight than whatever they trained.

This does not threaten the structural claim -- 0.1 against a first step of 9.9
means the first step still does essentially all the work, which is the thing
section 12 lacked. But it does mean "faithful to Eq. 7" and "same effective step
size" are not the same statement, and the ||w|| from the real activations should
be read off before the result is described as a reproduction. `midas-eval` now
prints it per layer.

81 -> 83 tests. Both the crash and the single-wording fallback are pinned, plus
the case with four wordings, which must still use the strict split.

---

## Iteration 6 — 2026-11-13 16:36 UTC — pre-flighted the rest of the stage

Kernel v16 `RUNNING` at 3 h 33 m. The ~3.5 h estimate covered the evaluation
only; the activation block adds roughly 1,200 forward passes (100 scenes x 8
answer-token rows + 4 prompt-only rows), so ~4 h total is the better figure.
Nothing wrong.

Pre-flighted the two remaining CLI calls in the steering stage against the
synthetic sweep.

**`midas-probes` runs, but reported a non-result as silence.** The strict
`scene+phrase` split printed its heading and then nothing at all under it --
which reads as "the probe learned nothing", a different claim from "this split
could not be evaluated". Same single-wording cause as the crash in iteration 5.
It now says which, and names the number of attack wordings that caused it.

**`midas-eval --both --style plain --defend` runs clean** and writes
`baseline_*` / `attack_*` into the config directory, which is what the summary
table reads back.

### Changed how the step sizes are chosen

The sweep was going to be 5/10/20, anchored on ARGUS's measured 9.9. That is the
wrong thing to copy: **9.9 means different things at different hidden sizes** --
a sizeable nudge in 2048 dimensions, a small one in 4096 -- and theirs was a
28-layer model, ours is 36 layers at 2048.

The stage now reads `||mean||` at the steered layer out of the fitted artifact
and sweeps 0.1x, 0.25x, 0.5x of it, printing what fraction ARGUS's 9.9 would
have been. The comparison this stage exists to make -- argus against uniform at
*matched* strength -- is unaffected, since both arms move by the same amount.

Related, and the reason this matters: the fitted `||w||` is ~1.9 against ARGUS's
~46 (iteration 5), so our Eq. 7 tail steps are already far more aggressive
relative to the score than theirs. Two quantities now differ from the paper by
large factors, and both are consequences of fitting our own probe rather than
porting theirs. Worth stating plainly in any writeup: this reproduces their
*design*, not their *hyperparameters*.

Also caught a brace collision while making the change -- the cell is an
f-string template, so `{...}` in the generated code interpolates at generation
time. Split into two cells. The generated-notebook syntax check caught it
immediately, which is the second time that check has paid for itself.

83 tests, all passing.
