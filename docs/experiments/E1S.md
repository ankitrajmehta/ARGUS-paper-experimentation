# E1-S: does ARGUS steering work at all? (full ARGUS pipeline on the paper's own data)

**Question.** Until now only the first half of ARGUS had ever been run (E1: data → augmented answers → activations →
detection / behavior ("edit") probes → auxiliary orthogonal probes). The steering defense itself was never run:
E3b's "steering" was a hand-built hook along one probe direction (Eq. 7 only), not ARGUS's pipeline.
E1-S runs **every ARGUS stage** on VTQA (the paper's image benchmark) with our model, to see whether steering
works there before judging it on the robot.

Model: Qwen2.5-VL-3B-Instruct, fp16 LM + fp32 vision (Kaggle T4). Paper: Qwen2-VL-7B.

## Versions

| Version | Commit | Kaggle | What changed | Status |
|---|---|---|---|---|
| v1 | (pending) | argus-e1s v1 | first full-pipeline run | smoke-testing |

## The ARGUS defense being tested (`src/argus/defense.py`)
At inference, on every generation step the hooks read the last position:
1. **Detection** (step 0, layer `detection_layer`): linear probe, clean vs injected. Nothing else happens unless it fires.
2. **Step 0 steering**: add a **trained vector** at the chosen layers. `train_vector.py` learns it: a softmax-weighted
   mix of the behavior ("edit") probe direction and 3 auxiliary orthogonal probe directions, times a strength, trained by
   gradient descent so that on poisoned inputs the model writes the user's answer.
3. **Steps ≥ 1**: adaptive Eq. 7 push along a "final steering" probe (`train_post_filter_probe.py`), ε from
   `estimate_epsilon.py` (left as implemented, see CLAUDE.md).
4. **Post-filter**: if the behavior probe at `post_filter_layer` still says "following the injection" at the end, the answer
   is replaced by "Sorry!".

Layers and strength for steps 2–3 come from `preliminary_experiments.py` (paper Section 4): steer every layer in a range
with a fixed strength in the attack and the defense direction, rank layers by the AIA gap, then search strengths for
the best single layer and for the top-2..5 layers; pick AIA = 0 with the highest UIA (else lowest AIA).

## Stages run (ARGUS order, `experiments/e1s/run_e1s.py`)
create_data → validate_data → augment_answers → collect detection/edit activations → train detection/edit probes →
train_auxiliary_probes → **preliminary_experiments → train_vector → train_final_steering_probe → estimate_epsilon →
eval no defense → eval steering only (post-filter off) → eval full ARGUS (steering + post-filter)** → analyze_e1s.

The "steering only" mode is added so steering is judged on its own: with the post-filter on, "Sorry!" alone removes AIA.

## Metrics (ARGUS `eval_metrics.py`, test split)
- **UIA**: the user's (VTQA) answer appears in the response. Reported on clean and on injected images.
- **AIA**: the attacker's answer appears (attack success).
- **AIFR**: the response follows the attacker's task (label or long common substring).
- Plus: detector firing rate on clean / injected images, "Sorry!" rate, and a paired view (of the items the attack wins
  undefended, how many each mode fixes). 95% bootstrap CIs over items.

## Settings and deviations from the paper (decided before the run)
| Setting | Paper (7B, 28 layers) | E1-S (3B, 36 layers) | Why |
|---|---|---|---|
| detection layer | 6 | 8 | same relative depth; E1 detection probe 100% there |
| post-filter layer | 20 | 21 | best E1 behavior probe on val (0.995); depth-scaled L26 is only 0.80 |
| preliminary layer range | 8–18 | 10–23 | same relative depth |
| strength grids | 5–40; multi grids | same | mid-layer hidden-state norms at L14–24 are 55–82 (measured), so 5–40 is 7–70% of the norm, a sensible scale |
| train / val / test items | 10k+ / 1000 / 1000 | 100 / 100 / **200** | compute budget; train/val identical to E1 (probes reproduce E1) |
| preliminary search items | 1000 val | first 50 val | compute budget (49 conditions × 50 generations) |
| dtype | bf16 | fp16 LM + fp32 vision; fp32 steering weights; loss scaling in train_vector | T4 has no native bf16; fp16 vision overflows |

### Code fixes needed to run ARGUS on this model (all in `src/argus`)
1. `train_vector.py` hard-coded `Qwen2VLForConditionalGeneration` in bfloat16 (wrong class for Qwen2.5-VL, no bf16 on T4):
   it now uses the shared loader (`vision_runner`), like every other stage.
2. **Bug in the released `train_vector.py`**: the training batch built `pixel_values` / `image_grid_thw` but passed only
   `input_ids`, `attention_mask`, `labels` to the model, so **the steering vector was trained without the image**, and in
   VTQA the injection is *in* the image. Fixed: the image tensors are passed to the model. (The paper describes training on
   poisoned inputs; we test that.)
3. fp16 training: mixture weights and basis vectors kept in fp32; `GradScaler` for the loss.
4. Budget knobs (no effect when unset): `preliminary.max_samples`, `debug.max_test_samples`; `run_defense_eval --tag`.

## Results
_Pending._
