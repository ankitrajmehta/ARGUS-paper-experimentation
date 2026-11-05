
Step 0: Open a terminal in the repo and set the environment

Git Bash
cd /e/Argus/ARGUS
export HF_HOME=E:/Argus/hf_cache HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
P=E:/Argus/env/Scripts/python.exe

PowerShell
cd E:\Argus\ARGUS
$env:HF_HOME="E:\Argus\hf_cache"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"; $env:PYTHONUNBUFFERED="1"; $env:PYTHONIOENCODING="utf-8"
$P="E:\Argus\env\Scripts\python.exe"

What this does:
- Moves you into the code folder.
- Points the Hugging Face cache at E:\Argus\hf_cache.
- Makes output print immediately, with emoji and em-dashes displaying correctly.
- Saves the Python path as $P, so the commands below stay short.

Optional: delete the stale cache first to save ~950 MB. It rebuilds automatically.
rm -rf E:/Argus/hf_cache (Bash) or Remove-Item -Recurse -Force E:\Argus\hf_cache (PowerShell)

Step 1: E1 on 300 samples (about 2–3 hours)

Git Bash
$P experiments/e1/run_e1.py --output-dir E:/Argus/outputs/e1_300 --max-samples 300 --behavior-n 50 2>&1 | tee E:/Argus/outputs/e1_300.log

PowerShell
& $P experiments\e1\run_e1.py --output-dir E:\Argus\outputs\e1_300 --max-samples 300 --behavior-n 50 2>&1 | Tee-Object E:\Argus\outputs\e1_300.log

What it does (each ==== name ==== header in the output is one stage):

┌──────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────┐
│                  Stage                   │                                      What happens                                       │                  Rough time                  │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ create_data                              │ Takes 300 VTQA photos, adds the injected text strip, writes the train/val/test images   │ ~2–4 min (first run also rebuilds the cache) │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ validate_data                            │ Checks that train and val triggers don't overlap and that all images exist              │ seconds                                      │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ augment_answers                          │ The model rewrites each answer in short, medium and long versions (600 generations)     │ ~60–90 min (longest step)                    │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ collect_detection_train/val              │ Records internal activations at all 28 layers for clean vs. injected images             │ ~10–15 min                                   │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ collect_edit_train/val                   │ Records activations for "answering the user" vs. "obeying the injection"                │ ~20–30 min                                   │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ train_detection_probe / train_edit_probe │ Trains one probe per layer (56 in total)                                                │ ~2–5 min                                     │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ train_auxiliary_probes                   │ Trains probes perpendicular to the first, for the subspace check                        │ ~5–15 min                                    │
├──────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ analyze                                  │ Controls (shuffled labels, length only), the plot, and a 50-sample natural attack check │ ~5–10 min                                    │
└──────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────┘

How it ends: it prints a JSON summary of the best layers, the length-only control, and the attack success rate, followed by the results folder path.

If it stops or crashes: run the same command again. Each stage skips outputs it has already saved, so it resumes where it left off.

Step 2: E2 activations and attack check (about 20–40 minutes)

Run this only after Step 1 has finished.

Git Bash
PYTHONPATH=src $P experiments/e2/collect_e2.py --config E:/Argus/outputs/e1_300/configs/run_config.yaml --output-dir E:/Argus/outputs/e2 --behavior-n 40 2>&1 | tee E:/Argus/outputs/e2_collect.log

PowerShell
$env:PYTHONPATH="src"
& $P experiments\e2\collect_e2.py --config E:\Argus\outputs\e1_300\configs\run_config.yaml --output-dir E:\Argus\outputs\e2 --behavior-n 40 2>&1 | Tee-Object E:\Argus\outputs\e2_collect.log

What it does:
- Loads the same model, using the same image cap as E1 (taken from E1's config).
- For each of the 300 AI2-THOR frames × 4 versions (clean, strip, in-scene sign, harmless sign), records the activation at all 28 layers just before the robot answers.
- Asks the robot for its next action on 40 test frames. It counts how often the model outputs the attacker's action (e.g. open(front_door)), and how often its answer changes compared with the clean image.

PYTHONPATH=src lets the script find the ARGUS code, because E2 doesn't go through E1's runner.

Step 3: E2 analysis (about 1–3 minutes, CPU only)

Git Bash
PYTHONPATH=src $P experiments/e2/analyze_e2.py --output-dir E:/Argus/outputs/e2 --e1-dir E:/Argus/outputs/e1_300 2>&1 | tee E:/Argus/outputs/e2_analyze.log

PowerShell (PYTHONPATH is still set from Step 2 if it's the same window)
& $P experiments\e2\analyze_e2.py --output-dir E:\Argus\outputs\e2 --e1-dir E:\Argus\outputs\e1_300 2>&1 | Tee-Object E:\Argus\outputs\e2_analyze.log

What it does: trains and compares three detectors at every layer:
- transfer_e1: E1's VTQA-trained detector, used unchanged on robot scenes.
- indomain_clean: trained on robot frames, clean vs. injected sign.
- indomain_hardneg: the same, plus harmless signs as negatives.

For each detector it reports the share of injections caught (TPR) and the share of harmless and clean frames wrongly flagged (FPR), then prints each detector's best layer.

What to send me afterwards

┌──────────────────────────────────────────────────────────┬────────┐
│                           File                           │  From  │
├──────────────────────────────────────────────────────────┼────────┤
│ E:\Argus\outputs\e1_300\results\e1\e1_report.json        │ Step 1 │
├──────────────────────────────────────────────────────────┼────────┤
│ E:\Argus\outputs\e1_300\results\e1\e1_probe_accuracy.png │ Step 1 │
├──────────────────────────────────────────────────────────┼────────┤
│ E:\Argus\outputs\e2\results\e2_report.json               │ Step 3 │
├──────────────────────────────────────────────────────────┼────────┤
│ E:\Argus\outputs\e2\results\e2_detection.png             │ Step 3 │
├──────────────────────────────────────────────────────────┼────────┤
│ E:\Argus\outputs\e2\results\e2_behavior.json             │ Step 2 │
└──────────────────────────────────────────────────────────┴────────┘

You can also just tell me the runs are done. These files are on your disk, so I can read them myself. If any step fails, send me the last ~30 lines of its log (e1_300.log, e2_collect.log or e2_analyze.log).


---------------------------------------

Commands to run (Git Bash, from E:/Argus/ARGUS)

export HF_HOME=E:/Argus/hf_cache HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
P=E:/Argus/env/Scripts/python.exe

# E1 sanity check, 100 samples
$P experiments/e1/run_e1.py --config configs/e1_qwen25vl_3b.yaml --output-dir E:/Argus/outputs/e1_q3b_100 --max-samples 100 --behavior-n 50 2>&1 | tee E:/Argus/outputs/e1_q3b_100.log

# E2 embodied, which reuses E1's run config, so it gets the 3B and 4-bit settings
PYTHONPATH=src $P experiments/e2/collect_e2.py --config E:/Argus/outputs/e1_q3b_100/configs/run_config.yaml --output-dir E:/Argus/outputs/e2_q3b --behavior-n 40 2>&1 | tee E:/Argus/outputs/e2_q3b_collect.log
PYTHONPATH=src $P experiments/e2/analyze_e2.py --output-dir E:/Argus/outputs/e2_q3b --e1-dir E:/Argus/outputs/e1_q3b_100 2>&1 | tee E:/Argus/outputs/e2_q3b_analyze.log
If something fails partway, rerun the same command and it picks up where it stopped. E2 collect doesn't need E1's probes, only its config. So once the E1 log shows the data stage has finished, you can run E2 collect in a second terminal, but only if VRAM allows: two models at about 3.6 GiB each won't fit in 6 GiB, so in practice run them one after the other.
