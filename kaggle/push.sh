#!/usr/bin/env bash
# Upload the committed HEAD as the private argus-code dataset, then (re)run a Kaggle kernel on T4.
# Usage (Git Bash, from the repo root): [KERNEL=e1|e1s|e2|e2b|e2c|e3a|e3b|e3b2] bash kaggle/push.sh [RUN_NAME MAX_SAMPLES BEHAVIOR_N]
# KERNEL=e1s runs the full ARGUS pipeline on VTQA (E1-S, steering test);
# KERNEL=e2 runs E2 (+E2b) in its own kernel, mounting the latest finished argus-e1 output;
# KERNEL=e2b runs only E2b, reusing the latest finished argus-e2 activations;
# KERNEL=e2c runs only E2c, reusing argus-e2 and argus-e2b activations.
# KERNEL=e3a builds the robot behavior probe; KERNEL=e3b compares responses after detection (mounts argus-e3a);
# KERNEL=e3b2 adds the grounding mask and strong steering (mounts argus-e3a and argus-e3b).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain -- src experiments configs kaggle data)" ]; then
  echo "Uncommitted changes: commit first so the Kaggle run matches a known commit." >&2
  exit 1
fi

DATASET=ankitrmehta/argus-code
BUILD=kaggle/.build
rm -rf "$BUILD" && mkdir -p "$BUILD/code" "$BUILD/kernel"
cp kaggle/code-dataset/dataset-metadata.json "$BUILD/code/"
git archive --format=tar.gz -o "$BUILD/code/argus_code.tar.gz" HEAD
git rev-parse HEAD > "$BUILD/code/COMMIT"

if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  MSG="$(git log -1 --format='%h %s')"
  # The Windows CLI builds temp file names from -p, so upload from inside the folder.
  (cd "$BUILD/code" && kaggle datasets version -p . -m "$MSG")
else
  (cd "$BUILD/code" && kaggle datasets create -p .)
fi
# The kernel must not start before the new dataset version is processed.
until kaggle datasets status "$DATASET" 2>/dev/null | grep -qi ready; do sleep 10; done

KERNEL=${KERNEL:-e1}
case "$KERNEL" in
  e1) META=kaggle/kernel ;;
  e1s|e2|e2b|e2c|e3a|e3b|e3b2) META=kaggle/kernel-$KERNEL ;;
  *) echo "KERNEL must be e1, e1s, e2, e2b, e2c, e3a, e3b or e3b2" >&2; exit 1 ;;
esac
cp "$META/kernel-metadata.json" kaggle/run_kaggle.py "$BUILD/kernel/"
case "$KERNEL" in
  e1) ;;
  # E1-S: full ARGUS pipeline on VTQA, same 100 train/val items as E1.
  e1s) sed -i -e 's/^EXPERIMENT = .*/EXPERIMENT = "e1s"/' -e 's/^RUN_NAME = .*/RUN_NAME = "e1s_3b_fp16"/' "$BUILD/kernel/run_kaggle.py" ;;
  # E2 defaults: all 94 test frames for the attack-success check.
  e2|e2b|e2c|e3a|e3b|e3b2) sed -i -e "s/^EXPERIMENT = .*/EXPERIMENT = \"$KERNEL\"/" -e "s/^RUN_NAME = .*/RUN_NAME = \"${KERNEL}_3b_fp16\"/"                  -e 's/^BEHAVIOR_N = .*/BEHAVIOR_N = 94/' "$BUILD/kernel/run_kaggle.py" ;;
esac
# Optional overrides for this push only, e.g. a smoke run: bash kaggle/push.sh e1_smoke 2 2
if [ $# -ge 3 ]; then
  sed -i -e "s/^RUN_NAME = .*/RUN_NAME = \"$1\"/" -e "s/^MAX_SAMPLES = .*/MAX_SAMPLES = $2/" \
         -e "s/^BEHAVIOR_N = .*/BEHAVIOR_N = $3/" "$BUILD/kernel/run_kaggle.py"
  grep -E "^(RUN_NAME|MAX_SAMPLES|BEHAVIOR_N) =" "$BUILD/kernel/run_kaggle.py"
fi
(cd "$BUILD/kernel" && kaggle kernels push -p . --accelerator NvidiaTeslaT4)
echo "Follow with: kaggle kernels status ankitrmehta/argus-$KERNEL"
