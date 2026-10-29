#!/usr/bin/env bash
# Upload the committed HEAD as the private argus-code dataset, then (re)run the E1 kernel on T4.
# Usage (Git Bash, from the repo root): bash kaggle/push.sh [RUN_NAME MAX_SAMPLES BEHAVIOR_N]
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
  kaggle datasets version -p "$BUILD/code" -m "$(git log -1 --format='%h %s')"
else
  kaggle datasets create -p "$BUILD/code"
fi
# The kernel must not start before the new dataset version is processed.
until kaggle datasets status "$DATASET" 2>/dev/null | grep -qi ready; do sleep 10; done

cp kaggle/kernel/kernel-metadata.json kaggle/run_kaggle.py "$BUILD/kernel/"
# Optional overrides for this push only, e.g. a smoke run: bash kaggle/push.sh e1_smoke 2 2
if [ $# -ge 3 ]; then
  sed -i -e "s/^RUN_NAME = .*/RUN_NAME = \"$1\"/" -e "s/^MAX_SAMPLES = .*/MAX_SAMPLES = $2/" \
         -e "s/^BEHAVIOR_N = .*/BEHAVIOR_N = $3/" "$BUILD/kernel/run_kaggle.py"
  grep -E "^(RUN_NAME|MAX_SAMPLES|BEHAVIOR_N) =" "$BUILD/kernel/run_kaggle.py"
fi
kaggle kernels push -p "$BUILD/kernel" --accelerator NvidiaTeslaT4
echo "Follow with: kaggle kernels status ankitrmehta/argus-e1"
