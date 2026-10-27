#!/usr/bin/env bash
# Fetch CHAI's released DriveLM artefacts into vendor/.
#
# We vendor rather than fork: the frames, phrase dictionary, optimiser output and
# sign renderer all come from upstream unchanged, so our numbers stay comparable
# to the paper's. Nothing here is modified; our code lives in src/midas.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/vendor/chai"

if [ -d "$DEST" ]; then
  echo "vendor/chai already present; delete it to re-clone."
else
  mkdir -p "$ROOT/vendor"
  echo "cloning CHAI (drivelm branch) ..."
  git clone --depth 1 --single-branch --branch drivelm \
    https://github.com/Cyphysecurity/chai.git "$DEST"
fi

echo
echo "checking released artefacts:"
for f in training.json testing.json attack_dicts_internvl.json \
         optimal_attacks/optimal.json attack/create_attack.py; do
  if [ -e "$DEST/$f" ]; then echo "  ok      $f"; else echo "  MISSING $f"; fi
done

frames=$(find "$DEST/DriveLM_dataset" -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
echo "  frames  $frames (expect 184)"
