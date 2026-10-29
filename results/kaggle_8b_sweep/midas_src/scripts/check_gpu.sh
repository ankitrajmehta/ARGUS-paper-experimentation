#!/usr/bin/env bash
# Record what the college GPU can do. Run this first and keep the output:
# it decides dtype, whether 4-bit is needed, and whether closed-loop simulation
# (which needs RT cores) is possible later.
set -uo pipefail

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
else
  echo "nvidia-smi not found - no NVIDIA GPU visible"
fi

echo
echo "=== host ==="
echo "RAM (GB): $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo '?')"
echo "OS      : $(lsb_release -ds 2>/dev/null || uname -sr)"
echo "python  : $(python3 --version 2>&1)"

echo
echo "=== torch ==="
python3 - <<'PY' 2>/dev/null || echo "torch not installed yet"
import torch
print("torch        :", torch.__version__)
print("cuda avail   :", torch.cuda.is_available())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("device       :", torch.cuda.get_device_name(i))
    print("capability   :", torch.cuda.get_device_capability(i))
    print("total VRAM GB:", round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1))
    # bf16 needs compute capability >= 8.0; a T4 (7.5) must use float16.
    print("bf16 ok      :", torch.cuda.is_bf16_supported())
PY

cat <<'EOF'

interpreting this:
  VRAM >= 20 GB  -> midas-eval --dtype bfloat16            (or float16 if bf16 ok = False)
  VRAM 12-20 GB  -> midas-eval --load-4bit
  bf16 ok False  -> add --dtype float16 --no-flash-attn    (e.g. Tesla T4, V100)
EOF
