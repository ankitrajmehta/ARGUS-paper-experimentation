#!/bin/bash
export DISPLAY=:0
for R in d3d12 llvmpipe; do
  GALLIUM_DRIVER=$R timeout 300 /opt/thor/bin/python /mnt/e/Argus/thor/smoke.py $R 2>&1 | grep -vE "^\s*$" | tail -4
done
