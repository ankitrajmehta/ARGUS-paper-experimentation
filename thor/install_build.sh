#!/bin/bash
set -e
N=thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917
D=/root/.ai2thor/releases/$N
mkdir -p "$D" && cd "$D"
unzip -q -o /mnt/e/Argus/thor/$N.zip
ls "$D" | head
