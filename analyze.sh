#!/bin/bash
set -e

for d in out-*/; do
    ckpt="${d}ckpt.pt"
    if [ -f "$ckpt" ]; then
        echo "=== analyzing $d ==="
        python analyze.py --ckpt "$ckpt" --device cuda
    else
        echo "skip $d (no ckpt.pt)"
    fi
done
