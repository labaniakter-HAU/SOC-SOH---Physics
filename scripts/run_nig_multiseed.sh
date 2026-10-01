#!/bin/bash
# Multi-seed v12-NIG training campaign (closes the "NIG accuracy-uncertainty
# trade-off not multi-seeded" limitation). Same hyperparameters as the
# canonical single-seed checkpoints/nasa_v12_nig/ run (nig_coeff=0.02,
# 400 epochs, intra_cell_random split), now across seeds 0-4, each to its
# own NEW checkpoint dir. Never touches checkpoints/nasa_v12_nig/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"

mkdir -p results/_multiseed_raw logs/_nig_multiseed

for seed in 0 1 2 3 4; do
  ckpt_dir="checkpoints/nasa_v12_nig_seed${seed}"
  if [ -e "${ckpt_dir}/best.pt" ]; then
    echo "SKIP seed=${seed}: ${ckpt_dir}/best.pt already exists"
    continue
  fi
  echo "=== NIG seed=${seed} START $(date) ==="
  "$PY" train.py --epochs 400 --batch-size 256 --lr 0.001 --device cuda \
    --split-mode intra_cell_random --nig-coeff 0.02 \
    --data-dir data/raw/nasa --checkpoint-dir "${ckpt_dir}" \
    --seed "${seed}" \
    > "logs/_nig_multiseed/seed${seed}.txt" 2>&1
  echo "=== NIG seed=${seed} DONE $(date) ==="

  "$PY" scripts/eval_nasa.py --checkpoint "${ckpt_dir}/best.pt" \
    --split-mode intra_cell_random \
    > "logs/_nig_multiseed/seed${seed}_eval.txt" 2>&1 || true
done

echo "NIG multiseed campaign complete $(date)"
