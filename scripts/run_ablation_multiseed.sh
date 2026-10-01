#!/bin/bash
# Multi-seed replication of the 5 remaining single-seed ablation rows
# (No-NIG, No-PRAP, No-DCT, No-CTBA, No-IC), closing the "Partial multi-seed
# coverage" limitation. Uses the identical protocol already applied to
# No-DTAG+CTBA and No-EITE (Section~\ref{sec:robustness}): 3 seeds {0,1,2},
# 200 epochs, scripts/ablation.py's exact train/eval code path via
# scripts/run_multiseed.py (new VARIANTS entries added 2026-07-11, smoke
# tested at 2 epochs on real NASA data for all 5 variants before launch).
# Never touches results/ablation_v12.csv or checkpoints/nasa_v12*/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"

mkdir -p results/_multiseed_raw logs/_ablation_multiseed

for variant in no_nig no_prap no_dct no_ctba no_ic; do
  out_json="results/_multiseed_raw/${variant}.json"
  if [ -e "${out_json}" ]; then
    echo "SKIP variant=${variant}: ${out_json} already exists"
    continue
  fi
  echo "=== ${variant} START $(date) ==="
  "$PY" scripts/run_multiseed.py --variant "${variant}" --seeds 0 1 2 \
    --epochs 200 --data-dir data/raw/nasa --device cuda \
    --out-json "${out_json}" \
    > "logs/_ablation_multiseed/${variant}.txt" 2>&1
  echo "=== ${variant} DONE $(date) ==="
done

echo "Ablation multiseed campaign complete $(date)"
