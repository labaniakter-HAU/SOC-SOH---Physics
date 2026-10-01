#!/bin/bash
# SP4 (five seeds everywhere): training runs for seeds 3 and 4 that did not exist yet.
#   queue A / B : ablation variants (scripts/run_multiseed.py, identical protocol to seeds 0-2:
#                 200 epochs, final-epoch weights, NASA random split). Outputs go to
#                 results/revision/ablation_s34/ and are merged into results/_multiseed_raw/
#                 and results/revision/ctba_self.json by scripts/revision_sp4_merge.py.
#                 Queue B starts with a re-run of Full seed 3, compared against the stored value
#                 to confirm the current environment reproduces the July campaign.
#   queue C     : fixed-100-epoch LOCO baselines (training-rule sensitivity), seeds 3 and 4,
#                 then their conformal scoring on CPU (device rule).
# Usage: bash scripts/revision_sp4_runs.sh A|B|C
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
OUT=results/revision/ablation_s34
LOG=logs/revision/sp4
mkdir -p "$OUT" "$LOG"

variant() {  # $1 = variant name
  if [ -e "$OUT/$1.json" ]; then echo "SKIP $1"; return; fi
  echo "=== $1 START $(date)"
  "$PY" scripts/run_multiseed.py --variant "$1" --seeds 3 4 --epochs 200 --data-dir data/raw/nasa \
    --device cuda --out-json "$OUT/$1.json" > "$LOG/abl_$1.log" 2>&1 || echo "FAILED $1"
  echo "=== $1 DONE $(date)"
}

case "$1" in
  A) for v in no_eite no_nig no_dtag_ctba no_prap; do variant "$v"; done ;;
  B) if [ ! -e "$OUT/full_check_s3.json" ]; then
       echo "=== full seed 3 check START $(date)"
       "$PY" scripts/run_multiseed.py --variant full --seeds 3 --epochs 200 --data-dir data/raw/nasa \
         --device cuda --out-json "$OUT/full_check_s3.json" > "$LOG/abl_full_check_s3.log" 2>&1 || echo "FAILED full check"
       echo "=== full seed 3 check DONE $(date)"
     fi
     for v in no_dct no_ctba no_ic ctba_self; do variant "$v"; done ;;
  C) echo "=== fixed100 baselines START $(date)"
     "$PY" scripts/revision_baselines.py --mode loco --seeds 3 4 --epochs 100 --patience 0 \
       --ckpt-tag _fixed100 --out results/revision/sensitivity/baselines_loco_fixed100.json \
       > "$LOG/baselines_loco_fixed100_s34.log" 2>&1 || echo "FAILED fixed100 training"
     for s in 3 4; do
       "$PY" scripts/revision_baseline_conformal.py --seed "$s" --device cpu --ckpt-tag _fixed100 \
         --out "results/revision/sensitivity/baseline_conformal_fixed100_s$s.json" \
         > "$LOG/baseline_conformal_fixed100_s$s.log" 2>&1 || echo "FAILED conformal fixed100 s$s"
     done
     echo "=== fixed100 DONE $(date)" ;;
esac
echo "queue $1 finished $(date)"
