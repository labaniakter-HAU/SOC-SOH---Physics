#!/bin/bash
# SP4 cross-chemistry chain for one NASA backbone seed (five-backbone design, user 2026-09-30).
# Same settings as the former single-backbone runs (scripts/reproduce_transfer.sh,
# scripts/run_mit_tri_campaign.sh, the SP2 Oxford relabel run), with the backbone
# checkpoints/nasa_v12_seed{s}/best.pt and fine-tuning seed s. Paths: scripts/revision_sp4_paths.py.
# Every step is skipped when its output exists, so an interrupted chain resumes.
# Usage: bash scripts/revision_sp4_chem.sh <seed>
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
s="$1"
BB="checkpoints/nasa_v12_seed${s}/best.pt"
L="logs/revision/sp4/chem_s${s}"
mkdir -p "$L" checkpoints/sp4 results/revision/seeds
[ -e "$BB" ] || { echo "missing backbone $BB"; exit 1; }
declare -A DIR=([calce]=data/raw/calce [oxford]=data/raw/oxford [mit_tri]=data/raw/mit_tri)

has_rmse() { [ -e "$1" ] && grep -q "SOC  RMSE" "$1"; }

evaluate() {  # $1 checkpoint, $2 dataset, $3 log
  has_rmse "$3" && { echo "skip $3"; return; }
  "$PY" scripts/evaluate.py --checkpoint "$1" --dataset "$2" --data-dir "${DIR[$2]}" \
    --split test --device cuda > "$3" 2>&1 || echo "FAILED eval $3"
}

finetune() {  # $1 dataset, $2 checkpoint dir, extra args...
  local ds="$1" out="$2"; shift 2
  [ -e "$out/best.pt" ] && [ -e "$out/.done" ] && { echo "skip $out"; return; }
  "$PY" train.py --dataset "$ds" --data-dir "${DIR[$ds]}" --ft --resume "$BB" --checkpoint-dir "$out" \
    --epochs 100 --mse-only --device cuda --seed "$s" "$@" > "$L/ft_$(basename "$out").log" 2>&1 \
    && touch "$out/.done" || echo "FAILED finetune $out"
}

echo "=== backbone s$s START $(date)"
for ds in calce oxford mit_tri; do evaluate "$BB" "$ds" "$L/zeroshot_${ds}.log"; done
finetune calce  "checkpoints/sp4/calce_ft_s${s}"  --lr 1e-4
evaluate "checkpoints/sp4/calce_ft_s${s}/best.pt" calce "$L/ft_eval_calce.log"
finetune oxford "checkpoints/sp4/oxford_ft_s${s}" --lr 5e-5 --batch-size 64
evaluate "checkpoints/sp4/oxford_ft_s${s}/best.pt" oxford "$L/ft_eval_oxford.log"
finetune mit_tri "checkpoints/sp4/mit_tri_ft_s${s}" --lr 1e-4
evaluate "checkpoints/sp4/mit_tri_ft_s${s}/best.pt" mit_tri "$L/ft_eval_mit_tri.log"
for ds in calce oxford mit_tri; do
  if [ -e "checkpoints/sp4/eot_transducer_s${s}_${ds}.pt" ]; then echo "skip EOT $ds"; continue; fi
  "$PY" scripts/eot_transfer.py --checkpoint "$BB" --backbone-seed "$s" --dataset "$ds" \
    --out "results/revision/seeds/eot_transfer_s${s}.json" \
    --save-transducer "checkpoints/sp4/eot_transducer_s${s}_{ds}.pt" > "$L/eot_${ds}.log" 2>&1 \
    || echo "FAILED EOT $ds"
done
for c in 1 5 10 20; do
  finetune calce "checkpoints/sp4/calce_ft_fewshot${c}_s${s}" --lr 1e-4 --ft-max-cycles "$c"
  evaluate "checkpoints/sp4/calce_ft_fewshot${c}_s${s}/best.pt" calce "$L/fewshot_eval_${c}.log"
done
echo "=== backbone s$s DONE $(date)"
