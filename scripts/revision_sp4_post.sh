#!/bin/bash
# SP4 per-backbone post-processing on CPU: chemistry coverage (revision_chemistry_coverage.py)
# and the Oxford / MIT per-cell statistics (revision_sp4_percell.py) for each finished backbone.
# Starts backbones 0-3 once lane 2 has finished (memory is then free of one fine-tuning job),
# backbone 4 once lane 1 has finished. Waits while any E:\Yolo-Thermal Python job runs.
# Every step is skipped when its output exists.
# Usage: bash scripts/revision_sp4_post.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
L="logs/revision/sp4"

yolo_running() {
  local n
  n=$(pwsh -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'Yolo' }).Count" 2>/dev/null | tr -d '\r')
  [ "${n:-0}" != "0" ]
}
wait_for() { until grep -q "finished" "$1" 2>/dev/null; do sleep 120; done; }
backbone_done() { grep -q "backbone s$1 DONE" logs/revision/sp4_lane1.log logs/revision/sp4_lane2.log 2>/dev/null; }

post() {  # $1 = backbone seed
  local s="$1"
  backbone_done "$s" || { echo "FAILED s$s: chain not done"; return; }
  while yolo_running; do echo "Yolo job running, waiting $(date)"; sleep 300; done
  if [ -e "results/revision/seeds/chemistry_coverage_s${s}.json" ]; then echo "skip coverage s$s"
  else "$PY" scripts/revision_chemistry_coverage.py --seed "$s" > "$L/chem_cov_s${s}.log" 2>&1 \
         || echo "FAILED coverage s$s"; fi
  if [ -e "results/revision/seeds/mit_tri_percell_s${s}.json" ]; then echo "skip percell s$s"
  else "$PY" scripts/revision_sp4_percell.py --seed "$s" > "$L/percell_s${s}.log" 2>&1 \
         || echo "FAILED percell s$s"; fi
  echo "post s$s done $(date)"
}

wait_for logs/revision/sp4_lane2.log
for s in 0 1 2 3; do post "$s"; done
wait_for logs/revision/sp4_lane1.log
post 4
echo "post finished $(date)"
