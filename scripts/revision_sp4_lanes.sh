#!/bin/bash
# SP4 scheduler: two GPU lanes of cross-chemistry chains (scripts/revision_sp4_chem.sh).
# Lane 1 starts after queue C (fixed-100 baselines) finished; lane 2 after queues A and B
# (ablation) finished. Before every backbone a lane waits while any E:\Yolo-Thermal Python
# job is running: the user's jobs are never interrupted and always go first.
# Usage: bash scripts/revision_sp4_lanes.sh 1|2
set -uo pipefail
cd "$(dirname "$0")/.."

yolo_running() {
  local n
  n=$(pwsh -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -match 'Yolo' }).Count" 2>/dev/null | tr -d '\r')
  [ "${n:-0}" != "0" ]
}
wait_for() {  # $1.. = log files that must contain "finished"
  for f in "$@"; do
    until grep -q "finished" "$f" 2>/dev/null; do sleep 120; done
  done
}

case "$1" in
  1) wait_for logs/revision/sp4_queueC.log; SEEDS="0 2 4" ;;
  2) wait_for logs/revision/sp4_queueA.log logs/revision/sp4_queueB.log; SEEDS="1 3" ;;
esac
for s in $SEEDS; do
  while yolo_running; do echo "Yolo job running, waiting $(date)"; sleep 300; done
  bash scripts/revision_sp4_chem.sh "$s"
done
echo "lane $1 finished $(date)"
