#!/bin/bash
# Protocol v3 (reduced scope, 2026-10-05), our method, unattended: discovery at 512 px for U = 100 and 1000, then parts +
# detection for both. Each step is skipped if its output exists, so the queue can be restarted.
# Start it detached:  systemd-run --user --unit=v3ours --working-directory=$PWD bash run_v3_ours.sh
set -euo pipefail
cd "$(dirname "$0")"
exec > >(tee -a run_v3_ours.log) 2>&1
export FFHQ_DIR=${FFHQ_DIR:-../ffhq_nvlabs}
PY=${PY:-python}
disc() {  # U res out
  [ -f "$3/clusters.pkl" ] && { echo "=== $(date) discovery $3 exists, skipped"; return; }
  echo "=== $(date) discovery U=$1 at $2 px -> $3"; $PY -u proto_leader_gpu.py --U $1 --level1 single --res $2 --out $3
}
parts() {  # run U res method
  [ -f "../v3/parts/$4_U$2_K32.npz" ] && { echo "=== $(date) parts $4 U=$2 exist, skipped"; return; }
  echo "=== $(date) parts + detection $4 U=$2 ($1)"; $PY -u run_ours.py $1 $2 --res $3 --method $4
}
disc 100 512 results/v3_U100_r512
disc 1000 512 results/v3_U1000_r512 > disc_U1000.log 2>&1 &                         # U = 1000 discovery (mostly CPU) alongside
D1000=$!
parts results/v3_U100_r512 100 512 ours                                              # U = 100 detection (mostly GPU)
wait $D1000; cat disc_U1000.log
parts results/v3_U1000_r512 1000 512 ours
echo "=== $(date) DONE"
