#!/bin/bash
# Extra analysis (2026-10-05): our parts chosen WITH LABELS from all discovered clusters, for U = 100 and 1000 (512 px).
# Per U: scan every cluster on the first 500 readout faces (crop 0) -> greedy label selection of 32 (readout faces only) ->
# scan the chosen 32 on all readout and test crops -> v3/parts/ourssel_U<U>_K<K>.npz. Steps are skipped if their output exists.
# Start it detached:  systemd-run --user --unit=v3sel --working-directory=$PWD bash run_v3_ours_sel.sh
set -euo pipefail
cd "$(dirname "$0")"
exec > >(tee -a run_v3_ours_sel.log) 2>&1
export FFHQ_DIR=${FFHQ_DIR:-../ffhq_nvlabs}
PY=${PY:-python}
for U in 100 1000; do
  R=results/v3_U${U}_r512
  [ -f $R/scan_all_500.npz ] || { echo "=== $(date) U=$U: scan all clusters on 500 readout faces"; $PY -u run_ours.py $R $U --scan_all 500; }
  [ -f $R/parts_labelsel.json ] || { echo "=== $(date) U=$U: label selection"; $PY -u select_parts.py $R 500; }
  [ -f ../v3/parts/ourssel_U${U}_K32.npz ] || { echo "=== $(date) U=$U: scan the chosen parts"; $PY -u run_ours.py $R $U --parts_json $R/parts_labelsel.json --method ourssel; }
done
echo "=== $(date) DONE"
