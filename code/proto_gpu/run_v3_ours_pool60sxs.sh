#!/bin/bash
# Second pool-matched row (added 2026-10-06 after seeing the planned one): pool = the 60 clusters ranked highest by size x
# orientation spread (our label-free part ranking), then the same label selection and scan -> v3/parts/ourssel60sxs_U<U>_K<K>.npz.
# Start it detached:  systemd-run --user --unit=v3pool2 --working-directory=$PWD bash run_v3_ours_pool60sxs.sh
set -euo pipefail
cd "$(dirname "$0")"
exec > >(tee -a run_v3_ours_pool60sxs.log) 2>&1
export FFHQ_DIR=${FFHQ_DIR:-../ffhq_nvlabs}
PY=${PY:-python}
while systemctl --user is-active --quiet v3pool; do echo "=== $(date) waiting for v3pool to finish"; sleep 300; done
for U in 100 1000; do
  R=results/v3_U${U}_r512
  [ -f $R/parts_labelsel_pool60_sxs.json ] || { echo "=== $(date) U=$U: label selection, pool 60 by size x spread"; $PY -u select_parts.py $R 500 60 sxs; }
  [ -f ../v3/parts/ourssel60sxs_U${U}_K32.npz ] || { echo "=== $(date) U=$U: scan the chosen parts"; $PY -u run_ours.py $R $U --parts_json $R/parts_labelsel_pool60_sxs.json --method ourssel60sxs; }
done
echo "=== $(date) DONE"
