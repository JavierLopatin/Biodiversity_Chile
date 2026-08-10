#!/bin/bash
# Parte GPU de la etapa de serie cruda. Se lanza cuando el MAE suelta las tarjetas.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/search
echo "=== modelos sobre la serie cruda  $(date +%H:%M:%S)"
i=0
for ngs in 68 100 156 196; do
  [ -f "data/derived/phenoshape_by_index_raw${ngs}.parquet" ] || { echo "[falta] _raw${ngs}"; continue; }
  for sub in serpentine reshape; do
    BIODIV_CURVES="_raw${ngs}" CUDA_VISIBLE_DEVICES=$((i%2)) \
      python scripts/11_run_conv.py --substrate $sub --index kndvi --scheme kfold5_window \
      --context clim+topo+area --seeds 3 --max-epochs 200 --patience 20 \
      > "logs/search/raw${ngs}_${sub}.log" 2>&1 &
    i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
  done
done
wait
echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/30_search_cnn.py --report
