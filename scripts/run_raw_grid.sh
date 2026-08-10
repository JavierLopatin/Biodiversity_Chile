#!/bin/bash
# Rejilla completa de la serie cruda, RE-CORRIDA.
#
# La primera pasada estaba mal medida: `STEP_COLS` estaba clavado en 52, asi que las curvas
# de 68 a 196 pasos se truncaban en silencio a sus primeros 52 -- comparando cuanto del
# periodo de 3 anos cabe en 52 columnas, no la resolucion que decia la etiqueta. Arreglado en
# `features.step_cols` y blindado en `tests/test_step_cols.py`.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/search
run() {
  BIODIV_CURVES="$1" CUDA_VISIBLE_DEVICES="$3" \
    python scripts/11_run_conv.py --substrate "$2" --index kndvi --scheme kfold5_window \
    --context clim+topo+area --seeds 3 --max-epochs 200 --patience 20 \
    > "logs/search/g${1}_${2}.log" 2>&1 || echo "    FALLO $1 $2"
}
echo "=== rejilla de resolucion  $(date +%H:%M:%S)"
i=0
for ngs in 24 36 48 60 68 100 156 196; do
  [ -f "data/derived/phenoshape_by_index_raw${ngs}.parquet" ] || { echo "[falta] _raw${ngs}"; continue; }
  for sub in serpentine reshape; do
    run "_raw${ngs}" "$sub" $((i%2)) &
    i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
  done
done
wait
python scripts/30_search_cnn.py --report > logs/search/report_grid.log 2>&1

BEST=$(python - <<'PY'
import pandas as pd, re
d = pd.read_csv("results/tables/cnn_search.csv")
r = d[d.run_id.str.contains("_raw")]
print("_raw" + re.search(r"_raw(\d+)", r.loc[r["mean"].idxmax(), "run_id"]).group(1))
PY
)
echo "=== mejor resolucion: $BEST -- sustratos  $(date +%H:%M:%S)"
i=0
for sub in hilbert gaf mtf ndi cwt cos2d spectrogram stack5 curve1d; do
  run "$BEST" "$sub" $((i%2)) &
  i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
done
wait
echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/30_search_cnn.py --report
