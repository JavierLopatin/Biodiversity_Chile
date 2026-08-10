#!/bin/bash
# Parte CPU de la etapa de serie cruda: construir las curvas. Se separa de la parte GPU
# porque no compite con nada -- corre mientras el MAE usa las tarjetas.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/search
echo "=== construyendo series crudas  $(date +%H:%M:%S)"
for ngs in 68 100 156 196; do
  sfx="_raw${ngs}"
  [ -f "data/derived/phenoshape_by_index${sfx}.parquet" ] && { echo "[skip] $sfx"; continue; }
  echo "--- ngs=$ngs  $(date +%H:%M:%S)"
  conda run -n phenopy python scripts/29_refit_curves_from_cubes.py --raw-series \
    --ngs "$ngs" --suffix "$sfx" --workers 10 > "logs/search/raw_${ngs}.log" 2>&1 \
    || echo "    FALLO en ngs=$ngs"
done
echo "=== curvas listas  $(date +%F\ %H:%M:%S)"
