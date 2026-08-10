#!/bin/bash
# Etapa B1 del plan: una variante de curva por (reconstructor x modo de borde).
#
# Se corre en el env `phenopy` porque es el unico donde estan las diez reconstrucciones:
# KDEpy y whittaker_eilers faltan en base, y sin ellos `KDE`, `whittaker` y
# `upper_envelope` devuelven NaN en silencio en vez de fallar.
#
# `wrap` queda deliberadamente fuera de la rejilla: ya esta medido que aplasta la tendencia
# interanual (pendiente -0,16 contra el -1 teorico), y ese es justo el eje que no queremos
# perder. Ver docs/13_phenology_year_boundary.md.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/curves
for recon in linear savgol whittaker RBF harmonic dlog_beck dlog_elmore agauss upper_envelope KDE; do
  for mode in shrink reflect; do
    sfx="_${recon}_${mode}"
    if [ -f "data/derived/phenoshape_by_index${sfx}.parquet" ]; then
      echo "[skip] $sfx"; continue
    fi
    echo "=== $recon / $mode  $(date +%H:%M:%S)"
    conda run -n phenopy python scripts/29_refit_curves_from_cubes.py \
      --recon "$recon" --roll-mode "$mode" --suffix "$sfx" --workers 10 \
      > "logs/curves/${recon}_${mode}.log" 2>&1 \
      || echo "    FALLO en $recon/$mode -- sigue con el resto"
  done
done
echo "=== fin $(date +%F\ %H:%M:%S)"
