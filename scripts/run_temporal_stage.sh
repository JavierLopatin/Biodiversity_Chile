#!/bin/bash
# Los cinco cabezas de familia bajo los esquemas que faltaban.
#
# TEMPORALES: kfold_time, kfold_loc_time y time_within_owner. Ninguno de los esquemas previos
# probaba la transferencia temporal -- todos bloquean el espacio y agrupan los anos.
#
# Y kfold5_block20, que no es temporal pero es el que el diagnostico de patron de muestreo
# senala como el unico cercano a lo que un mapa hace: evalua a 11,4 km del entrenamiento
# contra los 13,2 km a los que el dominio nativo dista de la parcela mas proxima.
# kfold5_window evalua a 0,47 km, o sea al 3,6 % de esa distancia.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/temporal

# `kfold_loc_time` es la validacion titular: la unica que retiene sitio Y tiempo a la vez,
# y la que evalua a 14,1 km, algo por encima de los 13,2 km del mapa.
#
# `kfold_time` se conserva porque el LLTO solo dice CUANTO cae, no por que: sin la caida
# temporal por separado, un R2 bajo en LLTO no distingue "sitio nuevo" de "tiempo nuevo".
# Cuesta 6 folds contra los 29 del otro.
SCHEMES="kfold_time kfold_loc_time"

for sc in $SCHEMES; do
  echo "=== $sc  $(date +%H:%M:%S)"

  # RF y B03 son CPU: van en paralelo a los de GPU
  BIODIV_CURVES=_raw100 python scripts/09_run_baselines.py --model RF06 --scheme "$sc" \
    --seeds 3 > "logs/temporal/${sc}_RF06.log" 2>&1 &
  rf=$!
  python scripts/09_run_baselines.py --model B03 --scheme "$sc" --seeds 3 \
    > "logs/temporal/${sc}_B03.log" 2>&1 &
  b3=$!

  BIODIV_CURVES=_raw100 CUDA_VISIBLE_DEVICES=0 python scripts/10_run_tabular_dl.py \
    --model MLP06 --index kndvi --scheme "$sc" --seeds 3 \
    > "logs/temporal/${sc}_MLP06.log" 2>&1 &
  BIODIV_CURVES=_raw36 CUDA_VISIBLE_DEVICES=1 python scripts/11_run_conv.py \
    --substrate serpentine --index kndvi --scheme "$sc" --context clim+topo+area \
    --seeds 3 --max-epochs 200 --patience 20 \
    > "logs/temporal/${sc}_C2D.log" 2>&1 &
  wait
  wait $rf $b3

  BIODIV_CURVES=_raw36 CUDA_VISIBLE_DEVICES=0 python scripts/11_run_conv.py \
    --substrate curve1d --index kndvi --scheme "$sc" --context clim+topo+area \
    --seeds 3 --max-epochs 200 --patience 20 \
    > "logs/temporal/${sc}_C1D.log" 2>&1
done

echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/39_temporal_report.py 2>/dev/null || \
  echo "(el informe se genera aparte)"
