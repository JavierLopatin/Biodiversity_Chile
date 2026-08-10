#!/bin/bash
# Dos preguntas, en este orden.
#
# 1. EQUIDAD. La serie cruda de 3 anos sube a la C2D de 0,362 a 0,383. La pregunta que decide
#    la historia no es si sube, sino a QUIEN sube. Si la ganancia es de la representacion
#    -- la imagen 2D puede leer la estructura interanual que una tabla de columnas no --,
#    entonces la C2D gana por un motivo. Si sube a todas las familias por igual, entonces la
#    serie cruda es simplemente mejor informacion y la C2D sigue detras. Es falsable y se
#    contesta corriendo MLP y RF sobre las mismas curvas.
#
# 2. CONFIRMACION. Se han buscado ~120 combinaciones con semillas {0,1,2} y se reporta el
#    maximo, que sobrestima. El top-5 se re-corre con semillas DISJUNTAS {10..14} -- y
#    tambien los competidores, o la contraccion se le aplicaria solo a la CNN.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/search

echo "=== 1. equidad: MLP y RF sobre la serie cruda  $(date +%H:%M:%S)"
for sfx in _raw36 _raw100; do
  # RF es CPU: va en paralelo a los MLP, que son GPU
  BIODIV_CURVES=$sfx python scripts/09_run_baselines.py --model RF06 \
    --scheme kfold5_window --seeds 3 > "logs/search/fair_RF06${sfx}.log" 2>&1 &
  rf=$!
  i=0
  for m in MLP06 MLP07; do
    extra=$([ "$m" = MLP06 ] && echo "--index kndvi" || echo "")
    BIODIV_CURVES=$sfx CUDA_VISIBLE_DEVICES=$((i%2)) \
      python scripts/10_run_tabular_dl.py --model $m $extra --scheme kfold5_window \
      --seeds 3 > "logs/search/fair_${m}${sfx}.log" 2>&1 &
    i=$((i+1))
  done
  wait $rf; wait
done

echo "=== 2. confirmacion con semillas disjuntas {10..14}  $(date +%H:%M:%S)"
i=0
for spec in "_raw36 serpentine" "_raw100 serpentine" "_raw24 serpentine" \
            "_raw36 hilbert" "_raw100 reshape" "_raw36 spectrogram"; do
  set -- $spec
  BIODIV_CURVES=$1 CUDA_VISIBLE_DEVICES=$((i%2)) \
    python scripts/11_run_conv.py --substrate $2 --index kndvi --scheme kfold5_window \
    --context clim+topo+area --seed-start 10 --seeds 5 --max-epochs 200 --patience 20 \
    > "logs/search/conf_$1_$2.log" 2>&1 &
  i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
done
wait
# los competidores con las MISMAS semillas nuevas
i=0
for spec in "MLP06 --index kndvi" "MLP07 "; do
  set -- $spec; m=$1; shift
  CUDA_VISIBLE_DEVICES=$((i%2)) python scripts/10_run_tabular_dl.py --model $m "$@" \
    --scheme kfold5_window --seed-start 10 --seeds 5 \
    > "logs/search/conf_${m}.log" 2>&1 &
  i=$((i+1))
done
wait
python scripts/09_run_baselines.py --model RF06 --scheme kfold5_window \
  --seed-start 10 --seeds 5 > logs/search/conf_RF06.log 2>&1 || echo "  RF06 sin --seed-start"
echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/30_search_cnn.py --report
