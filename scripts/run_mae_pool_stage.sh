#!/bin/bash
# MAE sobre el pool nuevo: 16.950 series de vegetacion nativa repartidas por las 1.449
# celdas de 10 km del territorio.
#
# El intento anterior fallo por una razon medida, no por el metodo: las 135.250 curvas de
# pixel salian de las ventanas 5x5 de las mismas 1.082 parcelas -- dimension de participacion
# 1,3 contra 1,2 -- asi que el autoencoder veia mil objetos repetidos veinticinco veces.
# Aprendia bien (MSE 0,0095 -> 0,0031) y no transferia nada (0,359 contra 0,362).
#
# Este pool es cobertura NUEVA, que es lo unico que faltaba. Si tampoco transfiere, la
# conclusion deja de ser "faltan datos sin etiquetar" y pasa a ser "el cuello de botella son
# las 1.082 etiquetas", que es una afirmacion mas fuerte y ya no refutable con mas pixeles.
#
# Tres ablaciones de cobertura, que el muestreo permite porque cada muestra lleva su clase:
#   todo        las 16.950, toda la vegetacion nativa
#   arbustivo   solo Shrubland, que es el 48 % de las parcelas etiquetadas
#   sin humedal Wetland es la clase peor cubierta por el AOA (66,8 % dentro)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/mae_pool results/mae

UP=data/derived/unlabelled
SUB=serpentine
NGS=100          # la curva del modelo titular (_raw100)

echo "=== preentrenamiento sobre el pool  $(date +%H:%M:%S)"
i=0
run_pre() {  # run_pre <etiqueta> <args...>
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$((i%2)) python scripts/31_pretrain_mae.py --substrate $SUB \
    --index kndvi --unlabelled "$UP" --ngs $NGS --epochs 40 --batch-size 512 "$@" \
    > "logs/mae_pool/pre_${tag}.log" 2>&1 || echo "    FALLO $tag"
  i=$((i+1))
}
run_pre todo &
run_pre arbustivo --landcover 'Shrubland' &
wait
run_pre sinhumedal --landcover 'Shrubland,Secondary Forest,Primary Forest,Forest,Grassland,Steppe,Dwarf forest'

echo "=== afinado  $(date +%H:%M:%S)"
# contra la MISMA corrida sin --init-from, que ya existe: la diferencia aisla el
# preentrenamiento y nada mas
i=0
for ck in results/mae/*.pt; do
  [ -f "$ck" ] || continue
  n=$(basename "$ck" .pt)
  BIODIV_CURVES=_raw100 CUDA_VISIBLE_DEVICES=$((i%2)) python scripts/11_run_conv.py \
    --substrate $SUB --index kndvi --scheme kfold5_window --context clim+topo+area \
    --init-from "$ck" --seeds 3 --max-epochs 200 --patience 20 \
    > "logs/mae_pool/ft_${n}.log" 2>&1 &
  i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
done
wait
echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/33_search_report.py
