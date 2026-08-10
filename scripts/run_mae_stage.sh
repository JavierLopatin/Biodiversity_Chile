#!/bin/bash
# Etapa B4: preentrenamiento por enmascarado y afinado.
#
# Es lo unico de la busqueda que ataca la restriccion real. Las etapas de curva, sustrato y
# arquitectura mueven el MODELO, y todas convergieron a ~0,36: los diez reconstructores se
# reparten en 0,010 (bajo el ruido), las combinaciones de los factores positivos de la
# ablacion 4c no se suman, y las tres arquitecturas nuevas no superan a la separable. Lo que
# limita no es la arquitectura sino que hay 1.082 etiquetas. El MAE gasta las 135.250 curvas
# de pixel que estan en disco y nadie ha usado.
#
# El afinado se compara contra la MISMA corrida sin `--init-from`, que ya existe, asi que la
# diferencia aisla el preentrenamiento y nada mas.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/search results/mae

# espera a que la etapa de arquitectura suelte los GPUs. El patron va entre corchetes para
# que este propio script no se matchee a si mismo -- ya paso con pgrep dos veces en el repo.
# NO esperar por patron de linea de comandos. Un `pgrep -f "30_search_cnn"` matchea
# cualquier proceso que MENCIONE esa cadena, incluido un shell de monitoreo cuyo cmdline la
# contiene. Paso tres veces en este repo: la cola de GDM bloqueada 40 min, un monitor de
# patchctx con falsos positivos 9 h, y estas dos etapas paradas 6,5 h con el log vacio.
# El encadenamiento va por dependencia explicita de quien lanza, no adivinando por texto.

SUB=serpentine
echo "=== preentrenamiento  $(date +%H:%M:%S)"
for cfg in "kndvi 0.6" "all 0.6" "all 0.75"; do
  set -- $cfg; ixs=$1; mask=$2
  arg=$([ "$ixs" = all ] && echo "--indices all" || echo "--index $ixs")
  echo "--- indices=$ixs mask=$mask"
  python scripts/31_pretrain_mae.py --substrate $SUB $arg --mask-ratio $mask \
    --epochs 40 --batch-size 512 > "logs/search/mae_pre_${ixs}_${mask}.log" 2>&1 \
    || echo "    FALLO"
done

echo "=== afinado  $(date +%H:%M:%S)"
i=0
for ck in results/mae/mae_${SUB}_*.pt; do
  [ -f "$ck" ] || continue
  n=$(basename "$ck" .pt)
  CUDA_VISIBLE_DEVICES=$((i%2)) python scripts/11_run_conv.py --substrate $SUB --index kndvi \
    --scheme kfold5_window --context clim+topo+area --init-from "$ck" \
    --seeds 3 --max-epochs 200 --patience 20 > "logs/search/ft_${n}.log" 2>&1 &
  i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
done
wait
echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/30_search_cnn.py --report
