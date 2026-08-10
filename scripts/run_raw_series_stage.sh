#!/bin/bash
# Etapa extra: la serie CRUDA de 3 anos como sustrato, en vez del ano compuesto.
#
# Idea de J. Lopatin. PhenoShape colapsa tres anos en un ano promedio, y eso borra la
# variacion interanual -- que medimos que es senal real (el escalon de la frontera sigue a la
# tendencia con pendiente -0,945 contra un -1 predicho, docs/13). Una serie en tiempo
# calendario nunca la destruye.
#
# Y de paso ataca el eje que el paper de Trait_2DCNN senala como no estudiado: el tamano de
# la imagen. Tres anos a paso semanal son 156 pasos -> 13x13, contra el 8x8 del compuesto.
#
# El costo, que hay que decir: ~0,8 observaciones reales por paso semanal contra ~2,4 del
# compuesto, asi que hay mas interpolacion. Si el extra de senal interanual compensa el extra
# de suavizado es exactamente lo que esto mide. Por eso se prueban cuatro resoluciones: 68
# pasos es ~16 dias, la revisita real de Landsat, y no interpola casi nada.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/search

# NO esperar por patron de linea de comandos. Un `pgrep -f "30_search_cnn"` matchea
# cualquier proceso que MENCIONE esa cadena, incluido un shell de monitoreo cuyo cmdline la
# contiene. Paso tres veces en este repo: la cola de GDM bloqueada 40 min, un monitor de
# patchctx con falsos positivos 9 h, y estas dos etapas paradas 6,5 h con el log vacio.
# El encadenamiento va por dependencia explicita de quien lanza, no adivinando por texto.

echo "=== construyendo series crudas  $(date +%H:%M:%S)"
for ngs in 68 100 156 196; do
  sfx="_raw${ngs}"
  [ -f "data/derived/phenoshape_by_index${sfx}.parquet" ] && { echo "[skip] $sfx"; continue; }
  conda run -n phenopy python scripts/29_refit_curves_from_cubes.py --raw-series \
    --ngs "$ngs" --suffix "$sfx" --workers 10 > "logs/search/raw_${ngs}.log" 2>&1 \
    || echo "    FALLO en ngs=$ngs"
done

echo "=== modelos sobre la serie cruda  $(date +%H:%M:%S)"
i=0
for ngs in 68 100 156 196; do
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
