#!/bin/bash
# Atacar el deficit diagnosticado, no buscar a ciegas.
#
# Bajo kfold5_block20 la C2D pierde la media por 0,058 y CASI 0,04 es diversidad oscura sola:
# 0,010 contra 0,207 del RF. Las otras cuatro facetas estan a tiro o ganadas.
#
# La oscura depende del NIVEL de productividad y del contexto regional, no de la forma de la
# curva. Y el contexto entra hoy por `--fusion late`: un vector concatenado en la cabeza,
# despues de que la convolucion ya resumio todo. Tres modos sin probar bajo este esquema:
#
#   film      modula las caracteristicas convolucionales con el contexto (FiLM). El contexto
#             puede fijar el nivel mientras la conv lee la forma -- el mecanismo que falta.
#   patch     la topografia como PARCHE 2D, no como vector.
#   patchctx  parche + vector.
#
# Y los dos sustratos cuyo segundo eje no es manufacturado (stack5, pxcube) nunca se corrieron
# con contexto climatico bajo block20.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/fusion
export CUDA_VISIBLE_DEVICES=0
SNAP="${SNAP:-scripts/11_run_conv.py}"

run() { local tag=$1 crv=$2; shift 2
  BIODIV_CURVES="$crv" python "$SNAP" --index kndvi --scheme kfold5_block20 \
    --context clim+topo+area --seeds 3 --max-epochs 200 --patience 20 "$@" \
    > "logs/fusion/${tag}.log" 2>&1 || echo "    FALLO $tag"
}

echo "=== fusion del contexto  $(date +%H:%M:%S)"
run film      _raw36 --substrate serpentine --fusion film
run patch     _raw36 --substrate serpentine --fusion patch
run patchctx  _raw36 --substrate serpentine --fusion patchctx

echo "=== fusion + la aumentacion ganadora  $(date +%H:%M:%S)"
run film_aug     _raw36 --substrate serpentine --fusion film --aug-slope 0.05 --aug-prob 0.5
run patchctx_aug _raw36 --substrate serpentine --fusion patchctx --aug-slope 0.05 --aug-prob 0.5
run film_aug_wd  _raw36 --substrate serpentine --fusion film --aug-slope 0.05 --aug-prob 0.5 --weight-decay 0.03

echo "=== segundo eje no manufacturado, con contexto  $(date +%H:%M:%S)"
run stack5_ctx _raw36 --substrate stack5
run pxcube_ctx _raw36 --substrate pxcube
run pxcube_film _raw36 --substrate pxcube --fusion film

echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/40_block20_report.py
