#!/bin/bash
# Las arquitecturas alternativas bajo el esquema que importa.
#
# res/se/multi se probaron SOLO bajo kfold5_window, donde perdieron contra la separable
# simple. Pero ese esquema evalua a 0,47 km y ya vimos que un factor puede invertir su signo
# al pasar a block20: la aumentacion estorbaba alli (+0,012 para noaug) y aqui es lo que mas
# ayuda (+0,021). No hay razon para suponer que la arquitectura se comporte igual en los dos.
#
# Se cruzan con la aumentacion fuerte, que es el unico factor que ya demostro ayudar aqui.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/arch20
export CUDA_VISIBLE_DEVICES=0
SNAP="${SNAP:-scripts/11_run_conv.py}"
AUG="--aug-slope 0.05 --aug-prob 0.5"

run() { local tag=$1; shift
  BIODIV_CURVES=_raw36 python "$SNAP" --substrate serpentine --index kndvi \
    --scheme kfold5_block20 --context clim+topo+area --seeds 3 \
    --max-epochs 200 --patience 20 "$@" > "logs/arch20/${tag}.log" 2>&1 \
    || echo "    FALLO $tag"
}

echo "=== arquitecturas, solas  $(date +%H:%M:%S)"
for a in res se multi; do run "$a" --arch "$a"; done

echo "=== arquitecturas + la aumentacion que aqui si ayuda  $(date +%H:%M:%S)"
for a in res se multi; do run "${a}_aug" --arch "$a" $AUG; done

echo "=== pixel central, que a los RF les ayuda bajo este esquema  $(date +%H:%M:%S)"
run ctr     --px center
run ctr_aug --px center $AUG
run se_ctr_aug --arch se --px center $AUG

echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/40_block20_report.py
