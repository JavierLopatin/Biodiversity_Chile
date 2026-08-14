#!/bin/bash
# Buscar la 2D-CNN bajo el esquema que importa para un mapa.
#
# Las ~195 combinaciones de `docs/14` se buscaron TODAS bajo kfold5_window, que evalua a
# 0,47 km. La CNN esta sintonizada para interpolar. Bajo kfold5_block20 -- 12,1 km, la
# distancia a la que un mapa predice -- solo hay UNA configuracion C2D corrida (0,180),
# contra 0,259 del mejor RF.
#
# LA HIPOTESIS, y es falsable: bajo `window` la aumentacion ESTORBABA (`noaug` ganaba por
# +0,012) porque el test estaba a 500 m y era casi la misma distribucion. Bajo `block20` el
# test es una region entera no vista, que es exactamente donde aumentar y regularizar deberian
# pagar. Si la prediccion es correcta, el orden de los factores se invierte respecto de
# docs/14; si no, la CNN no gana y eso tambien es un resultado.
#
# Un solo GPU (el 0): el 1 esta ocupado con las corridas del pool unificado.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/b20search
export CUDA_VISIBLE_DEVICES=0

run() {  # run <etiqueta> <curvas> <sustrato> <extra...>
  local tag=$1 crv=$2 sub=$3; shift 3
  BIODIV_CURVES="$crv" python "$SNAP" --substrate "$sub" --index kndvi \
    --scheme kfold5_block20 --context clim+topo+area --seeds 3 \
    --max-epochs 200 --patience 20 "$@" \
    > "logs/b20search/${tag}.log" 2>&1 || echo "    FALLO $tag"
}

echo "=== 1. el factor de la hipotesis: aumentacion  $(date +%H:%M:%S)"
run aug_on     _raw36 serpentine
run aug_off    _raw36 serpentine --no-augment
run aug_mixup  _raw36 serpentine --mixup
run aug_fuerte _raw36 serpentine --aug-slope 0.05 --aug-prob 0.5

echo "=== 2. regularizacion  $(date +%H:%M:%S)"
run wd003   _raw36 serpentine --weight-decay 0.03
run wd01    _raw36 serpentine --weight-decay 0.1
run drop    _raw36 serpentine --p-conv 0.2 --p-head 0.5
run wA      _raw36 serpentine --width A
run wC      _raw36 serpentine --width C

echo "=== 3. sustrato y curva  $(date +%H:%M:%S)"
for sub in reshape hilbert spectrogram; do run "sub_$sub" _raw36 "$sub"; done
for crv in _raw100 _raw24 ""; do
  run "crv${crv:-comp}" "$crv" serpentine
done
run idx5 _raw36 serpentine_5idx

echo "=== 4. combinar los positivos  $(date +%H:%M:%S)"
# se rellena tras leer el paso 1-3; aqui van las dos apuestas a priori
run comb_reg_aug _raw36 serpentine --weight-decay 0.03 --aug-prob 0.5 --aug-slope 0.05
run comb_wA_mix  _raw36 serpentine --width A --mixup

echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/40_block20_report.py 2>/dev/null || true
