#!/bin/bash
# Re-afinado del MAE con identidades distintas por checkpoint.
#
# La primera pasada colisiono: los tres checkpoints compartian la etiqueta `mae`, asi que dos
# se pisaron y el tercero salio `[skip]`. Corregido en `_variant_tags` y blindado en
# `tests/test_run_ids.py`, que exige que cada flag cambie el run_id.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/search
echo "=== re-afinado MAE  $(date +%H:%M:%S)"
i=0
for ck in results/mae/mae_serpentine_*.pt; do
  [ -f "$ck" ] || continue
  n=$(basename "$ck" .pt)
  CUDA_VISIBLE_DEVICES=$((i%2)) python scripts/11_run_conv.py --substrate serpentine \
    --index kndvi --scheme kfold5_window --context clim+topo+area --init-from "$ck" \
    --seeds 3 --max-epochs 200 --patience 20 > "logs/search/ft2_${n}.log" 2>&1 &
  i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
done
wait
echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/30_search_cnn.py --report
