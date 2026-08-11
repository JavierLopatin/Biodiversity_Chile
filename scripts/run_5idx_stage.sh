#!/bin/bash
# Los cinco indices como CANALES de la misma imagen.
#
# Corrige una asimetria que corria en contra de la convolucion: RF06 y MLP07 leen los cinco
# indices y cada corrida C2D leia uno. La comparacion no era 2D contra tabla, era un indice
# contra cinco. `stack5` no es lo mismo: alli los indices son cinco FILAS de una imagen y un
# kernel cruza indices y semanas a la vez; aqui cada indice conserva su disposicion espacial
# y la mezcla ocurre en los canales, igual que en los modelos tabulares.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p logs/search
echo "=== 5 indices como canales  $(date +%H:%M:%S)"
i=0
for sfx in _raw36 _raw100; do
  for sub in serpentine_5idx reshape_5idx hilbert_5idx; do
    BIODIV_CURVES=$sfx CUDA_VISIBLE_DEVICES=$((i%2)) \
      python scripts/11_run_conv.py --substrate $sub --scheme kfold5_window \
      --context clim+topo+area --seeds 3 --max-epochs 200 --patience 20 \
      > "logs/search/m5${sfx}_${sub}.log" 2>&1 || echo "    FALLO $sfx $sub"
    i=$((i+1)); [ $((i%2)) -eq 0 ] && wait
  done
done
wait
echo "=== confirmacion del mejor con semillas {10..14}  $(date +%H:%M:%S)"
read -r BSFX BSUB <<< "$(python - <<'PY'
import pandas as pd, numpy as np, sys, re
sys.path.insert(0, "src"); from biodiv import targets as tg
s = pd.read_csv("results/models/summary.csv"); s = s[s.scheme == "kfold5_window"]
r = s[s.run_id.str.contains("_5idx") & ~s.run_id.str.contains("_s10")]
sc = {rid: np.mean([g[g.target.isin(t)].R2.mean() for t in tg.FACETS.values()])
      for rid, g in r.groupby("run_id")}
best = max(sc, key=sc.get)
print("_raw" + re.search(r"_raw(\d+)", best).group(1),
      re.search(r"(\w+)_5idx", best).group(1) + "_5idx")
PY
)"
echo "mejor: $BSUB sobre $BSFX"
BIODIV_CURVES=$BSFX python scripts/11_run_conv.py --substrate $BSUB --scheme kfold5_window \
  --context clim+topo+area --seed-start 10 --seeds 5 --max-epochs 200 --patience 20 \
  > "logs/search/conf_5idx.log" 2>&1 || echo "    FALLO confirmacion"
echo "=== fin  $(date +%F\ %H:%M:%S)"
python scripts/33_search_report.py
