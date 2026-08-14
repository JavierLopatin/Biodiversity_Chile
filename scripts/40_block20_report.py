#!/usr/bin/env python3
"""Leaderboard bajo `kfold5_block20`: el esquema cuya distancia de evaluacion (12,1 km) se
parece a la del mapa (13,2 km).

Existe aparte de `33_search_report.py` porque responden preguntas distintas. Aquel ordena
modelos por interpolacion en un territorio muestreado; este los ordena por lo que un mapa va
a hacer. Medido, el orden **no es el mismo**, y publicar solo uno de los dos seria elegir la
pregunta por su respuesta.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biodiv import metrics as mx                 # noqa: E402
from biodiv import targets as tg                 # noqa: E402

s = pd.read_csv(ROOT / "results/models/summary.csv")
s = s[s.scheme == "kfold5_block20"]
m = mx.by_facet(s).sort_values("media", ascending=False)
cols = list(tg.FACETS) + ["media"]
print("=== todas las corridas bajo kfold5_block20 ===")
print(m[["family"] + cols].head(20).round(3).to_string())
best = m.loc[m.groupby("family")["media"].idxmax()].sort_values("media", ascending=False)
print("\n=== mejor de cada familia ===")
print(best[["family"] + cols].round(3).to_string())
print("\n=== quien gana CADA faceta ===")
print("La media entre facetas esconde el resultado que importa: las familias no ganan en las")
print("mismas respuestas. La convolucion lee la FORMA de la curva, que es lo que se relaciona")
print("con riqueza y estructura filogenetica; el bosque lee NIVELES absolutos, que es lo que")
print("gobierna composicion y diversidad oscura.\n")
rows = []
for f in list(tg.FACETS):
    t = m.sort_values(f, ascending=False).head(1)
    rid = t.index[0]
    second = m[m.family != t["family"].iloc[0]].sort_values(f, ascending=False).head(1)
    rows.append(dict(faceta=f, ganador=t["family"].iloc[0], valor=t[f].iloc[0],
                     corrida=rid[:46],
                     mejor_otra_familia=second["family"].iloc[0],
                     valor2=second[f].iloc[0]))
print(pd.DataFrame(rows).round(3).to_string(index=False))

c2d = m[m.family == "C2D"]["media"]
rf = m[m.family == "RF"]["media"]
if len(c2d) and len(rf):
    print(f"\nbrecha C2D contra RF: {c2d.max() - rf.max():+.3f} "
          f"(umbral de ruido 2 sd = 0,022)")
