#!/usr/bin/env python3
"""Las facetas leñosas del pool amplio, enmascaradas a Parcelas-CL (sufijo `_woodypclm`).

Para el test de descarte de la textura GLCM (scripts/123), que solo existe en PCL: con
BIODIV_TARGETS=_woodypclm cada bosque se entrena y se evalúa solo sobre las 1.082 parcelas
PCL. Los valores son los del pool leñoso completo (`_woody`) con las filas de Living Trees en
NaN; no se recalculan dentro de PCL. Distinto de `_woodypcl` (scripts/85), que recalcula
LCBD e iNEXT dentro de PCL y solo tiene 226-479 parcelas con valor, porque exige conteos.

Uso:
    python scripts/124_mask_unified_to_pcl.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "derived"
ID = "PlotObservationID"

pcl = set(pd.read_parquet(D / "plots_unified.parquet").query("source == 'parcelas_cl'")[ID])
for src in ("unified_diversity_responses", "unified_phylo_responses", "unified_dark_diversity"):
    d = pd.read_parquet(D / f"{src}_woody.parquet")
    lt = ~d[ID].isin(pcl)
    d.loc[lt, [c for c in d.columns if c != ID]] = float("nan")
    d.to_parquet(D / f"{src}_woodypclm.parquet", index=False)
    print(f"{src}_woodypclm: {(~lt).sum()} filas PCL con valor, {lt.sum()} en NaN")
