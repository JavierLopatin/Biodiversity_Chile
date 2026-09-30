#!/usr/bin/env python3
"""Isomap contra PCoA como ejes de composición: margen sobre el piso de coordenadas (B03c).

Lee los OOF de las siete corridas unified_all leñosas (kfold5_block20_unified) y, con las
lecturas de scripts/100, da por target y corrida el R² global y dentro de banda de 2° (LT),
y la diferencia pareada por semilla contra B03c (lon, lat, elevación). Criterio fijado antes
de ver los números: Isomap sustituye a PCoA solo si su margen sobre B03c dentro de banda
supera al de PCoA.

Escribe results/tables/isomap_vs_pcoa.csv.

Uso:
    python scripts/118_isomap_vs_pcoa.py
"""
import importlib.util, pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parents[1]; ID="PlotObservationID"; B="kfold5_block20_unified"
s=importlib.util.spec_from_file_location("l",R/"scripts/100_paired_lenses.py"); l=importlib.util.module_from_spec(s); s.loader.exec_module(l)
G="results/models_gate"; sf="raw100_unified-all_unified_woody"
runs={"B03c":f"B03c_coords_{sf}","B01c":f"B01c_topo_ctr-area_{sf}","RFG1c":f"RFG1c_clim-topo_ctr-area_{sf}",
"RFG2c":f"RFG2c_curve-topo_ctr-area_kndvi_{sf}","RFG4c":f"RFG4c_gm-topo_ctr-area_{sf}","RFG5c":f"RFG5c_gm-clim-topo_ctr-area_{sf}","RFG7c":f"RFG7c_lspu-topo_ctr-area_kndvi_{sf}"}
plots=pd.read_parquet(R/"data/derived/plots_unified.parquet")[[ID,"Owner","source","lat"]]
oof={k:pd.read_csv(R/G/v/B/"oof_predictions.csv").merge(plots,on=ID,how="left") for k,v in runs.items()}
T=[f"{a}{i}_{f}_unified" for f in ("pa","freq") for a,ii in (("pcoa",(1,2)),("isomap",(1,2,3))) for i in ii]+["lcbd_pa_unified","hill_q0_unified"]
rows=[]
for t in T:
    L={k:l.lenses(o,t,2.0) for k,o in oof.items()}
    for k in runs:
        d=L[k]-L["B03c"]
        rows.append(dict(target=t,run=k,n=int(L[k].n_todas.iloc[0]),R2=L[k].todas.mean(),R2_banda=L[k].lt_dentro_banda.mean(),
          marg=d.todas.mean(),marg_banda=d.lt_dentro_banda.mean(),marg_banda_DE=d.lt_dentro_banda.std(ddof=1)))
out=pd.DataFrame(rows); out.to_csv(R/"results/tables/isomap_vs_pcoa.csv",index=False)
pd.set_option("display.width",250)
print(out.pivot(index="target",columns="run",values="R2_banda").loc[T].round(3))
print(out.pivot(index="target",columns="run",values="marg_banda").loc[T].round(3))
print(out.pivot(index="target",columns="run",values="R2").loc[T].round(3))
