#!/usr/bin/env python3
"""Que distingue a las parcelas que el modelo acierta de las que falla.

El paper tiene R2 y no tiene ecologia. Esta es la capa que falta, y es deliberadamente una
pregunta de residuos y no una exploracion: "en zonas con distintas caracteristicas de
biodiversidad" se convierte en pesca; "que distingue lo bien predicho de lo mal predicho" esta
acotado por construccion y alimenta la Discusion.

Se parte el OOF del mejor bloque (gm+clima) sobre `lcbd_count_sorensen` en cuartiles de residuo
absoluto, y se describe cada cuartil con tres cosas:

    composicion    generos y especies dominantes
    estructura     numero de especies, y que fraccion de la parcela se lleva la dominante
    nicho          fraccion de generalistas altitudinales contra especialistas estrechos,
                   con el rasgo del catalogo de Rodriguez et al. 2018 (`scripts/81`)

La tercera columna usa el rasgo como DESCRIPCION, no como respuesta nueva a modelar. Si el
modelo falla donde dominan los especialistas estrechos, eso es una frase de Discusion con
respaldo; convertirlo en un target seria otro paper (`docs/27`).

Dos cautelas que van en la lectura, no en el codigo:

- El residuo absoluto mezcla dos cosas distintas: parcelas cuyo LCBD es extremo y el modelo
  regresa a la media, y parcelas donde el predictor no informa. Por eso se reporta tambien el
  LCBD observado por cuartil: si el mal predicho es simplemente el de LCBD alto, la lectura es
  regresion a la media y no una propiedad ecologica.
- El rasgo de nicho cubre el 70 % de las lenosas del pool. La fraccion se calcula sobre las
  especies con rasgo, y se reporta la cobertura por cuartil para que se vea si el hueco esta
  repartido o concentrado.

Salida: `results/tables/residuos_composicion.csv` y `residuos_taxones.csv`.

Uso:
    python scripts/111_residuos_composicion.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
GATE = ROOT / "results" / "models_gate"
OUT = ROOT / "results" / "tables" / "residuos_composicion.csv"
SCHEME = "kfold5_block20_unified"
RUN = "RFG5c_gm-clim-topo_ctr-area_raw100_pg-all_unified_woody"
TARGET = "lcbd_count_sorensen"
N_TAXONES = 8      #: cuantos generos/especies listar por cuartil


def cuartiles() -> pd.DataFrame:
    """Parcelas con su residuo absoluto medio entre semillas, en cuartiles."""
    f = GATE / RUN / SCHEME / "oof_predictions.csv"
    if not f.exists():
        raise SystemExit(f"No esta {f}. Es la corrida gm+clima sobre el pool lenoso.")
    d = pd.read_csv(f)
    d = d[d[f"{TARGET}_obs"].notna()].copy()
    d["res"] = (d[f"{TARGET}_pred"] - d[f"{TARGET}_obs"]).abs()
    g = (d.groupby("PlotObservationID")
         .agg(res=("res", "mean"), obs=(f"{TARGET}_obs", "first"),
              pred=(f"{TARGET}_pred", "mean")).reset_index())
    g["cuartil"] = pd.qcut(g.res, 4, labels=["Q1 mejor", "Q2", "Q3", "Q4 peor"])
    return g


def rasgo_nicho() -> pd.Series:
    """species -> tercil de amplitud altitudinal, sobre las lenosas con rasgo.

    Los terciles se calculan sobre las especies del pool, no sobre el catalogo entero: el corte
    tiene que separar el pool que se esta describiendo. Es un corte por cuantil y por tanto
    arbitrario; se declara como tal.
    """
    cat = pd.read_csv(DERIVED / "rodriguez2018_habitos.csv")
    lk = pd.read_csv(DERIVED / "growth_form_lookup.csv")
    w = lk[lk.forma == "lenosa"][["species"]].merge(
        cat[["species", "amp_alt"]].drop_duplicates("species"), on="species", how="left")
    w = w[w.amp_alt.notna()]
    if len(w) < 50:
        raise SystemExit("Menos de 50 lenosas con rango altitudinal: corre scripts/81 primero.")
    q = pd.qcut(w.amp_alt, 3, labels=["estrecho", "medio", "amplio"])
    return pd.Series(q.to_numpy(), index=w.species.to_numpy())


def main() -> None:
    g = cuartiles()
    occ = pd.read_parquet(DERIVED / "occurrences_unified_counts_woody.parquet")
    occ = occ[occ.PlotObservationID.isin(set(g.PlotObservationID))].copy()
    occ["genero"] = occ.species.str.split().str[0]
    nicho = rasgo_nicho()
    occ["nicho"] = occ.species.map(nicho)
    occ = occ.merge(g[["PlotObservationID", "cuartil"]], on="PlotObservationID")

    # --- estructura por parcela, y despues promediada por cuartil
    por_parcela = (occ.groupby("PlotObservationID")
                   .agg(n_sp=("species", "nunique"),
                        dom=("Value", lambda v: v.max() / v.sum() if v.sum() else np.nan))
                   .reset_index())
    m = g.merge(por_parcela, on="PlotObservationID", how="left")

    filas = []
    for q, s in m.groupby("cuartil", observed=True):
        o = occ[occ.cuartil == q]
        con = o[o.nicho.notna()]
        filas.append(dict(
            cuartil=q, n_parcelas=len(s),
            residuo_medio=s.res.mean(), lcbd_obs_medio=s.obs.mean(),
            n_especies=s.n_sp.mean(), dominancia=s.dom.mean(),
            n_taxones_distintos=o.species.nunique(),
            cobertura_rasgo=len(con) / len(o) if len(o) else np.nan,
            frac_estrecho=(con.nicho == "estrecho").mean() if len(con) else np.nan,
            frac_amplio=(con.nicho == "amplio").mean() if len(con) else np.nan))
    t = pd.DataFrame(filas)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print(f"-> {OUT}")
    print(t.round(3).to_string(index=False))

    # --- que taxones dominan cada cuartil, por frecuencia de aparicion en parcelas
    tx = []
    for q, o in occ.groupby("cuartil", observed=True):
        npl = o.PlotObservationID.nunique()
        for col in ("genero", "species"):
            fr = (o.groupby(col).PlotObservationID.nunique() / npl).sort_values(ascending=False)
            for nombre, f in fr.head(N_TAXONES).items():
                tx.append(dict(cuartil=q, nivel=col, taxon=nombre, frac_parcelas=f))
    x = pd.DataFrame(tx)
    x.to_csv(OUT.with_name("residuos_taxones.csv"), index=False)
    print(f"\n-> {OUT.with_name('residuos_taxones.csv')}")
    print("\nGeneros mas frecuentes, mejor contra peor cuartil:")
    w = (x[x.nivel == "genero"].pivot_table(index="taxon", columns="cuartil",
                                            values="frac_parcelas")
         .reindex(columns=["Q1 mejor", "Q4 peor"]).dropna(how="all"))
    w["dif"] = w["Q4 peor"].fillna(0) - w["Q1 mejor"].fillna(0)
    print(w.sort_values("dif").round(3).to_string())


if __name__ == "__main__":
    main()
