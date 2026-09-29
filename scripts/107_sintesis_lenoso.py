#!/usr/bin/env python3
"""El cuadro completo del pool lenoso: 20 facetas x 8 representaciones x estres hidrico.

Cerrada la decision de hierbas (`docs/24`, 2026-09-29), el paper es el pool lenoso y nada mas.
Este script arma el resultado general sobre esa base, en una sola tabla, para que la Tabla 3 del
manuscrito se escriba leyendo un archivo y no doce carpetas.

Dos familias de targets, que NO cubren las mismas parcelas y por eso no se comparan entre si:

    pg_all        7 facetas de Perez-Giraldo: LCBD Sorensen y PD/TD Hill q=0,1,2
                  estandarizados por cobertura. n: LCBD 2.499, PD 877-888, TD 881-895.
    unified_all   13 facetas sobre el pool ancho: LCBD p/a y de frecuencia, PCoA 1-2 de las
                  dos, riqueza Hill q0, MPD, MNTD, los tres SES, y riqueza oscura. n 3.102.

El eje de estres sale del OOF, no de corridas aparte: es una particion de las predicciones ya
hechas por tercil de SPI-12. El tercil se calcula DENTRO de cada ano de censo y sobre el SPI
residualizado contra latitud, porque el SPI crudo dentro de un ano es en dos tercios latitud
(rho -0,67) y partir por SPI crudo seria volver a medir el gradiente latitudinal que este
proyecto ya descarto cuatro veces. Solo Living Trees: es un protocolo, area fija, y los anos con
potencia suficiente.

El control que hace legible el eje de estres es el piso. Si el R2 de los bloques remotos cambia
entre terciles PERO el piso topografia+area no se mueve, el cambio no puede ser varianza del
target -- tendria que arrastrar al piso igual. Ese contraste se emite como columna aparte.

Salidas:
    results/tables/sintesis_lenoso.csv          faceta x representacion x estres
    results/tables/sintesis_lenoso_resumen.csv  la mejor representacion por faceta

Uso:
    python scripts/107_sintesis_lenoso.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
GATE = ROOT / "results" / "models_gate"
OUT = ROOT / "results" / "tables" / "sintesis_lenoso.csv"
SCHEME = "kfold5_block20_unified"

REPS = {
    "area_sola":   "B02c_area_raw100",
    "piso":        "B01c_topo_ctr-area_raw100",
    "clima":       "RFG1c_clim-topo_ctr-area_raw100",
    "curva":       "RFG2c_curve-topo_ctr-area_kndvi_raw100",
    "clima_curva": "RFG3c_clim-curve-topo_ctr-area_kndvi_raw100",
    "gm":          "RFG4c_gm-topo_ctr-area_raw100",
    "gm_clima":    "RFG5c_gm-clim-topo_ctr-area_raw100",
    "lsp":         "RFG7c_lspu-topo_ctr-area_kndvi_raw100",
}
#: (sufijo, familia). Las dos familias van por separado en toda la tabla.
FAMILIAS = [("pg-all_unified_woody", "pg_all"),
            ("unified-all_unified_woody", "unified_all")]

#: Como se agrupan las facetas al resumir. La distincion importa: composicion y riqueza se
#: comportan al revés entre si bajo estres, y mezclarlas en un promedio borra el resultado.
GRUPO = {
    "lcbd_count_sorensen": "composicion", "lcbd_pa_unified": "composicion",
    "lcbd_freq_unified": "composicion", "pcoa1_pa_unified": "composicion",
    "pcoa2_pa_unified": "composicion", "pcoa1_freq_unified": "composicion",
    "pcoa2_freq_unified": "composicion",
    "td_inext_q0": "riqueza", "td_inext_q1": "riqueza", "td_inext_q2": "riqueza",
    "hill_q0_unified": "riqueza", "dark_n_unified": "riqueza",
    "pd_inext_q0": "filogenetica", "pd_inext_q1": "filogenetica",
    "pd_inext_q2": "filogenetica", "mpd_unified": "filogenetica",
    "mntd_unified": "filogenetica", "ses_pd_unified": "filogenetica",
    "ses_mpd_unified": "filogenetica", "ses_mntd_unified": "filogenetica",
}
MIN_N_TERCIL = 40


BANDA = 2.0     #: grados de latitud del control


def r2(o, p) -> float:
    o, p = np.asarray(o, float), np.asarray(p, float)
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def r2_en_banda(g: pd.DataFrame, oc: str, pc: str) -> float:
    """R2 tras centrar observado y predicho dentro de cada banda de latitud.

    El control que hace falta porque las dos bases solo se solapan entre 30 y 38 S y varios
    targets de `unified_all` difieren por inventario -- la riqueza media es 3,6 en Living Trees
    y 5,4 en Parcelas-CL, y la diversidad oscura 40,0 contra 69,4. Sin centrar, parte del R2
    agrupado es el gradiente latitudinal y la procedencia, no la relacion que se quiere medir.
    La referencia pasa a ser 0 y no la media global, igual que en `scripts/90` y `scripts/104`.
    """
    o = g[oc] - g.groupby("banda")[oc].transform("mean")
    p = g[pc] - g.groupby("banda")[pc].transform("mean")
    o, p = o.to_numpy(), p.to_numpy()
    return float(1 - ((p - o) ** 2).sum() / (o ** 2).sum())


def contexto() -> pd.DataFrame:
    p = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lat", "Year", "source"]]
    spi = pd.read_parquet(DERIVED / "spi_unified.parquet")
    if spi.index.name:
        spi = spi.reset_index()
    col = [c for c in spi.columns if "spi12" in c and "mean" in c][0]
    return p.merge(spi[["PlotObservationID", col]], on="PlotObservationID",
                   how="left").rename(columns={col: "spi"})


def terciles(g: pd.DataFrame) -> pd.Series:
    """Tercil de SPI dentro del ano de censo, quitado antes el gradiente latitudinal."""
    out = pd.Series(index=g.index, dtype="float64")
    for _, gy in g.groupby("Year"):
        if len(gy) < 3 * MIN_N_TERCIL or gy.lat.nunique() < 3:
            continue
        res = gy.spi - np.polyval(np.polyfit(gy.lat, gy.spi, 1), gy.lat)
        out.loc[gy.index] = pd.qcut(res, 3, labels=[0, 1, 2]).astype(float)
    return out


def main() -> None:
    ctx = contexto().set_index("PlotObservationID")
    rows = []
    for sfx, familia in FAMILIAS:
        for rep, spec in REPS.items():
            f = GATE / f"{spec}_{sfx}" / SCHEME / "oof_predictions.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            nseeds = d.seed.nunique() if "seed" in d else 1
            e = d.join(ctx, on="PlotObservationID")
            e["banda"] = np.floor(e.lat / BANDA) * BANDA
            for t in sorted(c[:-4] for c in d.columns if c.endswith("_obs")):
                oc, pc = f"{t}_obs", f"{t}_pred"
                z = e[e[oc].notna()]
                if len(z) < 50:
                    continue
                per = [r2(g[oc], g[pc]) for _, g in z.groupby("seed")]
                ban = [r2_en_banda(g, oc, pc) for _, g in z.groupby("seed")]
                rows.append(dict(familia=familia, faceta=t, grupo=GRUPO.get(t, "otra"),
                                 representacion=rep, estres="todas",
                                 n=z.PlotObservationID.nunique(), n_seeds=nseeds,
                                 R2=float(np.mean(per)),
                                 R2_sd=float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
                                 R2_en_banda=float(np.mean(ban))))
                lt = z[(z.source == "living_trees") & z.spi.notna()].copy()
                lt["t"] = terciles(lt)
                for k, nombre in enumerate(["seco", "medio", "humedo"]):
                    s = lt[lt.t == k]
                    if s.PlotObservationID.nunique() < MIN_N_TERCIL:
                        continue
                    per = [r2(g[oc], g[pc]) for _, g in s.groupby("seed")]
                    rows.append(dict(familia=familia, faceta=t, grupo=GRUPO.get(t, "otra"),
                                     representacion=rep, estres=nombre,
                                     n=s.PlotObservationID.nunique(), n_seeds=nseeds,
                                     R2=float(np.mean(per)),
                                     R2_sd=float(np.std(per, ddof=1)) if len(per) > 1 else 0.0))

    f = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT, index=False)
    print(f"-> {OUT}  ({len(f)} filas)")

    # --- la mejor representacion por faceta, con el piso al lado para que se lea contra algo
    g = f[f.estres == "todas"]
    pi = g[g.representacion == "piso"].set_index(["familia", "faceta"])
    best = (g[g.representacion != "piso"]
            .sort_values("R2", ascending=False)
            .groupby(["familia", "faceta"]).first().reset_index())
    k = list(zip(best.familia, best.faceta))
    best["R2_piso"] = [pi.R2.get(x, np.nan) for x in k]
    best["R2_piso_en_banda"] = [pi.R2_en_banda.get(x, np.nan) for x in k]
    best["sobre_piso"] = best.R2 - best.R2_piso
    best["sobre_piso_en_banda"] = best.R2_en_banda - best.R2_piso_en_banda
    best = best[["familia", "grupo", "faceta", "n", "representacion", "R2", "R2_sd",
                 "R2_piso", "sobre_piso", "R2_en_banda", "R2_piso_en_banda",
                 "sobre_piso_en_banda"]].sort_values(["grupo", "sobre_piso_en_banda"],
                                                     ascending=[True, False])
    best.to_csv(OUT.with_name("sintesis_lenoso_resumen.csv"), index=False)
    print(f"-> {OUT.with_name('sintesis_lenoso_resumen.csv')}")
    print("\nMejor representacion por faceta. `en_banda` centra dentro de 2 grados de latitud:")
    print(best.round(3).to_string(index=False))

    print()
    print(contraste_estres(f).round(3).to_string(index=False))


def contraste_estres(f: pd.DataFrame, bloque: str = "gm_clima") -> pd.DataFrame:
    """Seco contra humedo, descontando al piso su propio cambio entre terciles.

    La columna que se lee es `neto`, no `bruto`. El bruto de una faceta puede cambiar porque
    cambia la varianza del target entre terciles, y eso arrastraria al piso igual que al bloque
    remoto; la resta lo cancela. Es la diferencia entre "el predictor remoto funciona distinto
    segun el estado hidrico" y "el target es mas facil en un tercil".

    Sin esta resta, `hill_q0_unified` parece predecirse mejor en seco (bruto -0,084) y sostiene
    la idea de que la sequia se invierte entre riqueza y composicion. Su piso se mueve -0,132, o
    sea MAS, asi que el neto es +0,049 y apunta al mismo lado que todo lo demas.
    """
    w = f[f.estres.isin(["seco", "humedo"])]
    piso = w[w.representacion == "piso"].set_index(["faceta", "estres"]).R2
    r = w[w.representacion == bloque].set_index(["faceta", "estres"])
    out = []
    for fc in r.index.get_level_values(0).unique():
        try:
            bs, bh = r.loc[(fc, "seco"), "R2"], r.loc[(fc, "humedo"), "R2"]
            ps, ph = piso.loc[(fc, "seco")], piso.loc[(fc, "humedo")]
        except KeyError:
            continue
        out.append(dict(faceta=fc, grupo=r.loc[(fc, "seco"), "grupo"], bloque=bloque,
                        R2_seco=bs, R2_humedo=bh, bruto=bh - bs,
                        piso_seco=ps, piso_humedo=ph, piso_delta=ph - ps,
                        neto=(bh - ph) - (bs - ps)))
    o = pd.DataFrame(out).sort_values("neto", ascending=False)
    o.to_csv(OUT.with_name("contraste_estres_lenoso.csv"), index=False)
    print(f"-> {OUT.with_name('contraste_estres_lenoso.csv')}")
    print(f"\nSeco contra humedo, {bloque}, descontado el piso "
          f"({int((o.neto > 0).sum())} de {len(o)} facetas mejoran en humedo):")
    return o[["grupo", "faceta", "R2_seco", "R2_humedo", "bruto", "piso_delta", "neto"]]


if __name__ == "__main__":
    main()
