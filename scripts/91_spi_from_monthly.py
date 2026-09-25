#!/usr/bin/env python3
"""SPI por parcela, y el valor que le toca a la ventana causal de cada censo.

Por que. Las parcelas se censaron entre 2003 y 2026 y el predictor de cada una es la
ventana de tres anos anterior a su censo. En un ano seco la reflectancia cambia mucho y la
diversidad lenosa no -- los arboles no mueren en una sequia de un ano --, asi que juntar
parcelas de anos distintos mete en el predictor una variacion que la respuesta no acompana.
`docs/05_data_acquisition.md:100-119` ya planteaba medir esto y lo dejo sin implementar.

Por que el SPI y no el ano. En el eje temporal, ano de censo, era del sensor y periodo de
sequia son inseparables con este diseno: medido, rho(%Landsat8, R2 del ano) = +0,64. El SPI
sirve porque varia tambien en el ESPACIO: dos parcelas censadas el mismo ano, una en una
cuenca golpeada y otra no, comparten era del sensor y difieren en estres hidrico. Eso es lo
unico que rompe la colinealidad.

Definicion. SPI de escala k: se acumula la precipitacion en ventanas moviles de k meses, se
ajusta una gamma a la serie de cada mes calendario sobre el periodo de referencia, y se
transforma a normal estandar. Los ceros se tratan aparte (distribucion mixta de Thom), que
importa en el norte arido donde meses secos completos son comunes.

Entrada: la serie mensual que escribe `scripts/22_extract_climate.py --monthly-out`.

Uso:
    python scripts/22_extract_climate.py --plots data/derived/plots_unified.parquet \
        --out data/derived/climate_unified.parquet \
        --monthly-out data/derived/precip_monthly_unified.parquet \
        --start 1979 --end 2021
    python scripts/91_spi_from_monthly.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

ROOT = Path(__file__).resolve().parents[1]
ID = "PlotObservationID"


def spi_series(pr: np.ndarray, scale: int) -> np.ndarray:
    """SPI de escala `scale` para una serie mensual continua de una parcela.

    `pr` es (n_meses,) en orden cronologico. Devuelve el mismo largo, con NaN en los
    primeros `scale-1` meses, que no tienen ventana completa.
    """
    acc = pd.Series(pr).rolling(scale).sum().to_numpy()
    out = np.full_like(acc, np.nan, dtype=float)
    for m in range(12):
        idx = np.arange(m, len(acc), 12)
        v = acc[idx]
        ok = np.isfinite(v)
        if ok.sum() < 20:                      # menos de 20 anos: no se ajusta
            continue
        x = v[ok]
        # Thom: P(0) aparte, gamma sobre los positivos
        zero = x <= 0
        q = zero.mean()
        pos = x[~zero]
        if len(pos) < 10:
            continue
        a, loc, sc = gamma.fit(pos, floc=0)
        cdf = np.where(x <= 0, q, q + (1 - q) * gamma.cdf(x, a, loc=loc, scale=sc))
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        out[idx[ok]] = norm.ppf(cdf)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly", default="data/derived/precip_monthly_unified.parquet")
    ap.add_argument("--plots", default="data/derived/plots_unified.parquet")
    ap.add_argument("--scales", default="12,24")
    ap.add_argument("--out", default="data/derived/spi_unified.parquet")
    a = ap.parse_args()

    f = ROOT / a.monthly
    if not f.exists():
        raise SystemExit(
            f"{f} no existe. Corre antes:\n"
            "  python scripts/22_extract_climate.py "
            "--plots data/derived/plots_unified.parquet "
            "--out data/derived/climate_unified.parquet "
            "--monthly-out data/derived/precip_monthly_unified.parquet "
            "--start 1979 --end 2021\n"
            "(necesita conexion a Data Cube Chile)"
        )
    m = pd.read_parquet(f).sort_values([ID, "year", "month"])
    plots = pd.read_parquet(ROOT / a.plots).set_index(ID)
    scales = [int(s) for s in a.scales.split(",")]

    rows = []
    for pid, g in m.groupby(ID, sort=False):
        rec = {ID: pid}
        ym = g.year.to_numpy() * 12 + g.month.to_numpy() - 1
        for k in scales:
            s = spi_series(g.pr.to_numpy(float), k)
            # el SPI que le toca al censo: el del ultimo mes de su ventana causal
            we = plots.at[pid, "win_end"] if pid in plots.index else np.nan
            rec[f"spi{k}"] = np.nan
            if np.isfinite(we):
                target = int(we) * 12 + 11        # diciembre del ano de cierre
                j = np.where(ym == target)[0]
                if len(j):
                    rec[f"spi{k}"] = s[j[0]]
            # y la media del SPI sobre los 36 meses de la ventana, mas robusta
            if np.isfinite(we):
                lo = (int(we) - 2) * 12
                hi = int(we) * 12 + 11
                w = s[(ym >= lo) & (ym <= hi)]
                rec[f"spi{k}_win_mean"] = np.nanmean(w) if np.isfinite(w).any() else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    fo = ROOT / a.out
    out.to_parquet(fo, index=False)
    print(f"-> {fo}  ({len(out)} parcelas, {out.shape[1]-1} columnas)")
    print(out.drop(columns=[ID]).describe().T.round(3).to_string())
    na = out.drop(columns=[ID]).isna().any(axis=1).sum()
    print(f"\nparcelas con algun SPI faltante: {int(na)}")
    if ID in plots.index.names or True:
        j = out.set_index(ID).join(plots[["source", "Year"]])
        print("\npor fuente:")
        print(j.groupby("source")[[c for c in out.columns if c.startswith("spi")]]
               .mean().round(3).to_string())


if __name__ == "__main__":
    main()
