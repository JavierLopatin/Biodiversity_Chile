#!/usr/bin/env python3
"""El patrón de muestreo, y a qué distancia pone a prueba cada esquema de validación.

Campo obligatorio de STeMP (`Sampling pattern`), pero la parte que decide algo es la
segunda: **comparar la distancia a la que cada esquema de CV evalúa contra la distancia a la
que el mapa va a predecir de verdad.**

    Gj    de cada parcela a la parcela mas proxima
    Gij   de cada punto del dominio nativo a la parcela mas proxima  <- lo que el mapa hara
    CV    de cada parcela de test a la parcela de entrenamiento mas proxima de SU fold

Si la curva `CV` de un esquema cae encima de `Gij`, ese esquema mide lo que el mapa va a
hacer. Si cae encima de `Gj`, mide interpolación a corta distancia y su R² es optimista para
el mapa por construcción -- que es el argumento de Milà et al. (2022) y el criterio con el
que `kNNDM` construye folds.

El dominio de predicción son las **16.950 muestras del pool no etiquetado**
(`data/derived/unlabelled/`), que ya están en disco y son una muestra de la cobertura nativa
enmascarada con MapBiomas repartida por las 1.449 celdas del territorio. Usarlas evita
depender de los rásters, que pesan 3,1 GB y viven en la máquina del datacube.

Uso:
    python scripts/36_sampling_pattern.py
    python scripts/36_sampling_pattern.py --schemes kfold5_window kfold5_random
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import cv as cvmod                   # noqa: E402
from biodiv import figures as fg                 # noqa: E402
from biodiv import spatial_diag as sd            # noqa: E402

DEFAULT_SCHEMES = ["kfold5_random", "kfold5_owner", "kfold5_block20", "kfold5_window"]


def cv_distances(plots: pd.DataFrame, folds: pd.DataFrame, scheme: str) -> np.ndarray:
    """De cada parcela de test, la distancia a la parcela de entrenamiento más próxima.

    Es la distancia de predicción que el esquema **realmente** pone a prueba, y no coincide
    con la nominal: `kfold5_owner` agrupa por contribuyente, pero como los contribuyentes
    están agrupados en el espacio, la distancia efectiva que produce es otra cosa.
    """
    xy = plots.set_index(cvmod.ID_COL)[["X", "Y"]]
    out = []
    for _, _, tr, te in cvmod.iter_folds(folds, scheme):
        a = xy.reindex(te).dropna().to_numpy()
        b = xy.reindex(tr).dropna().to_numpy()
        if len(a) and len(b):
            out.append(sd.nnd(a, b))
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--schemes", nargs="+", default=DEFAULT_SCHEMES)
    p.add_argument("--out", default="results/tables/sampling_pattern.json")
    p.add_argument("--fig-dir", default="results/figures", dest="fig_dir")
    args = p.parse_args()

    d = Path(args.derived)
    plots = pd.read_parquet(d / "plots_subset.parquet")
    pool = d / "unlabelled" / "manifest.csv"
    if not pool.exists():
        raise SystemExit(f"{pool} no existe -- ver docs/15, la extraccion corre en la "
                         "maquina con datacube")
    dom = pd.read_csv(pool)
    dom = dom[dom["X"].notna() & dom["Y"].notna()]

    pat = sd.sampling_pattern(plots[["X", "Y"]].to_numpy(), dom[["X", "Y"]].to_numpy())
    s = pat.summary()
    print(f"patron: {s['label'].upper()}   Clark-Evans R = {s['clark_evans_R']}")
    print(f"  {s['n_samples']} parcelas contra {s['n_domain']} puntos de dominio nativo, "
          f"{s['area_km2']:,.0f} km2")
    print(f"  Gj  mediana {s['Gj_p50_km']:.2f} km   (p5 {s['Gj_p5_km']:.2f}, "
          f"p95 {s['Gj_p95_km']:.2f})")
    print(f"  Gij mediana {s['Gij_p50_km']:.2f} km   (p5 {s['Gij_p5_km']:.2f}, "
          f"p95 {s['Gij_p95_km']:.2f})")
    print(f"  el mapa predice {s['median_ratio_Gij_Gj']}x mas lejos que la distancia "
          f"entre parcelas\n")

    folds = cvmod.load_schemes(d / "cv_folds_modelling.parquet")
    have = set(folds["scheme"])
    rows, curves = [], {"Gj (parcela->parcela)": pat.gj, "Gij (dominio->parcela)": pat.gij}
    for sc in args.schemes:
        if sc not in have:
            print(f"  [falta] {sc}")
            continue
        dist = cv_distances(plots, folds, sc)
        curves[sc] = dist
        rows.append(dict(scheme=sc, n=len(dist),
                         p50_km=float(np.median(dist)) / 1000,
                         p95_km=float(np.percentile(dist, 95)) / 1000,
                         # 1.0 = el esquema evalua justo a la distancia del mapa
                         vs_Gij=float(np.median(dist) / np.median(pat.gij))))
    tab = pd.DataFrame(rows).sort_values("p50_km")
    print("distancia de prediccion que cada esquema pone a prueba:")
    print(tab.round(3).to_string(index=False))
    print("\n  vs_Gij = 1 significa que el esquema evalua a la misma distancia a la que el "
          "mapa predice;\n  < 1 lo evalua mas cerca, y su R2 es optimista para el mapa.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pattern": s, "schemes": rows}, indent=2))

    # --- figura: ECDF de todas las distancias, que es como STeMP la presenta
    import matplotlib.pyplot as plt
    fg.set_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, v in curves.items():
        v = np.sort(v[np.isfinite(v)]) / 1000
        style = dict(lw=1.8, zorder=3) if name.startswith("G") else dict(lw=1.0, ls="--")
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=name, **style)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("distancia al vecino de entrenamiento más próximo (km)")
    ax.set_ylabel("proporción acumulada")
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Lo que la validación pone a prueba, contra lo que el mapa hará", fontsize=8)
    fg.save_figure(fig, "sampling_pattern", args.fig_dir)
    print(f"\n  -> {args.out}\n  -> {args.fig_dir}/sampling_pattern.pdf")


if __name__ == "__main__":
    main()
