#!/usr/bin/env python3
"""Cuánto del territorio nativo cae fuera del espacio que las 1.082 parcelas cubren.

Aplica el área de aplicabilidad (Meyer & Pebesma 2021, `src/biodiv/aoa.py`) al pool no
etiquetado: 16.950 series de kNDVI repartidas por las 1.449 celdas de 10 km con cobertura
nativa. No es todavía el mapa, pero es la mejor aproximación disponible a su dominio, y
responde antes de gastar un solo cómputo de mapa a la pregunta que un revisor hará después:
**¿dónde tiene este modelo derecho a predecir?**

El espacio de predictores es la **curva de kNDVI**, que es el bloque que se puede calcular
para cualquier píxel desde Landsat. Se deja fuera a propósito lo que un píxel no tiene:

* `log10_area` y `stratum` son propiedades de la parcela, no del terreno. Están en todos los
  diseños del proyecto y **no son mapeables**; un mapa tendrá que fijarlos a un valor
  convencional y eso es una limitación que hay que declarar, no un detalle.
* clima y topografía sí son mapeables, pero no se extrajeron para el pool. Incluirlos
  ampliaría el diagnóstico y sólo cuesta una extracción; queda anotado.

Las dos curvas se construyen con la **misma** interpolación (`biodiv.curves.interp_grid`,
vía `scripts/31_pretrain_mae.unlabelled_curves`), o el índice mediría la diferencia entre dos
gridados en vez de la diferencia entre dos territorios.

Uso:
    python scripts/37_aoa.py
    python scripts/37_aoa.py --ngs 100 --index kndvi --scheme kfold5_block20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import aoa as aoamod                 # noqa: E402
from biodiv import cv as cvmod                   # noqa: E402
from biodiv import features as feat              # noqa: E402
from biodiv import figures as fg                 # noqa: E402
from biodiv import substrates as sub             # noqa: E402


def _pretrain_mod():
    """`scripts/31_pretrain_mae.py`, cuyo nombre no es un identificador válido."""
    spec = importlib.util.spec_from_file_location(
        "pretrain_mae", ROOT / "scripts" / "31_pretrain_mae.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default="kndvi")
    p.add_argument("--ngs", type=int, default=100,
                   help="pasos de la grilla; 100 es la curva del modelo titular (_raw100)")
    p.add_argument("--scheme", default="kfold5_block20",
                   help="folds de los que sale el umbral. Por omision block20, que es el "
                        "esquema cuya distancia de evaluacion se parece a la del mapa "
                        "(11-12 km contra 13,2); con kfold5_window el umbral saldria de "
                        "vecinos a 0,5 km y el AOA seria demasiado permisivo")
    p.add_argument("--curves", default="_raw100",
                   help="sufijo BIODIV_CURVES de las curvas de parcela")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--out", default="results/tables/aoa_unlabelled.csv")
    p.add_argument("--fig-dir", default="results/figures", dest="fig_dir")
    args = p.parse_args()

    import os
    os.environ[feat.CURVE_SUFFIX_ENV] = args.curves

    # --- entrenamiento: las curvas de las 1.082 parcelas
    train, ids = sub.load_curves(args.index, args.derived)
    train = train[:, 0, :]
    print(f"entrenamiento: {train.shape[0]} parcelas x {train.shape[1]} pasos "
          f"(BIODIV_CURVES={args.curves!r})")

    # --- dominio: el pool enmascarado con MapBiomas
    updir = Path(args.derived) / "unlabelled"
    if not updir.exists():
        raise SystemExit(f"{updir} no existe -- ver docs/15")
    new, man = _pretrain_mod().unlabelled_curves(args.index, updir, train.shape[1])
    print(f"dominio: {len(new)} muestras nativas de {len(pd.read_csv(updir/'manifest.csv'))}"
          f" ({man['cell'].nunique()} celdas de 10 km)")

    # --- folds del esquema elegido, como indices posicionales
    cv = cvmod.load_schemes(Path(args.derived) / "cv_folds_modelling.parquet")
    pos = {pid: i for i, pid in enumerate(ids)}
    folds = []
    for _, _, _, te in cvmod.iter_folds(cv, args.scheme):
        idx = [pos[i] for i in te if i in pos]
        if idx:
            folds.append(np.array(idx))

    r = aoamod.aoa(train, new, folds)
    man = man.assign(DI=r.di, inside=r.inside)
    print(f"\numbral (Q3 + 1,5 IQR del DI entre folds de {args.scheme}) = {r.threshold:.3f}")
    print(f"DI del dominio: mediana {np.median(r.di):.3f}, p95 {np.percentile(r.di, 95):.3f}")
    print(f"\n**dentro del area de aplicabilidad: {100 * r.fraction_inside:.1f} %** "
          f"({int(r.inside.sum())} de {len(r.di)})")

    if "mb_class" in man:
        by = man.groupby("mb_class")["inside"].agg(["size", "mean"])
        by["% dentro"] = (100 * by["mean"]).round(1)
        print("\npor clase de cobertura nativa:")
        print(by[["size", "% dentro"]].sort_values("% dentro").to_string())

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(out, index=False)
    (out.parent / "aoa_summary.json").write_text(json.dumps({
        "index": args.index, "ngs": int(train.shape[1]), "scheme": args.scheme,
        "curves": args.curves, "threshold": r.threshold,
        "mean_pairwise": r.mean_pairwise, "n_train": int(len(train)),
        "n_domain": int(len(new)), "fraction_inside": r.fraction_inside,
        "di_median": float(np.median(r.di)),
        "di_p95": float(np.percentile(r.di, 95)),
    }, indent=2))

    # --- figura: dónde cae el DI del dominio contra el del entrenamiento
    import matplotlib.pyplot as plt
    fg.set_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(0, max(np.percentile(r.di, 99), r.threshold * 1.5), 60)
    ax.hist(r.di_train, bins=bins, density=True, alpha=0.55,
            label=f"entrenamiento entre folds ({args.scheme})")
    ax.hist(r.di, bins=bins, density=True, alpha=0.55, label="dominio nativo")
    ax.axvline(r.threshold, color="k", lw=1.2, ls="--",
               label=f"umbral {r.threshold:.2f}")
    ax.set_xlabel("índice de disimilitud (DI)")
    ax.set_ylabel("densidad")
    ax.legend(fontsize=6.5)
    ax.set_title(f"Área de aplicabilidad: {100 * r.fraction_inside:.0f} % del dominio dentro",
                 fontsize=8)
    fg.save_figure(fig, "aoa_di", args.fig_dir)
    print(f"\n  -> {args.out}\n  -> {args.fig_dir}/aoa_di.pdf")


if __name__ == "__main__":
    main()
