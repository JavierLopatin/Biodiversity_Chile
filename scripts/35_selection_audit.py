#!/usr/bin/env python3
"""Cuánto del R² titular es sesgo de selección, medido sin reentrenar nada.

**El problema.** El proyecto eligió el mejor modelo entre ~200 corridas mirando
`kfold5_window`, y después reportó el `kfold5_window` de ese ganador. Seleccionar y evaluar
sobre la misma partición sesga el número hacia arriba (Hastie et al. 2009, §7.10.2), y es el
primer aviso que STeMP levanta en su propio ejemplo. La confirmación con semillas {10..14}
que ya se corrió **no** cubre esto: acota la lotería de la semilla, pero los folds eran los
mismos, así que la ventaja que un modelo saque *en esos cinco folds concretos* se conserva
íntegra al cambiar de semilla.

**Por qué sale gratis.** `dl_runner` guarda `oof_predictions.csv` con la columna `fold`, así
que la puntuación de cualquier subconjunto de folds se recalcula desde disco. No hay
entrenamiento de por medio y el resultado es determinista.

**El estimador.** Para un conjunto de folds ``S``, ``score(corrida, S)`` es el R² calculado
sobre las filas OOF cuyo fold está en ``S`` -- agrupando, no promediando R² por fold, que es
otro estimador. Entonces, para cada fold ``f``:

    ganador_f = argmax_corrida  score(corrida, todos \\ f)      <- f no participa
    auditado  = media_f  score(ganador_f, {f})
    ingenuo   = media_f  score(ganador_global, {f})

Los dos son medias de puntuaciones de un solo fold, así que son comparables entre sí; su
diferencia es el sesgo de selección. El R² agrupado sobre los 5 folds se imprime aparte como
referencia, pero **no** se compara contra estos dos: no es el mismo estimador.

Uso:
    python scripts/35_selection_audit.py
    python scripts/35_selection_audit.py --scheme kfold5_window --min-runs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import metrics as mx                 # noqa: E402
from biodiv import targets as tg                 # noqa: E402

TARGETS = [t for ts in tg.FACETS.values() for t in ts]


def load_oof(scheme: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """``run_id -> OOF table`` y ``run_id -> familia``, para cada corrida bajo ``scheme``.

    Las corridas de confirmación (`_s10`) se excluyen del conjunto de selección: usan otras
    semillas y competir contra ellas mezclaría dos experimentos.

    La familia se lee de `config.json`, que es donde el corredor la escribió. Deducirla del
    prefijo del `run_id` produce etiquetas como `RF0` y separa `B03_coords` de su propia
    familia.
    """
    out, fam = {}, {}
    for f in sorted((ROOT / "results" / "models").glob(f"*/{scheme}/oof_predictions.csv")):
        rid = f.parts[-3]
        if "_s10" in rid:
            continue
        d = pd.read_csv(f)
        if "fold" not in d.columns:
            continue
        if any(f"{t}_obs" not in d.columns for t in TARGETS):
            continue                    # corridas viejas con menos targets
        out[rid] = d
        cfg = f.parent / "config.json"
        fam[rid] = (json.loads(cfg.read_text()).get("family", "?")
                    if cfg.exists() else "?")
    return out, fam


def facet_mean(m: pd.DataFrame) -> float:
    """Media sobre las cinco facetas, no sobre los 15 targets.

    Promediar targets le daría a beta (3 por estrato) el triple de peso que a diversidad
    oscura (1 target). Es el mismo criterio que `scripts/33_search_report.py`.
    """
    per = [m[m["target"].isin(ts)]["R2"].mean() for ts in tg.FACETS.values()]
    return float(np.mean(per))


def score(oof: pd.DataFrame, folds: set[int]) -> float:
    """R² agrupado sobre las filas de esos folds, promediado entre semillas."""
    sub = oof[oof["fold"].isin(folds)]
    if sub.empty:
        return np.nan
    vals = []
    for _, g in sub.groupby("seed"):
        pred = np.column_stack([g[f"{t}_pred"].to_numpy(float) for t in TARGETS])
        obs = np.column_stack([g[f"{t}_obs"].to_numpy(float) for t in TARGETS])
        vals.append(facet_mean(mx.compute_metrics(pred, obs, TARGETS)))
    return float(np.mean(vals))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scheme", default="kfold5_window")
    p.add_argument("--out", default="results/tables/selection_audit.csv")
    args = p.parse_args()

    # el R2 de un solo fold puede tocar un target constante en ese fold; spearman avisa y
    # no afecta al R2, que es lo unico que se usa aqui
    warnings.filterwarnings("ignore", message="An input array is constant")
    oofs, fams = load_oof(args.scheme)
    if not oofs:
        raise SystemExit(f"sin corridas con oof_predictions.csv bajo {args.scheme!r}")
    folds = sorted({int(f) for d in oofs.values() for f in d["fold"].unique()})
    print(f"{len(oofs)} corridas, {len(folds)} folds, {len(TARGETS)} targets\n")

    # score(corrida, subconjunto) para los subconjuntos que hacen falta: cada fold solo, y
    # cada complemento. 2k evaluaciones por corrida, todas desde disco.
    single = {r: {f: score(d, {f}) for f in folds} for r, d in oofs.items()}
    loo = {r: {f: score(d, set(folds) - {f}) for f in folds} for r, d in oofs.items()}
    allf = {r: score(d, set(folds)) for r, d in oofs.items()}

    naive_winner = max(allf, key=lambda r: allf[r])
    print(f"ganador ingenuo (elegido sobre los {len(folds)} folds): {naive_winner}")
    print(f"  R² agrupado sobre los {len(folds)} folds = {allf[naive_winner]:.4f}   "
          "<- el numero de docs/14\n")

    rows = []
    for f in folds:
        w = max(loo, key=lambda r: loo[r][f] if np.isfinite(loo[r][f]) else -np.inf)
        rows.append(dict(fold=f, ganador_sin_f=w,
                         score_ganador_sin_f_en_f=single[w][f],
                         score_ganador_ingenuo_en_f=single[naive_winner][f],
                         mismo_ganador=int(w == naive_winner)))
    a = pd.DataFrame(rows)
    audit = a["score_ganador_sin_f_en_f"].mean()
    naive = a["score_ganador_ingenuo_en_f"].mean()

    print(a.to_string(index=False))
    print(f"\nseleccion honesta (ganador elegido sin ver f)  : {audit:.4f}")
    print(f"seleccion ingenua (ganador elegido viendo todo) : {naive:.4f}")
    print(f"sesgo de seleccion                              : {naive - audit:+.4f}")
    print(f"el ganador cambia en {len(folds) - int(a.mismo_ganador.sum())} de {len(folds)} "
          "folds")

    # el mismo cálculo por familia, que es como se reporta la tabla titular
    fam_rows = []
    for fam in sorted(set(fams.values())):
        pool = [r for r in oofs if fams[r] == fam]
        if len(pool) < 2:
            continue
        nw = max(pool, key=lambda r: allf[r])
        au = np.mean([single[max(pool, key=lambda r: loo[r][f])][f] for f in folds])
        na = np.mean([single[nw][f] for f in folds])
        fam_rows.append(dict(familia=fam, n_corridas=len(pool), ganador=nw,
                             honesta=au, ingenua=na, sesgo=na - au))
    if fam_rows:
        print("\npor familia:")
        print(pd.DataFrame(fam_rows).round(4).to_string(index=False))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    a.assign(scheme=args.scheme, n_runs=len(oofs),
             honesta=audit, ingenua=naive, sesgo=naive - audit).to_csv(out, index=False)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
