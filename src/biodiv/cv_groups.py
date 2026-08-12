"""Cross-validation schemes that hold out whole units (dataset or site).

Never random CV: Parcelas-CL is strongly clustered by project and locality, so a random
fold places plots from the same site on both sides of the split and inflates R². The
partition unit here is always a complete group.

Two grouping axes, with different meanings:

- ``metadata_id`` (project/dataset): deliberately confounds sampling protocol, site and
  year. It is the strictest transferability test and answers "does the model work on data
  we did not collect ourselves?". In this dataset most projects are single-year, so
  leave-one-dataset-out is simultaneously leave-one-year-out.
- ``Location`` (site): splits by place while leaving protocols mixed. Finer-grained
  (169 levels) and measures spatial generalisation within known protocols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def union_groups(plots: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Merge two groupings: two plots share a group if they share *either* label.

    Vive aqui y no en `scripts/08` porque `cv.ensure_group_col` tiene que reconstruir
    **la misma** agrupacion para la particion interna de la parada temprana. Pegar las dos
    etiquetas (`f"{owner}|{window}"`) da una agrupacion mas FINA, no mas gruesa, y dejaria
    la particion interna partiendo grupos que la externa mantiene unidos -- en silencio.

    Not the same as pasting the two labels together, which would make the grouping *finer*
    rather than coarser and would let a shared window straddle a fold boundary whenever the
    two plots have different owners — exactly the 47-plot case this is meant to close.
    """
    idx = {v: i for i, v in enumerate(pd.unique(plots[a]))}
    off = len(idx)
    idx.update({v: off + i for i, v in enumerate(pd.unique(plots[b]))})
    parent = np.arange(len(idx))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for va, vb in zip(plots[a], plots[b]):
        ra, rb = find(idx[va]), find(idx[vb])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return pd.Series([f"u{find(idx[v])}" for v in plots[a]], index=plots.index)



def leave_one_group_out(df: pd.DataFrame, group_col: str, min_size: int = 20) -> pd.DataFrame:
    """One row per (fold, plot). Every group with >= min_size plots becomes a test fold.

    Groups smaller than ``min_size`` are never used as test —a test metric over 3 plots is
    not interpretable— but they still contribute to training.
    """
    sizes = df[group_col].value_counts()
    testable = sizes[sizes >= min_size].index.tolist()
    rows = []
    for fold, g in enumerate(sorted(testable)):
        is_test = df[group_col] == g
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "held_out": str(g),
                    "PlotObservationID": df["PlotObservationID"],
                    "split": np.where(is_test, "test", "train"),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out.attrs["scheme"] = f"leave-one-{group_col}-out"
    out.attrs["n_folds"] = len(testable)
    out.attrs["excluded_from_test"] = sorted(set(sizes.index) - set(testable))
    return out


def grouped_kfold(df: pd.DataFrame, group_col: str, k: int = 5, seed: int = 42,
                  stratify_on: str | None = None,
                  balance_weight: float = 1.0) -> pd.DataFrame:
    """K folds assigning **whole groups** to a fold, balanced by size and (optionally) by a
    response variable.

    Greedy packing: groups are sorted largest-first and each goes to the fold that minimises
    the resulting imbalance. With very unequal groups (md022 has 181 plots while others have
    11) this balances considerably better than a random assignment.

    ``stratify_on`` (e.g. ``"richness"``) additionally balances the *distribution* of that
    variable across folds. This is not cosmetic. R-squared is a variance-normalised metric,
    so a test fold whose response has almost no variance returns R-squared near zero however
    good the model is, and a fold concentrating the high-richness plots returns a high value
    almost for free. Measured on this dataset, the per-dataset groups span median richness
    from 1 (md006) to 27 (md023), so unbalanced folds make per-fold scores incomparable.

    Balancing the split on the response is not leakage: no information crosses from test to
    train within a fold, group integrity is preserved, and the only effect is that folds
    become comparable to each other. What it *does* hide is genuine distribution shift, so
    the unstratified leave-one-group-out scheme should be reported alongside it — there the
    shift is the quantity of interest, not a nuisance.

    **Know the limit before relying on this.** Measured on this dataset, richness variance
    decomposes as 80% *between* metadata_id groups and 20% within (82/18 for Location).
    Richness is largely a property OF the group — md023 holds 6% of plots but 22% of all
    species occurrences, at mean richness 28 against a dataset median of 5 — and groups are
    atomic, so stratification can only rebalance the within-group fifth. Expect it to reduce
    fold-to-fold imbalance modestly, not to remove it.

    The imbalance that survives is best handled at the metric, not the split:

    - Score pooled out-of-fold predictions rather than averaging per-fold R-squared. One
      score over all plots uses the full response variance and sidesteps comparability
      entirely; keep per-fold values as diagnostics.
    - Report RMSE/MAE, which are in response units, next to any R-squared.
    - Always publish ``fold_report`` beside the scores.
    """
    sizes = df[group_col].value_counts().sort_values(ascending=False)
    load = np.zeros(k, dtype=int)
    assignment: dict[str, int] = {}

    if stratify_on is None:
        for g, n in sizes.items():
            f = int(np.argmin(load))
            assignment[g] = f
            load[f] += n
    else:
        # Size balance is a hard constraint, response balance a soft one. Optimising both
        # in a single weighted sum lets the response term win whenever groups are small and
        # numerous (169 Locations, median a handful of plots each), which produced folds of
        # 96 vs 370 plots. Instead: restrict candidates to the folds that are still near the
        # minimum load, then among those pick the one that best balances the response.
        gsum = df.groupby(group_col)[stratify_on].sum()
        gsum2 = df.groupby(group_col)[stratify_on].apply(lambda s: float((s**2).sum()))
        scale = float(df[stratify_on].std()) or 1.0
        total = float(len(df))
        tol = max(1.0, 0.10 * total / k)   # allowed slack over the least-loaded fold
        acc = np.zeros(k)    # sum of the response per fold
        acc2 = np.zeros(k)   # sum of squares, so variance can be balanced too

        def spread(loads, s1, s2):
            """Spread of per-fold means and sds, in units of the overall sd."""
            with np.errstate(invalid="ignore", divide="ignore"):
                nz = np.maximum(loads, 1)
                means = s1 / nz
                var = np.maximum(s2 / nz - means**2, 0.0)
            return means.std() / scale + np.sqrt(var).std() / scale

        for g, n in sizes.items():
            candidates = [f for f in range(k) if load[f] <= load.min() + tol] or [int(np.argmin(load))]
            best, best_cost = candidates[0], np.inf
            for f in candidates:
                tl, t1, t2 = load.copy(), acc.copy(), acc2.copy()
                tl[f] += n; t1[f] += gsum[g]; t2[f] += gsum2[g]
                cost = balance_weight * spread(tl, t1, t2)
                if cost < best_cost:
                    best, best_cost = f, cost
            assignment[g] = best
            load[best] += n
            acc[best] += gsum[g]
            acc2[best] += gsum2[g]

    fold_of_plot = df[group_col].map(assignment)
    rows = []
    for fold in range(k):
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "held_out": f"fold{fold}",
                    "PlotObservationID": df["PlotObservationID"],
                    "split": np.where(fold_of_plot == fold, "test", "train"),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    suffix = f"-stratified-on-{stratify_on}" if stratify_on else ""
    out.attrs["scheme"] = f"grouped-{k}fold-by-{group_col}{suffix}"
    out.attrs["n_folds"] = k
    out.attrs["fold_sizes"] = load.tolist()
    if stratify_on:
        fold_mean = df.groupby(fold_of_plot)[stratify_on].mean()
        out.attrs["fold_response_means"] = [round(float(v), 2) for v in fold_mean]
    return out


#: Cortes de los bloques temporales, elegidos para igualar tamaños sin partir un año de
#: censo. Los 16 años de censo (2003-2026) están muy desiguales -- 2 parcelas en 2003 contra
#: 254 en 2022 -- así que un corte cada 3 años calendario daría folds de 2 y de 350.
TIME_EDGES = [2002, 2011, 2015, 2018, 2021, 2024, 2027]


def time_block(years: pd.Series, edges: list[int] | None = None) -> pd.Series:
    """Año de censo -> índice de bloque temporal."""
    e = TIME_EDGES if edges is None else edges
    return pd.cut(years.astype(int), e, labels=False).astype(int)


def _buffered_train(plots: pd.DataFrame, test_mask: pd.Series) -> pd.Series:
    """Entrenamiento válido para ese test: fuera del test y **sin solape de ventana causal**.

    La ventana causal dura 3 años (`win_start`..`win_end`), así que una parcela de 2013 y una
    de 2014 comparten observaciones Landsat de 2012 y 2013. Un bloque temporal sin este
    filtro deja entrar por la puerta del tiempo la misma fuga que `window_components` cierra
    en el espacio: el modelo vería el año que se le está pidiendo predecir.

    El precio es real -- entre 31 y 306 parcelas de entrenamiento según el fold -- y se paga
    a propósito: sin él, el esquema mediría interpolación temporal disfrazada de
    extrapolación.
    """
    lo = int(plots.loc[test_mask, "win_start"].min())
    hi = int(plots.loc[test_mask, "win_end"].max())
    overlaps = (plots["win_end"] >= lo) & (plots["win_start"] <= hi)
    return ~test_mask & ~overlaps


def _emit(plots: pd.DataFrame, folds: list[tuple[str, pd.Series, pd.Series]]) -> pd.DataFrame:
    """Tabla larga (fold, held_out, id, split) a partir de máscaras test/train.

    Una parcela que no aparece en ningún lado de un fold simplemente **no tiene fila** en él:
    es el buffer, y `cv.iter_folds` la deja fuera de los dos conjuntos sin más.
    """
    rows = []
    for i, (name, te, tr) in enumerate(folds):
        for mask, split in ((te, "test"), (tr, "train")):
            ids = plots.loc[mask, "PlotObservationID"]
            if len(ids):
                rows.append(pd.DataFrame({"fold": i, "held_out": name,
                                          "PlotObservationID": ids, "split": split}))
    if not rows:
        # Que ningún fold califique es un resultado legítimo -- p.ej. `time_within_owner` en
        # un conjunto donde ningún contribuyente abarca dos periodos. Devolver la tabla vacía
        # con sus columnas deja que el consumidor lo vea; reventar aquí lo convertiría en un
        # fallo de programa en vez de en un dato sobre el diseño de muestreo.
        return pd.DataFrame({"fold": pd.Series(dtype=int),
                             "held_out": pd.Series(dtype=str),
                             "PlotObservationID": pd.Series(dtype=object),
                             "split": pd.Series(dtype=str)})
    return pd.concat(rows, ignore_index=True)


def time_kfold(plots: pd.DataFrame, edges: list[int] | None = None) -> pd.DataFrame:
    """Leave-time-out con bloques rodantes: cada parcela es test exactamente una vez.

    Meyer et al. (2018) lo llaman LTO y es el esquema que un revisor pedirá por su nombre
    cuando el modelo se ofrezca para predecir en años no muestreados.
    """
    tb = time_block(plots["Year"], edges)
    folds = []
    for b in sorted(tb.unique()):
        te = tb == b
        yrs = plots.loc[te, "Year"]
        folds.append((f"t{b}_{int(yrs.min())}-{int(yrs.max())}", te,
                      _buffered_train(plots, te)))
    return _emit(plots, folds)


def loc_time_kfold(plots: pd.DataFrame, loc_fold: pd.Series,
                   edges: list[int] | None = None, min_test: int = 5) -> pd.DataFrame:
    """LLTO: test = grupo espacial x bloque temporal, y el entrenamiento excluye los dos.

    El caso duro y el que se parece a un mapa proyectado a un año nuevo: sitio que el modelo
    no vio, en un tiempo que tampoco vio.

    ``loc_fold`` tiene que venir de un esquema que separe **de verdad** en el espacio. En
    este proyecto eso es `kfold5_block20` (bloques de 20 km, mediana test-entrenamiento
    11,4 km) y **no** `kfold5_window`: los componentes de ventana cierran la fuga de píxeles
    pero reparten componentes vecinos entre folds, así que retener un fold deja la parcela de
    test a 0,47 km de su entrenamiento -- un "sitio nuevo" que no es nuevo. Medido en
    `results/tables/sampling_pattern.json`. `block20` también deja 0 de 135 componentes
    partidos, así que no se pierde el cierre de fuga (`docs/10` §1b).
    """
    tb = time_block(plots["Year"], edges)
    folds = []
    for s in sorted(pd.unique(loc_fold.dropna())):
        for b in sorted(tb.unique()):
            te = (loc_fold == s) & (tb == b)
            if int(te.sum()) < min_test:
                continue
            folds.append((f"s{int(s)}t{b}", te,
                          _buffered_train(plots, te) & (loc_fold != s)))
    return _emit(plots, folds)


def time_within_owner(plots: pd.DataFrame, edges: list[int] | None = None,
                      min_test: int = 15) -> pd.DataFrame:
    """Holdout temporal para contribuyentes que abarcan varios periodos.

    **El control del confundido.** En estos datos, contribuyente y año están casi
    superpuestos (Cramér's V = 0,732), así que un LTO puro no distingue no-estacionariedad
    temporal del salto de nivel entre contribuyentes, que ya se sabe que domina alfa
    (eta^2 = 0,72). Aquí el contribuyente del test **sí** está en el entrenamiento, con sus
    parcelas de otro periodo, de modo que su nivel es aprendible y lo único retenido es el
    tiempo.

    Sólo entran contribuyentes presentes dentro y fuera del bloque de test; en estos datos
    son Galleguillos, Ovalle y Miranda (507 parcelas).
    """
    tb = time_block(plots["Year"], edges)
    folds = []
    for b in sorted(tb.unique()):
        inb = tb == b
        # contribuyentes con parcelas dentro y fuera del bloque
        multi = {o for o in plots.loc[inb, "Owner"].unique()
                 if (plots["Owner"].eq(o) & ~inb).any()}
        te = inb & plots["Owner"].isin(multi)
        if int(te.sum()) < min_test:
            continue
        folds.append((f"t{b}_within", te, _buffered_train(plots, te)))
    return _emit(plots, folds)


def fold_report(df: pd.DataFrame, cv: pd.DataFrame, response: str = "richness") -> pd.DataFrame:
    """Per-fold composition of the response — always report this next to any score.

    ``pct_out_of_range`` counts test plots whose response falls outside the training range.
    It is usually near zero here because the pooled training set spans the full range; the
    quantity that actually breaks comparability is the difference in *variance* between
    test folds, which is why sd is reported.
    """
    m = df.set_index("PlotObservationID")[response]
    rows = []
    for (scheme, fold), g in cv.groupby(["scheme", "fold"]):
        te = m[g.loc[g["split"] == "test", "PlotObservationID"]]
        tr = m[g.loc[g["split"] == "train", "PlotObservationID"]]
        rows.append(dict(
            scheme=scheme, fold=fold, held_out=g["held_out"].iloc[0], n_test=len(te),
            test_median=te.median(), test_mean=round(te.mean(), 2), test_sd=round(te.std(), 2),
            train_mean=round(tr.mean(), 2), train_sd=round(tr.std(), 2),
            pct_out_of_range=round(float(((te < tr.min()) | (te > tr.max())).mean() * 100), 1),
        ))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Size and composition of each group, so the scheme is chosen with the data in view."""
    g = df.groupby(group_col)
    out = pd.DataFrame(
        {
            "n_plots": g.size(),
            "n_locations": g["Location"].nunique(),
            "years": g["Year"].apply(lambda s: f"{int(s.min())}-{int(s.max())}"),
            "richness_median": g["richness"].median(),
            "strata": g["stratum"].apply(lambda s: ",".join(sorted(s.unique()))),
            "owner": g["Owner"].first(),
        }
    ).sort_values("n_plots", ascending=False)
    return out.reset_index()
