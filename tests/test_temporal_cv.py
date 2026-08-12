"""Los esquemas de validación temporal, y sobre todo el buffer de ventana causal.

El buffer es la razón de ser de estos esquemas y también lo único que puede romperse en
silencio: si se cae, el esquema sigue produciendo folds, sigue dando un R², y ese R² mide
interpolación temporal disfrazada de extrapolación. Por eso el invariante se comprueba
parcela a parcela y no por conteos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import cv_groups as cg              # noqa: E402

ID = "PlotObservationID"


def synth(n_per_year: int = 10, years=range(2005, 2026), owners=("A", "B", "C")) -> pd.DataFrame:
    """Parcelas con ventana causal de 3 años, repartidas entre contribuyentes y años."""
    rows = []
    for i, y in enumerate(years):
        for j in range(n_per_year):
            rows.append(dict(**{ID: f"P{y}_{j}"}, Year=y, Owner=owners[(i + j) % len(owners)],
                             win_start=y - 2, win_end=y, richness=5.0 + j))
    return pd.DataFrame(rows)


def as_folds(tab: pd.DataFrame, plots: pd.DataFrame):
    """(fold, test_ids, train_ids) desde la tabla larga."""
    for f, g in tab.groupby("fold", sort=True):
        yield (f, set(g.loc[g.split == "test", ID]), set(g.loc[g.split == "train", ID]))


def assert_buffer_holds(tab: pd.DataFrame, plots: pd.DataFrame) -> None:
    """Ninguna parcela de entrenamiento comparte años con la ventana del test."""
    w = plots.set_index(ID)[["win_start", "win_end"]]
    for f, te, tr in as_folds(tab, plots):
        lo, hi = w.loc[list(te), "win_start"].min(), w.loc[list(te), "win_end"].max()
        bad = w.loc[list(tr)]
        clash = bad[(bad.win_end >= lo) & (bad.win_start <= hi)]
        assert clash.empty, (
            f"fold {f}: {len(clash)} parcelas de entrenamiento con ventana dentro de "
            f"[{lo}, {hi}], p.ej. {clash.index[0]}")


# ----------------------------------------------------------------- bloques

def test_time_block_is_monotone_in_year():
    y = pd.Series([2003, 2010, 2012, 2019, 2026])
    b = cg.time_block(y)
    assert list(b) == sorted(b), "un año posterior no puede caer en un bloque anterior"


def test_time_block_covers_every_census_year():
    plots = pd.read_parquet(ROOT / "data/derived/plots_subset.parquet")
    b = cg.time_block(plots["Year"])
    assert b.notna().all(), "algún año de censo cae fuera de TIME_EDGES"


# ----------------------------------------------------------------- LTO

def test_time_kfold_no_overlap_and_full_coverage():
    p = synth()
    t = cg.kfold_time(p)
    seen = []
    for f, te, tr in as_folds(t, p):
        assert not (te & tr), f"fold {f}: parcela en train y test a la vez"
        seen += list(te)
    assert sorted(seen) == sorted(p[ID]), "cada parcela debe ser test exactamente una vez"


def test_time_kfold_buffer_holds():
    p = synth()
    assert_buffer_holds(cg.kfold_time(p), p)


def test_buffer_actually_removes_plots():
    """Si el buffer no quitara nada, el test de arriba pasaría por vacuidad."""
    p = synth()
    t = cg.kfold_time(p)
    dropped = [len(p) - len(g) for _, g in t.groupby("fold")]
    assert max(dropped) > 0, "el buffer no excluyó ninguna parcela en ningún fold"


def test_without_buffer_the_invariant_would_fail():
    """Control: el mismo esquema sin buffer sí mete años del test en el entrenamiento.

    Fija que el test de buffer detecta lo que dice detectar.
    """
    p = synth()
    tb = cg.time_block(p["Year"])
    naive = pd.concat([
        pd.DataFrame({"fold": i, "held_out": f"t{b}", ID: p[ID],
                      "split": np.where(tb == b, "test", "train")})
        for i, b in enumerate(sorted(tb.unique()))], ignore_index=True)
    with pytest.raises(AssertionError, match="ventana dentro de"):
        assert_buffer_holds(naive, p)


# ----------------------------------------------------------------- LLTO

def test_loc_time_excludes_both_the_place_and_the_time():
    p = synth()
    loc = pd.Series(np.arange(len(p)) % 5, index=p.index)
    t = cg.kfold_loc_time(p, loc, min_test=1)
    lookup = dict(zip(p[ID], loc))
    tb = dict(zip(p[ID], cg.time_block(p["Year"])))
    for f, te, tr in as_folds(t, p):
        s = {lookup[i] for i in te}
        b = {tb[i] for i in te}
        assert len(s) == 1 and len(b) == 1, "el test debe ser un solo (sitio, tiempo)"
        assert not any(lookup[i] in s for i in tr), "entrenamiento en el sitio retenido"
        assert not any(tb[i] in b for i in tr), "entrenamiento en el tiempo retenido"


def test_loc_time_buffer_holds():
    p = synth()
    loc = pd.Series(np.arange(len(p)) % 5, index=p.index)
    assert_buffer_holds(cg.kfold_loc_time(p, loc, min_test=1), p)


# ----------------------------------------------------------------- within-owner

def test_within_owner_test_contributors_are_present_in_training():
    """El punto entero del esquema: el nivel del contribuyente es aprendible."""
    p = synth()
    t = cg.time_within_owner(p, min_test=1)
    owner = dict(zip(p[ID], p["Owner"]))
    for f, te, tr in as_folds(t, p):
        assert {owner[i] for i in te} <= {owner[i] for i in tr}, (
            f"fold {f}: un contribuyente de test no aparece en el entrenamiento, así que "
            "su caída sería confundido de contribuyente y no efecto temporal")


def test_within_owner_skips_single_period_contributors():
    """Un contribuyente que sólo censó en un bloque no puede entrar: no hay con qué anclarlo."""
    p = pd.concat([synth(owners=("A",), years=range(2005, 2010)),
                   synth(owners=("solo",), years=range(2022, 2024))], ignore_index=True)
    t = cg.time_within_owner(p, min_test=1)
    owner = dict(zip(p[ID], p["Owner"]))
    tested = {owner[i] for _, te, _ in as_folds(t, p) for i in te}
    assert "solo" not in tested


# ----------------------------------------------------------------- datos reales

@pytest.mark.parametrize("scheme", ["kfold_time", "kfold_loc_time", "time_within_owner"])
def test_real_schemes_are_leakage_free(scheme):
    f = ROOT / "data/derived/cv_folds_modelling.parquet"
    if not f.exists():
        pytest.skip("cv_folds_modelling.parquet no está construido")
    cv = pd.read_parquet(f)
    if scheme not in set(cv["scheme"]):
        pytest.skip(f"{scheme} no está en la tabla; correr scripts/08")
    plots = pd.read_parquet(ROOT / "data/derived/plots_subset.parquet")
    assert_buffer_holds(cv[cv["scheme"] == scheme], plots)
