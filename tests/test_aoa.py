"""El AOA contra su definición, con geometrías donde la respuesta se sabe de antemano.

El riesgo de una implementación así no es que falle: es que devuelva un número plausible
—una fracción entre 0 y 1— que no signifique lo que dice. Por eso cada test fija una
propiedad de la definición de Meyer & Pebesma (2021) y no un valor observado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import aoa as aoamod                 # noqa: E402

RNG = np.random.default_rng(0)


def kfolds(n: int, k: int = 5) -> list[np.ndarray]:
    idx = RNG.permutation(n)
    return [np.sort(a) for a in np.array_split(idx, k)]


@pytest.fixture
def train():
    return RNG.normal(0, 1, (300, 6))


# ----------------------------------------------------------------- DI

def test_a_training_point_has_di_zero(train):
    """Un punto que ya está en el entrenamiento dista cero de sí mismo."""
    r = aoamod.aoa(train, train[:10], kfolds(len(train)))
    assert np.allclose(r.di[:10], 0.0)
    assert r.inside[:10].all()


def test_a_duplicate_is_inside(train):
    dup = train[:5] + 1e-9
    assert aoamod.aoa(train, dup, kfolds(len(train))).inside.all()


def test_a_far_point_is_outside(train):
    far = np.full((3, train.shape[1]), 10.0)     # 10 sd en todas las variables
    r = aoamod.aoa(train, far, kfolds(len(train)))
    assert not r.inside.any()
    assert (r.di > r.threshold).all()


def test_di_grows_with_distance(train):
    """El índice tiene que ser monótono en la distancia, o no es un índice de disimilitud."""
    steps = np.array([0.0, 1.0, 3.0, 10.0])
    pts = np.stack([np.full(train.shape[1], s) for s in steps])
    di = aoamod.aoa(train, pts, kfolds(len(train))).di
    assert np.all(np.diff(di) > 0), di


def test_threshold_is_q3_plus_iqr_of_the_cv_di(train):
    folds = kfolds(len(train))
    di_tr, thr, _ = aoamod.train_di(train, folds)
    q1, q3 = np.percentile(di_tr, [25, 75])
    assert thr == pytest.approx(q3 + 1.5 * (q3 - q1))


def test_train_di_uses_neighbours_outside_the_fold(train):
    """Si mirase dentro del fold, el DI de entrenamiento sería casi cero en todas partes."""
    folds = kfolds(len(train))
    di_tr, _, _ = aoamod.train_di(train, folds)
    assert di_tr.min() > 0, "un punto no puede ser su propio vecino entre folds"


# ----------------------------------------------------------------- pesos

def test_weights_change_which_dimension_matters(train):
    """Un punto lejano sólo en una variable ignorada debe entrar; en una que pesa, no."""
    p = np.zeros((1, train.shape[1]))
    p[0, 0] = 6.0                                  # lejos sólo en la variable 0
    w_ignora = np.array([0.001, 1, 1, 1, 1, 1.0])
    w_importa = np.array([1000.0, 1, 1, 1, 1, 1])
    folds = kfolds(len(train))
    assert aoamod.aoa(train, p, folds, weights=w_ignora).inside[0]
    assert not aoamod.aoa(train, p, folds, weights=w_importa).inside[0]


def test_uniform_weights_match_no_weights(train):
    folds = kfolds(len(train))
    new = RNG.normal(0, 1, (50, train.shape[1]))
    a = aoamod.aoa(train, new, folds)
    b = aoamod.aoa(train, new, folds, weights=np.ones(train.shape[1]))
    assert np.allclose(a.di, b.di)


@pytest.mark.parametrize("bad", [np.array([-1.0, 1, 1, 1, 1, 1]),
                                 np.zeros(6),
                                 np.ones(3)])
def test_bad_weights_raise(train, bad):
    with pytest.raises(ValueError):
        aoamod.aoa(train, train[:5], kfolds(len(train)), weights=bad)


# ----------------------------------------------------------------- escalado

def test_scaling_uses_training_statistics(train):
    """Un dominio con mucha más varianza no debe parecerse más al entrenamiento por eso."""
    folds = kfolds(len(train))
    wide = RNG.normal(0, 8, (400, train.shape[1]))
    frac = aoamod.aoa(train, wide, folds).fraction_inside
    same = aoamod.aoa(train, RNG.normal(0, 1, (400, train.shape[1])), folds).fraction_inside
    assert frac < same


def test_constant_column_does_not_break_it(train):
    t = np.column_stack([train, np.ones(len(train))])
    n = np.column_stack([train[:20], np.ones(20)])
    assert aoamod.aoa(t, n, kfolds(len(t))).inside.all()


def test_shape_mismatch_raises(train):
    with pytest.raises(ValueError, match="incompatibles"):
        aoamod.aoa(train, RNG.normal(0, 1, (5, 3)), kfolds(len(train)))


def test_most_of_an_in_distribution_domain_is_inside(train):
    """Cordura: puntos del mismo proceso generador caen mayoritariamente dentro."""
    same = RNG.normal(0, 1, (500, train.shape[1]))
    assert aoamod.aoa(train, same, kfolds(len(train))).fraction_inside > 0.9
