"""El diagnóstico de patrón de muestreo, contra casos con respuesta conocida.

Se prueba con geometrías sintéticas —rejilla regular, uniforme aleatorio, dos grumos— porque
ahí la respuesta correcta se sabe de antemano y el test no depende de que los rásters de
MapBiomas estén en la máquina.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import spatial_diag as sd            # noqa: E402

RNG = np.random.default_rng(0)


def grid(n_side: int, step: float = 1000.0) -> np.ndarray:
    a = np.arange(n_side) * step
    return np.stack(np.meshgrid(a, a), -1).reshape(-1, 2).astype(float)


def uniform(n: int, side: float = 100_000.0) -> np.ndarray:
    return RNG.uniform(0, side, (n, 2))


def clumped(n: int, side: float = 100_000.0, k: int = 4, sd_m: float = 500.0) -> np.ndarray:
    centres = RNG.uniform(0, side, (k, 2))
    idx = RNG.integers(0, k, n)
    return centres[idx] + RNG.normal(0, sd_m, (n, 2))


# ----------------------------------------------------------------- nnd

def test_nnd_self_excludes_the_point_itself():
    """Un punto no es su propio vecino: en una rejilla, todos los Gj valen el paso."""
    d = sd.nnd(grid(5, step=1000.0))
    assert np.allclose(d, 1000.0)


def test_nnd_duplicates_give_zero():
    """Dos parcelas en el mismo pixel distan cero, y eso es el dato, no un error."""
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [5000.0, 0.0]])
    assert sd.nnd(xy)[0] == 0.0


def test_nnd_cross_is_directional():
    a = np.array([[0.0, 0.0], [10_000.0, 0.0]])
    b = np.array([[1000.0, 0.0]])
    assert np.allclose(sd.nnd(a, b), [1000.0, 9000.0])


@pytest.mark.parametrize("bad", [np.zeros((3,)), np.zeros((3, 3))])
def test_nnd_rejects_wrong_shape(bad):
    with pytest.raises(ValueError, match=r"se esperaba"):
        sd.nnd(bad)


def test_nnd_needs_two_points():
    with pytest.raises(ValueError, match="al menos 2"):
        sd.nnd(np.zeros((1, 2)))


# ----------------------------------------------------------------- Clark-Evans

def test_clark_evans_regular_grid_above_one():
    xy = grid(20, step=1000.0)
    side = 19 * 1000.0
    assert sd.clark_evans(xy, side ** 2) > 1.5      # la rejilla es el caso regular extremo


def test_clark_evans_uniform_near_one():
    xy = uniform(2000)
    assert 0.9 < sd.clark_evans(xy, 100_000.0 ** 2) < 1.1


def test_clark_evans_clumped_below_one():
    xy = clumped(2000)
    assert sd.clark_evans(xy, 100_000.0 ** 2) < 0.5


# ----------------------------------------------------------------- clasificacion

def test_clumped_samples_over_wide_domain_are_clustered():
    """El caso que importa: muestras en grumos, dominio repartido."""
    p = sd.sampling_pattern(clumped(500), uniform(3000))
    assert p.label == "clustered"
    assert p.summary()["median_ratio_Gij_Gj"] > 1.5
    assert p.ks_p < 0.01


def test_samples_drawn_from_the_domain_are_not_clustered():
    """Si las muestras salen del mismo proceso que el dominio, Gj y Gij se parecen."""
    dom = uniform(3000)
    p = sd.sampling_pattern(dom[RNG.choice(len(dom), 500, replace=False)], dom)
    assert p.label != "clustered"


def test_summary_is_json_friendly():
    p = sd.sampling_pattern(clumped(200), uniform(1000))
    s = p.summary()
    assert isinstance(s["label"], str)
    assert all(isinstance(v, (int, float, str)) for v in s.values())
    assert s["n_samples"] == 200 and s["n_domain"] == 1000


def test_area_defaults_to_the_domain_not_the_samples():
    """Unas muestras en una esquina no deben salir 'aleatorias' respecto de su esquina."""
    corner = RNG.uniform(0, 5_000, (300, 2))
    wide = uniform(3000, side=200_000.0)
    assert sd.sampling_pattern(corner, wide).clark_evans < 0.5
