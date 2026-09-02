"""Invariants of the DEM derivatives in `scripts/03_extract_topography.py`.

The one that matters is the **aspect convention**. `terrain` receives the DEM as a plain
array whose rows run north -> south, so the sign of the row derivative decides whether
`aspect` names the direction the slope *faces* (the cartographic convention, and the one
`northness`/`eastness`/`heat_load` are interpreted with) or the direction it climbs. The two
differ by exactly 180 degrees, every value stays inside [0, 360), nothing looks broken, and
`northness` comes out with the sign flipped: south-facing slopes read as north-facing and the
heat load index peaks on the coolest slopes. That is the error these tests exist to catch.

The expected values are the ones `gdaldem aspect`/`gdaldem slope` return for the same planes
(checked once against GDAL; hard-coded here so the test needs no GDAL binary).

All of it is local: synthetic surfaces, no datacube.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RES = 30.0
N = 21
C = N // 2
LAT = -40.0


@pytest.fixture(scope="module")
def terrain():
    spec = importlib.util.spec_from_file_location(
        "s03", ROOT / "scripts" / "03_extract_topography.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.terrain


def plane(dz_north: float = 0.0, dz_east: float = 0.0) -> np.ndarray:
    """A tilted plane, rising `dz_*` metres per metre towards north / east.

    Row 0 is the northernmost, as in a DEM loaded with a negative y resolution.
    """
    rows, cols = np.mgrid[0:N, 0:N].astype(float)
    return 1000.0 - dz_north * RES * rows + dz_east * RES * cols


def facing(bearing_deg: float, tilt: float = 0.2) -> np.ndarray:
    """A plane that *faces* `bearing_deg`, i.e. climbs in the opposite direction."""
    b = np.radians(bearing_deg)
    return plane(dz_north=-tilt * np.cos(b), dz_east=-tilt * np.sin(b))


@pytest.mark.parametrize("dz_north,dz_east,expected", [
    (0.10, 0.0, 180.0),     # climbs north  -> faces south
    (-0.10, 0.0, 0.0),      # climbs south  -> faces north
    (0.0, 0.10, 270.0),     # climbs east   -> faces west
    (0.0, -0.10, 90.0),     # climbs west   -> faces east
    (0.10, 0.10, 225.0),    # climbs NE     -> faces southwest
])
def test_aspect_points_downslope(terrain, dz_north, dz_east, expected):
    """`gdaldem aspect` values for the same planes. 180 degrees off = the sign flip."""
    t = terrain(plane(dz_north, dz_east), RES, LAT)
    got = t["aspect"][C, C]
    assert np.isclose((got - expected + 180) % 360 - 180, 0.0, atol=0.1), \
        f"aspect {got:.1f} vs {expected:.1f} (a 180 gap means the uphill bearing)"


def test_northness_is_positive_on_north_facing_slopes(terrain):
    """In Chile these are the slopes exposed to the sun; the sign is the whole point."""
    assert terrain(facing(0), RES, LAT)["northness"][C, C] == pytest.approx(1.0, abs=1e-3)
    assert terrain(facing(180), RES, LAT)["northness"][C, C] == pytest.approx(-1.0, abs=1e-3)
    assert terrain(facing(90), RES, LAT)["eastness"][C, C] == pytest.approx(1.0, abs=1e-3)
    assert terrain(facing(270), RES, LAT)["eastness"][C, C] == pytest.approx(-1.0, abs=1e-3)


def test_heat_load_peaks_on_the_northwest_slope(terrain):
    """McCune & Keon folded about 315: warmest NW, coolest SE, southern hemisphere."""
    hl = {b: terrain(facing(b), RES, LAT)["heat_load"][C, C] for b in (0, 90, 135, 270, 315)}
    assert hl[315] == max(hl.values())
    assert hl[135] == min(hl.values())
    assert hl[0] > hl[90]                     # north-facing warmer than east-facing
    assert hl[315] > hl[270]


def test_slope_matches_the_tilt(terrain):
    for tilt, deg in ((0.10, 5.71), (0.5, 26.57), (1.0, 45.0)):
        t = terrain(plane(dz_north=tilt), RES, LAT)
        assert t["slope"][C, C] == pytest.approx(deg, abs=0.05)


def test_aspect_is_nan_where_there_is_no_slope(terrain):
    """Flat ground has no orientation; inventing one would hand the model pure noise."""
    t = terrain(plane(), RES, LAT)
    assert np.isnan(t["aspect"]).all()
    assert np.isnan(t["northness"]).all()


def test_tpi_separates_a_peak_from_a_pit(terrain):
    rows, cols = np.mgrid[0:N, 0:N].astype(float)
    r2 = ((rows - C) * RES) ** 2 + ((cols - C) * RES) ** 2
    peak = terrain(1000.0 - r2 / 2000.0, RES, LAT)
    pit = terrain(1000.0 + r2 / 2000.0, RES, LAT)
    assert peak["tpi"][C, C] > 0 > pit["tpi"][C, C]
    assert peak["curvature"][C, C] < 0 < pit["curvature"][C, C]   # convex is negative
    assert peak["tri"][C, C] >= 0


def test_a_plane_is_neither_rugged_nor_curved(terrain):
    t = terrain(plane(dz_north=0.10), RES, LAT)
    assert t["tpi"][C, C] == pytest.approx(0.0, abs=1e-6)
    assert t["curvature"][C, C] == pytest.approx(0.0, abs=1e-9)
    # TRI is a sum over the 8 neighbours of a 3 m step each, not zero, but constant
    assert np.ptp(t["tri"][2:-2, 2:-2]) == pytest.approx(0.0, abs=1e-6)
