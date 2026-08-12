"""The MapBiomas legend is anchored by evidence, and these tests are the anchor.

The rasters carry no embedded legend, so the mapping from raster integer to land-cover class
was established empirically: sampled at the 1,082 Parcelas-CL plots, which are native
vegetation by design, and at point probes on known ground. If MapBiomas re-codes a class in a
future collection, or a file is replaced by one from a different product, nothing raises --
the mask just quietly starts selecting orchards. These tests are what makes that loud.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import mapbiomas as mb            # noqa: E402

PLOTS = ROOT / "data" / "derived" / "plots_subset.parquet"

pytestmark = pytest.mark.skipif(
    not mb.available_years(), reason="no MapBiomas rasters on disk")


def test_legend_covers_every_native_code():
    lg = mb.legend()
    for code in mb.NATIVE:
        assert code in lg.index, f"native code {code} missing from legend.csv"
        assert lg.loc[code, "native"] == 1, f"code {code} in NATIVE but not flagged in csv"
    flagged = set(lg.index[lg["native"] == 1])
    assert flagged == set(mb.NATIVE), "legend.csv and NATIVE disagree"


def test_rocky_outcrop_is_excluded():
    """29 is a natural non-forest formation but is not vegetation. Decided, not incidental."""
    assert 29 not in mb.NATIVE
    assert mb.legend().loc[29, "native"] == 0


def test_unassigned_codes_are_not_native():
    """67, 79 and 80 were never confirmed. What is not recognised must not enter the mask."""
    for code in (67, 79, 80):
        assert code not in mb.NATIVE


@pytest.mark.skipif(not PLOTS.exists(), reason="plots_subset.parquet not available")
def test_class_distribution_at_plots_reproduces_the_anchor():
    """The measurement the whole legend rests on.

    Shrubland dominates the labelled set, then the forest subclasses. If these shares move,
    the integer-to-class mapping moved and every downstream mask is suspect.
    """
    p = pd.read_parquet(PLOTS)
    codes, _, _ = mb.sample_at(p["lon"].to_numpy(), p["lat"].to_numpy(), 2014)
    share = pd.Series(codes).value_counts(normalize=True)
    assert share.get(66, 0) == pytest.approx(0.483, abs=0.02), "66 is shrubland"
    assert share.get(60, 0) == pytest.approx(0.229, abs=0.02), "60 is a forest subclass"
    assert share.get(59, 0) == pytest.approx(0.076, abs=0.02), "59 is a forest subclass"
    assert share.get(12, 0) == pytest.approx(0.075, abs=0.02), "12 is grassland"
    assert share.idxmax() == 66


@pytest.mark.skipif(not PLOTS.exists(), reason="plots_subset.parquet not available")
def test_mask_is_stricter_than_the_plots():
    """~10.7 % of the plots sit on classes the mask excludes. Declared, not hidden.

    Land-use change between census and map, geolocation error, or plantation edge -- the
    cause is not settled, but the consequence is: the unlabelled pool is drawn under a
    stricter rule than the labelled set it pretrains for.
    """
    p = pd.read_parquet(PLOTS)
    codes, _, _ = mb.sample_at(p["lon"].to_numpy(), p["lat"].to_numpy(), 2014)
    excluded = 1.0 - mb.is_native(codes).mean()
    assert 0.05 < excluded < 0.20, f"excluded share {excluded:.3f} left its measured range"


@pytest.mark.parametrize("name,lon,lat,code", [
    ("Santiago centre", -70.650, -33.437, 24),
    ("Vina del Mar", -71.552, -33.024, 24),
    ("Constitucion pine plantation", -72.320, -35.500, 9),
])
def test_known_ground_probes(name, lon, lat, code):
    """Urban and silviculture, confirmed against ground nobody disputes."""
    got, _, _ = mb.sample_at([lon], [lat], 2014)
    assert int(got[0]) == code, f"{name}: expected {code}, got {int(got[0])}"
    assert not mb.is_native(got)[0], f"{name} must not be native"


def test_year_map_exact_when_present():
    for year in mb.available_years():
        path, used, delta = mb.year_map(year)
        assert used == year and delta == 0


def test_year_map_falls_back_to_nearest():
    have = sorted(mb.available_years())
    missing = [y for y in range(min(have), max(have) + 1) if y not in have]
    for year in missing:
        _, used, delta = mb.year_map(year)
        assert used in have
        assert abs(delta) == min(abs(h - year) for h in have)
        assert used - year == delta


def test_year_map_beyond_the_collection_clamps():
    """2025 and 2026 hold 109 plots and do not exist in MapBiomas."""
    have = sorted(mb.available_years())
    _, used, delta = mb.year_map(max(have) + 2)
    assert used == max(have)
    assert delta == -2


def test_window_native_is_the_intersection_of_its_own_years():
    """Stability is per-window. The AND is the whole point -- verify it is really an AND."""
    bounds = (-71.60, -33.50, -71.40, -33.30)
    per_year = []
    for y in (2016, 2017, 2018):
        m, _ = mb.window_native(bounds, y, y)
        per_year.append(m)
    win, used = mb.window_native(bounds, 2016, 2018)
    assert len(used) == 3
    np.testing.assert_array_equal(win, per_year[0] & per_year[1] & per_year[2])
    assert win.sum() <= min(m.sum() for m in per_year)


def test_window_native_excludes_anthropic_classes():
    """No excluded class may survive the mask -- silviculture, agriculture and urban above all."""
    bounds = (-72.40, -35.60, -72.20, -35.40)          # plantation country
    mask, _ = mb.window_native(bounds, 2016, 2018)
    path, _, _ = mb.year_map(2017)
    codes = mb._read(path, bounds)
    for bad in (9, 18, 24, 15, 33, 34, 29, 0):
        assert not mask[codes == bad].any(), f"class {bad} survived the native mask"


def test_roundtrip_projection():
    x, y = 243614.41, 6605072.0                        # a real plot, UTM 19S
    lon, lat = mb.utm_to_ll(x, y)
    assert -71.7 < lon < -71.6 and -30.7 < lat < -30.6
    bx, by = mb.ll_to_utm(lon, lat)
    assert abs(bx - x) < 1e-3 and abs(by - y) < 1e-3
