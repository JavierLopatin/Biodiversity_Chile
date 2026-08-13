"""OLI-side, level-dependent correction applied by `biodiv.harmonize`, from the tercile
offsets measured by `scripts/40_sensor_harmonization_test.py`. No datacube, no network --
synthetic fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import harmonize                     # noqa: E402


def _da(sensors):
    n = len(sensors)
    return xr.DataArray(
        np.zeros(n), dims="time",
        coords={"time": np.arange(n), "sensor": ("time", np.array(sensors))},
    )


def _payload(bins):
    return {
        "status": "ok",
        "per_index": {
            "kndvi": {"n_parcelas": 800, "bins": bins},
            "nbr": {"status": "PENDIENTE", "reason": "sin observaciones elegibles"},
        },
    }


THREE_BINS = [
    {"n_parcelas": 380, "level_mediana": 0.05, "offset_oli_menos_tm_etm_mediana": 0.018,
    "wilcoxon_p": 1e-60},
    {"n_parcelas": 540, "level_mediana": 0.20, "offset_oli_menos_tm_etm_mediana": 0.045,
    "wilcoxon_p": 1e-80},
    {"n_parcelas": 450, "level_mediana": 0.50, "offset_oli_menos_tm_etm_mediana": 0.055,
    "wilcoxon_p": 1e-70},
]


def test_apply_offset_only_touches_oli():
    da = _da(["landsat7", "landsat8", "landsat5", "landsat9"])
    offsets = {"kndvi": (np.array([0.0, 1.0]), np.array([0.04, 0.04]))}
    level = np.array([0.5, 0.5, 0.5, 0.5])
    out = harmonize.apply_offset(da, "kndvi", offsets, level)
    expected = np.array([0.0, -0.04, 0.0, -0.04])
    assert np.allclose(out.values, expected)


def test_apply_offset_scales_with_level():
    """Low-level and high-level OLI observations get different corrections."""
    da = _da(["landsat8", "landsat8"])
    offsets = _offsets_from_payload(_payload(THREE_BINS))
    level = np.array([0.05, 0.50])   # matches the low and high tercile midpoints
    out = harmonize.apply_offset(da, "kndvi", offsets, level)
    low_correction = -out.values[0]
    high_correction = -out.values[1]
    assert low_correction == pytest.approx(0.018, abs=1e-6)
    assert high_correction == pytest.approx(0.055, abs=1e-6)
    assert high_correction > low_correction


def _offsets_from_payload(payload):
    out = {}
    for index, row in payload["per_index"].items():
        bins = [b for b in row.get("bins", []) if b.get("status") != "PENDIENTE"]
        if len(bins) < 2:
            continue
        bins.sort(key=lambda b: b["level_mediana"])
        out[index] = (np.array([b["level_mediana"] for b in bins]),
                      np.array([b["offset_oli_menos_tm_etm_mediana"] for b in bins]))
    return out


def test_apply_offset_noop_when_index_not_in_offsets():
    da = _da(["landsat7", "landsat8"])
    offsets = {"kndvi": (np.array([0.0, 1.0]), np.array([0.04, 0.04]))}
    out = harmonize.apply_offset(da, "nbr", offsets, np.array([0.5, 0.5]))
    assert np.allclose(out.values, da.values)


def test_apply_offset_noop_without_sensor_coord():
    da = xr.DataArray(np.array([1.0, 2.0]), dims="time")
    offsets = {"kndvi": (np.array([0.0, 1.0]), np.array([0.04, 0.04]))}
    out = harmonize.apply_offset(da, "kndvi", offsets, np.array([0.5, 0.5]))
    assert np.allclose(out.values, da.values)


def test_load_offsets_reads_ok_status_and_skips_pendiente_bins(tmp_path):
    payload = _payload(THREE_BINS + [{"status": "PENDIENTE", "reason": "muestra insuficiente"}])
    p = tmp_path / "sensor_harmonization.json"
    p.write_text(json.dumps(payload))
    offsets = harmonize.load_offsets(p)
    assert set(offsets.keys()) == {"kndvi"}
    levels, vals = offsets["kndvi"]
    assert list(levels) == [0.05, 0.20, 0.50]
    assert list(vals) == [0.018, 0.045, 0.055]


def test_load_offsets_drops_index_with_fewer_than_two_usable_bins(tmp_path):
    payload = _payload([THREE_BINS[0], {"status": "PENDIENTE", "reason": "insuficiente"}])
    p = tmp_path / "sensor_harmonization.json"
    p.write_text(json.dumps(payload))
    offsets = harmonize.load_offsets(p)
    assert offsets == {}


def test_load_offsets_raises_when_bias_never_measured(tmp_path):
    p = tmp_path / "sensor_harmonization.json"
    p.write_text(json.dumps({"status": "PENDIENTE", "reason": "sin parcelas elegibles"}))
    with pytest.raises(ValueError, match="sesgo no medido"):
        harmonize.load_offsets(p)
