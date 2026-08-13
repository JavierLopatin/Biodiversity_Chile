"""Cross-sensor (TM/ETM+ vs OLI) level correction, from the offsets measured empirically by
`scripts/40_sensor_harmonization_test.py`.

The offset is measured PER TERCILE of the plot's own curve level (low/mid/high vegetation),
not as one flat number -- the bias scales with vegetation level (a gain/amplitude mismatch,
confirmed by opposite-signed level-vs-residual regression slopes per sensor family), so a
single additive constant overcorrects the low season and undercorrects the growth peak. See
that script's docstring for the diagnosis and `notebooks/08_sensor_harmonization.ipynb` for
the before/after that caught it.

Correcting only the OLI side and leaving TM/ETM+ untouched keeps the older, longer half of
the record (1999-2015, TM/ETM+) as the reference level -- an arbitrary but immaterial choice,
since only the *relative* offset between sensors matters for a curve fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

SENSOR_FAMILY = {
    "landsat5": "tm_etm", "landsat7": "tm_etm",
    "landsat8": "oli", "landsat9": "oli",
}


def load_offsets(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """index -> (levels, offsets), sorted by level, ready for `np.interp`.

    Raises if the bias was never measured (status != "ok") rather than silently applying no
    correction, so a typo'd path fails loudly instead of producing an uncorrected "corrected"
    run. Terciles with too few plots (status "PENDIENTE") are skipped; an index needs at
    least 2 usable terciles to interpolate.
    """
    data = json.loads(Path(path).read_text())
    if data.get("status") != "ok":
        raise ValueError(
            f"{path}: sesgo no medido (status={data.get('status')!r}, "
            f"reason={data.get('reason')!r})")
    out = {}
    for index, row in data["per_index"].items():
        bins = [b for b in row.get("bins", []) if b.get("status") != "PENDIENTE"]
        if len(bins) < 2:
            continue
        bins.sort(key=lambda b: b["level_mediana"])
        levels = np.array([b["level_mediana"] for b in bins])
        offsets = np.array([b["offset_oli_menos_tm_etm_mediana"] for b in bins])
        out[index] = (levels, offsets)
    return out


def apply_offset(da: xr.DataArray, index: str, offsets: dict[str, tuple[np.ndarray, np.ndarray]],
                 level: xr.DataArray | np.ndarray) -> xr.DataArray:
    """Subtract the level-dependent OLI offset for `index`; TM/ETM+ passes through.

    `level` is the plot's own (uncorrected) curve value at each observation's DOY -- the same
    reference the bias was measured against, so the correction is looked up on a matching
    scale. Outside the measured tercile range, `np.interp` holds the nearest endpoint flat
    rather than extrapolating.
    """
    if index not in offsets or "sensor" not in da.coords:
        return da
    sensor = np.asarray(da["sensor"].values)
    is_oli = np.array([SENSOR_FAMILY.get(s) == "oli" for s in sensor], dtype=float)
    levels, level_offsets = offsets[index]
    lvl = np.asarray(level)
    correction_at_level = np.interp(lvl, levels, level_offsets)
    correction = xr.DataArray(is_oli * correction_at_level, dims=da["sensor"].dims,
                              coords={d: da.coords[d] for d in da["sensor"].dims})
    return da - correction
