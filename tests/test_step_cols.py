"""A curve table with any number of steps must flow through, or fail loudly.

The raw 3-year series has 24 to 196 steps where the composite year has 52. Indexing a curve
table with the hardcoded ``STEP_COLS`` list did the worst possible thing to the longer ones:
pandas returned the first 52 columns and said nothing. Eight runs scored, ranked, and were
reported as "68-step" and "196-step" curves when both were in fact 52 columns covering
different amounts of real time. The shorter tables at least raised `KeyError`, which is how
the truncation was found at all.

So the invariant is: the step columns come from the DataFrame, never from a constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import features as feat            # noqa: E402


def _frame(n: int, extra: bool = True) -> pd.DataFrame:
    cols = {f"s{i:02d}": np.arange(3, dtype=float) + i for i in range(n)}
    if extra:                                    # non-step columns must be ignored
        cols |= {"plot_id": [1, 2, 3], "index": "kndvi", "px": "mean5x5", "slope": 0.0,
                 "sos": 1.0, "stratum": "cover"}
    return pd.DataFrame(cols)


@pytest.mark.parametrize("n", [24, 36, 52, 68, 100, 156, 196])
def test_all_steps_are_returned(n):
    cols = feat.step_cols(_frame(n))
    assert len(cols) == n, f"{n}-step table yielded {len(cols)} columns"
    assert cols == [f"s{i:02d}" for i in range(n)], "step columns out of order"


def test_non_step_columns_are_ignored():
    """`slope`, `sos` and `stratum` all begin with 's' and are not steps."""
    assert feat.step_cols(_frame(5)) == ["s00", "s01", "s02", "s03", "s04"]


def test_a_table_with_no_steps_raises():
    with pytest.raises(ValueError, match="no step columns"):
        feat.step_cols(pd.DataFrame({"plot_id": [1], "sos": [2.0]}))


def test_order_is_numeric_not_lexicographic():
    """s9 before s10. Sorting as text would put s10 first and scramble the time axis."""
    df = pd.DataFrame({f"s{i}": [0.0] for i in (0, 1, 2, 9, 10, 11)})
    assert feat.step_cols(df) == ["s0", "s1", "s2", "s9", "s10", "s11"]


@pytest.mark.parametrize("n", [24, 68, 196])
def test_substrates_adapt_to_the_step_count(n):
    """The end-to-end invariant: a non-52-step curve table produces a correctly sized image.

    Skipped when the refit for that resolution is not on disk, so the suite still runs on a
    fresh clone.
    """
    import os
    sfx = f"_raw{n}"
    if not (ROOT / "data" / "derived" / f"phenoshape_by_index{sfx}.parquet").exists():
        pytest.skip(f"{sfx} curves not built")
    old = os.environ.get(feat.CURVE_SUFFIX_ENV)
    os.environ[feat.CURVE_SUFFIX_ENV] = sfx
    try:
        from biodiv import substrates as sub
        curves, _ = sub.load_curves("kndvi", derived=str(ROOT / "data" / "derived"))
        assert curves.shape[-1] == n, f"{sfx}: got {curves.shape[-1]} steps, expected {n}"
    finally:
        if old is None:
            os.environ.pop(feat.CURVE_SUFFIX_ENV, None)
        else:
            os.environ[feat.CURVE_SUFFIX_ENV] = old
        feat._load_tables.cache_clear()
