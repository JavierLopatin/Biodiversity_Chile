#!/usr/bin/env python3
"""How many directions a set of curves actually occupies. The gate before any pretraining.

The first masked-autoencoder attempt spent a full pretraining run to learn something this
measurement says in seconds: the 135,250 pixel curves carry **participation dimension 1.3**
against the 1,082 plot curves' 1.2 -- 125x more images spanning 1.08x more space. It gained
nothing (0.359 against 0.362, `docs/14` section 4).

So this is the gate. If the MapBiomas pool does not clear 1.3 by a clear margin, the coverage
problem is not solved and no amount of GPU will fix it.

The participation ratio of the covariance eigenvalues,

    PR = (sum lambda)^2 / sum lambda^2

is 1 when one direction holds everything and N when all N are equal. It is preferred here to
"number of PCs for 95 % variance" because it needs no threshold, and a threshold is exactly
the kind of free parameter that would let this gate be argued past.

Usage:
    python scripts/34_participation_dimension.py
    python scripts/34_participation_dimension.py --ngs 156
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import curves                      # noqa: E402


def participation_ratio(x: np.ndarray) -> float:
    """PR of the covariance spectrum of ``x`` (rows = samples, cols = steps)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 2:
        return float("nan")
    lam = np.linalg.eigvalsh(np.cov(x - x.mean(0), rowvar=False))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def pool_curves(updir: Path, index: str, ngs: int) -> np.ndarray:
    """The unlabelled pool, gridded over each sample's own causal window."""
    man = pd.read_csv(updir / "manifest.csv")
    if "error" in man:
        man = man[man["error"].isna()]
    ser = pd.read_parquet(updir / "series.parquet")
    ser["t"] = ser["time"].values.astype("datetime64[D]").astype(float)
    win = man.set_index("sample_id")[["win_start", "win_end"]]
    out = []
    for sid, g in ser.groupby("sample_id", sort=False):
        if sid not in win.index:
            continue
        w = win.loc[sid]
        lo = np.datetime64(f"{int(w.win_start)}-01-01", "D").astype(float)
        hi = np.datetime64(f"{int(w.win_end)}-12-31", "D").astype(float)
        out.append(curves.interp_grid(g["t"].to_numpy(), g[index].to_numpy(), ngs,
                                      t_min=lo, t_max=hi))
    return np.asarray(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unlabelled", default="data/derived/unlabelled")
    p.add_argument("--derived", default="data/derived")
    p.add_argument("--index", default="kndvi")
    p.add_argument("--ngs", type=int, default=curves.NGS)
    args = p.parse_args()

    rows = []

    pool = pool_curves(Path(args.unlabelled), args.index, args.ngs)
    ok = np.isfinite(pool).all(axis=1)
    rows.append(("MapBiomas pool (new)", int(ok.sum()), participation_ratio(pool)))

    # the two populations the new pool has to beat, rebuilt the same way
    d = Path(args.derived)
    px = d / "phenoshape_pixels.parquet"
    if px.exists():
        t = pd.read_parquet(px)
        t = t[t["index"] == args.index]
        step = [c for c in t.columns if c.startswith("s") and c[1:].isdigit()]
        rows.append((f"pixel curves ({args.index})", len(t),
                     participation_ratio(t[step].to_numpy())))

    by = d / "phenoshape_by_index.parquet"
    if by.exists():
        t = pd.read_parquet(by)
        t = t[(t["index"] == args.index) & (t.get("px", "mean5x5") == "mean5x5")]
        step = [c for c in t.columns if c.startswith("s") and c[1:].isdigit()]
        rows.append((f"plot curves ({args.index})", len(t),
                     participation_ratio(t[step].to_numpy())))

    print(f"participation dimension  (index={args.index}, {args.ngs} steps)\n")
    print(f"{'population':<28} {'n':>9} {'PR':>7}")
    print("-" * 47)
    for name, n, pr in rows:
        print(f"{name:<28} {n:>9,} {pr:>7.2f}")
    print("-" * 47)

    new = rows[0][2]
    old = next((r[2] for r in rows if r[0].startswith("pixel")), 1.3)

    # The reference populations are stored as fitted 52-step tables and are NOT re-gridded:
    # the observations they came from live in the .nc cubes, not in the parquet. So at any
    # other --ngs this compares a pool of N steps against a reference stuck at 52, and PR
    # grows with the number of steps available. That comparison is not one -- and left
    # unguarded it flips the verdict: pool PR is 1.24 at 24 steps, 1.48 at 52, 1.71 at 156,
    # against a reference frozen at 1.28 throughout.
    if args.ngs != 52:
        print(f"\nNO VERDICT at {args.ngs} steps.")
        print(f"  The reference curves are stored fitted at 52 steps and cannot be re-gridded")
        print(f"  from this parquet, so pool@{args.ngs} vs reference@52 compares different")
        print(f"  objects: more steps means more directions available, whatever the coverage.")
        print(f"  Re-run without --ngs for the gate.")
        return

    print(f"\ngate: the new pool must clear the pixel curves' {old:.2f}")
    print("PASS" if new > old * 1.15 else "FAIL",
          f"-- pool {new:.2f} vs {old:.2f}  ({new/old:.2f}x)")
    print("\nNote the margin. The 1.15x threshold is not the project's -- it was chosen when")
    print("this script was written and nothing validates it. A 16 % gain in occupied")
    print("directions is real but modest, and it is the honest read that this pool improves")
    print("coverage without transforming it.")
    if new <= old * 1.15:
        print("\nThe coverage problem is NOT solved. Do not spend GPU on pretraining.")


if __name__ == "__main__":
    main()
