#!/usr/bin/env python3
"""Centre pixel against the 5x5 patch mean, head to head.

Every predictor in this project is available at two footprints: the single Landsat pixel
containing the plot centroid, and a summary of the 5x5 window around it. The default is the
5x5 mean, and that was a judgement rather than a measurement. This script measures it.

The comparison switches the **whole design** at once — LSP, curve and topography together.
Mixing footprints (LSP at the centre, topography at the 5x5 mean) would confound the two
choices and the result would not answer the question.

Three things are being weighed against each other, and they do not all point the same way:

- **Footprint match.** Plots are 78.5-10,000 m2 against 900 m2 per Landsat pixel. For a small
  plot the centre pixel is the closer match and the 5x5 window mostly samples vegetation the
  botanist never looked at; for a 1 ha plot the reverse holds.
- **Noise.** A single pixel carries the full per-pixel reconstruction error. Averaging 25 of
  them cuts it, at the cost of blurring the plot boundary.
- **Missingness.** Measured on this dataset the centre-pixel LSP has 113 NaN in each of
  `los/ios/sw/mos` against 2 for the 5x5 means, because a degenerate single-pixel season
  leaves the metric undefined. The centre design carries 200 NaN cells against 2.

Results go to a separate root so the main matrix is untouched.

Usage:
    python scripts/15_pixel_ablation.py --run
    python scripts/15_pixel_ablation.py --report
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import features as feat      # noqa: E402
from biodiv import metrics as mx         # noqa: E402
from biodiv import runlog                # noqa: E402

PY = sys.executable
OUT = ROOT / "results" / "px_ablation"
LOGS = ROOT / "logs" / "px_ablation"
SCHEME = "kfold5_owner"
BETA = ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"]
ALPHA = ["hill_q0", "hill_q1", "hill_q2"]

#: RF01 isolates the LSP block, RF03 the curve, RF04 both. The 1-D CNN is included because a
#: convolution over a noisier single-pixel curve may behave differently from a forest over the
#: same values — averaging 25 pixels smooths exactly the high-frequency structure a kernel of
#: width 5 is looking at.
RF_MODELS = ["RF01", "RF03", "RF04"]


def jobs(seeds: int, epochs: int, patience: int) -> list[tuple[str, list[str], bool]]:
    out = []
    for px in ("mean5x5", "center"):
        for m in RF_MODELS:
            for ix in feat.INDICES:
                out.append((f"{m}_{ix}_{px}",
                            [PY, "scripts/09_run_baselines.py", "--model", m, "--index", ix,
                             "--scheme", SCHEME, "--seeds", str(seeds), "--px", px,
                             "--out", str(OUT)], False))
        for ix in feat.INDICES:
            out.append((f"C1D_{ix}_{px}",
                        [PY, "scripts/11_run_conv.py", "--substrate", "curve1d", "--index", ix,
                         "--scheme", SCHEME, "--seeds", str(seeds), "--px", px,
                         "--max-epochs", str(epochs), "--patience", str(patience),
                         "--out", str(OUT)], True))
    return out


def run(seeds: int, epochs: int, patience: int, devices: list[str], workers: int) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = jobs(seeds, epochs, patience)
    print(f"{len(todo)} jobs -> {OUT}")
    import os
    import time

    done = {"n": 0}

    def launch(item):
        name, cmd, gpu = item
        env = dict(os.environ)
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = devices[done["n"] % len(devices)]
        env["PYTHONPATH"] = str(ROOT / "src") + ":" + env.get("PYTHONPATH", "")
        log = LOGS / f"{name}.log"
        t0 = time.time()
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        done["n"] += 1
        print(f"  [{done['n']:>2}/{len(todo)}] {name:24s} "
              f"{'ok' if rc == 0 else f'FAILED({rc})':12s} {(time.time() - t0) / 60:4.1f}m",
              flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(launch, todo))

    runlog.rebuild_summary(OUT)


def _px_of(run_id: str) -> str:
    return "center" if "_ctr" in run_id or run_id.split("_")[0].endswith("c") else "mean5x5"


def _stem(run_id: str) -> str:
    """Strip the pixel-level marks so the two variants of a model line up in one row."""
    parts = run_id.split("_")
    head = parts[0][:-1] if parts[0].endswith("c") and parts[0][:-1] in RF_MODELS else parts[0]
    tail = [p for p in parts[1:] if p != "ctr"]
    tail = [t.replace("lsp_ctr", "lsp").replace("topo_ctr", "topo") for t in tail]
    ix = next((t for t in tail if t in feat.INDICES), "")
    return f"{head}_{ix}"


def report() -> None:
    s = pd.read_csv(OUT / "summary.csv")
    s["px"] = s["run_id"].map(_px_of)
    s["stem"] = s["run_id"].map(_stem)

    rows = []
    for block, names in (("beta", BETA), ("alpha", ALPHA)):
        d = s[s["target"].isin(names)]
        piv = d.pivot_table(index="stem", columns="px", values="R2", aggfunc="mean")
        piv = piv.dropna(how="any")
        piv["delta_center_minus_5x5"] = piv["center"] - piv["mean5x5"]
        rows.append(piv.assign(block=block).reset_index())
    tbl = pd.concat(rows, ignore_index=True)

    tdir = ROOT / "results" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(tdir / "pixel_ablation.csv", index=False)

    for block in ("beta", "alpha"):
        d = tbl[tbl["block"] == block].sort_values("delta_center_minus_5x5")
        print(f"\n=== {block} ===   (delta > 0 favours the centre pixel)")
        print(d[["stem", "mean5x5", "center", "delta_center_minus_5x5"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        print(f"  mean delta = {d['delta_center_minus_5x5'].mean():+.4f}   "
              f"centre wins in {int((d['delta_center_minus_5x5'] > 0).sum())}/{len(d)} models")

    # Paired test over the (model x target x seed) scores. The pairing is exact: the same
    # model, index, fold assignment and seed, differing only in the pixel level.
    w = (s[s["target"].isin(BETA)]
         .pivot_table(index=["stem", "target", "seed"], columns="px", values="R2")
         .dropna())
    if len(w) > 3:
        from scipy import stats
        t = stats.ttest_rel(w["center"], w["mean5x5"])
        wil = stats.wilcoxon(w["center"], w["mean5x5"])
        print(f"\npaired over {len(w)} (model x target x seed) composition scores: "
              f"mean delta = {(w['center'] - w['mean5x5']).mean():+.4f}, "
              f"t p = {t.pvalue:.4f}, Wilcoxon p = {wil.pvalue:.4f}")

    d = tbl[tbl["block"] == "beta"].sort_values("mean5x5")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(d))))
    y = np.arange(len(d))
    ax.barh(y - 0.19, d["mean5x5"], height=0.36, label="5x5 patch mean", color="#4c72b0")
    ax.barh(y + 0.19, d["center"], height=0.36, label="centre pixel", color="#dd8452")
    ax.set_yticks(y)
    ax.set_yticklabels(d["stem"], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$R^2$ agrupado fuera de fold — targets de composición")
    ax.set_title("Píxel central contra media 5×5, mismo diseño en todo lo demás", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "fig14_pixel_ablation.pdf", dpi=200)
    print(f"\n-> {tdir / 'pixel_ablation.csv'}")
    print("-> results/figures/fig14_pixel_ablation.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--devices", nargs="+", default=["0", "1"])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if args.run:
        run(args.seeds, args.max_epochs, args.patience, args.devices, args.workers)
    if args.report or not args.run:
        report()


if __name__ == "__main__":
    main()
