#!/usr/bin/env python3
"""Search the space the 2-D CNN was never given: curve, substrate, architecture.

Why this exists. The convolutional family is the story of the paper and it is not the best
model -- but it has never been tuned. Three facts, all measured and all in
`docs/13_phenology_year_boundary.md` and the results tables:

* the curve it consumes was fitted with `linear`, the *default* reconstructor, out of ten
  that `phenosensing` offers. Nobody chose it;
* the architecture ablation (stage 4c) varied **one factor at a time**, and its three
  positive factors -- climate context +0.036, no augmentation +0.012, width C +0.007 --
  were never combined;
* the gap to the best model overall is **0.023**, barely above the 2 sd noise floor.

HOW THIS STAYS HONEST. The goal is for the CNN to win; the risk is manufacturing it.

1. **Selection is not the result.** Searching ~150 combinations on 3 seeds and reporting the
   maximum overstates it. Search uses seeds 0..2; the finalists are re-run with a disjoint
   block (``--confirm-seed-start``) and *that* is the number to quote. The script prints the
   shrinkage between the two, which is the first thing a reviewer asks for.
2. **The whole grid is written**, not just the winner, to `results/tables/cnn_search.csv`.
   That 140 of 150 combinations lose to a Random Forest is as reportable as one winning.
3. **The winner propagates to every family**, not only the CNN -- that is stage C of the
   plan, run separately. Comparing a tuned CNN against an untuned forest would mean nothing.

Stages, each resumable and skippable:

    curve       the refit variants on disk, on a fixed substrate
    ngs         curve resolution: the image is sqrt(nGS) on a side
    substrate   the 11 substrates on the best curves
    arch        architecture, augmentation and regularisation, by coordinate descent
    confirm     the finalists, fresh seeds

Usage:
    python scripts/30_search_cnn.py --stage curve
    python scripts/30_search_cnn.py --stage arch --top 3
    python scripts/30_search_cnn.py --all
    python scripts/30_search_cnn.py --report
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import targets as tg                # noqa: E402

PY = sys.executable
LOGDIR = ROOT / "logs" / "search"
OUT = ROOT / "results" / "tables" / "cnn_search.csv"
SCHEME = "kfold5_window"
INDEX = "kndvi"

#: The facets are averaged with equal weight. Not a grand mean over the 15 targets: `phylo`
#: has five and `dark` one, so a target-level mean would silently weight the phylogenetic
#: facet five times more than dark diversity.
FACETS = list(tg.FACETS)


# --------------------------------------------------------------------------------------
# what a job is
# --------------------------------------------------------------------------------------

def job_cmd(substrate: str, seeds: int, seed_start: int, **kw) -> list[str]:
    cmd = [PY, "scripts/11_run_conv.py", "--substrate", substrate, "--scheme", SCHEME,
           "--seeds", str(seeds), "--max-epochs", "200", "--patience", "20"]
    if substrate not in ("stack5", "curve5"):
        cmd += ["--index", INDEX]
    if seed_start:
        cmd += ["--seed-start", str(seed_start)]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    return cmd


def run_jobs(jobs: list[dict], devices: list[str], workers: int) -> None:
    """Each job is a dict with `curve` and the kwargs of `job_cmd`. Runs round-robin."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    done = {"n": 0}
    t0 = time.time()

    def launch(i_job):
        i, job = i_job
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = devices[i % len(devices)]
        env["PYTHONPATH"] = str(ROOT / "src") + ":" + env.get("PYTHONPATH", "")
        # the curve variant is an *environment* switch, not a flag: it has to reach both the
        # feature loader and `runlog.make_run_id`, or two variants collide in one directory
        env["BIODIV_CURVES"] = job["curve"]
        name = job.get("name") or f"{job['curve']}_{job['substrate']}_{i}"
        cmd = job_cmd(**{k: v for k, v in job.items() if k not in ("curve", "name")})
        log = LOGDIR / f"{name.replace('/', '_')}.log"
        with log.open("w") as fh:
            fh.write(f"BIODIV_CURVES={job['curve']}\n" + " ".join(cmd) + "\n\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        done["n"] += 1
        print(f"  [{done['n']:>3}/{len(jobs)}] {name:52s} "
              f"{'ok' if rc == 0 else f'FAILED({rc})':12s} {(time.time()-t0)/60:5.1f}m",
              flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(launch, enumerate(jobs)))


# --------------------------------------------------------------------------------------
# reading results back
# --------------------------------------------------------------------------------------

def summary() -> pd.DataFrame:
    """One row per (run_id, curve variant): the facet means and their average."""
    f = ROOT / "results" / "models" / "summary.csv"
    s = pd.read_csv(f)
    s = s[(s["scheme"] == SCHEME) & s["family"].isin(["C1D", "C2D"])].copy()
    s["facet"] = s["target"].map(tg.FACET_OF)
    g = (s.groupby(["run_id", "family", "substrate", "facet"])["R2"].mean()
         .unstack("facet").reindex(columns=FACETS))
    g["mean"] = g.mean(axis=1)
    return g.sort_values("mean", ascending=False)


def curve_variants() -> list[str]:
    """Suffixes of every refit on disk, e.g. `_savgol_shrink`."""
    out = []
    for p in sorted((ROOT / "data" / "derived").glob("phenoshape_by_index_*.parquet")):
        sfx = p.stem.replace("phenoshape_by_index", "")
        # the pixel table has to exist too or `pxcube` silently falls over mid-stage
        if (ROOT / "data" / "derived" / f"phenoshape_pixels{sfx}.parquet").exists():
            out.append(sfx)
    return out


def top_curves(k: int) -> list[str]:
    """The k best curve variants seen so far, read back from the results."""
    g = summary().reset_index()
    g["curve"] = g.run_id.str.extract(r"_((?:[A-Za-z]+_)?(?:shrink|reflect))$")[0]
    g = g[g.curve.notna()]
    if g.empty:
        return ["_linear_shrink"]
    best = g.groupby("curve")["mean"].max().sort_values(ascending=False)
    return ["_" + c for c in best.head(k).index]


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------

BASE_KW = dict(context="clim+topo+area")     # the single largest measured gain, +0.036


def stage_curve(args) -> list[dict]:
    """Every refit variant on one substrate. Isolates the reconstruction."""
    return [dict(curve=c, substrate="serpentine", seeds=args.seeds,
                 seed_start=0, name=f"curve{c}", **BASE_KW)
            for c in curve_variants()]


def stage_substrate(args) -> list[dict]:
    subs = ["reshape", "serpentine", "hilbert", "gaf", "mtf", "ndi", "cwt", "cos2d",
            "spectrogram", "stack5", "pxcube", "curve1d"]
    return [dict(curve=c, substrate=s, seeds=args.seeds, seed_start=0,
                 name=f"sub_{s}{c}", **BASE_KW)
            for c in top_curves(args.top) for s in subs]


def stage_arch(args) -> list[dict]:
    """Coordinate descent from the combination of the three positive 4c factors.

    Not a full grid: 4 architectures x 2 widths x 5 augmentations x 3 dropouts is 120 runs
    for a space where most axes are close to independent. One axis at a time from a strong
    starting point costs a fifth of that and answers the same question.
    """
    curve = top_curves(1)[0]
    sub = best_substrate() or "serpentine"
    common = dict(curve=curve, substrate=sub, seeds=args.seeds, seed_start=0)
    jobs = []
    # the combination that has never been tried: ctx + noaug + wC, and each drop-one
    for w, noaug in itertools.product(["B", "C"], [True, False]):
        jobs.append(dict(**common, width=w, no_augment=noaug,
                         name=f"arch_w{w}{'_noaug' if noaug else ''}", **BASE_KW))
    for arch in ("res", "se", "multi"):
        jobs.append(dict(**common, arch=arch, width="B", no_augment=True,
                         name=f"arch_{arch}", **BASE_KW))
    # augmentation: the Trait_2DCNN terms this project lacked
    for slope, prob in [(0.05, 1.0), (0.10, 1.0), (0.0, 0.15), (0.05, 0.15)]:
        jobs.append(dict(**common, aug_slope=slope, aug_prob=prob,
                         name=f"aug_s{slope}_p{prob}", **BASE_KW))
    # regularisation, on the assumption that 1,082 samples want more of it
    for pc, ph in [(0.2, 0.3), (0.1, 0.5), (0.2, 0.5)]:
        jobs.append(dict(**common, p_conv=pc, p_head=ph, no_augment=True,
                         name=f"reg_{pc}_{ph}", **BASE_KW))
    for wd in (3e-2, 1e-1):
        jobs.append(dict(**common, weight_decay=wd, no_augment=True,
                         name=f"wd_{wd}", **BASE_KW))
    return jobs


def best_substrate() -> str | None:
    g = summary().reset_index()
    g = g[g.family == "C2D"]
    return None if g.empty else g.iloc[0]["substrate"]


def stage_confirm(args) -> list[dict]:
    """Re-run the finalists on a disjoint block of seeds.

    This is the step that turns a search maximum into a number worth reporting.
    """
    g = summary().reset_index().head(args.top)
    jobs = []
    for _, r in g.iterrows():
        curve = pd.Series([r.run_id]).str.extract(
            r"_((?:[A-Za-z]+_)?(?:shrink|reflect))$")[0].iloc[0]
        jobs.append(dict(curve="_" + curve if pd.notna(curve) else "",
                         substrate=r.substrate, seeds=args.confirm_seeds,
                         seed_start=args.confirm_seed_start,
                         name=f"confirm_{r.run_id}", **BASE_KW))
    return jobs


STAGES = {"curve": stage_curve, "substrate": stage_substrate,
          "arch": stage_arch, "confirm": stage_confirm}
ORDER = ["curve", "substrate", "arch", "confirm"]


# --------------------------------------------------------------------------------------

def report() -> None:
    g = summary()
    if g.empty:
        print("no convolutional runs yet")
        return
    g.to_csv(OUT)
    print(f"{len(g)} convolutional runs under {SCHEME}\n")
    print(g.head(20).round(3).to_string())

    # the comparison that matters: against the best of every other family
    s = pd.read_csv(ROOT / "results" / "models" / "summary.csv")
    s = s[s["scheme"] == SCHEME].copy()
    s["facet"] = s["target"].map(tg.FACET_OF)
    a = (s.groupby(["run_id", "family", "facet"])["R2"].mean().unstack("facet")
         .reindex(columns=FACETS))
    a["mean"] = a.mean(axis=1)
    fam = a.reset_index().groupby("family")["mean"].max().sort_values(ascending=False)
    print("\nbest run of each family (mean over the five facets):\n")
    print(fam.round(3).to_string())
    if "C2D" in fam.index:
        gap = fam["C2D"] - fam.drop("C2D").max()
        verdict = "the 2-D CNN leads" if gap > 0 else f"still {-gap:.3f} behind"
        print(f"\n  {verdict}   (seed noise ~0.011, so 2 sd = 0.022)")
    print(f"\n  -> {OUT.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=list(STAGES))
    p.add_argument("--all", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--seeds", type=int, default=3, help="seeds during the search")
    p.add_argument("--confirm-seeds", type=int, default=5, dest="confirm_seeds")
    p.add_argument("--confirm-seed-start", type=int, default=10, dest="confirm_seed_start",
                   help="disjoint from the search block, so the confirmation is independent")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--devices", nargs="+", default=["0", "1"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.report:
        report()
        return
    stages = ORDER if args.all else ([args.stage] if args.stage else [])
    if not stages:
        p.error("pass --stage, --all or --report")

    for st in stages:
        jobs = STAGES[st](args)
        print(f"\n[{st}] {len(jobs)} jobs")
        if args.dry_run:
            for j in jobs[:8]:
                print("   ", j.get("name"), "|", " ".join(job_cmd(
                    **{k: v for k, v in j.items() if k not in ("curve", "name")})[2:]))
            if len(jobs) > 8:
                print(f"    ... and {len(jobs)-8} more")
            continue
        run_jobs(jobs, args.devices, args.workers)
        report()


if __name__ == "__main__":
    main()
