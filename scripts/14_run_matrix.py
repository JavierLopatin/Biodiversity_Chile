#!/usr/bin/env python3
"""Run the whole model matrix unattended, in the order the plan specifies.

Design constraints this script exists to satisfy:

**Resumable.** Every job is skipped if its `pooled_metrics.csv` already exists, so an
interrupted night is continued by re-running the same command. Nothing is recomputed and
nothing is duplicated.

**Ordered by information, not by convenience.** Random Forest first, because it costs minutes
and decides which vegetation index and which feature block the expensive tiers should use.
The 2-D screen runs on one index only; the top substrates are then re-expanded over the
remaining four. Running the full crossing would be 34,000 fits.

**Two GPUs, one CPU pool.** The RF tier is CPU-bound and runs concurrently with the first GPU
stage. GPU jobs are handed out round-robin via `CUDA_VISIBLE_DEVICES`.

**Deadline-aware.** `--deadline-hours` stops *launching* new jobs after the budget is spent;
running jobs finish. An overnight run therefore ends with a coherent set of results rather
than a truncated one, and the next invocation picks up where it stopped.

Usage:
    python scripts/14_run_matrix.py --all --deadline-hours 11
    python scripts/14_run_matrix.py --stage rf
    python scripts/14_run_matrix.py --dry-run --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biodiv import features as feat            # noqa: E402
from biodiv import targets as tg               # noqa: E402

PY = sys.executable
LOGDIR = ROOT / "logs" / "modelling"
PROGRESS = ROOT / "docs" / "08_modelling_progress.md"

PRIMARY = "kfold5_owner"
SPATIAL = "kfold5_block20"
OPTIMISTIC = "kfold5_random"
OTHER_SCHEMES = ["kfold5_dataset", "kfold5_location", "lodo_owner"]

SCREEN_SEEDS = 3     # screening tiers: enough to see through seed noise, cheap
FINAL_SEEDS = 5      # finalists: the number reported in the paper


@dataclass
class Job:
    name: str
    cmd: list[str]
    gpu: bool = True


def _dl_common(args) -> list[str]:
    return ["--max-epochs", str(args.max_epochs), "--patience", str(args.patience)]


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------

def stage_rf(args) -> list[Job]:
    """Tier 0 + Tier 1. CPU only, ~1-2 h, and it decides everything downstream."""
    # One job per model rather than a single `--all` invocation: the CPU pool has few slots,
    # and a multi-hour omnibus process blocks one of them and loses everything on a failure.
    jobs = [Job(f"rf/{m}@{PRIMARY}",
                [PY, "scripts/09_run_baselines.py", "--model", m, "--scheme", PRIMARY,
                 "--seeds", str(SCREEN_SEEDS)], gpu=False)
            for m in ("B00", "B01", "B02", "B03", "RF01", "RF02", "RF03", "RF04",
                      "RF05", "RF06", "RF08")]
    # the optimism gap needs the same models under the two contrasting schemes
    for scheme in (SPATIAL, OPTIMISTIC):
        for model in ("B00", "B01", "B02", "B03"):
            jobs.append(Job(f"rf/{model}@{scheme}",
                            [PY, "scripts/09_run_baselines.py", "--model", model,
                             "--scheme", scheme, "--seeds", str(SCREEN_SEEDS)], gpu=False))
        for model in ("RF01", "RF03"):
            jobs.append(Job(f"rf/{model}@{scheme}",
                            [PY, "scripts/09_run_baselines.py", "--model", model,
                             "--scheme", scheme, "--seeds", str(SCREEN_SEEDS)], gpu=False))
    return jobs


def stage_c1d(args) -> list[Job]:
    """Tier 3 — the 1-D control, over all five indices. Runs early because every 2-D claim
    is a claim relative to it."""
    jobs = []
    for ix in feat.INDICES:
        jobs.append(Job(f"c1d/{ix}",
                        [PY, "scripts/11_run_conv.py", "--substrate", "curve1d",
                         "--index", ix, "--scheme", PRIMARY,
                         "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)))
    # the five indices as channels of one 1-D signal: the control that isolates what
    # `stack5` gains by making the index axis a spatial dimension instead of a channel
    jobs.append(Job("c1d/curve5",
                    [PY, "scripts/11_run_conv.py", "--substrate", "curve5",
                     "--scheme", PRIMARY,
                     "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)))
    return jobs


def stage_mlp(args) -> list[Job]:
    """Tier 2 — tabular MLP."""
    jobs = []
    for model in ("MLP01", "MLP02"):
        for ix in feat.INDICES:
            jobs.append(Job(f"mlp/{model}/{ix}",
                            [PY, "scripts/10_run_tabular_dl.py", "--model", model,
                             "--index", ix, "--scheme", PRIMARY,
                             "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)))
    bi = best_index(args)
    for model in ("MLP03", "MLP05a", "MLP05c"):
        jobs.append(Job(f"mlp/{model}/{bi}",
                        [PY, "scripts/10_run_tabular_dl.py", "--model", model,
                         "--index", bi, "--scheme", PRIMARY,
                         "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)))
    jobs.append(Job("mlp/MLP04",
                    [PY, "scripts/10_run_tabular_dl.py", "--model", "MLP04",
                     "--scheme", PRIMARY, "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)))
    return jobs


def stage_4a(args) -> list[Job]:
    """Tier 4a — the substrate screen, on the index the RF tier selected."""
    substrates = ["reshape", "serpentine", "gaf", "mtf", "ndi", "cwt", "hilbert",
                  "cos2d", "spectrogram", "stack5", "pxcube"]
    bi = best_index(args)
    jobs = []
    for s in substrates:
        cmd = [PY, "scripts/11_run_conv.py", "--substrate", s, "--scheme", PRIMARY,
               "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)
        if s != "stack5":
            cmd += ["--index", bi]
        jobs.append(Job(f"4a/{s}", cmd))
    return jobs


def stage_4b(args) -> list[Job]:
    """Tier 4b — the top substrates re-expanded over the remaining indices."""
    tops = best_substrates(args, k=3)
    jobs = []
    for s in tops:
        if s in ("stack5", "curve5"):
            continue
        for ix in feat.INDICES:
            jobs.append(Job(f"4b/{s}/{ix}",
                            [PY, "scripts/11_run_conv.py", "--substrate", s,
                             "--index", ix, "--scheme", PRIMARY,
                             "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)))
    return jobs


def stage_4c(args) -> list[Job]:
    """Tier 4c — one factor at a time around the winning substrate."""
    s = best_substrates(args, k=1)[0]
    bi = best_index(args)
    base = [PY, "scripts/11_run_conv.py", "--substrate", s, "--scheme", PRIMARY,
            "--seeds", str(SCREEN_SEEDS)] + _dl_common(args)
    if s not in ("stack5", "curve5"):
        base += ["--index", bi]
    jobs = []
    for rot in ("calendar",):
        jobs.append(Job(f"4c/rotation-{rot}", base + ["--rotation", rot]))
    for norm in ("global", "perSample"):
        jobs.append(Job(f"4c/normalize-{norm}", base + ["--normalize", norm]))
    for fus in ("none", "film", "patch"):
        jobs.append(Job(f"4c/fusion-{fus}", base + ["--fusion", fus]))
    for w in ("A", "C", "X"):
        jobs.append(Job(f"4c/width-{w}", base + ["--width", w]))
    jobs.append(Job("4c/mixup", base + ["--mixup"]))
    jobs.append(Job("4c/no-augment", base + ["--no-augment"]))
    return jobs


def stage_final(args) -> list[Job]:
    """Tier 5 — the finalists under every remaining CV scheme, at full seed count."""
    fin = finalists(args)
    jobs = []
    for run in fin:
        for scheme in [SPATIAL, OPTIMISTIC] + OTHER_SCHEMES:
            cmd = _rebuild_cmd(run, scheme, FINAL_SEEDS, args)
            if cmd:
                jobs.append(Job(f"final/{run['run_id']}@{scheme}", cmd,
                                gpu=run["family"] != "RF"))
    return jobs


# --------------------------------------------------------------------------------------
# choices driven by results already on disk
# --------------------------------------------------------------------------------------

#: Ranking targets. Not `TARGETS_MAIN`: measured on the first complete tier, alpha diversity
#: is not predictable across contributors (`hill_q0`, `hill_q1` negative for every model) while
#: composition is (`pcoa1_pa` up to +0.41). Averaging the two would pick the winning index and
#: substrate on the strength of the null half, i.e. on noise.
RANK_TARGETS = ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"]


def _summary(args) -> pd.DataFrame:
    f = Path(args.out) / "summary.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def best_index(args) -> str:
    """The vegetation index with the best mean RF score. Falls back to ndvi before Tier 1."""
    s = _summary(args)
    if s.empty:
        return "ndvi"
    s = s[(s["family"] == "RF") & (s["scheme"] == PRIMARY)
          & (s["target"].isin(RANK_TARGETS)) & s["index"].notna()]
    s = s[s["index"].astype(str).isin(feat.INDICES)]
    if s.empty:
        return "ndvi"
    return str(s.groupby("index")["R2"].mean().sort_values(ascending=False).index[0])


def best_substrates(args, k: int = 3) -> list[str]:
    s = _summary(args)
    if s.empty:
        return ["reshape", "stack5", "pxcube"][:k]
    s = s[(s["family"] == "C2D") & (s["scheme"] == PRIMARY)
          & (s["target"].isin(RANK_TARGETS))]
    if s.empty:
        return ["reshape", "stack5", "pxcube"][:k]
    rank = s.groupby("substrate")["R2"].mean().sort_values(ascending=False)
    return list(rank.index[:k])


def finalists(args) -> list[dict]:
    """Best run of each family, plus the two non-manufactured substrates regardless of rank."""
    s = _summary(args)
    if s.empty:
        return []
    s = s[(s["scheme"] == PRIMARY) & (s["target"].isin(RANK_TARGETS))]
    out, seen = [], set()
    for fam in ("RF", "MLP", "C1D", "C2D"):
        g = s[s["family"] == fam]
        if g.empty:
            continue
        rid = g.groupby("run_id")["R2"].mean().sort_values(ascending=False).index[0]
        row = g[g["run_id"] == rid].iloc[0]
        out.append(row.to_dict())
        seen.add(rid)
    for sname in ("stack5", "pxcube"):
        g = s[s["substrate"] == sname]
        if g.empty:
            continue
        rid = g.groupby("run_id")["R2"].mean().sort_values(ascending=False).index[0]
        if rid not in seen:
            out.append(g[g["run_id"] == rid].iloc[0].to_dict())
            seen.add(rid)
    return out


def _rebuild_cmd(run: dict, scheme: str, seeds: int, args) -> list[str] | None:
    fam, ix = run["family"], run.get("index")
    ix = "" if not isinstance(ix, str) or ix in ("", "nan") else ix
    if fam == "RF":
        model = str(run["run_id"]).split("_")[0]
        cmd = [PY, "scripts/09_run_baselines.py", "--model", model, "--scheme", scheme,
               "--seeds", str(min(seeds, 3))]
        return cmd + (["--index", ix] if ix else [])
    if fam == "MLP":
        model = str(run["run_id"]).split("_")[0]
        cmd = [PY, "scripts/10_run_tabular_dl.py", "--model", model, "--scheme", scheme,
               "--seeds", str(seeds)] + _dl_common(args)
        return cmd + (["--index", ix] if ix else [])
    sub_name = run.get("substrate")
    if not isinstance(sub_name, str) or not sub_name:
        return None
    cmd = [PY, "scripts/11_run_conv.py", "--substrate", sub_name, "--scheme", scheme,
           "--seeds", str(seeds)] + _dl_common(args)
    return cmd + (["--index", ix] if ix else [])


# --------------------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------------------

def run_jobs(jobs: list[Job], args, deadline: datetime | None, stage: str) -> list[dict]:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    gpu_jobs = [j for j in jobs if j.gpu]
    cpu_jobs = [j for j in jobs if not j.gpu]
    results: list[dict] = []
    counter = {"i": 0}

    def launch(job: Job, device: str | None) -> dict:
        if deadline and datetime.now() > deadline:
            return dict(name=job.name, status="skipped-deadline", seconds=0)
        env = dict(os.environ)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONPATH"] = str(ROOT / "src") + ":" + env.get("PYTHONPATH", "")
        log = LOGDIR / f"{job.name.replace('/', '_')}.log"
        t0 = time.time()
        with log.open("w") as fh:
            fh.write(" ".join(job.cmd) + "\n\n")
            fh.flush()
            rc = subprocess.run(job.cmd, cwd=ROOT, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        counter["i"] += 1
        dt = time.time() - t0
        state = "ok" if rc == 0 else f"FAILED(rc={rc})"
        print(f"  [{counter['i']:>3}/{len(jobs)}] {job.name:38s} {state:14s} "
              f"{dt / 60:5.1f}m  -> {log.name}", flush=True)
        return dict(name=job.name, status=state, seconds=round(dt, 1), log=str(log))

    n_workers = args.gpu_workers + (args.cpu_workers if cpu_jobs else 0)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = []
        for i, j in enumerate(gpu_jobs):
            dev = str(args.devices[i % len(args.devices)])
            futs.append(ex.submit(launch, j, dev))
        for j in cpu_jobs:
            futs.append(ex.submit(launch, j, ""))
        for f in futs:
            results.append(f.result())

    write_progress(stage, results)
    return results


def write_progress(stage: str, results: list[dict]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    head = "" if PROGRESS.exists() else (
        "# Registro de ejecución del modelamiento\n\n"
        "Generado por `scripts/14_run_matrix.py`. Una sección por etapa, en el orden en que "
        "se corrieron. El estado `skipped-deadline` significa que el trabajo no alcanzó a "
        "lanzarse dentro del presupuesto de tiempo y queda pendiente para la próxima "
        "invocación (el script es reanudable).\n")
    ok = sum(r["status"] == "ok" for r in results)
    lines = [head,
             f"\n## Etapa `{stage}` — {datetime.now():%Y-%m-%d %H:%M}\n",
             f"{ok}/{len(results)} trabajos completados, "
             f"{sum(r['seconds'] for r in results) / 3600:.1f} h de cómputo acumulado.\n",
             "\n| trabajo | estado | minutos |",
             "|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["seconds"]):
        lines.append(f"| `{r['name']}` | {r['status']} | {r['seconds'] / 60:.1f} |")
    with PROGRESS.open("a") as fh:
        fh.write("\n".join(lines) + "\n")


STAGES = {
    "rf": stage_rf,
    "c1d": stage_c1d,
    "mlp": stage_mlp,
    "4a": stage_4a,
    "4b": stage_4b,
    "4c": stage_4c,
    "final": stage_final,
}
def stage_rf_c1d(args) -> list[Job]:
    """The forests (CPU) and the 1-D control (GPU) have no dependency on one another.

    Run as separate stages, both GPUs would sit idle for the ~2 h the forests take. Merged,
    the pool fills every slot from the start.
    """
    return stage_rf(args) + stage_c1d(args)


STAGES["rf+c1d"] = stage_rf_c1d

ORDER = ["rf+c1d", "4a", "mlp", "4b", "4c", "final"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=list(STAGES), default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--devices", nargs="+", default=["0", "1"])
    p.add_argument("--gpu-workers", type=int, default=2)
    p.add_argument("--cpu-workers", type=int, default=2,
                   help="concurrent Random Forest jobs; each is already n_jobs=-1, but a good "
                        "share of every fit is single-threaded (target transform, smearing)")
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--deadline-hours", type=float, default=None)
    p.add_argument("--out", default="results/models")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true", default=True,
                   help="run scripts 12 and 13 at the end")
    args = p.parse_args()

    deadline = (datetime.now() + timedelta(hours=args.deadline_hours)
                if args.deadline_hours else None)
    stages = ORDER if args.all else ([args.stage] if args.stage else [])
    if not stages:
        p.error("give --stage or --all")

    print(f"start {datetime.now():%Y-%m-%d %H:%M}"
          + (f"  deadline {deadline:%H:%M}" if deadline else ""))
    print(f"devices={args.devices}  epochs<={args.max_epochs}  patience={args.patience}\n")

    for stage in stages:
        jobs = STAGES[stage](args)
        print(f"[{stage}] {len(jobs)} jobs")
        if args.dry_run:
            for j in jobs:
                print("   ", " ".join(j.cmd[1:]))
            continue
        if deadline and datetime.now() > deadline:
            print(f"  deadline reached, stopping before stage {stage}")
            break
        run_jobs(jobs, args, deadline, stage)

    if args.dry_run:
        return

    if args.report:
        print("\n[report]")
        for cmd in ([PY, "scripts/12_model_report.py"],
                    [PY, "scripts/13_interpretability.py", "--auto"]):
            log = LOGDIR / (Path(cmd[1]).stem + ".log")
            with log.open("w") as fh:
                subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            print(f"  {' '.join(cmd[1:]):40s} -> {log}")

    print(f"\ndone {datetime.now():%Y-%m-%d %H:%M}")
    print(f"progress log: {PROGRESS}")


if __name__ == "__main__":
    main()
