"""Run identity and artefact layout.

Every fit writes the same five things to the same place, so the reporting layer
(`scripts/12_model_report.py`) is a glob over CSVs and never needs to know what produced
them. Concretely::

    results/models/
      summary.csv                       one row per (run_id, scheme, target, seed)
      <run_id>/<scheme>/
          config.json                   full config + git hash + package versions
          oof_predictions.csv           PlotObservationID, fold, seed, <t>_obs, <t>_pred
          per_seed_metrics.csv          pooled score per seed — the paired unit for tests
          pooled_metrics.csv            seed-mean +/- sd, plus the seed-ensemble score
          importance.csv | model_seed*.pt

`summary.csv` is appended under a lock-free read-modify-write, which is safe because runs
are launched sequentially by the driver scripts. Re-running a run_id replaces its rows
rather than duplicating them, so a partially completed sweep can simply be re-launched.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results/models")
SUMMARY = RESULTS / "summary.csv"


def _git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:                                     # noqa: BLE001 - provenance is best effort
        return "unknown"


def _versions() -> dict[str, str]:
    import sklearn
    import scipy
    mods = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__}
    try:
        import torch
        mods["torch"] = torch.__version__
        mods["cuda"] = str(torch.cuda.is_available())
    except ImportError:
        pass
    return mods


@dataclass
class RunConfig:
    """Everything needed to reproduce one run, and nothing that varies within it."""
    run_id: str
    family: str                       # RF | MLP | C1D | C2D | BASE
    scheme: str
    features: str = ""                # feature-block spec, e.g. "lsp+topo+area"
    substrate: str = ""               # curve1d | reshape | stack5 | pxcube | ...
    index: str = ""                   # vegetation index, or "stack"/"" when not applicable
    target_set: str = "all"
    seeds: tuple[int, ...] = (0, 1, 2)
    fusion: str = "late"              # none | late | film | patch
    model: str = ""
    params: dict = field(default_factory=dict)
    notes: str = ""

    def outdir(self, root: Path = RESULTS) -> Path:
        return Path(root) / self.run_id / self.scheme

    def to_json(self) -> dict:
        return asdict(self) | {"git": _git_hash(), "versions": _versions()}


def make_run_id(family: str, tag: str, index: str = "", substrate: str = "",
                fusion: str = "") -> str:
    parts = [family, tag]
    if index:
        parts.append(index)
    if substrate:
        parts.append(substrate)
    if fusion and fusion != "late":
        parts.append(fusion)
    # The curve version has to be part of the identity. Without it a run on the refitted
    # curves lands in the same directory as the original, `already_done` reports it as
    # finished and the job is skipped in silence -- the trap that made five jobs of the 4c
    # and clim stages report `ok` in 0.0 minutes without computing anything.
    sfx = os.environ.get("BIODIV_CURVES", "")
    if sfx:
        parts.append(sfx.lstrip("_"))
    return "_".join(p.replace("+", "-").replace("/", "-") for p in parts if p)


def write_run(cfg: RunConfig, oof: pd.DataFrame, per_seed: pd.DataFrame,
              pooled: pd.DataFrame, extra: dict[str, pd.DataFrame] | None = None,
              root: Path = RESULTS) -> Path:
    out = cfg.outdir(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_json(), indent=2, default=str))
    oof.to_csv(out / "oof_predictions.csv", index=False)
    per_seed.to_csv(out / "per_seed_metrics.csv", index=False)
    pooled.to_csv(out / "pooled_metrics.csv", index=False)
    for name, df in (extra or {}).items():
        df.to_csv(out / f"{name}.csv", index=False)
    append_summary(cfg, per_seed, root=root)
    return out


def _summary_rows(cfg: RunConfig, per_seed: pd.DataFrame) -> pd.DataFrame:
    return per_seed.assign(run_id=cfg.run_id, scheme=cfg.scheme, family=cfg.family,
                           features=cfg.features, substrate=cfg.substrate,
                           index=cfg.index, fusion=cfg.fusion, model=cfg.model)


def append_summary(cfg: RunConfig, per_seed: pd.DataFrame, root: Path = RESULTS) -> None:
    """Upsert this run's per-seed scores into the global table, safely under concurrency.

    The driver runs four jobs at once, so a plain read-modify-write races: one process can
    read the file in the instant another has truncated it, which is exactly what happened
    on the first overnight run — one job died on ``EmptyDataError`` and its rows never made
    it into the table even though its artefacts were on disk.

    Two guards. An exclusive ``flock`` serialises the read-modify-write, and the write goes
    to a temporary file that is then renamed, so the visible file is never a partial one.
    :func:`rebuild_summary` remains the backstop: the per-run CSVs are the source of truth
    and this table is a cache of them.
    """
    import fcntl
    import os

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = _summary_rows(cfg, per_seed)
    path = root / "summary.csv"
    lock = root / ".summary.lock"

    with lock.open("w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if path.exists() and path.stat().st_size > 0:
                old = pd.read_csv(path)
                old = old[~((old["run_id"] == cfg.run_id) & (old["scheme"] == cfg.scheme))]
                rows = pd.concat([old, rows], ignore_index=True)
            tmp = path.with_suffix(".csv.tmp")
            rows.to_csv(tmp, index=False)
            os.replace(tmp, path)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def rebuild_summary(root: Path = RESULTS) -> pd.DataFrame:
    """Regenerate `summary.csv` from the per-run artefacts, which are the source of truth.

    Use after an interrupted or concurrent sweep: a run whose `per_seed_metrics.csv` exists
    but whose rows are absent from the table is recovered rather than recomputed.
    """
    root = Path(root)
    frames = []
    for f in sorted(root.glob("*/*/per_seed_metrics.csv")):
        run_id, scheme = f.parts[-3], f.parts[-2]
        cfg_f = f.parent / "config.json"
        meta = json.loads(cfg_f.read_text()) if cfg_f.exists() else {}
        frames.append(pd.read_csv(f).assign(
            run_id=run_id, scheme=scheme, family=meta.get("family", ""),
            features=meta.get("features", ""), substrate=meta.get("substrate", ""),
            index=meta.get("index", ""), fusion=meta.get("fusion", ""),
            model=meta.get("model", "")))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(root / "summary.csv", index=False)
    return out


def already_done(cfg: RunConfig, root: Path = RESULTS) -> bool:
    """True if this run's artefacts exist — lets an interrupted sweep resume cheaply."""
    return (cfg.outdir(root) / "pooled_metrics.csv").exists()


def summarise_pooled(per_seed: pd.DataFrame, ensemble: pd.DataFrame) -> pd.DataFrame:
    """Seed mean +/- sd next to the seed-ensemble score, one row per target.

    Both are reported because they answer different questions: the mean +/- sd says how
    stable a single fit is (which for a 15k-parameter model on n=1082 is the honest number),
    the ensemble says what the deployed model would achieve.
    """
    agg = (per_seed.groupby("target", sort=False)
           .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                MAE_mean=("MAE", "mean"), spearman_mean=("spearman", "mean"),
                n=("n", "first"), n_seeds=("R2", "size"))
           .reset_index())
    ens = ensemble.rename(columns={"R2": "R2_ensemble", "RMSE": "RMSE_ensemble",
                                   "spearman": "spearman_ensemble"})
    return agg.merge(ens[["target", "R2_ensemble", "RMSE_ensemble", "spearman_ensemble"]],
                     on="target", how="left")
