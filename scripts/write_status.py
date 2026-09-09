#!/usr/bin/env python3
"""Write results/maps/STATUS.json: how far the map run has got, for a reader outside the pod.

The run takes days and nothing inside the pod can outlive the pod, so the machine that watches
it (pop-os, which also wakes the server hourly) has to see progress from outside. It reads
this file through the Jupyter server's *contents* API -- a read-only GET with the hub token,
no kernel and no terminal:

    GET https://hub.datacubechile.cl/user/jlopatin/api/contents/
        temp/Biodiversity_Chile/results/maps/STATUS.json?content=1
        Authorization: token <hub token>

`updated_utc` is as much of the answer as the counts are: a timestamp that stops advancing
means the pod went down, which looks nothing like a run that finished.
"""
from __future__ import annotations

import glob
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HALVES = {"chile_gw": ("results/figures/tiles_run_gateway.csv", ""),
          "chile_pod": ("results/figures/tiles_run_pod.csv", "_parts/manifest_*.csv")}
YEARS = 27


def half_progress(tag: str, tiles_csv: str, parts: str) -> dict:
    tiles = set(pd.read_csv(ROOT / tiles_csv).tile_id)
    files = [str(ROOT / f"results/maps/{tag}/manifest.csv")]
    if parts:
        files += glob.glob(str(ROOT / f"results/maps/{tag}/{parts}"))
    frames = [pd.read_csv(f, low_memory=False) for f in files if Path(f).exists()]
    done = set()
    if frames:
        m = pd.concat(frames, ignore_index=True)
        ok = m[m.status == "ok"]
        done = {t for t, n in ok.groupby("tile_id").year.nunique().items() if n == YEARS}
    alive = subprocess.run(["pgrep", "-f", f"run_maps_supervised.sh "
                            f"{'gateway' if tag == 'chile_gw' else 'pod'}"],
                           capture_output=True).returncode == 0
    return dict(assigned=len(tiles), done=len(done), remaining=len(tiles - done),
                supervisor_alive=alive)


def main() -> None:
    halves = {tag: half_progress(tag, *cfg) for tag, cfg in HALVES.items()}
    total = sum(h["assigned"] for h in halves.values())
    done = sum(h["done"] for h in halves.values())
    status = dict(
        updated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        tiles_total=total, tiles_done=done,
        pct=round(100 * done / total, 2) if total else 0.0,
        tile_years_done=done * YEARS, tile_years_total=total * YEARS,
        complete=bool(total and done == total),
        halves=halves,
        note=("complete=true means every assigned tile has all 27 years written. "
              "If updated_utc stops advancing the pod is down, which is not the same "
              "as finished -- check complete, not the timestamp alone."),
    )
    out = ROOT / "results" / "maps" / "STATUS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")           # atomic: a reader never sees half a file
    tmp.write_text(json.dumps(status, indent=1))
    tmp.replace(out)
    print(json.dumps({k: status[k] for k in ("updated_utc", "tiles_done", "tiles_total",
                                             "pct", "complete")}))


if __name__ == "__main__":
    main()
