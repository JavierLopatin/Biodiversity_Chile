#!/bin/bash
# Start (or resume) both halves of the map run.   Usage: scripts/start_maps.sh
#
# This is the one command to run after the container restarts, which it does on its own: the
# hub stops a server it considers idle, and it measures idleness by kernels and HTTP requests,
# not by CPU -- so fifteen processes saturating the pod look like nothing at all. Everything
# already written is in the manifests and --resume counts only tiles with all 27 years, so
# running this again costs nothing but the tiles that were in flight when the pod went down.
#
# Idempotent on purpose. Starting a half twice would run the same tiles in two places, burning
# compute and racing on the same manifest, so a half that is already alive is left alone.
set -uo pipefail
cd "$(dirname "$0")/.."

start_half() {
    local half="$1" log="$2" pattern="$3"
    if pgrep -f "run_maps_supervised.sh $half" > /dev/null; then
        echo "  $half: already supervised, left alone"
    elif pgrep -f "$pattern" > /dev/null; then
        echo "  $half: a run is alive but unsupervised; leaving it (kill it first to supervise)"
    else
        setsid nohup scripts/run_maps_supervised.sh "$half" > "$log" 2>&1 < /dev/null &
        echo "  $half: started, pid $!, log $log"
    fi
}

echo "starting the map run ($(date -u +%F' '%H:%M:%S) UTC)"
start_half gateway logs/chile_gw.log  "tiles_run_gateway"
start_half pod     logs/chile_pod.log "tiles_run_pod.csv"

sleep 2
python - <<'PY'
import glob
import pandas as pd
files = glob.glob("results/maps/chile_gw/manifest.csv") + \
        glob.glob("results/maps/chile_pod/_parts/manifest_*.csv")
if files:
    m = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    ok = m[m.status == "ok"]
    per = ok.groupby("tile_id").year.nunique()
    done = int((per == 27).sum())
    print(f"\n{done:,} of 5,769 tiles complete ({100*done/5769:.1f} %)")
PY
