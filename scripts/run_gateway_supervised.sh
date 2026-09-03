#!/bin/bash
# The gateway half, relaunched until it finishes.
#
# It has now lost its cluster twice mid-run -- once around a second cluster being raised, once
# with the hub rejecting the JupyterHub API token -- and each loss ends the driver. Over a
# four-day run that is not something to babysit, so this restarts it. Every attempt passes
# --resume, which counts only rows with status "ok", so nothing already written is redone and
# nothing that failed is skipped.
#
# The token: dask_gateway authenticates with JUPYTERHUB_API_TOKEN. When the pod's own token is
# rejected, a fresh one from https://hub.datacubechile.cl/hub/token in ~/.gw_token is used
# instead -- read from the file, never passed on a command line where `ps` would show it.
set -uo pipefail
cd /home/jovyan/temp/Biodiversity_Chile

[ -r "$HOME/.gw_token" ] && export JUPYTERHUB_API_TOKEN="$(cat "$HOME/.gw_token")" \
    && echo "using the token from ~/.gw_token"

UID_S3=$(python -c "import boto3;print(boto3.client('sts').get_caller_identity()['UserId'])")
B=s3://easido-prod-user-scratch/${UID_S3}/biodiv
CK=results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr_FINAL_alldata/final
OOF=results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr/kfold5_block20_unified/oof_predictions.csv

MAX=200
for attempt in $(seq 1 $MAX); do
    left=$(python - <<'PY'
import pandas as pd
from pathlib import Path
tiles = set(pd.read_csv("results/figures/tiles_run_gateway.csv").tile_id)
man = Path("results/maps/chile_gw/manifest.csv")
done = set()
if man.exists():
    m = pd.read_csv(man, low_memory=False)
    ok = m[m.status == "ok"]
    done = {t for t, n in ok.groupby("tile_id").year.nunique().items() if n == 27}
print(len(tiles - done))
PY
)
    echo "=== attempt $attempt/$MAX: $left tiles left ($(date -u +%H:%M:%S)) ==="
    [ "$left" -eq 0 ] && { echo "gateway half complete"; break; }

    BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python -u scripts/73_map_inference.py \
        --tiles-file results/figures/tiles_run_gateway.csv --years 2000-2026 \
        --area-m2 900 --stratum basal --mask mapbiomas \
        --ckpt-dir "$CK" --oof-csv "$OOF" \
        --gw-workers 5 --worker-cores 8 --worker-memory 28 --worker-threads 4 \
        --gw-batch 250 --load-threads 2 --torch-threads 2 --workers 0 \
        --dest "$B/maps/chile_30m_2000_2026" --mapbiomas-dir "$B/MapBiomas" \
        --out results/maps --tag chile_gw --resume
    rc=$?
    echo "=== attempt $attempt exited $rc; pausing 120 s ==="
    sleep 120
done
