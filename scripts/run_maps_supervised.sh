#!/bin/bash
# One half of the map run, relaunched until it finishes.   Usage: run_maps_supervised.sh gateway|pod
#
# Both halves have now been lost more than once, for three different reasons: the gateway's
# cluster vanished mid-run (twice -- once with the hub rejecting the JupyterHub API token),
# and the whole container was restarted overnight, which killed the pod's 14 processes and
# every driver with them. `setsid` survives a Claude session ending; nothing inside the
# container survives the container. So the loop lives here, and after a pod restart this
# script is the one thing to run again.
#
# Every attempt passes --resume, which counts only rows with status "ok": nothing already
# written is redone, and nothing that failed is skipped.
set -uo pipefail
cd /home/jovyan/temp/Biodiversity_Chile

HALF="${1:?usage: run_maps_supervised.sh gateway|pod}"
case "$HALF" in
  gateway) TILES=results/figures/tiles_run_gateway.csv; TAG=chile_gw;  PARTS="" ;;
  pod)     TILES=results/figures/tiles_run_pod.csv;     TAG=chile_pod; PARTS="_parts/manifest_*.csv" ;;
  *) echo "half must be 'gateway' or 'pod'"; exit 2 ;;
esac

# A fresh token from https://hub.datacubechile.cl/hub/token, used only if the pod's own is
# rejected. Read from the file, never passed on a command line where `ps` would show it.
[ -r "$HOME/.gw_token" ] && export JUPYTERHUB_API_TOKEN="$(cat "$HOME/.gw_token")"

UID_S3=$(python -c "import boto3;print(boto3.client('sts').get_caller_identity()['UserId'])")
B=s3://easido-prod-user-scratch/${UID_S3}/biodiv
CK=results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr_FINAL_alldata/final
OOF=results/models_unified_topofix/C2D02_serpentine_kndvi_raw100_pg-all_unified_maekndvi_m06_ctr/kfold5_block20_unified/oof_predictions.csv

for attempt in $(seq 1 200); do
    left=$(TILES="$TILES" TAG="$TAG" PARTS="$PARTS" python - <<'PY'
import glob, os
import pandas as pd
tiles = set(pd.read_csv(os.environ["TILES"]).tile_id)
tag, parts = os.environ["TAG"], os.environ["PARTS"]
files = glob.glob(f"results/maps/{tag}/manifest.csv") + (
    glob.glob(f"results/maps/{tag}/{parts}") if parts else [])
done = set()
if files:
    m = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    ok = m[m.status == "ok"]
    done = {t for t, n in ok.groupby("tile_id").year.nunique().items() if n == 27}
print(len(tiles - done))
PY
)
    echo "=== $HALF attempt $attempt: $left tiles left ($(date -u +%F' '%H:%M:%S)) ==="
    [ "$left" -eq 0 ] && { echo "$HALF half complete"; break; }

    if [ "$HALF" = gateway ]; then
        BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python -u scripts/73_map_inference.py \
            --tiles-file "$TILES" --years 2000-2026 \
            --area-m2 900 --stratum basal --mask mapbiomas \
            --ckpt-dir "$CK" --oof-csv "$OOF" \
            --gw-workers 5 --worker-cores 8 --worker-memory 28 --worker-threads 4 \
            --gw-batch 250 --load-threads 2 --torch-threads 2 --workers 0 \
            --dest "$B/maps/chile_30m_2000_2026" --mapbiomas-dir "$B/MapBiomas" \
            --out results/maps --tag "$TAG" --resume
    else
        BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python -u scripts/73_map_inference.py \
            --tiles-file "$TILES" --years 2000-2026 \
            --area-m2 900 --stratum basal --mask mapbiomas \
            --ckpt-dir "$CK" --oof-csv "$OOF" \
            --jobs 14 --workers 0 --load-threads 4 --torch-threads 1 \
            --dest "$B/maps/chile_30m_2000_2026" --mapbiomas-dir "$B/MapBiomas" \
            --out results/maps --tag "$TAG" --resume
    fi
    echo "=== $HALF attempt $attempt exited $?; pausing 120 s ==="
    sleep 120
done
