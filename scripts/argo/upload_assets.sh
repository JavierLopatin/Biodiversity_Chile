#!/usr/bin/env bash
# Stage the private inference assets into the S3 prefix the Argo pods read.
#
# The pods git-sync the PUBLIC repo (code only) and pull everything else from
# S3, so this is what makes biodiv-map-inference runnable. It is idempotent:
# re-running syncs only what changed.
#
# Requires AWS credentials with write access to the team prefix. The Argo
# pods get theirs from IRSA, which is a cluster identity and cannot be copied
# to a laptop -- so this script needs a real profile (aws login / ~/.aws).
#
#   ./argo/upload_assets.sh --dry-run          # show what would be sent
#   ./argo/upload_assets.sh --skip-mapbiomas   # the 4.4 MB that unblock the run
#   ./argo/upload_assets.sh                    # everything, incl. 3.6 GB
#
# --skip-mapbiomas is the useful first move: the small assets are what
# check-access and list-missing need, while the rasters may already be in S3
# from the pod's own runs (argo/probe-s3.yaml answers that) and would then be
# referenced in place with -p s3-mapbiomas=<prefix> instead of re-uploaded.
#
# Override the sources with env vars if the paths differ. MAPB defaults under DISK
# (Javier's Mac layout: checkpoints and MapBiomas share the external drive); a machine
# where they live apart -- e.g. the GPU workstation, checkpoints under the repo's own
# results/, MapBiomas on a separate data mount -- overrides MAPB on its own:
#   DISK=/Volumes/... REPO=~/GitHub/Biodiversity_Chile ./argo/upload_assets.sh
#   DISK=results/models_unified_topofix REPO=. MAPB=/mnt/data/GIS/MapBiomas ./argo/upload_assets.sh
set -euo pipefail

DISK="${DISK:-/Volumes/Lopatin 1TB}"
REPO="${REPO:-$HOME/Documents/GitHub/Biodiversity_Chile}"
MAPB="${MAPB:-$DISK/MapBiomas}"
A="${A:-s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv/assets}"
M="${M:-s3://easido-prod-dc-data-projects/easi-workflows-team/biodiv/MapBiomas}"

# The deployed model is now the 1D-CNN (C1D01_curve1d_kndvi), not the 2D-CNN
# (C2D02_serpentine_kndvi) -- essentially tied on skill (td_inext_q0 0.781 vs
# 0.783, pd_inext_q0 0.602 vs 0.616) but simpler: no MAE-pretrained trunk to
# carry into deployment, no serpentine reshape, at essentially the same
# parameter count (14,535 vs 15,175 -- matched by design). See scripts/77 for the
# all-data refit on corrected topography (models_unified_topofix), 149,139
# bytes per checkpoint, dated 8 Sep. The oof MUST come from the same parent:
# smearing back-transforms with the checkpoint's own target scaler, so an oof
# from a different run (or the old C2D02 one) is a silently wrong correction,
# not an error. `biodiv.mapinfer.FacetEnsemble` detects the architecture from
# the checkpoint's own `input_shape`, so this is the only file that needs to
# know which model is deployed.
CKPT="$DISK/C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr_FINAL_alldata/final"
OOF="$DISK/C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr/kfold5_block20_unified/oof_predictions.csv"
TILES="$REPO/results/figures/tiles_native_10km.csv"
DERIVED="$REPO/data/derived"

DRY=""
SKIP_MB=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)        DRY="--dryrun"; echo "DRY RUN -- nothing will be written" ;;
    --skip-mapbiomas) SKIP_MB=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "== identity =="
aws sts get-caller-identity || {
  echo "no AWS credentials. Run 'aws login' (or set a profile) first." >&2
  exit 1
}

echo; echo "== checking sources =="
fail=0
for f in "$OOF" "$TILES"; do
  [ -f "$f" ] || { echo "MISSING: $f" >&2; fail=1; }
done
check_dirs=("$CKPT" "$DERIVED")
# Only require MapBiomas when it is actually going to be uploaded. Without this the
# `--skip-mapbiomas` path -- the documented first move, the 4.4 MB that unblock a run --
# aborts on any machine that does not keep the 3.1 GB of rasters locally, which is every
# machine except the one holding the external drive.
[ "$SKIP_MB" = "1" ] || check_dirs+=("$MAPB")
for d in "${check_dirs[@]}"; do
  [ -d "$d" ] || { echo "MISSING: $d" >&2; fail=1; }
done
n_ckpt=$(ls "$CKPT"/model_seed*.pt 2>/dev/null | wc -l | tr -d ' ')
[ "$n_ckpt" = "5" ] || { echo "expected 5 model_seed*.pt in $CKPT, found $n_ckpt" >&2; fail=1; }
# _scan() only sees flat *.tif whose name starts with <4 digits>_ , so count
# exactly what the pipeline will actually be able to use.
if [ "$SKIP_MB" = "1" ]; then
  n_tif="skipped"
else
  n_tif=$(find "$MAPB" -maxdepth 1 -name '[0-9][0-9][0-9][0-9]_*.tif' | wc -l | tr -d ' ')
  [ "$n_tif" -gt 0 ] || { echo "no <year>_*.tif directly under $MAPB" >&2; fail=1; }
fi
[ "$fail" = "0" ] || exit 1
echo "5 checkpoints, $n_tif MapBiomas rasters, oof + tiles csv + derived present"

# Everything below is `cp`, never `sync`. sync lists the destination to work
# out what changed, and the hub identity carries an explicit deny on
# s3:ListBucket for this bucket (policy user-easido-prod-eks-easihub-list) --
# so sync dies on ListObjectsV2 before it ever tries to write, while a plain
# cp only needs PutObject. The Argo pods do not hit this: they run as
# easi-workflows-team-sa-argo through IRSA, a different identity.
put() {  # put <local file> <s3 key>
  if [ -n "$DRY" ]; then
    echo "  would put: $(basename "$1") -> $2"
  else
    aws s3 cp --only-show-errors "$1" "$2" && echo "  put: $2"
  fi
}

echo; echo "== 1/4 checkpoints -> $A/models/final =="
for f in "$CKPT"/model_seed*.pt; do
  put "$f" "$A/models/final/$(basename "$f")"
done

echo; echo "== 2/4 oof_predictions.csv -> $A/models/ =="
put "$OOF" "$A/models/oof_predictions.csv"

echo; echo "== 3/4 tile list + derived parquet -> $A =="
put "$TILES" "$A/tiles/$(basename "$TILES")"
for f in "$DERIVED"/plots_unified.parquet "$DERIVED"/*_unified_padded.parquet; do
  put "$f" "$A/derived/$(basename "$f")"
done

if [ "$SKIP_MB" = "1" ]; then
  echo; echo "== 4/4 MapBiomas SKIPPED (--skip-mapbiomas) =="
  echo "   process-chunk needs the rasters: either run this again without the"
  echo "   flag, or pass -p s3-mapbiomas=<prefix where they already live>."
else
  echo; echo "== 4/4 MapBiomas rasters -> $M  (3.6 GB, slowest step) =="
  # Flat keys: _scan() drops anything with a "/" left in the key after the
  # prefix, so each raster goes at the top level of $M, not in a subfolder.
  for f in "$MAPB"/[0-9][0-9][0-9][0-9]_*.tif; do
    put "$f" "$M/$(basename "$f")"
  done
fi

[ -n "$DRY" ] && { echo; echo "dry run done"; exit 0; }

echo; echo "== verifying =="
# head-object asks about ONE key and needs s3:GetObject, not s3:ListBucket --
# so it survives the same deny that rules out `aws s3 ls` here. If GetObject
# is denied too, this identity simply cannot read back what it wrote: submit
# argo/probe-s3.yaml instead, which checks from inside the cluster.
check() {  # check <s3 uri>
  key="${1#s3://}"; bucket="${key%%/*}"; key="${key#*/}"
  if aws s3api head-object --bucket "$bucket" --key "$key" \
       --query 'ContentLength' --output text 2>/dev/null; then
    echo "    ok: $1"
  else
    echo "    UNVERIFIED: $1"
  fi
}
for i in 0 1 2 3 4; do check "$A/models/final/model_seed$i.pt"; done
check "$A/models/oof_predictions.csv"
check "$A/tiles/$(basename "$TILES")"
check "$A/derived/plots_unified.parquet"
echo; echo "done -- if every line says ok, check-access should now pass"
