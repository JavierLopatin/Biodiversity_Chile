#!/bin/bash
# Unattended: wait for the cube download, extract it, re-fit the curves, and run the one
# experiment that settles whether the year-boundary artefact mattered.
#
# Written to run with nobody watching, so every step is guarded and nothing destructive
# happens on its own:
#
#   - the originals are never overwritten. The refit writes `phenoshape_*_v2.parquet` and
#     the models run with BIODIV_CURVES=_v2, which `runlog.make_run_id` appends to every run
#     id. Old and new results coexist and stay comparable.
#   - the download is only considered finished when the file size stops changing AND the
#     cube count is plausible. A truncated zip that happens to exist is the obvious way to
#     ruin an overnight run.
#   - if a stage fails the script stops rather than feeding garbage to the next one.
#
#   nohup bash scripts/run_refit_when_ready.sh >> logs/refit_chain.log 2>&1 &
#   tail -f logs/refit_chain.log
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CUBES=data/derived/phenology
EXPECTED=1082          # plots in plots_subset.parquet
POLL=120               # seconds between checks
MAX_WAIT=21600         # 6 h, then give up rather than spin forever
SEARCH_DIRS=("$HOME/Descargas" "$HOME/Downloads" "$CUBES" "data/derived" ".")
# `find` on this machine is bfs, which rejects relative timestamps like -newermt '-1 day'
# and errors out instead of matching nothing. Unattended, that silently disables the whole
# search. An absolute ISO stamp works on both bfs and GNU find.
SINCE=$(date -d '1 day ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)

say() { echo "[$(date +%F\ %H:%M:%S)] $*"; }

n_cubes() { find "$CUBES" -maxdepth 1 -name '*.nc' 2>/dev/null | wc -l; }

find_zip() {
  # newest .zip that looks like the cube archive and is not a partial download
  for d in "${SEARCH_DIRS[@]}"; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 1 -name '*.zip' -newermt "$SINCE" 2>/dev/null
  done | sort -u | while read -r f; do
    case "$f" in *.part|*.crdownload|*.tmp) continue;; esac
    # must contain .nc entries to be ours
    if unzip -l "$f" 2>/dev/null | grep -q '\.nc$'; then echo "$f"; fi
  done | head -1
}

stable_size() {   # $1 = path; true when the size is unchanged across two polls
  local a b
  a=$(stat -c%s "$1" 2>/dev/null || echo 0)
  sleep "$POLL"
  b=$(stat -c%s "$1" 2>/dev/null || echo 0)
  [ "$a" = "$b" ] && [ "$a" -gt 0 ]
}

# ----------------------------------------------------------------------------- wait
say "=== waiting for cubes. have $(n_cubes) of $EXPECTED"
waited=0
while [ "$waited" -lt "$MAX_WAIT" ]; do
  if [ "$(n_cubes)" -ge "$EXPECTED" ]; then
    say "all $EXPECTED cubes present, no archive needed"; break
  fi
  zip=$(find_zip)
  if [ -n "${zip:-}" ]; then
    say "found archive: $zip ($(du -h "$zip" | cut -f1)) -- checking it is complete"
    if stable_size "$zip" && unzip -t "$zip" >/dev/null 2>&1; then
      say "archive complete and valid, extracting into $CUBES"
      mkdir -p "$CUBES"
      # -j flattens any directory prefix, -n never clobbers an existing cube
      unzip -j -n -q "$zip" '*.nc' -d "$CUBES" || { say "ABORT: unzip failed"; exit 1; }
      say "extracted. now $(n_cubes) cubes"
      [ "$(n_cubes)" -ge "$EXPECTED" ] && break
      say "still short of $EXPECTED -- the archive may be partial; continuing to wait"
    else
      say "archive still growing or not yet valid; waiting"
    fi
  else
    sleep "$POLL"
  fi
  waited=$((waited + POLL))
done

have=$(n_cubes)
if [ "$have" -lt 100 ]; then
  say "ABORT: only $have cubes after $((waited/60)) min. Nothing worth running."
  exit 1
fi
[ "$have" -lt "$EXPECTED" ] && say "WARNING: $have of $EXPECTED cubes. Proceeding on what is here."

# ----------------------------------------------------------------------------- refit
say "=== re-fitting curves (harmonic, k=3)"
python scripts/29_refit_curves_from_cubes.py --workers 10 --suffix _v2 \
  || { say "ABORT: refit failed"; exit 1; }

python - <<'EOF' || { echo "ABORT: refit output failed its checks"; exit 1; }
import sys, pandas as pd
sys.path.insert(0, "src")
n = pd.read_parquet("data/derived/phenoshape_by_index_v2.parquet")
plots = pd.read_parquet("data/derived/plots_subset.parquet")
have = set(n.plot_id); want = set(plots.PlotObservationID)
miss = want - have
print(f"refit covers {len(have & want)} of {len(want)} plots in plots_subset")
if miss:
    print(f"  {len(miss)} missing, e.g. {sorted(miss)[:5]}")
# the models need every plot: a partial curve table would silently drop rows
if len(miss) > 0:
    print("  -> not switching the models over; the comparison needs the full set")
    sys.exit(1)
EOF

say "=== curves refitted and complete"

# ------------------------------------------------------------------- the decisive test
# Pre-registered in docs/13 section 5: does closing the year change what the convolutional
# models can do? The comparison is one factor -- same substrates, indices, folds, seeds --
# and BIODIV_CURVES makes both the inputs and the run ids differ.
say "=== convolutional models on the corrected curves"
export BIODIV_CURVES=_v2
mkdir -p logs/modelling

i=0
for ix in ndvi evi kndvi nbr savi; do
  dev=$((i % 2)); i=$((i + 1))
  CUDA_VISIBLE_DEVICES=$dev python scripts/11_run_conv.py --substrate curve1d --index "$ix" \
    --scheme kfold5_window --seeds 3 --max-epochs 200 --patience 20 \
    > "logs/modelling/v2_c1d_$ix.log" 2>&1 &
  [ $((i % 2)) -eq 0 ] && wait
done
wait
say "1-D done"

for sub in serpentine hilbert reshape; do
  dev=$((i % 2)); i=$((i + 1))
  CUDA_VISIBLE_DEVICES=$dev python scripts/11_run_conv.py --substrate "$sub" --index kndvi \
    --scheme kfold5_window --seeds 3 --max-epochs 200 --patience 20 \
    > "logs/modelling/v2_2d_$sub.log" 2>&1 &
  [ $((i % 2)) -eq 0 ] && wait
done
wait
say "2-D done"

# a Random Forest on the same curves: it cannot be affected by the boundary (column order is
# irrelevant to it), so it is the control that says whether any change is about periodicity
# or about the curves having moved in general
python scripts/09_run_baselines.py --model RF03 --scheme kfold5_window --seeds 3 \
  > logs/modelling/v2_rf03.log 2>&1
say "RF control done"

# ----------------------------------------------------------------------------- report
python scripts/28_results_tables.py > logs/modelling/v2_tables.log 2>&1
python - <<'EOF'
import pandas as pd
f = pd.read_csv("results/tables/results_by_facet.csv")
f["v2"] = f.run_id.str.endswith("_v2")
base = f[~f.v2].copy()
new = f[f.v2].copy()
new["stem"] = new.run_id.str.replace(r"_v2$", "", regex=True)
m = new.merge(base, left_on=["stem", "facet"], right_on=["run_id", "facet"],
              suffixes=("_new", "_old"))
if m.empty:
    print("no paired runs yet")
else:
    m["delta"] = m.R2_new - m.R2_old
    print("R2 on the corrected curves minus the original, per run and facet:\n")
    print(m.pivot_table(index="stem", columns="facet", values="delta").round(3).to_string())
    print(f"\nmean delta over everything: {m.delta.mean():+.4f}")
    print("Seed noise is ~0.011, so 2 sd ~ 0.022. Below that, the boundary did not matter.")
EOF

say "=== chain finished"
