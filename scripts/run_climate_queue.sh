#!/bin/bash
# Climate rows of the screening matrix, the aggregation factor, and the GDM runs that put
# climate in the design. data/derived/climate.parquet already exists, so the wait loop of the
# original queue is gone. 19_screen_blocks.py resumes by (row, scheme, agg), so re-running
# this after a kill re-does only the block that was in flight.
set -u
cd /home/jovyan/temp/Biodiversity_Chile

export LOKY_MAX_CPU_COUNT=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [ ! -f data/derived/climate.parquet ]; then
  echo "ABORTA: climate.parquet no existe"; exit 1
fi

echo "=== arranca cribado de clima $(date +%F\ %H:%M:%S)"
python scripts/19_screen_blocks.py --rows X15 X16 X17 X18 X19

# The aggregation factor (row X10 of docs/09_predictors.md): does the rule that collapses
# the 25 pixels matter at all? Only worth asking on a block that is actually made of pixels.
echo "=== factor de agregacion $(date +%H:%M:%S)"
for agg in mean trimmed center; do
  python scripts/19_screen_blocks.py --rows X07 X12 --schemes kfold5_owner --agg "$agg"
done

# Both queues append to results/tables/gdm_comparison.csv, and 21_run_gdm.py writes its header
# only when the file is absent — two appends racing on an empty file would emit it twice. The
# screening above almost certainly outlasts the other queue, but "almost certainly" is not a
# guarantee worth taking on a table this expensive to rebuild.
# The pattern is anchored to the actual command line. An unanchored `pgrep -f run_gdm_queue`
# also matches any *other* process that merely mentions the name — a `tail`, a watcher, an
# editor — and then this loop never exits. That is not hypothetical: it blocked this queue
# for 40 minutes after the GDM runs had already finished, because a monitoring shell was
# waiting on the same string.
while pgrep -f '^bash scripts/run_gdm_queue\.sh' >/dev/null; do sleep 60; done

# BLAS gets its threads back: the sCCA grid search is dense linear algebra, not forests.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== GDM con clima $(date +%H:%M:%S)"
python scripts/21_run_gdm.py --scheme kfold5_owner --spec "clim+topo"
python scripts/21_run_gdm.py --scheme kfold5_owner --spec "curve+clim+topo" --index kndvi
python scripts/21_run_gdm.py --scheme kfold5_owner --sgdm-grid \
    --spec "lsp+curve+gm+seas_all+contrast+clim+topo" --index kndvi
echo "=== todo terminado $(date +%F\ %H:%M:%S)"
