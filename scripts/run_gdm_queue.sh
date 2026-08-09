#!/bin/bash
# GDM comparison over the specs the RF screen was deciding between. The screen (X00-X14) is
# already in results/tables/block_screen.csv, so this no longer waits for anything — it runs
# alongside run_climate_queue.sh. Each queue is capped at half the machine so the two do not
# oversubscribe the 8 cores; 21_run_gdm.py appends, so a kill only costs the run in flight.
set -u
cd /home/jovyan/temp/Biodiversity_Chile

export LOKY_MAX_CPU_COUNT=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

run () {  # spec, index, label
  echo ""
  echo "############ $3   spec=$1 index=$2   $(date +%H:%M:%S)"
  if [ -z "$2" ]; then
    python scripts/21_run_gdm.py --scheme kfold5_owner --spec "$1"
  else
    python scripts/21_run_gdm.py --scheme kfold5_owner --spec "$1" --index "$2"
  fi
}

echo "=== arranca GDM $(date +%F\ %H:%M:%S)"
run "curve+topo"                                         kndvi "fenologia: la curva de 52 pasos"
run "composite_all+topo"                                 ""    "compuesto anual (sin forma)"
run "gm+topo"                                            ""    "geomediana + MADs"
run "gm+obscomp_all+seas_all+contrast+topo"              ""    "todo lo sin forma"
run "curve+gm+obscomp_all+seas_all+contrast+topo"        kndvi "todo junto"
echo ""
echo "=== GDM terminado $(date +%F\ %H:%M:%S)"
