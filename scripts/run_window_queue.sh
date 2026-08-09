#!/bin/bash
# La cola de CPU que faltaba, ahora bajo el esquema unico.
#
# Reemplaza a `run_climate_queue.sh`, que tenia el `cd` clavado a la ruta de la maquina con
# datacube (/home/jovyan/temp) y corria el factor de agregacion bajo `kfold5_owner`, que era
# el primario cuando se escribio. Aqui todo va bajo `kfold5_window`.
#
# Lo que quedaba pendiente de docs/11_next_steps.md seccion 1:
#   - el factor de agregacion (X07, X12) bajo el esquema nuevo: solo estaba bajo owner
#   - las tres corridas de GDM con clima, que nunca arrancaron
#   - la grilla de penalizaciones del SGDM, sin la cual no se puede afirmar que el SGDM sea
#     peor que el GDM simple: con las penalizaciones por defecto lo es en las cinco filas
#     corridas, pero nunca se lo sintonizo
#
# Los dos scripts que invoca reanudan por clave -- `19_screen_blocks.py` por
# (fila, esquema, agregacion) y `21_run_gdm.py` hace append -- asi que relanzar es seguro:
# lo hecho se salta y un kill cuesta como mucho la corrida en vuelo.
#
#   nohup bash scripts/run_window_queue.sh >> logs/window_queue.log 2>&1 &
#   pgrep -af '^bash scripts/run_window_queue\.sh'    # vacio = termino
set -u
cd "$(dirname "$0")/.."

# Esta cola convive con `14_run_matrix.py`, que ya tiene su propio pool. Sin este limite los
# bosques de las dos colas piden las 32 hebras cada uno y las dos van mas lento que en serie.
export LOKY_MAX_CPU_COUNT=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

SCHEME=kfold5_window

echo "=== factor de agregacion bajo $SCHEME  $(date +%F\ %H:%M:%S)"
# `median` ya esta corrido; faltan las otras tres reglas. La pregunta es si la regla que
# colapsa los 25 pixeles importa, y solo tiene sentido sobre bloques hechos de pixeles.
for agg in mean trimmed center; do
  python scripts/19_screen_blocks.py --rows X07 X12 --schemes "$SCHEME" --agg "$agg"
done

echo "=== GDM bajo $SCHEME  $(date +%H:%M:%S)"
# Las cinco filas que ya existen estan bajo kfold5_owner y no son comparables con nada de lo
# demas; se rehacen. Las dos con clima son las que nunca corrieron.
for spec in \
    "curve+topo" \
    "gm+topo" \
    "composite_all+topo" \
    "gm+obscomp_all+seas_all+contrast+topo" \
    "curve+gm+obscomp_all+seas_all+contrast+topo" \
    "clim+topo" \
    "curve+clim+topo"; do
  echo "--- $spec  $(date +%H:%M:%S)"
  python scripts/21_run_gdm.py --scheme "$SCHEME" --spec "$spec"
done

# El BLAS recupera sus hebras: la grilla de sCCA es algebra lineal densa, no bosques.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

echo "=== grilla de penalizaciones del SGDM  $(date +%H:%M:%S)"
python scripts/21_run_gdm.py --scheme "$SCHEME" \
  --spec "curve+gm+obscomp_all+seas_all+contrast+topo" --sgdm-grid

echo "=== fin  $(date +%F\ %H:%M:%S)"
