#!/usr/bin/env python3
"""Matrix filtrado (kNDVI, pixel central, kfold5_block20), modelos guardados.

Motivado por el chequeo de sensibilidad al area (`scripts/42_area_sensitivity.py`): el
mapa final predice sobre celdas de 900 m2 (1 pixel Landsat), pero el entrenamiento usaba
la media de una ventana 5x5. Esta corrida entrena con el pixel central en su lugar, bajo
`kfold5_block20` (el esquema espacial "honesto" en distancia de la sesion), restringido a
kNDVI, y guarda los modelos ajustados -- ninguna corrida anterior lo hacia (ver
`src/biodiv/models_tabular.py::save_rf` y los cambios de hoy en `dl_runner.py`).

Precedente que no se descarta: `docs/08_modelling.md` S7.5 ya midio esto bajo
`kfold5_owner` (no `kfold5_block20`) para RF01/03/04 + CNN 1D. CNN-1D colapsa en alfa
(-0.268, 0/5 configs a favor); RF casi indiferente en composicion (-0.009), mejora en alfa
(+0.043, sin mecanismo validado). Esta corrida sirve tambien para ver si el patron se
sostiene bajo block20.

Excluidos, y por que (no pueden restringirse a un solo indice o no soportan pixel
central):
  - RF05, RF06 (spec `curve_all`/`lsp_all`, apilan las 5 curvas)
  - MLP04, MLP07 (spec `curve_all`)
  - C1D02 / substrate `curve5` (apila las 5 curvas)
  - substrate `stack5` (idem)
  - substrate `pxcube` (`substrates.py:110-131` siempre carga los 25 pixeles, `px` no se
    propaga ahi -- limitacion real del sustrato, no una eleccion de esta corrida)

Uso:
    python scripts/43_matrix_center_kndvi_block20.py --dry-run
    python scripts/43_matrix_center_kndvi_block20.py --group rf
    python scripts/43_matrix_center_kndvi_block20.py --group all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCHEME = "kfold5_block20"
INDEX = "kndvi"
PX = "center"

RF_MODELS = ["RF01", "RF02", "RF03", "RF04", "RF08"]
MLP_MODELS = ["MLP01", "MLP02", "MLP03", "MLP05a", "MLP05c", "MLP06"]
C2D_SUBSTRATES = ["reshape", "serpentine", "gaf", "mtf", "ndi", "cwt", "hilbert",
                  "cos2d", "spectrogram"]


def build_jobs() -> list[tuple[str, list[str]]]:
    jobs = []
    for m in RF_MODELS:
        jobs.append((f"rf/{m}", [
            PY, "scripts/09_run_baselines.py", "--model", m, "--scheme", SCHEME,
            "--index", INDEX, "--px", PX, "--seeds", "3", "--force", "--save-state",
        ]))
    for m in MLP_MODELS:
        jobs.append((f"mlp/{m}", [
            PY, "scripts/10_run_tabular_dl.py", "--model", m, "--index", INDEX,
            "--scheme", SCHEME, "--px", PX, "--seeds", "3", "--force",
        ]))
    jobs.append(("c1d/curve1d", [
        PY, "scripts/11_run_conv.py", "--substrate", "curve1d", "--index", INDEX,
        "--scheme", SCHEME, "--px", PX, "--seeds", "3", "--force",
    ]))
    for s in C2D_SUBSTRATES:
        jobs.append((f"c2d/{s}", [
            PY, "scripts/11_run_conv.py", "--substrate", s, "--index", INDEX,
            "--scheme", SCHEME, "--px", PX, "--seeds", "3", "--force",
        ]))
    return jobs


GROUPS = {
    "rf": lambda name: name.startswith("rf/"),
    "mlp": lambda name: name.startswith("mlp/"),
    "cnn": lambda name: name.startswith(("c1d/", "c2d/")),
    "all": lambda name: True,
}


def is_gpu_job(name: str) -> bool:
    return name.startswith(("mlp/", "c1d/", "c2d/"))


def run_job(name: str, cmd: list[str], gpu: int | None) -> None:
    t0 = time.time()
    tag = f"GPU{gpu}" if gpu is not None else "CPU"
    print(f"=== {name} [{tag}]  {time.strftime('%H:%M:%S')}", flush=True)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"    {name} [{tag}] {status}  [{dt/60:.1f} min]", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=sorted(GROUPS), default="all")
    p.add_argument("--gpus", type=int, default=1,
                   help="GPU jobs (mlp/c1d/c2d) run this many at once, one per "
                        "CUDA_VISIBLE_DEVICES id 0..gpus-1. RF stays sequential/CPU "
                        "(sklearn's own n_jobs=-1 already saturates the cores).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = [(n, c) for n, c in build_jobs() if GROUPS[args.group](n)]
    print(f"{len(jobs)} jobs (grupo={args.group}), scheme={SCHEME} index={INDEX} px={PX}, "
         f"gpus={args.gpus}\n")
    for name, cmd in jobs:
        print(f"  {name:16s} {' '.join(cmd[1:])}")
    if args.dry_run:
        return

    print()
    cpu_jobs = [(n, c) for n, c in jobs if not is_gpu_job(n)]
    gpu_jobs = [(n, c) for n, c in jobs if is_gpu_job(n)]

    for name, cmd in cpu_jobs:
        run_job(name, cmd, gpu=None)

    if gpu_jobs:
        if args.gpus <= 1:
            for name, cmd in gpu_jobs:
                run_job(name, cmd, gpu=0 if args.gpus == 1 else None)
        else:
            with ThreadPoolExecutor(max_workers=args.gpus) as ex:
                futs = [ex.submit(run_job, name, cmd, i % args.gpus)
                       for i, (name, cmd) in enumerate(gpu_jobs)]
                for f in futs:
                    f.result()

    print(f"\n=== fin {time.strftime('%F %H:%M:%S')}")


if __name__ == "__main__":
    main()
