#!/usr/bin/env python3
"""Pixel central (kNDVI, kfold5_block20) contra la media 5x5, por facies y por modelo.

Companero de `scripts/43_matrix_center_kndvi_block20.py`. Compara cada run `*c_..._kndvi`
(pixel central) contra su equivalente sin el sufijo `c`/`_ctr` bajo el MISMO esquema e
indice (`kfold5_block20`, `kndvi`) -- el unico factor que cambia es el pixel. Donde no
existe el baseline de media 5x5 bajo exactamente ese esquema+indice, la fila queda
PENDIENTE en vez de comparar contra un numero de otro esquema/config (que confundiria el
efecto de pixel con el de cualquier otra cosa que tambien haya cambiado).

Uso:
    python scripts/44_compare_center_vs_5x5.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# center_run_id -> baseline_run_id (mean5x5, mismo scheme+index). RF usa prefijo "c" en el
# nombre del modelo (`scripts/09_run_baselines.py`: `model + ("c" if px=="center" else "")`);
# MLP/C1D/C2D usan sufijo "_ctr" al final del run_id (`scripts/10`/`11`: `ident += "_ctr"`)
# -- confirmado contra los run_ids reales en logs/matrix43_*.log, no adivinado.
PAIRS = {
    "RF01c_lsp_ctr-topo_ctr-area_kndvi": "RF01_lsp-topo-area_kndvi",
    "RF02c_lsp_ctr-qc-topo_ctr-area_kndvi": None,      # sin baseline mean5x5+block20+kndvi
    "RF03c_curve-topo_ctr-area_kndvi": "RF03_curve-topo-area_kndvi",
    "RF04c_lsp_ctr-curve-topo_ctr-area_kndvi": None,
    "RF08c_lsp_ctr-curve-topo_ctr_kndvi": None,
    "MLP01_lsp_ctr_kndvi_ctr": None,
    "MLP02_curve_kndvi_ctr": "MLP02_curve_kndvi",
    "MLP03_lsp_ctr-curve-qc_kndvi_ctr": None,
    "MLP05a_curve_kndvi_ctr": None,
    "MLP05c_curve_kndvi_ctr": None,
    "MLP06_curve_kndvi_ctr": None,     # baseline existente usa sufijo _raw100, curva distinta
    "C1D01_curve1d_kndvi_ctr": "C1D01_curve1d_kndvi",
    "C2D02_serpentine_kndvi_ctr": None,  # baseline existente usa sufijo _raw36
    "C2D01_reshape_kndvi_ctr": None,
    "C2D03_gaf_kndvi_ctr": None,
    "C2D04_mtf_kndvi_ctr": None,
    "C2D05_ndi_kndvi_ctr": None,
    "C2D06_cwt_kndvi_ctr": None,
    "C2D07_hilbert_kndvi_ctr": None,
    "C2D08_cos2d_kndvi_ctr": None,
    "C2D09_spectrogram_kndvi_ctr": None,
}

FACETS = {
    "alpha": ["hill_q0", "hill_q1", "hill_q2"],
    "beta_pa": ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"],
    "beta_cover": ["lcbd_cover", "pcoa1_cover", "pcoa2_cover"],
    "phylo": ["mpd", "mntd", "ses_pd", "ses_mpd", "ses_mntd"],
    "dark": ["dark_n"],
}
FACET_OF = {t: f for f, ts in FACETS.items() for t in ts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default="results/models/summary.csv")
    p.add_argument("--out", default="results/tables/center_vs_5x5.csv")
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    df = df[df["scheme"] == "kfold5_block20"]
    df["facet"] = df["target"].map(FACET_OF)

    rows = []
    for center_id, base_id in PAIRS.items():
        c = df[df["run_id"] == center_id]
        if c.empty:
            print(f"  {center_id:45s} PENDIENTE -- no corrio (todavia) en este esquema")
            continue
        c_r2 = c.groupby("facet")["R2"].mean()
        if base_id is None:
            for f in FACETS:
                rows.append({"model": center_id, "facet": f,
                            "r2_center": c_r2.get(f), "r2_5x5": None, "delta": None,
                            "status": "PENDIENTE (sin baseline mean5x5 en block20+kndvi)"})
            print(f"  {center_id:45s} sin baseline -- reportado solo el valor central")
            continue
        b = df[df["run_id"] == base_id]
        if b.empty:
            print(f"  {center_id:45s} baseline {base_id!r} no encontrado")
            continue
        b_r2 = b.groupby("facet")["R2"].mean()
        for f in FACETS:
            cr, br = c_r2.get(f), b_r2.get(f)
            rows.append({"model": center_id, "facet": f, "r2_center": cr, "r2_5x5": br,
                        "delta": None if cr is None or br is None else cr - br,
                        "status": "ok"})

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    ok = out[out["status"] == "ok"]
    if len(ok):
        print("\n=== comparacion valida (mismo scheme+index, solo cambia el pixel) ===")
        piv = ok.pivot_table(index="facet", columns="model", values="delta")
        print(piv.round(4))
    pend = out[out["status"] != "ok"]["model"].nunique()
    if pend:
        print(f"\n{pend} modelos sin baseline mean5x5 comparable bajo kfold5_block20+kndvi "
             f"-- ver columna 'status' en {args.out}")


if __name__ == "__main__":
    main()
