#!/usr/bin/env python3
"""Tabla larga única de resultados de modelos, desde los archivos versionados de cada corrida.

Recorre results/models_*/<run>/<esquema>/{config.json, pooled_metrics.csv} y escribe
results/tables/resumen_modelos.csv con una fila por corrida × target. Es la fuente única
para las tablas del manuscrito: nada se transcribe de mensajes.

Columnas principales:
  carpeta, run_id, familia, bloques, indice, esquema
  variante_target      nombre explícito de BIODIV_TARGETS (con_hierbas, lenoso,
                       lenoso_solo_pcl_lcbd_del_pool, lenoso_solo_pcl_lcbd_propio, estrato_*)
  version_curvas, version_lspu, version_gmo   versión de reflectancia (nc, tcfe, ... null*)
  target, faceta_publicada, n, R2_mean, R2_sd, R2_ensemble, RMSE_mean
  vigente (bool) y motivo_no_vigente: POR QUÉ, con el commit que lo explica.

Reglas de vigencia, en orden:
  1. El bloque gm sobre un pool con filas de Living Trees, con código anterior al arreglo
     del cruce sitio -> parcela (6e9f8e8): "gm de LT mal cruzado, ver 6e9f8e8".
  2. Predictores o targets cuyo hash (data_sha1 en config.json, runlog desde 4b78fab) no
     coincide con el archivo actual. Sin hash y con predictor de curva: "curvas
     pre-interp_grid, ver f769622". El hash se compara con el archivo en disco, así que la
     vigencia responde "¿esta corrida vio los datos de hoy?" en cualquier máquina, sin
     depender de fechas.
  3. Notas que NO invalidan: bloque gm con gm_count (antes de que saliera del bloque por
     defecto) y facetas q1/q2 (R² de bloque <= 0,04; no se publican: el ruido de los arreglos
     se concentra ahí).
El piso topo+área (B01) va como fila más: la mitad de las conclusiones se leen contra él.

Uso:
    python scripts/101_resumen_modelos.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
GM_FIX = "6e9f8e8"
GMCOUNT_OUT = "gm_count fuera del bloque gm por defecto"   # commit que lo introduce (ver git log)
VARIANTES = {
    "": "con_hierbas", "herbcommon": "con_hierbas_parcelas_comunes", "woody": "lenoso",
    "woodypcl": "lenoso_solo_pcl_lcbd_del_pool", "woodypclown": "lenoso_solo_pcl_lcbd_propio",
    "strat_11forest": "estrato_Forest", "strat_nobosque": "estrato_NoBosque",
}
PUBLICADAS = {"lcbd_count_sorensen", "pd_inext_q0", "td_inext_q0", "lcbd_pa_unified",
              "hill_q0_unified", "lcbd_freq_unified"}
_sha_cache: dict[str, str] = {}


def sha(name: str) -> str | None:
    f = DERIVED / name
    if not f.exists():
        return None
    if name not in _sha_cache:
        _sha_cache[name] = hashlib.sha1(f.read_bytes()).hexdigest()[:12]
    return _sha_cache[name]


def is_ancestor(fix: str, commit: str) -> bool | None:
    c = (commit or "").replace("-dirty", "")
    if not c:
        return None
    r = subprocess.run(["git", "merge-base", "--is-ancestor", fix, c], cwd=ROOT,
                       capture_output=True)
    return {0: True, 1: False}.get(r.returncode)


def parse(run: str) -> dict:
    v = re.search(r"_unified_(woodypclown|woodypcl|herbcommon|woody|strat_11forest|strat_nobosque)", run)
    cur = re.search(r"_raw100(nc|tcfe|tccsall|tccs|null\d+)?_", run + "_")
    gmo = re.search(r"_gmo(nc|tcfe|tccsall|tccs|null\d+)", run)
    lspu = re.search(r"_lspu(nc|tcfe|tccsall|tccs|null\d+)", run)
    return {"variante_target": VARIANTES[v.group(1) if v else ""],
            "version_curvas": (cur.group(1) or "") if cur else "",
            "version_gmo": gmo.group(1) if gmo else "",
            "version_lspu": lspu.group(1) if lspu else ""}


def main() -> None:
    rows = []
    for cfg_f in sorted(glob.glob(str(ROOT / "results" / "models_*" / "*" / "*" / "config.json"))):
        cfg_f = Path(cfg_f)
        pm = cfg_f.with_name("pooled_metrics.csv")
        if not pm.exists():
            continue
        cfg = json.loads(cfg_f.read_text())
        run, carpeta, esquema = cfg_f.parts[-3], cfg_f.parts[-4], cfg_f.parts[-2]
        feats = str(cfg.get("features", ""))
        substrate = str(cfg.get("substrate", ""))
        meta = parse(run)
        uses_curve = "curve" in feats or substrate in ("curve1d", "serpentine", "hilbert", "reshape")
        uses_gm = bool(re.search(r"(^|[+-])gm($|[+-])", feats.replace("_ctr", "")))
        pool_has_lt = meta["variante_target"] not in ("lenoso_solo_pcl_lcbd_del_pool",
                                                       "lenoso_solo_pcl_lcbd_propio")
        motivo, notas = "", []
        if uses_gm and pool_has_lt and is_ancestor(GM_FIX, cfg.get("git", "")) is False:
            motivo = f"gm de LT mal cruzado, ver {GM_FIX}"
        ds = cfg.get("data_sha1") or {}
        if not motivo:
            for key, val in ds.items():
                name, h = val.split(":")
                cur = sha(name)
                if cur is not None and cur != h and (key != "cube_lt" or uses_gm):
                    motivo = f"{key} ({name}) distinto del archivo actual"
                    break
        if not motivo and not ds and uses_curve:
            motivo = "curvas pre-interp_grid, ver f769622"
        if uses_gm and "gmoall" not in feats and "gmomed" not in feats and "gmomad" not in feats:
            notas.append("bloque gm con gm_count si la corrida es anterior a su salida del bloque")
        pmd = pd.read_csv(pm)
        for _, r in pmd.iterrows():
            t = r["target"]
            nota_t = list(notas)
            if re.search(r"_q[12]$", t):
                nota_t.append("q1/q2 no publicada (R² de bloque <= 0,04)")
            rows.append(dict(carpeta=carpeta, run_id=run, familia=cfg.get("family"),
                             bloques=feats, sustrato=substrate, indice=cfg.get("index", ""),
                             esquema=esquema, target_set=cfg.get("target_set"), **meta,
                             target=t, faceta_publicada=t in PUBLICADAS,
                             n=r.get("n"), n_seeds=r.get("n_seeds"), R2_mean=r.get("R2_mean"),
                             R2_sd=r.get("R2_sd"), R2_ensemble=r.get("R2_ensemble"),
                             RMSE_mean=r.get("RMSE_mean"), git=cfg.get("git", ""),
                             vigente=motivo == "", motivo_no_vigente=motivo,
                             notas="; ".join(nota_t)))
    out = pd.DataFrame(rows)
    f = ROOT / "results" / "tables" / "resumen_modelos.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(f, index=False)
    print(f"-> {f}  ({len(out)} filas, {out.run_id.nunique()} corridas)")
    print(out.groupby(["carpeta", "vigente"]).run_id.nunique().unstack(fill_value=0).to_string())
    print("\nmotivos:\n", out[~out.vigente].drop_duplicates("run_id").motivo_no_vigente.value_counts().to_string())


if __name__ == "__main__":
    main()
