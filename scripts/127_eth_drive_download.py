#!/usr/bin/env python3
"""Baja de Google Drive las teselas de altura de dosel ETH exportadas por scripts/126 --modo chile.

Espera a que terminen las tareas de Earth Engine (ETH_CH_height_Chile, ETH_CH_sd_Chile), lista
los GeoTIFF de la carpeta de Drive y los baja a --dest, saltando los que ya están con el mismo
tamaño. Usa las credenciales de Earth Engine, que incluyen el alcance de Drive. Al final arma un
VRT por banda (ETH_CH_height_Chile.vrt, ETH_CH_sd_Chile.vrt) para leer Chile como un raster.

Correr con el entorno `phenopy` (earthengine-api):
    /home/javier/miniconda3/envs/phenopy/bin/python scripts/127_eth_drive_download.py
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import requests

DEST = "/mnt/rapidita_4T/datos/ETH_GlobalCanopyHeight_10m_2020_version1"
PREFIX = "ETH_CH_"            # todas las franjas de scripts/126 --modo chile
API = "https://www.googleapis.com/drive/v3/files"


def token(ee):
    import google.auth.transport.requests as gr
    c = ee.data.get_persistent_credentials()
    c.refresh(gr.Request())
    return {"Authorization": f"Bearer {c.token}"}


def wait_tasks(ee, poll: int) -> None:
    """Espera a las franjas lanzadas por scripts/126; para cada descripción vale su tarea más
    reciente (las canceladas de antes no cuentan si se relanzaron)."""
    while True:
        latest = {}
        for t in ee.data.getTaskList():                 # más reciente primero
            d = t.get("description", "")
            if d.startswith(PREFIX) and "_S" in d and d not in latest:
                latest[d] = t["state"]
        n = {s: sum(v == s for v in latest.values()) for s in set(latest.values())}
        print(time.strftime("%H:%M"), f"{len(latest)} franjas", n, flush=True)
        bad = [d for d, s in latest.items() if s in ("FAILED", "CANCELLED")]
        if bad:
            raise SystemExit(f"franjas fallidas, relanzar con scripts/126 --only: {bad}")
        if latest and all(s == "COMPLETED" for s in latest.values()):
            return
        time.sleep(poll)


def list_files(ee, folder: str) -> list[dict]:
    h = token(ee)
    q = f"name = '{folder}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    fids = requests.get(API, params={"q": q, "fields": "files(id)"}, headers=h, timeout=60).json()["files"]
    if not fids:
        raise SystemExit(f"no está la carpeta {folder!r} en Drive")
    # Earth Engine puede crear más de una carpeta con el mismo nombre (una por tarea que
    # arranca a la vez), así que se buscan los archivos en todas.
    parents = " or ".join(f"'{f['id']}' in parents" for f in fids)
    out, page = [], None
    while True:
        r = requests.get(API, headers=h, timeout=60, params={
            "q": f"({parents}) and trashed = false and name contains 'ETH_CH_'",
            "fields": "nextPageToken, files(id, name, size)", "pageSize": 1000, "pageToken": page}).json()
        out += r["files"]
        page = r.get("nextPageToken")
        if not page:
            return out


def download(ee, f: dict, dest: Path) -> None:
    p = dest / f["name"]
    if p.exists() and p.stat().st_size == int(f["size"]):
        return
    tmp = p.with_suffix(p.suffix + ".part")
    with requests.get(f"{API}/{f['id']}", params={"alt": "media"}, headers=token(ee),
                      stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(8 << 20):
                fh.write(chunk)
    if tmp.stat().st_size != int(f["size"]):
        raise RuntimeError(f"{f['name']}: tamaño distinto al de Drive")
    tmp.rename(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="ee-javierlopatin")
    ap.add_argument("--folder", default="ETH_CH_Chile")
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    import ee
    ee.Initialize(project=a.project)
    if not a.no_wait:
        wait_tasks(ee, a.poll)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(list_files(ee, a.folder), key=lambda f: f["name"])
    tot = sum(int(f["size"]) for f in files)
    print(f"{len(files)} archivos, {tot / 1e9:.1f} GB", flush=True)
    for k, f in enumerate(files, 1):
        for intento in range(4):
            try:
                download(ee, f, dest)
                break
            except Exception as e:
                print(f"  {f['name']}: {e} (reintento {intento + 1})", flush=True)
                time.sleep(30)
        print(f"  {k}/{len(files)} {f['name']}", flush=True)
    for band in ("height", "sd"):
        tifs = sorted(str(p) for p in dest.glob(f"ETH_CH_{band}_Chile*.tif"))
        subprocess.run(["gdalbuildvrt", str(dest / f"ETH_CH_{band}_Chile.vrt"), *tifs], check=True)
        print(f"VRT {band}: {len(tifs)} teselas", flush=True)


if __name__ == "__main__":
    main()
