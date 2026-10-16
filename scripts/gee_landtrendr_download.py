"""LandTrendr Chile (Coquimbo a Magallanes) descargado por computePixels, sin cola de exportación.

Usa el endpoint de alto volumen de Earth Engine: cada trozo de CHUNK x CHUNK píxeles
(30 m, EPSG:32719) se calcula y descarga como GeoTIFF directo a disco, en paralelo.
Misma lógica de LandTrendr que gee_landtrendr_chile.py; aquí cada trozo trae las 36
bandas juntas (mayor perturbación + todos los segmentos), así LT corre una sola vez.

Bandas (Int16, sin datos = -32768):
    yod, mag, dur, preval, rate, dsnr            mayor perturbación (filtrada)
    yod_sK, end_sK, mag_sK, dur_sK, preval_sK    segmentos K = 1..6

Reanudable: los trozos ya escritos se saltan. Al final arma un VRT por todo el área.

Uso:
    python scripts/gee_landtrendr_download.py --project ee-javierlopatin --dry-run
    python scripts/gee_landtrendr_download.py --project ee-javierlopatin --limit 4
    python scripts/gee_landtrendr_download.py --project ee-javierlopatin --regiones Maule
"""

import argparse
import io
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import rasterio
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

import gee_landtrendr_chile as lt

CHUNK = 256          # píxeles por lado (256 x 30 m = 7.68 km)
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "landtrendr_chile"
HV_URL = "https://earthengine-highvolume.googleapis.com"
BAND_NAMES = ["yod", "mag", "dur", "preval", "rate", "dsnr"] + [
    f"{r}_s{k}" for k in range(1, lt.LT_PARAMS["maxSegments"] + 1)
    for r in ["yod", "end", "mag", "dur", "preval"]
]


def chunk_grid(regions):
    """Trozos alineados en EPSG:32719 que tocan las regiones (geometría simplificada a 1 km)."""
    to_utm = Transformer.from_crs("EPSG:4326", lt.CRS, always_xy=True).transform
    shapes = []
    for _, fc in regions:
        gj = fc.map(lambda f: f.simplify(1000)).geometry(1000).getInfo()
        shapes.append(transform(to_utm, shape(gj)))
    mainland = transform(to_utm, box(*lt.MAINLAND))
    area = unary_union(shapes).intersection(mainland).buffer(1000)

    size = CHUNK * lt.SCALE
    xmin, ymin, xmax, ymax = area.bounds
    i0, i1 = math.floor(xmin / size), math.ceil(xmax / size)
    j0, j1 = math.floor(ymin / size), math.ceil(ymax / size)
    out = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            if area.intersects(box(i * size, j * size, (i + 1) * size, (j + 1) * size)):
                out.append((i, j))
    return out


def chunk_image(i, j, mask):
    size = CHUNK * lt.SCALE
    aoi = ee.Geometry.Rectangle(
        [i * size, j * size, (i + 1) * size, (j + 1) * size], lt.CRS, False
    )
    res = lt.run_landtrendr(aoi)
    img = lt.greatest_disturbance(res).addBands(lt.all_segments(res))
    return img.updateMask(mask).unmask(lt.NODATA).toInt16()


def fetch(i, j, mask, retries=6):
    """Descarga un trozo; devuelve 'ok', 'vacio' o 'existe'. Reintenta con backoff."""
    name = f"LT_{i:+05d}_{j:+05d}"
    tif, empty = OUT_DIR / "chunks" / f"{name}.tif", OUT_DIR / "chunks" / f"{name}.empty"
    if tif.exists() or empty.exists():
        return "existe"
    size = CHUNK * lt.SCALE
    request = {
        "expression": chunk_image(i, j, mask),
        "fileFormat": "GEO_TIFF",
        "grid": {
            "dimensions": {"width": CHUNK, "height": CHUNK},
            "affineTransform": {
                "scaleX": lt.SCALE, "shearX": 0, "translateX": i * size,
                "shearY": 0, "scaleY": -lt.SCALE, "translateY": (j + 1) * size,
            },
            "crsCode": lt.CRS,
        },
    }
    for attempt in range(retries):
        try:
            raw = ee.data.computePixels(request)
            break
        except Exception as e:  # 429 / cuota / red: esperar y reintentar
            if attempt == retries - 1:
                raise RuntimeError(f"{name}: {e}") from e
            time.sleep(2 ** attempt * 5 + random.random() * 5)

    with rasterio.MemoryFile(io.BytesIO(raw)) as mem, mem.open() as src:
        data = src.read()
        if (data == lt.NODATA).all():
            empty.touch()
            return "vacio"
        profile = src.profile | {
            "driver": "GTiff", "compress": "deflate", "predictor": 2,
            "tiled": True, "blockxsize": CHUNK, "blockysize": CHUNK, "nodata": lt.NODATA,
        }
        tmp = tif.with_suffix(".part")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(data)
            for b, n in enumerate(BAND_NAMES, 1):
                dst.set_band_description(b, n)
        tmp.rename(tif)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--regiones", nargs="*", help="filtrar por nombre (sin tildes, parcial)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="solo los primeros N trozos (prueba)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ee.Initialize(project=args.project, opt_url=HV_URL)
    ee.data.setDeadline(300_000)  # 5 min por petición: sin esto, un corte de red cuelga hilos

    regions = lt.chile_regions()
    if args.regiones:
        wanted = [w.lower() for w in args.regiones]
        regions = [(n, r) for n, r in regions if any(w in lt.ascii_name(n).lower() for w in wanted)]
    print("Regiones:", ", ".join(lt.ascii_name(n) for n, _ in regions))

    chunks = chunk_grid(regions)
    if args.limit:
        mid = len(chunks) // 2  # trozos del centro: más probable que tengan tierra
        chunks = chunks[mid: mid + args.limit]
    print(f"{len(chunks)} trozos de {CHUNK}x{CHUNK} px")
    if args.dry_run:
        return

    (OUT_DIR / "chunks").mkdir(parents=True, exist_ok=True)
    mask = lt.region_mask(ee.FeatureCollection([r for _, r in regions]).flatten())

    t0, counts, errors = time.time(), {"ok": 0, "vacio": 0, "existe": 0}, []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, i, j, mask): (i, j) for i, j in chunks}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                counts[fut.result()] += 1
            except Exception as e:
                errors.append(str(e)[:300])
            if n % 50 == 0 or n == len(chunks):
                rate = n / (time.time() - t0)
                eta = (len(chunks) - n) / rate / 60 if rate else float("nan")
                print(f"{n}/{len(chunks)} {counts} errores={len(errors)} "
                      f"{rate * 60:.1f} trozos/min, faltan {eta:.0f} min", flush=True)

    for e in errors[:10]:
        print("ERROR", e)
    (OUT_DIR / "errores.txt").write_text("\n".join(errors))

    print(f"Listo en {(time.time() - t0) / 60:.1f} min. Mosaico (son >10 mil archivos, va por lista):")
    print(f"  find {OUT_DIR / 'chunks'} -name '*.tif' > {OUT_DIR / 'chunks_list.txt'}")
    print(f"  gdalbuildvrt -input_file_list {OUT_DIR / 'chunks_list.txt'} {OUT_DIR / 'LT_Chile.vrt'}")


if __name__ == "__main__":
    main()
