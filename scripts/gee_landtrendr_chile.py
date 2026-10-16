"""LandTrendr (GEE) para Chile, de Coquimbo a Magallanes, con Landsat 5/7/8/9 desde 2000.

Corre ee.Algorithms.TemporalSegmentation.LandTrendr sobre un índice NBR anual
(compuesto de verano austral, diciembre a marzo) y exporta a una carpeta de
Google Drive, por región y por teselas, dos GeoTIFF:

  * LT_<region>_<tile>_mayor.tif  -> la mayor perturbación de cada píxel
        yod      año de detección (primer año después del vértice inicial)
        mag      magnitud (caída de NBR x 1000; positivo = pérdida de vegetación)
        dur      duración del segmento (años; solo segmentos de <= MAX_DUR = 4 años)
        preval   NBR x 1000 antes del cambio
        rate     mag / dur
        dsnr     mag / RMSE del ajuste, x 100
  * LT_<region>_<tile>_segmentos.tif -> todos los segmentos ajustados, en orden
        cronológico, s1..sN (N = maxSegments):
        yod_sK, end_sK, mag_sK, dur_sK, preval_sK
        mag > 0 = pérdida (perturbación), mag < 0 = recuperación;
        yod_sK = 0 indica que el píxel tiene menos de K segmentos.

Valor sin datos: -32768 (Int16).

Uso:
    pip install earthengine-api
    earthengine authenticate
    python scripts/gee_landtrendr_chile.py --project mi-proyecto-gee          # lanza tareas
    python scripts/gee_landtrendr_chile.py --project mi-proyecto-gee --dry-run  # solo lista
    python scripts/gee_landtrendr_chile.py --project X --regiones Maule --max-teselas 1
"""

import argparse
import re
import unicodedata

import ee

# --------------------------------------------------------------------------
# Parámetros
# --------------------------------------------------------------------------
START_YEAR = 2000
END_YEAR = 2026
SEASON_START_MONTH = 12  # diciembre del año anterior ...
SEASON_END_MONTH = 3     # ... hasta marzo del año del compuesto
DRIVE_FOLDER = "LandTrendr_Chile"
CRS = "EPSG:32719"
SCALE = 30
TILE_DEG = 1.0           # tamaño de tesela de exportación (grados)
NODATA = -32768

# Filtro de la mayor perturbación
MIN_MAG = 100            # NBR x 1000 (0.10 de NBR)
MAX_DUR = 4              # solo segmentos abruptos compiten por "mayor"; None = todos
MIN_PREVAL = None        # p. ej. 300 para exigir vegetación antes del cambio
MMU = 11                 # píxeles conectados mínimos (~1 ha); None desactiva

LT_PARAMS = {
    "maxSegments": 6,
    "spikeThreshold": 0.9,
    "vertexCountOvershoot": 3,
    "preventOneYearRecovery": True,
    "recoveryThreshold": 0.25,
    "pvalThreshold": 0.05,
    "bestModelProportion": 0.75,
    "minObservationsNeeded": 6,
}

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
TM_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
OLI_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]


# --------------------------------------------------------------------------
# Landsat Collection 2 Level 2
# --------------------------------------------------------------------------
def _mask_scale(img, src_bands):
    qa = img.select("QA_PIXEL")
    # bits: 1 nube dilatada, 3 nube, 4 sombra, 5 nieve
    clear = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    unsat = img.select("QA_RADSAT").eq(0)
    sr = img.select(src_bands, BANDS).multiply(0.0000275).add(-0.2)
    valid = sr.reduce(ee.Reducer.min()).gt(0).And(sr.reduce(ee.Reducer.max()).lt(1))
    return ee.Image(
        sr.updateMask(clear).updateMask(unsat).updateMask(valid).toFloat()
        .copyProperties(img, ["system:time_start"])
    )


def _oli_to_etm(img):
    # Roy et al. (2016), coeficientes OLS usados en LT-GEE: ETM+ = (OLI - b) / m
    slopes = ee.Image.constant([0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949])
    itcp = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029])
    out = img.select(BANDS).subtract(itcp).divide(slopes).rename(BANDS).toFloat()
    return ee.Image(out.copyProperties(img, ["system:time_start"]))


def landsat_collection(aoi, start, end):
    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(lambda i: _mask_scale(i, TM_BANDS))
    # Landsat 7 hasta 2021: después deriva de órbita; SLC-off lo resuelve la mediana
    l7 = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        .filterDate("1999-01-01", "2022-01-01")
        .map(lambda i: _mask_scale(i, TM_BANDS))
    )
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(
        lambda i: _oli_to_etm(_mask_scale(i, OLI_BANDS))
    )
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").map(
        lambda i: _oli_to_etm(_mask_scale(i, OLI_BANDS))
    )
    return l5.merge(l7).merge(l8).merge(l9).filterBounds(aoi).filterDate(start, end)


def annual_nbr(aoi):
    """Colección anual: banda 'NBR_inv' = -NBR x 1000 (pérdida = aumento, como pide LT)."""
    empty = ee.Image.constant([0] * len(BANDS)).rename(BANDS).toFloat().updateMask(ee.Image(0))
    images = []
    for year in range(START_YEAR, END_YEAR + 1):
        start = ee.Date.fromYMD(year - 1, SEASON_START_MONTH, 1)
        end = ee.Date.fromYMD(year, SEASON_END_MONTH, 1).advance(1, "month")
        comp = (
            landsat_collection(aoi, start, end)
            .merge(ee.ImageCollection([empty]))
            .median()
        )
        nbr = comp.normalizedDifference(["nir", "swir2"]).multiply(-1000).toInt16()
        images.append(
            nbr.rename("NBR_inv").set("system:time_start", ee.Date.fromYMD(year, 2, 1).millis())
        )
    return ee.ImageCollection(images)


# --------------------------------------------------------------------------
# LandTrendr y extracción de cambios
# --------------------------------------------------------------------------
def run_landtrendr(aoi):
    params = dict(LT_PARAMS, timeSeries=annual_nbr(aoi))
    return ee.Algorithms.TemporalSegmentation.LandTrendr(**params)


def segment_info(lt):
    """Matriz 6 x nSeg: yod, end, mag, dur, preval, rate (espacio NBR normal, x1000)."""
    arr = lt.select("LandTrendr")
    vertices = arr.arrayMask(arr.arraySlice(0, 3, 4))
    left = vertices.arraySlice(1, 0, -1)
    right = vertices.arraySlice(1, 1, None)
    start_year = left.arraySlice(0, 0, 1)
    end_year = right.arraySlice(0, 0, 1)
    start_val = left.arraySlice(0, 2, 3)  # ajustado, espacio invertido
    end_val = right.arraySlice(0, 2, 3)
    dur = end_year.subtract(start_year)
    mag = end_val.subtract(start_val)     # > 0 = pérdida de vegetación
    rate = mag.divide(dur)
    return ee.Image.cat(
        [start_year.add(1), end_year, mag, dur, start_val.multiply(-1), rate]
    ).toArray(0)


def greatest_disturbance(lt):
    seg = segment_info(lt)
    rmse = lt.select("rmse")
    mag = seg.arraySlice(0, 2, 3)
    if MAX_DUR is not None:
        # declinaciones lentas (dur > MAX_DUR) no compiten; siguen en los segmentos
        abrupt = seg.arraySlice(0, 3, 4).lte(MAX_DUR)  # 1/0; where() no acepta arrays
        mag = mag.multiply(abrupt).add(abrupt.subtract(1).multiply(1e6))
    big = seg.arraySort(mag.multiply(-1)).arraySlice(1, 0, 1)  # mag descendente
    names = ["yod", "end", "mag", "dur", "preval", "rate"]
    img = big.arrayProject([0]).arrayFlatten([names])
    img = img.addBands(img.select("mag").divide(rmse).multiply(100).rename("dsnr"))

    keep = img.select("mag").gte(MIN_MAG)
    if MAX_DUR is not None:
        keep = keep.And(img.select("dur").lte(MAX_DUR))  # píxeles sin segmento abrupto
    if MIN_PREVAL is not None:
        keep = keep.And(img.select("preval").gte(MIN_PREVAL))
    img = img.updateMask(keep)
    if MMU:
        patch = img.select("yod").toInt16().connectedPixelCount(MMU, True)
        img = img.updateMask(patch.gte(MMU))
    return img.select(["yod", "mag", "dur", "preval", "rate", "dsnr"]).toInt16()


def all_segments(lt):
    n = LT_PARAMS["maxSegments"]
    seg = segment_info(lt).toFloat()
    pad = ee.Image(ee.Array([[0.0] * n] * 6)).toFloat()
    fixed = seg.arrayCat(pad, 1).arraySlice(1, 0, n)
    rows = ["yod", "end", "mag", "dur", "preval", "rate"]
    cols = [f"s{k}" for k in range(1, n + 1)]
    img = fixed.arrayFlatten([rows, cols])  # nombres: yod_s1, ..., rate_s6
    keep = [f"{r}_{c}" for c in cols for r in ["yod", "end", "mag", "dur", "preval"]]
    return img.select(keep).toInt16()


# --------------------------------------------------------------------------
# Regiones y exportación
# --------------------------------------------------------------------------
def ascii_name(name):
    name = re.sub(r"^Regi[oó]n (De(l)? )?", "", name)
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


MAINLAND = [-76.5, -56.5, -66.0, -28.5]  # corta Isla de Pascua, Juan Fernández y Antártica


def chile_regions():
    """Regiones GAUL 2025 nivel 1 con centroide al sur de 29.2 S (Coquimbo a Magallanes).

    Devuelve (nombre, FeatureCollection de la región). Las geometrías costeras del sur
    son muy complejas, así que no se unen ni intersectan: se simplifican para elegir
    teselas y se rasterizan (paint) para enmascarar.
    """
    fc = ee.FeatureCollection("FAO/GAUL/2025/level1").filter(ee.Filter.eq("ISO3_CODE", "CHL"))
    fc = fc.map(
        lambda f: f.set("lat", f.geometry().centroid(1000).coordinates().get(1))
    ).filter(ee.Filter.lt("lat", -29.2))
    names = fc.aggregate_array("GAUL1_NAME").distinct().getInfo()
    return [(name, fc.filter(ee.Filter.eq("GAUL1_NAME", name))) for name in names]


def tiles_for(region):
    """Teselas de TILE_DEG grados, dentro de MAINLAND, que tocan la región."""
    xmin, ymin, xmax, ymax = MAINLAND
    rects = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            rects.append(ee.Feature(ee.Geometry.Rectangle(
                [x, y, min(x + TILE_DEG, xmax), min(y + TILE_DEG, ymax)], None, False)))
            y += TILE_DEG
        x += TILE_DEG
    simple = region.map(lambda f: f.simplify(1000))
    hits = ee.FeatureCollection(rects).filterBounds(simple.geometry(1000))
    return [ee.Geometry(g, None, False) for g in hits.geometry().geometries().getInfo()]


def region_mask(region):
    return ee.Image(0).paint(region, 1)


def export(img, name, region):
    task = ee.batch.Export.image.toDrive(
        image=img.unmask(NODATA).toInt16(),
        description=name[:100],
        folder=DRIVE_FOLDER,
        fileNamePrefix=name,
        region=region,
        scale=SCALE,
        crs=CRS,
        maxPixels=1e13,
        formatOptions={"cloudOptimized": True},
    )
    task.start()
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="proyecto Cloud con Earth Engine")
    ap.add_argument("--regiones", nargs="*", help="filtrar por nombre (sin tildes, parcial)")
    ap.add_argument("--max-teselas", type=int, help="teselas por región (piloto)")
    ap.add_argument("--dry-run", action="store_true", help="listar teselas sin lanzar tareas")
    args = ap.parse_args()

    ee.Initialize(project=args.project)

    regions = chile_regions()
    if args.regiones:
        wanted = [w.lower() for w in args.regiones]
        regions = [(n, r) for n, r in regions if any(w in ascii_name(n).lower() for w in wanted)]

    # no relanzar lo que ya está hecho o en cola (Drive duplicaría archivos)
    done = {
        t["description"] for t in ee.data.getTaskList()
        if t["state"] in ("COMPLETED", "READY", "RUNNING")
    }

    n_tasks = 0
    for name, region in regions:
        tag = ascii_name(name)
        tiles = tiles_for(region)[: args.max_teselas]
        mask = region_mask(region)
        print(f"{name}: {len(tiles)} teselas")
        if args.dry_run:
            continue
        for k, tile in enumerate(tiles, 1):
            lt = run_landtrendr(tile)
            prefix = f"LT_{tag}_{k:03d}"
            for suffix, img in (("mayor", greatest_disturbance(lt)), ("segmentos", all_segments(lt))):
                if f"{prefix}_{suffix}" in done:
                    continue
                export(img.updateMask(mask), f"{prefix}_{suffix}", tile)
                n_tasks += 1
    print(f"{n_tasks} tareas lanzadas a Drive/{DRIVE_FOLDER}")


if __name__ == "__main__":
    main()
