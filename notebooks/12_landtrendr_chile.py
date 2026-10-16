#!/usr/bin/env python3
# ---
# Source for `notebooks/12_landtrendr_chile.ipynb`.
#
#     python notebooks/build_notebook.py notebooks/12_landtrendr_chile.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # LandTrendr Chile: carga, unificación y análisis
#
# Lee los trozos GeoTIFF que descargó `scripts/gee_landtrendr_download.py` (LandTrendr de
# Google Earth Engine sobre NBR anual de verano, Landsat 5/7/8/9, 2000-2026, Coquimbo a
# Magallanes), los unifica en un mosaico y resume cuándo, dónde y con qué intensidad cambió
# la vegetación.
#
# | | |
# |---|---|
# | método LandTrendr | `scripts/gee_landtrendr_chile.py` |
# | descarga (computePixels) | `scripts/gee_landtrendr_download.py` |
# | datos | `data/landtrendr_chile/chunks/*.tif` (256 x 256 px, 30 m, EPSG:32719) |
# | regiones | `data/landtrendr_chile/regiones_gaul2025.gpkg` (FAO GAUL 2025, simplificadas a 100 m) |
#
# ## Bandas de cada trozo (Int16, sin datos = -32768)
#
# | banda | qué es |
# |---|---|
# | `yod` | año de la mayor perturbación **abrupta** (segmento de <= 4 años) |
# | `mag` | su intensidad: caída de NBR x 1000 (positivo = pérdida de vegetación) |
# | `dur` | duración del segmento (años) |
# | `preval` | NBR x 1000 antes del cambio |
# | `rate` | `mag / dur` |
# | `dsnr` | `mag / RMSE` del ajuste, x 100 |
# | `yod_sK, end_sK, mag_sK, dur_sK, preval_sK` | cada segmento ajustado, K = 1..6, en orden cronológico; `mag < 0` es recuperación, `yod_sK = 0` indica que no hay segmento K |
#
# La mayor perturbación ya viene filtrada en GEE: `mag >= 100` (0.10 de NBR) y una mancha
# mínima de 11 píxeles conectados (~1 ha).
#
# ## Dos cosas a tener en cuenta al leer los resultados
#
# * **Borde inicial de la serie.** 2001 y 2002 tienen más perturbaciones que los años
#   siguientes. Es el artefacto típico de LandTrendr al inicio de la serie: el primer
#   vértice no tiene años previos que lo anclen. Esos dos años van sombreados en las figuras.
# * **Solo cambios abruptos en `yod`/`mag`.** Las declinaciones lentas (> 4 años) no
#   compiten por la mayor perturbación; siguen disponibles en las bandas `_sK`.

# %%
import re
import subprocess
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "data" / "landtrendr_chile"
CHUNKS = DATA / "chunks"
OUT = DATA / "analisis"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

NODATA = -32768
CHUNK = 256
SCALE = 30
SIZE = CHUNK * SCALE          # 7680 m por trozo
CRS = "EPSG:32719"
YEARS = np.arange(2001, 2027)  # yod posible: año siguiente al primer vértice (2000) .. 2026
EDGE_YEARS = (2001, 2002)
N_SEG = 6
BAND_NAMES = ["yod", "mag", "dur", "preval", "rate", "dsnr"] + [
    f"{r}_s{k}" for k in range(1, N_SEG + 1) for r in ["yod", "end", "mag", "dur", "preval"]
]
B = {n: i for i, n in enumerate(BAND_NAMES)}

# umbrales para contar perturbaciones abruptas en los segmentos (mismos que la mayor)
MIN_MAG, MAX_DUR = 100, 4
MAG_BINS = np.arange(100, 2050, 50)      # histograma de magnitud (NBR x 1000)
PRE_BINS = np.arange(-1000, 1050, 50)    # histograma de NBR previo
COARSE = 32                              # bloque para mapas gruesos: 32 px = 960 m
FORCE = False                            # True recalcula las estadísticas aunque existan

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# %% [markdown]
# ## 1. Carga
#
# Cada trozo se llama `LT_<i>_<j>.tif`, donde `(i, j)` es su posición en una grilla de
# 7680 m alineada al origen de EPSG:32719. Los trozos sin tierra dentro de las regiones
# quedaron como `.empty`. Esa grilla permite ubicar cada trozo sin abrirlo.

# %%
def parse_ij(p):
    i, j = re.match(r"LT_([+-]\d+)_([+-]\d+)", p.stem).groups()
    return int(i), int(j)

tifs = sorted(CHUNKS.glob("*.tif"))
empties = sorted(CHUNKS.glob("*.empty"))
chunks = pd.DataFrame([(p, *parse_ij(p)) for p in tifs], columns=["path", "i", "j"])
print(f"{len(tifs)} trozos con datos + {len(empties)} vacíos = {len(tifs) + len(empties)}")
print(f"grilla i: {chunks.i.min()}..{chunks.i.max()}, j: {chunks.j.min()}..{chunks.j.max()}")
print(f"disco: {sum(p.stat().st_size for p in tifs) / 1e9:.1f} GB")

# %%
regions = gpd.read_file(DATA / "regiones_gaul2025.gpkg").to_crs(CRS)
regions["norte"] = regions.geometry.centroid.y      # en UTM, para ordenar de norte a sur
regions = regions.sort_values("norte", ascending=False).reset_index(drop=True)
regions["rid"] = np.arange(1, len(regions) + 1)     # 0 = fuera de región
SHORT = {
    "Coquimbo": "Coquimbo", "Valparaiso": "Valparaíso", "Metropolitana_De_Santiago": "Metropolitana",
    "Libertador_Bernardo_O_Higgins": "O'Higgins", "Maule": "Maule", "Nuble": "Ñuble",
    "Bio_Bio": "Biobío", "La_Araucania": "Araucanía", "Los_Rios": "Los Ríos",
    "Los_Lagos": "Los Lagos", "Aysen_Del_Gral_Ibanez_Del_Campo": "Aysén",
    "Magallanes_Y_Antartica_Chilena": "Magallanes",
}
regions["corto"] = regions.region.map(SHORT)
NAMES = regions.corto.tolist()
NR = len(regions)
regions[["rid", "corto", "nombre"]]

# %% [markdown]
# Un trozo cualquiera, para ver que las bandas son lo que dicen ser.

# %%
demo = chunks.iloc[len(chunks) // 2].path
with rasterio.open(demo) as src:
    print(demo.name, src.crs, src.res, src.count, src.dtypes[0], src.nodata)
    print(src.descriptions[:6], "...", src.descriptions[-5:])
    arr = src.read()

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, band, cmap in zip(axs, ["yod", "mag", "preval"], ["viridis", "magma", "YlGn"]):
    a = np.ma.masked_equal(arr[B[band]], NODATA)
    im = ax.imshow(a, cmap=cmap)
    ax.set_title(band); ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()

# %% [markdown]
# ## 2. Unificación
#
# ### 2a. Mosaico virtual (VRT)
#
# Un VRT referencia los 10 mil trozos sin copiarlos: se abre en QGIS o rasterio como un solo
# raster de 36 bandas. Son demasiados archivos para la línea de comandos, así que
# `gdalbuildvrt` los recibe por lista. `gdalbuildvrt` no copia los nombres de banda; se
# escriben después.

# %%
VRT = DATA / "LT_Chile.vrt"
if not VRT.exists() or FORCE:
    lst = DATA / "chunks_list.txt"
    lst.write_text("\n".join(str(p) for p in tifs))
    subprocess.run(["gdalbuildvrt", "-q", "-a_srs", CRS, "-input_file_list", str(lst), str(VRT)],
                   check=True)
    with rasterio.open(VRT, "r+") as dst:
        for b, n in enumerate(BAND_NAMES, 1):
            dst.set_band_description(b, n)
with rasterio.open(VRT) as src:
    print(VRT.name, src.width, "x", src.height, "px,", src.count, "bandas,",
          f"{src.width * src.height * 900 / 1e10:.1f} M ha de extensión")

# %% [markdown]
# ### 2b. Una pasada por todos los trozos
#
# Todo lo que sigue sale de una sola lectura de cada trozo, en paralelo. Por trozo se
# rasteriza la región de cada píxel y se acumulan:
#
# * píxeles válidos por región (el denominador de los porcentajes);
# * mayor perturbación por región y año: píxeles, histograma de magnitud, de duración y de NBR previo;
# * cuántas perturbaciones abruptas tuvo cada píxel en toda la serie (segmentos con
#   `mag >= 100` y `dur <= 4`), y en qué años, para no quedarse solo con la mayor;
# * si al segmento de la mayor perturbación le sigue uno de recuperación;
# * mapas gruesos de 960 m: fracción perturbada, año mediano y magnitud media.
#
# El resultado se guarda en `analisis/stats.npz`; volver a correr el notebook lo reutiliza
# salvo que `FORCE = True`.

# %%
NY, NMAG, NPRE = len(YEARS), len(MAG_BINS) - 1, len(PRE_BINS) - 1
NC = CHUNK // COARSE


def chunk_stats(paths_ij, geoms):
    acc = dict(
        valid=np.zeros(NR + 1, np.int64),
        dist=np.zeros((NR + 1, NY), np.int64),
        mag_hist=np.zeros((NR + 1, NY, NMAG), np.int64),
        dur_hist=np.zeros((NR + 1, MAX_DUR + 1), np.int64),
        pre_hist=np.zeros((NR + 1, NPRE), np.int64),
        recov=np.zeros((NR + 1, NY), np.int64),
        n_events=np.zeros((NR + 1, N_SEG + 1), np.int64),
        seg_events=np.zeros((NR + 1, NY), np.int64),
    )
    coarse = []
    for path, i, j in paths_ij:
        with rasterio.open(path) as src:
            a = src.read()
        tr = from_origin(i * SIZE, (j + 1) * SIZE, SCALE, SCALE)
        cbox = box(i * SIZE, j * SIZE, (i + 1) * SIZE, (j + 1) * SIZE)
        shapes = [(g.intersection(cbox), rid) for g, rid in geoms if g.intersects(cbox)]
        rid = (rasterize(shapes, out_shape=(CHUNK, CHUNK), transform=tr, fill=0, dtype="uint8")
               if shapes else np.zeros((CHUNK, CHUNK), np.uint8))

        valid = a[B["yod_s1"]] != NODATA
        rid = np.where(valid, rid, 0)
        acc["valid"] += np.bincount(rid.ravel(), minlength=NR + 1)

        yod, mag, dur, pre = (a[B[k]] for k in ("yod", "mag", "dur", "preval"))
        d = (yod != NODATA) & (rid > 0)
        if d.any():
            r, y = rid[d], yod[d] - YEARS[0]
            np.add.at(acc["dist"], (r, y), 1)
            mb = np.clip(np.digitize(mag[d], MAG_BINS) - 1, 0, NMAG - 1)
            np.add.at(acc["mag_hist"], (r, y, mb), 1)
            np.add.at(acc["dur_hist"], (r, np.clip(dur[d], 0, MAX_DUR)), 1)
            pb = np.clip(np.digitize(pre[d], PRE_BINS) - 1, 0, NPRE - 1)
            np.add.at(acc["pre_hist"], (r, pb), 1)

            # ¿el segmento siguiente al de la mayor perturbación es de recuperación?
            rec = np.zeros(d.sum(), bool)
            yd = yod[d]
            for k in range(1, N_SEG):
                this = a[B[f"yod_s{k}"]][d] == yd
                nxt = a[B[f"mag_s{k + 1}"]][d] < 0
                rec |= this & nxt
            np.add.at(acc["recov"], (r[rec], y[rec]), 1)

        # todas las perturbaciones abruptas de la serie, segmento por segmento
        n_ev = np.zeros((CHUNK, CHUNK), np.int8)
        for k in range(1, N_SEG + 1):
            sy, sm, sd = a[B[f"yod_s{k}"]], a[B[f"mag_s{k}"]], a[B[f"dur_s{k}"]]
            ev = valid & (rid > 0) & (sy > 0) & (sm >= MIN_MAG) & (sd <= MAX_DUR)
            n_ev += ev
            if ev.any():
                np.add.at(acc["seg_events"], (rid[ev], np.clip(sy[ev], YEARS[0], YEARS[-1]) - YEARS[0]), 1)
        inr = valid & (rid > 0)
        np.add.at(acc["n_events"], (rid[inr], n_ev[inr]), 1)

        # mapas gruesos 960 m
        blk = lambda x: x.reshape(NC, COARSE, NC, COARSE).swapaxes(1, 2).reshape(NC, NC, -1)
        nv = blk(inr).sum(-1)
        nd = blk(d).sum(-1)
        yod_f = np.where(d, yod, np.nan).astype("float32")
        mag_f = np.where(d, mag, np.nan).astype("float32")
        with warnings.catch_warnings():  # bloques sin perturbación: nanmedian de todo NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            coarse.append((i, j, nv, nd, np.nanmedian(blk(yod_f), -1), np.nanmean(blk(mag_f), -1)))
    return acc, coarse


STATS = OUT / "stats.npz"
if STATS.exists() and not FORCE:
    z = np.load(STATS)
    acc = {k: z[k] for k in z.files if not k.startswith("c_")}
    grid = {k[2:]: z[k] for k in z.files if k.startswith("c_")}
    print("estadísticas cargadas de", STATS)
else:
    geoms = list(zip(regions.geometry, regions.rid))
    items = list(zip(chunks.path, chunks.i, chunks.j))
    batches = [items[k:k + 100] for k in range(0, len(items), 100)]
    res = Parallel(n_jobs=-1, verbose=5)(delayed(chunk_stats)(b, geoms) for b in batches)

    acc = {k: sum(r[0][k] for r in res) for k in res[0][0]}
    # ensamblar la grilla gruesa nacional (fila 0 = norte)
    i0, j1 = chunks.i.min(), chunks.j.max()
    W = (chunks.i.max() - i0 + 1) * NC
    H = (j1 - chunks.j.min() + 1) * NC
    grid = {k: np.full((H, W), np.nan, "float32") for k in ("nvalid", "ndist", "yod_med", "mag_mean")}
    for _, cs in res:
        for i, j, nv, nd, ym, mm in cs:
            r0, c0 = (j1 - j) * NC, (i - i0) * NC
            for k, v in zip(grid, (nv, nd, ym, mm)):
                grid[k][r0:r0 + NC, c0:c0 + NC] = v
    grid["origin"] = np.array([i0 * SIZE, (j1 + 1) * SIZE], "float64")
    np.savez_compressed(STATS, **acc, **{f"c_{k}": v for k, v in grid.items()})
    print("guardado", STATS)

print({k: v.shape for k, v in acc.items()})

# %% [markdown]
# ### 2c. Productos unificados livianos
#
# Los mapas gruesos de 960 m se escriben como GeoTIFF para abrirlos en un SIG junto al VRT
# de 30 m: fracción de píxeles con perturbación abrupta, año mediano y magnitud media.

# %%
def write_coarse(name, arr, dtype="float32"):
    tr = from_origin(grid["origin"][0], grid["origin"][1], SCALE * COARSE, SCALE * COARSE)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                crs=CRS, transform=tr, nodata=np.nan, compress="deflate")
    with rasterio.open(OUT / f"{name}.tif", "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)

with np.errstate(invalid="ignore", divide="ignore"):
    frac = np.where(grid["nvalid"] > 0, grid["ndist"] / grid["nvalid"], np.nan)
write_coarse("frac_perturbado_960m", frac)
write_coarse("yod_mediano_960m", grid["yod_med"])
write_coarse("mag_media_960m", grid["mag_mean"])
print(sorted(p.name for p in OUT.glob("*.tif")))

# %% [markdown]
# Un recorte de 30 m por región, como GeoTIFF independiente (solo las 6 bandas de la mayor
# perturbación), sirve para compartir una región sin el VRT. Escribir las 12 regiones lleva
# tiempo y varios GB, así que queda apagado: poner `WRITE_REGION_TIFS = True`.

# %%
WRITE_REGION_TIFS = False
if WRITE_REGION_TIFS:
    (OUT / "regiones").mkdir(exist_ok=True)
    for _, reg in regions.iterrows():
        cut = OUT / "regiones" / f"{reg.region}.gpkg"
        gpd.GeoDataFrame(geometry=[reg.geometry], crs=CRS).to_file(cut)
        subprocess.run(["gdalwarp", "-q", "-overwrite", "-cutline", str(cut), "-crop_to_cutline",
                        *sum([["-b", str(b)] for b in range(1, 7)], []),
                        "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", "-co", "BIGTIFF=YES",
                        str(VRT), str(OUT / "regiones" / f"LT_mayor_{reg.region}.tif")], check=True)
        print(reg.corto)

# %% [markdown]
# ## 3. Análisis
#
# Toda superficie se calcula como píxeles x 0.09 ha. Los porcentajes son sobre los píxeles
# válidos de cada región (con serie LandTrendr), no sobre su superficie administrativa.

# %%
HA = 0.09
valid_ha = acc["valid"][1:] * HA
dist_ha = acc["dist"][1:] * HA                     # región x año
seg_ha = acc["seg_events"][1:] * HA

summary = pd.DataFrame({
    "región": NAMES,
    "superficie válida (kha)": valid_ha / 1e3,
    "con perturbación (kha)": dist_ha.sum(1) / 1e3,
    "% perturbado": 100 * dist_ha.sum(1) / valid_ha,
    "% perturbado sin 2001-02": 100 * dist_ha[:, 2:].sum(1) / valid_ha,
    "eventos abruptos totales (kha)": seg_ha.sum(1) / 1e3,
})
tot = summary.select_dtypes("number").sum()
tot["% perturbado"] = 100 * dist_ha.sum() / valid_ha.sum()
tot["% perturbado sin 2001-02"] = 100 * dist_ha[:, 2:].sum() / valid_ha.sum()
summary.loc[len(summary)] = ["Total", *tot.values]
summary.to_csv(OUT / "resumen_regiones.csv", index=False)
summary.round(1)

# %% [markdown]
# ### 3a. Superficie perturbada por año
#
# Año de la mayor perturbación abrupta de cada píxel, apilado por región. Cada píxel cuenta
# una sola vez. La línea negra cuenta en cambio **todas** las perturbaciones abruptas de
# todos los segmentos, así un píxel cortado en 2008 y quemado en 2017 suma en ambos años.

# %%
colors = plt.cm.tab20(np.linspace(0, 1, NR))
fig, ax = plt.subplots(figsize=(11, 4.5))
bottom = np.zeros(NY)
for r in range(NR):
    ax.bar(YEARS, dist_ha[r] / 1e3, bottom=bottom, color=colors[r], label=NAMES[r], width=0.85)
    bottom += dist_ha[r] / 1e3
ax.plot(YEARS, seg_ha.sum(0) / 1e3, "k.-", lw=1.2, label="todos los eventos abruptos")
ax.axvspan(EDGE_YEARS[0] - 0.5, EDGE_YEARS[1] + 0.5, color="0.85", zorder=-1)
ax.text(2001.5, ax.get_ylim()[1] * 0.95, "borde de\nla serie", ha="center", va="top", fontsize=8)
ax.set_ylabel("superficie (miles de ha)")
ax.set_title("Mayor perturbación abrupta por año, Coquimbo a Magallanes")
ax.legend(ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout(); plt.savefig(FIGS / "area_por_anio.png", dpi=200)

# %% [markdown]
# ### 3b. Región x año
#
# Porcentaje de la superficie válida de cada región perturbada en cada año. Normalizar por
# región evita que Aysén y Magallanes, las más grandes, dominen la figura.

# %%
pct = 100 * dist_ha / valid_ha[:, None]
fig, ax = plt.subplots(figsize=(11, 4.8))
im = ax.imshow(pct, aspect="auto", cmap="YlOrRd", vmax=np.nanpercentile(pct, 99))
ax.set_yticks(range(NR), NAMES)
ax.set_xticks(range(0, NY, 2), YEARS[::2])
for k, y in enumerate(YEARS):
    if y in EDGE_YEARS:
        ax.axvline(k, color="0.5", lw=0.5, ls=":")
fig.colorbar(im, ax=ax, label="% de la superficie válida")
ax.set_title("Superficie con su mayor perturbación en cada año (regiones de norte a sur)")
plt.tight_layout(); plt.savefig(FIGS / "region_x_anio.png", dpi=200)
pd.DataFrame(pct, index=NAMES, columns=YEARS).round(3).to_csv(OUT / "pct_region_anio.csv")

# %% [markdown]
# ### 3c. Intensidad
#
# Magnitud (caída de NBR) de la mayor perturbación: por año a nivel nacional (mediana y
# rango intercuartil, desde los histogramas) y por región.

# %%
centers = (MAG_BINS[:-1] + MAG_BINS[1:]) / 2 / 1000   # en unidades de NBR

def hist_quantiles(h, qs=(0.25, 0.5, 0.75)):
    c = np.cumsum(h, -1)
    tot = c[..., -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = c / tot
    return np.stack([centers[np.argmax(f >= q, -1)] for q in qs], -1).astype(float) * \
        np.where(tot > 0, 1, np.nan)

q_year = hist_quantiles(acc["mag_hist"][1:].sum(0))          # año x 3
q_reg = hist_quantiles(acc["mag_hist"][1:].sum(1))           # región x 3

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
a1.fill_between(YEARS, q_year[:, 0], q_year[:, 2], color="tab:red", alpha=0.25, label="P25-P75")
a1.plot(YEARS, q_year[:, 1], "o-", color="tab:red", ms=3, label="mediana")
a1.axvspan(EDGE_YEARS[0] - 0.5, EDGE_YEARS[1] + 0.5, color="0.85", zorder=-1)
a1.set_ylabel("magnitud (caída de NBR)"); a1.set_title("Intensidad por año"); a1.legend(frameon=False)
yy = np.arange(NR)[::-1]
a2.hlines(yy, q_reg[:, 0], q_reg[:, 2], color="tab:red", lw=4, alpha=0.35)
a2.plot(q_reg[:, 1], yy, "o", color="tab:red")
a2.set_yticks(yy, NAMES); a2.set_xlabel("magnitud (caída de NBR)"); a2.set_title("Intensidad por región")
plt.tight_layout(); plt.savefig(FIGS / "intensidad.png", dpi=200)

# %% [markdown]
# ### 3d. Duración, vegetación previa y recuperación
#
# * Duración del segmento de mayor perturbación (1 a 4 años, por el filtro de abruptos).
# * NBR antes del cambio: separa pérdidas sobre vegetación densa (bosque, plantación) de
#   cambios sobre vegetación rala.
# * Fracción de píxeles cuyo segmento siguiente a la perturbación es de recuperación (NBR
#   subiendo). Los años finales tienen poca serie posterior para mostrarla.

# %%
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
dh = acc["dur_hist"][1:].sum(0)[1:]
axs[0].bar(range(1, MAX_DUR + 1), 100 * dh / dh.sum(), color="tab:blue")
axs[0].set_xlabel("duración (años)"); axs[0].set_ylabel("% de píxeles perturbados")
axs[0].set_title("Duración")

pre_c = (PRE_BINS[:-1] + PRE_BINS[1:]) / 2 / 1000
for r in range(NR):
    h = acc["pre_hist"][1 + r]
    if h.sum():
        axs[1].plot(pre_c, h / h.sum(), color=colors[r], lw=1, label=NAMES[r])
axs[1].set_xlabel("NBR antes del cambio"); axs[1].set_title("Vegetación previa")
axs[1].legend(fontsize=6, ncol=2, frameon=False)

with np.errstate(invalid="ignore", divide="ignore"):
    rec = acc["recov"][1:].sum(0) / acc["dist"][1:].sum(0)
axs[2].plot(YEARS, 100 * rec, "o-", color="tab:green", ms=3)
axs[2].axvspan(EDGE_YEARS[0] - 0.5, EDGE_YEARS[1] + 0.5, color="0.85", zorder=-1)
axs[2].set_ylabel("% con recuperación posterior"); axs[2].set_title("Recuperación tras la perturbación")
plt.tight_layout(); plt.savefig(FIGS / "duracion_previa_recuperacion.png", dpi=200)

# %% [markdown]
# ### 3e. Frecuencia de perturbación
#
# Cuántas perturbaciones abruptas tuvo cada píxel en 2000-2026, contando todos los
# segmentos. Más de una suele indicar rotación forestal o incendios repetidos.

# %%
ne = acc["n_events"][1:]
ne_pct = 100 * ne / ne.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 4))
left = np.zeros(NR)
cmap = plt.cm.Oranges(np.linspace(0.15, 1, N_SEG + 1))
for k in range(N_SEG + 1):
    ax.barh(NAMES[::-1], ne_pct[::-1, k], left=left[::-1], color=cmap[k],
            label=f"{k}" if k < N_SEG else f"{k}+")
    left += ne_pct[:, k]
ax.set_xlabel("% de la superficie válida"); ax.set_xlim(0, 100)
ax.legend(title="n.º de eventos", ncol=N_SEG + 1, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / "frecuencia.png", dpi=200)
pd.DataFrame(ne_pct, index=NAMES).round(2).to_csv(OUT / "frecuencia_eventos_pct.csv")

# %% [markdown]
# ### 3f. Mapas (960 m)
#
# Fracción de píxeles con perturbación abrupta, año mediano de la mayor perturbación y su
# magnitud media, en bloques de 960 m. Chile es largo y angosto: los tres paneles van lado
# a lado con el mismo eje.

# %%
H, W = frac.shape
ext = [grid["origin"][0], grid["origin"][0] + W * SCALE * COARSE,
       grid["origin"][1] - H * SCALE * COARSE, grid["origin"][1]]
fig, axs = plt.subplots(1, 3, figsize=(12, 15), sharey=True)
panels = [
    (frac, "Fracción perturbada", "YlOrRd", dict(vmin=0, vmax=0.5)),
    (np.where(grid["ndist"] >= 5, grid["yod_med"], np.nan), "Año mediano", "viridis",
     dict(vmin=2001, vmax=2026)),
    (np.where(grid["ndist"] >= 5, grid["mag_mean"] / 1000, np.nan), "Magnitud media (ΔNBR)",
     "magma_r", dict(vmin=0.1, vmax=0.8)),
]
for ax, (arr, title, cmap, kw) in zip(axs, panels):
    im = ax.imshow(arr, extent=ext, cmap=cmap, interpolation="nearest", **kw)
    regions.boundary.plot(ax=ax, color="0.4", lw=0.3)
    ax.set_title(title); ax.set_xticks([])
    fig.colorbar(im, ax=ax, shrink=0.35, pad=0.01)
axs[0].set_ylabel("UTM 19S norte (m)")
plt.tight_layout(); plt.savefig(FIGS / "mapas_960m.png", dpi=200)

# %% [markdown]
# ## Archivos producidos
#
# | archivo | contenido |
# |---|---|
# | `data/landtrendr_chile/LT_Chile.vrt` | mosaico 30 m, 36 bandas con nombre |
# | `analisis/stats.npz` | todas las estadísticas de la pasada (se reutiliza) |
# | `analisis/resumen_regiones.csv` | superficie válida y perturbada por región |
# | `analisis/pct_region_anio.csv` | % perturbado por región y año |
# | `analisis/frecuencia_eventos_pct.csv` | n.º de eventos abruptos por píxel, por región |
# | `analisis/*_960m.tif` | mapas gruesos: fracción perturbada, año mediano, magnitud media |
# | `analisis/figs/*.png` | figuras de este notebook |
