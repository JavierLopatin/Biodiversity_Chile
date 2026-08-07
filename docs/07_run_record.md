# Registro del run final de extracción (2026-08-06 / 2026-08-07)

Este documento describe el run que produjo los predictores usados de aquí en adelante:
curvas fenológicas, métricas LSP y variables topográficas para 1.082 parcelas de
Chile central. Su propósito es que el run pueda **describirse y reproducirse** sin
depender de la sesión en que se ejecutó.

El inventario exacto de archivos, con tamaños y `sha256`, está en
[`docs/run_manifest.json`](run_manifest.json). Este `.md` da el porqué; el `.json` da el qué.

---

## 1. Estado: completo

| Etapa | Script | Salida | Estado |
|---|---|---|---|
| Subset y folds de CV | `01_build_subset.py` | `plots_subset.parquet`, `cv_folds.parquet`, `cv_fold_report.csv`, `filter_cascade.csv`, `groups_*.csv` | ✅ 1.082 parcelas |
| Topografía | `03_extract_topography.py` | `topography/topography.parquet`, `topography_patches.nc` | ✅ 1.082 parcelas, 240 celdas |
| Fenología | `02_extract_phenology.py` | `phenology/*.nc` (1.082), `phenology/manifest.csv` | ✅ 1.082 cubos, 294 cargas |
| LSP re-anclado | `04_recompute_lsp.py` | `lsp_all_auto.parquet`, `lsp_all_by_phase.parquet`, `phase_audit_full.csv`, `lsp_*` dentro de cada `.nc` | ✅ 5.410 filas (1.082 × 5 índices) |
| Figuras | `06_paper_figures.py` | `results/figures/*` (11) | ✅ regeneradas 2026-08-07 |

No hay `scripts/05_*`: la numeración de `scripts/` y la de `docs/` son independientes.

---

## 2. Comandos

Los comandos se reconstruyen desde los defaults de cada script y desde las cabeceras de
los logs; no quedó historial de shell. Donde el log fija un valor (nº de parcelas, celdas,
workers, índice, anclaje) el valor está confirmado; el resto son los defaults, que no se
sobrescribieron.

```bash
# 1. subset espacio-temporal + folds de CV agrupados
python scripts/01_build_subset.py --zip data/20602096.zip --out-dir data/derived
#    lat 30-38S, año >= 1999, ventana 3 años, celda 5 km, k=5, estratificado por richness

# 2. topografía (DEM Copernicus vía datacube, halo 20 px para vecindad)
python scripts/03_extract_topography.py                       # log: logs_topo.log

# 3. fenología: 5 índices, parche 5x5, 52 pasos, reconstructor lineal, 4 workers dask
python scripts/02_extract_phenology.py --hemisphere auto --resume    # log: logs_pheno.log

# 4. re-cómputo de LSP sobre las curvas ya almacenadas (sin volver al cubo)
python scripts/04_recompute_lsp.py --hemisphere auto --write-back    # log: logs_lsp.log
python scripts/04_recompute_lsp.py --hemisphere auto --compare       # log: logs_lsp2.log

# 5. figuras
python scripts/06_paper_figures.py --out-dir results/figures
```

Orden real por timestamps: `01` (06-ago 21:34) → `03` (06-ago 20:51) → `02` (07-ago 04:24)
→ `04 --write-back` (07-ago 04:34–04:44) → `04 --compare` (07-ago 05:05) → `06` (07-ago 13:07–13:13).

---

## 3. Parámetros que definen el run

**Subset** — Parcelas-CL ([10.5281/zenodo.20602096](https://doi.org/10.5281/zenodo.20602096)),
cascada de filtros en `filter_cascade.csv`:

```
0. todas las parcelas            1485
1. + año conocido                1443
2. + latitud 30.0-38.0 S         1254
3. + año >= 1999                 1253
4. + coordenada única            1082   <- n del run
5. (flag) richness >= 2           969
```

**Fenología** — parche 5×5 px, 30 m, ventana **causal** de 3 años `y−2 … y` respecto al
año del censo, `ngs=52` (~semanal), reconstructor `linear`, 4 workers dask.

La ventana causal es la que se ejecutó, confirmado en los datos: `win_end − Year = 0`
para las 1.082 parcelas. `docs/05_data_acquisition.md` §5 discute una ventana centrada en
`y+1` y advierte que mira al futuro; **este run no la usa**, así que esa advertencia no
aplica aquí. La ventana centrada queda como análisis de sensibilidad pendiente, no como
lo ejecutado. `01_build_subset.py` también nota que la centrada no existiría para las
parcelas de 2026.

Índices (`src/biodiv/cube.py:INDEX_NAMES`): `ndvi, evi, kndvi, nbr, savi`.
Bandas: `blue, green, red, nir, swir1, swir2, qa_pixel`.
Productos Landsat C2 L2 SR: `landsat5` (1984–2011), `landsat7` (1999–2022),
`landsat8` (2013–), `landsat9` (2021–); el sensor se trata como estrato.

**Anclaje de fase** — `hemisphere="auto"` (fase armónica por píxel). La justificación
ecológica y la evidencia están en `docs/06_phase_and_2d_transform.md` y en el comentario
largo de `scripts/02_extract_phenology.py`. Resumen: Chile central contiene fenologías de
fase opuesta en la misma escena (esclerófilo primaveral vs. matorral que verdea con las
lluvias de invierno), por lo que una rotación global sesga sistemáticamente a la minoría
invernal, y esa minoría es un tipo de comunidad ecológicamente distinto.

**Rotación 2D** — `doy_order = "trough_anchored"`, `doy_anchor = 108` (no el DOY 183
austral de manual): medido sobre las 1.082 parcelas, el pico mediano cae en DOY 243 y el
valle en DOY 108, así que cortar en 183 metería el borde del arreglo dentro de la estación
de crecimiento.

**Topografía** — parche 5×5 px a 30 m, halo de 20 px para derivadas de vecindad.
Variables: `elevation, slope, aspect, northness, eastness, heat_load, tpi, tri`,
cada una con `_mean` y `_std` sobre el parche.

---

## 4. Salidas

### Cubos por parcela — `data/derived/phenology/{plot_id}.nc` (1.082 archivos, 360 MB)

```
dims        index=5, doy=52, y=5, x=5, time=<nº de observaciones>
data_vars   phenoshape                        curva reconstruida (index, doy, y, x)
            lsp_{sos,pos,eos,vsos,vpos,veos,los,msp,mau,vmsp,vmau,
                 ampl,ios,rog,ros,sw,trough,mos}    18 métricas LSP
            obs_{ndvi,evi,kndvi,nbr,savi}     serie cruda de índices
            obs_band_{blue,green,red,nir,swir1,swir2}
coords      y, x, spatial_ref, doy, index, time, sensor, year
attrs       plot_id, census_year, win_start, win_end, n_obs, max_doy_gap, mean_doy_gap,
            frac_valid_px, n_obs_per_year, n_years_with_obs, patch_y, patch_x, sensors,
            indices_ok, plot_size_m2, n_landsat_px_covered, metadata_id, Location,
            lat, lon, phase_anchoring, doy_order, doy_anchor
```

`phenoshape` y las `obs_*` son **independientes del anclaje**. Solo las `lsp_*` dependen
de él, y por eso `04_recompute_lsp.py` puede re-anclarlas sin volver al datacube.

### Tablas — `data/derived/`

| Archivo | Forma | Contenido |
|---|---|---|
| `plots_subset.parquet` | 1082 × 19 | parcelas del run, con `cell`, `pixel_id`, ventana |
| `lsp_all_auto.parquet` | 5410 × 50 | LSP bajo `auto`; píxel central + `*_mean5x5` |
| `lsp_all_by_phase.parquet` | 4145 × 45 | comparación entre anclajes |
| `topography/topography.parquet` | 1082 × 28 | 8 variables × (valor, mean, std) |
| `topography/topography_patches.nc` | — | parches 5×5 crudos |
| `cv_folds.parquet` | 27050 × 5 | folds de todos los esquemas |
| `cv_fold_report.csv` | — | balance de richness por fold |
| `filter_cascade.csv` | — | cascada de filtros |
| `phase_audit_full.csv` | — | auditoría de anclaje por parcela |
| `phenology/manifest.csv` | 1082 filas | una fila por cubo, con sus atributos |

### Figuras — `results/figures/` (11 archivos)

`fig01_study_area`, `fig02_sampling`, `fig03_predictor_quality`, `fig04_curves_by_index`,
`fig05_seasonal_phase`, `fig06_lsp_distributions`, `fig07_lsp_topography`,
`figS1_example_plot`, `figS2_cv_folds`, `figS3_project_effect`, `figS4_environment_maps`.

Regeneradas el 2026-08-07 13:07–13:13 **después** del `--write-back`. Las versiones
anteriores (04:11–04:15) precedían al re-anclaje y quedaron obsoletas.

---

## 5. Diagnósticos del run

Coherencia del orden estacional (SOS → POS → EOS, se permite un wrap), sobre `auto`:

```
evi 96.9%   kndvi 98.2%   nbr 98.8%   ndvi 96.6%   savi 96.6%
píxeles degenerados: 1.3%
```

⚠️ **Los dos logs de LSP reportan cifras distintas y hay que leerlos en orden.**
`logs_lsp.log` (`--write-back`, 04:48) reporta coherencia ~90% y 0.2% de degenerados;
`logs_lsp2.log` (`--compare`, 05:05) reporta lo de arriba. Son dos etapas distintas del
mismo pipeline: el primero describe el estado **previo** a la rotación, el segundo el
estado **final**. Las cifras válidas para los artefactos tal como están hoy son las de
`logs_lsp2.log`.

El estado final es internamente consistente: se verificó que las `lsp_*` guardadas dentro
de los `.nc` y las columnas de `lsp_all_auto.parquet` coinciden exactamente (p. ej. parcela
32477, ndvi: `sos=175`, `pos=274` en ambos). Las LSP están en DOY verdadero, no en el
índice rotado, aunque la coordenada `doy` del cubo sí esté rotada (empieza en 109).

`--write-back` confirmó: `1082 files updated, 0 failed`.

Coherencia del anclaje **dentro** de cada parcela — el riesgo conocido del anclaje por
píxel. Desviación estándar circular del ancla entre los 25 píxeles, en días:

```
mediana=14   q75=27   q90=43
parcelas con sd circular > 15 d:  2634 (49%)
                        > 30 d:  1130 (21%)
                        > 60 d:   207 (4%)
fracción mediana de píxeles aestacionales: 0.08
fuerza estacional mediana:                 0.258
```

**Consecuencia operativa:** las parcelas con sd circular alta tienen píxeles anclados a
años fenológicos distintos; su ventana 5×5 no es internamente comparable y debe
**marcarse antes** de usarse para aumentación. Los campos para hacerlo están en
`phase_audit_full.csv` y en las coords por píxel de cada cubo.

Topografía, 1.082 parcelas:

```
            elevation   slope   heat_load     tpi
mediana        602.94   14.55        0.98   -0.17
rango        0-3820.5  0-40.3   0.54-1.13   -27.3-35.7
```

(`heat_load` tiene 1.079 valores no nulos: 3 parcelas en terreno plano donde no está definido.)

---

## 6. Entorno

Python 3.12.13. Versiones exactas de las 14 librerías relevantes en
`docs/run_manifest.json` → `environment`. Las principales:

```
numpy 2.3.5   pandas 3.0.3   xarray 2026.4.0   scipy 1.17.1
datacube 1.9.18   rasterio 1.5.0   geopandas 1.1.3   netCDF4 1.7.4
```

Acceso a datos satelitales: Data Cube Chile (EASI/ODC). `usgs-landsat` es un bucket
*requester-pays*; `src/biodiv/cube.py` documenta la configuración de S3 necesaria.

### Dependencias propias

| Repo | Commit usado | Rol |
|---|---|---|
| [PhenoSensing](https://github.com/JavierLopatin/PhenoSensing) | `1a800a8` (2026-05-27) | `PhenoShape`, `PhenoLSP`, `season_phase` |
| [Trait_2DCNN](https://github.com/JavierLopatin/Trait_2DCNN) | `4d2caed` (2026-07-30) | transformaciones 1D→2D, pérdida enmascarada, MAE (aún no usado en este run) |

⚠️ **Discrepancia de ruta, a resolver.** `scripts/02` y `scripts/04` hacen
`sys.path.insert(0, "/home/jovyan/PhenoSensing")` — una ruta absoluta **fuera del repo**.
Esa copia es anterior (mayo 2026) a la copia versionada en `./PhenoSensing` (`1a800a8`).
La única diferencia es `phenosensing/utils.py::_replaceElements`:

- copia en `/home/jovyan/` (la que **corrió**): implementación histórica, búsqueda por
  pertenencia a conjunto, ~O(n³);
- copia en `./PhenoSensing` (`1a800a8`): reescritura de una sola pasada, O(n).

Ambas producen un vector estrictamente creciente sobre entrada ordenada, que es lo único
que el llamador (`_getPheno`) requiere, así que **los resultados no cambian**; la
diferencia es de rendimiento. Aun así, para reproducir el run hay que apuntar el
`sys.path` a una copia equivalente, y lo correcto a futuro es importar desde
`./PhenoSensing` e instalarlo como dependencia declarada.

---

## 7. Advertencias para quien retome esto

1. **`data/derived/phenology_4idx_superseded/`** (353 archivos, 65 MB) es un run previo
   con **4** índices, sin `savi`. Está **superado** por el run de 5 índices. No mezclar.
   Su log es `logs_pheno_4idx_superseded.log`.
2. **La ventana es causal, pese a lo que sugiere `docs/05`.** Ese documento razona sobre
   una ventana centrada en `y+1` y sus riesgos; el código implementó `y−2 … y` y eso es lo
   que corrió. Al leer `docs/05` §5, tener presente que describe una alternativa
   considerada, no este run. La variante centrada sigue pendiente como sensibilidad.
3. **`richness >= 2`** es un *flag*, no un filtro: las 1.082 parcelas incluyen 113 con
   richness < 2. Filtrar en el momento de modelar, no antes.
4. **El anclaje por píxel debe auditarse antes de aumentar** (§5).
5. **Los `.nc` no están versionados en git** (360 MB). Se regeneran con `scripts/02` +
   `scripts/04`, que requieren acceso al datacube. `docs/run_manifest.json` trae el
   `sha256` de cada uno para verificar que una regeneración coincide con este run.
