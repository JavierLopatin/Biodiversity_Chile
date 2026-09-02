# Living_Trees_Chile — series Landsat de 3 años y topografía por parcela

Series temporales Landsat Collection 2 Level-2 de la **ventana causal de 3 años anterior al
censo** de cada una de las **2.021 parcelas** de `data/Living_Trees_Chile.xlsx`, a nivel de
observación, más las **derivadas topográficas del DEM Copernicus GLO-30** para esas mismas
parcelas (`topography/`, §4).

| | |
|---|---:|
| parcelas (sitios) | 2.021 |
| tablas de serie | 22 |
| filas totales | 2.709.402 |
| rango temporal | 2009-01-03 a 2020-12-31 |
| sensores | Landsat 5, 7, 8 |
| variables topográficas | 9 × (centro, media 5×5, sd 5×5) |
| tamaño en disco | 21 MB + 2,3 MB de topografía |
| generado | 2026-08-13 `scripts/35_extract_living_trees.py`; 2026-08-14 `scripts/03_extract_topography.py` |

Generado con `scripts/35_extract_living_trees.py` sobre Data Cube Chile: 1.764 cargas, 6 h 10
min con 16 workers de dask-gateway. La guía completa está en `docs/16_living_trees_extraction.md`.

---

## 1. La unidad es la coordenada, no el `um`

El Excel de origen es **una fila por árbol** (59.408 tallos) con la coordenada de la parcela
repetida en cada uno. Un sitio es una tripleta `(Latitude, Longitude, Date)` → 2.021 sitios.

**No agrupar por `um`.** Las dos particiones no están anidadas: 78 de los 1.943 `um` tienen más
de una coordenada, y 2 coordenadas están compartidas por más de un `um`. Agrupar por `um`
promediaría parcelas que caen en píxeles Landsat distintos. El `um` está en `sites.parquet`
como atributo (`um_ids`), nunca como clave.

Ningún par de sitios comparte píxel de 30 m, así que cada `site_id` tiene su propia serie.

**Se descartaron 46 filas de árbol** (un sitio, `um` 17048, Aysén) que no traían `Date`: sin
año de censo no hay ventana que definir.

## 2. La ventana

`y-2 .. y`, **causal**: no entra información posterior al censo. Con `Date` entre 2011 y 2020,
las ventanas van de 2009–2011 a 2018–2020. Es la misma convención de `plots_subset.parquet`
(Parcelas-CL), así que ambos conjuntos son comparables.

Las 194 parcelas con `Date`=2012 apoyan su tramo 2012 sólo en Landsat 7 SLC-off (L5 terminó en
noviembre de 2011, L8 empezó en abril de 2013). Se ve en `n_obs_per_year` del manifiesto.

---

## 3. Las tablas

### `sites.parquet` — 2.021 filas × 24 columnas

Una fila por parcela. Es la tabla de join: todo lo demás se une por `site_id`.

| columna | tipo | contenido |
|---|---|---|
| `site_id` | str | `LT0000`…`LT2020`, ordenado por `(lat, lon, year)`. **Clave primaria.** |
| `lat`, `lon` | float | grados decimales, WGS84 (EPSG:4326), tal cual vienen del Excel |
| `X`, `Y` | float | metros, WGS84 / UTM 19S (EPSG:32719) |
| `year` | int | año de establecimiento de la parcela (la columna `Date` del Excel) |
| `win_start`, `win_end`, `win_years` | int | `year-2`, `year`, `3` |
| `pixel_id` | str | identificador del píxel Landsat de 30 m; único en las 2.021 filas |
| `cell` | str | celda de 5 km usada para agrupar las cargas al cubo |
| `n_trees` | int | árboles medidos en la parcela |
| `n_um`, `um_ids` | int, str | unidades de muestreo de terreno que caen en esta coordenada |
| `plot_size_m2` | float | `10000/exp`: 250 o 500 m² |
| `region` | str | región administrativa |
| `relief_strip` | str | macroforma del relieve |
| `elevation`, `slope` | int | m s.n.m.; pendiente en % medida en terreno |
| `forest_type` | str | tipo forestal chileno (Donoso 1981) |
| `stand_development`, `stand_development_code` | str | estado de desarrollo del rodal (CONAF) |
| `pft` | str | tipo funcional de planta (Ma et al. 2023) |
| `exp` | int | factor de expansión, árboles/ha representados por cada tallo |

> **Ojo con la superficie.** Las parcelas miden 250–500 m², **menos que un píxel Landsat de
> 900 m²**. Ésa es la razón de que existan las dos lecturas espaciales de abajo.

### `series_<variable>_center.parquet` — 20 tablas, ~114.700 filas cada una

El valor del **píxel que contiene la coordenada**. Lectura estricta.

| columna | tipo | contenido |
|---|---|---|
| `site_id` | str | join contra `sites.parquet` |
| `time` | datetime64 | fecha y hora real de adquisición (no una grilla) |
| `sensor` | str | `landsat5`, `landsat7` o `landsat8` |
| `<variable>` | float32 | el valor, ver la tabla de variables |

### `series_<variable>_mean5x5.parquet` — 20 tablas, ~131.600 filas cada una

La **media de los 25 píxeles** de la ventana de 5×5 (150 × 150 m) centrada en la coordenada,
sobre los píxeles despejados de esa fecha. Da la señal de rodal y amortigua el error de GPS.

Mismas columnas que `center`, más:

| columna | tipo | contenido |
|---|---|---|
| `n_px` | int16 | píxeles despejados de los 25 que entraron en la media (1–25, mediana 25) |

> **`n_px` no es decoración.** Sin él, una media de 1 píxel y una de 25 son indistinguibles.
> Para trabajo exigente, filtrar por `n_px >= 20`.

> **Las tablas `mean5x5` tienen ~15 % más filas que las `center`**, y no es un error: en una
> fecha donde el píxel central quedó enmascarado pero alguno de los 25 estaba despejado, hay
> media 5×5 y no hay valor central.

### Las 11 variables

| variable | rango observado | qué es |
|---|---|---|
| `ndvi` | −1,000 … 1,000 | `(NIR−R)/(NIR+R)`, recortado a [−1, 1] |
| `evi` | −1,000 … 1,000 | `2,5·(NIR−R)/(NIR+6R−7,5B+1)`, recortado a [−1, 1] |
| `kndvi` | 0,000 … 0,762 | `tanh(NDVI²)`. El techo 0,7616 es `tanh(1)`, no un recorte |
| `nbr` | −1,000 … 1,000 | `(NIR−SWIR2)/(NIR+SWIR2)`, recortado a [−1, 1] |
| `savi` | −0,491 … 0,904 | `((NIR−R)/(NIR+R+0,5))·1,5` (Huete 1988, L=0,5) |
| `band_blue` … `band_swir2` | −0,20 … 1,602 | reflectancia de superficie escalada, **sin recortar** |

Las seis bandas se guardan además de los índices porque los índices son *lossy*: SAVI no se
puede reconstruir desde un NDVI guardado. Cualquier índice nuevo (MSAVI, NDMI, NIRv…) es un
cálculo local, sin volver al cubo.

### `manifest.csv` — 2.021 filas

Una fila por sitio extraído, con las covariables de calidad. **Es donde se decide qué sitios
sirven para qué.**

| columna | contenido |
|---|---|
| `site_id`, `X`, `Y`, `lon`, `lat`, `year`, `win_start`, `win_end`, `cell` | idem `sites.parquet` |
| `n_obs_center` | observaciones despejadas en el píxel central |
| `n_obs_mean5x5` | fechas con al menos un píxel despejado en el 5×5 |
| `max_doy_gap` | **el hueco más grande, en días del año.** Importa más que el total |
| `mean_doy_gap` | separación media entre observaciones consecutivas, en días |
| `frac_valid_px` | fracción de celdas (fecha × píxel) despejadas en el cubo 5×5 |
| `n_obs_per_year` | JSON `{año: observaciones}`, para ver si algún año quedó vacío |
| `n_years_with_obs` | años de la ventana con al menos una observación |
| `patch_y`, `patch_x` | tamaño real del parche; 5×5 en los 2.021 sitios |
| `sensors` | sensores presentes en la ventana |
| `error` | motivo de falla. **Vacía en los 2.021 sitios de esta versión** |

### `extraction.json`

Los parámetros exactos de la corrida: argumentos del script, envolvente geográfica, años,
lista de variables y conteos. Es lo que hace la corrida reproducible.

---

## 4. `topography/` — las derivadas del DEM

Las mismas 9 variables topográficas que usa Parcelas-CL, con las mismas definiciones y el
mismo código (`scripts/03_extract_topography.py`), para los 2.021 sitios. Fuente:
`copernicus_dem_30` (Copernicus GLO-30) vía Data Cube Chile, a 30 m, EPSG:32719.

| archivo | contenido |
|---|---|
| `topography/topography.parquet` | 2.021 × 28: `site_id` + 9 variables × (píxel central, `_mean`, `_std` del 5×5) |
| `topography/topography_patches.nc` | parches crudos, dims `(site_id: 2021, py: 5, px: 5)` × 9 variables |

| variable | unidad | qué es |
|---|---|---|
| `elevation` | m | altura sobre el nivel del mar |
| `slope` | grados | pendiente |
| `aspect` | grados | orientación, 0 = norte, horario, **hacia donde mira la ladera** |
| `northness`, `eastness` | −1…1 | `cos`/`sin` del aspecto; continuas, sin la discontinuidad 0/360 |
| `heat_load` | índice | McCune & Keon (2002), plegado en 315° para el hemisferio sur (máximo en laderas NW) |
| `tpi` | m | posición topográfica: cota menos la media de la vecindad (7×7 px) |
| `tri` | m | rugosidad de Riley sobre los 8 vecinos |
| `curvature` | m⁻¹ | curvatura total (laplaciano); positiva = divergente |

**Las derivadas se calculan sobre una ventana ancha y *después* se recorta el 5×5.** Pendiente,
aspecto y curvatura son operadores de vecindad: calcularlos sobre un recorte de 5×5 dejaría la
mitad de los píxeles con efecto de borde. El halo es de 20 px por lado (750 m con el parche).

**`aspect` apunta cuesta abajo**, verificado contra `gdaldem aspect` (correlación circular
0,9993 sobre 38.435 píxeles de terreno andino real; `tests/test_topography.py` fija la
convención con planos sintéticos). Esto importa: `data/derived/topography/` (Parcelas-CL) se
generó con la convención **antigua**, girada 180°, y por lo tanto con `northness`, `eastness`
y el término de aspecto de `heat_load` de signo invertido. **Las dos tablas no son comparables
hasta que Parcelas-CL se vuelva a extraer** — ver el aviso en `docs/07_run_record.md` §3.

**14 sitios con `NaN` en `aspect`, `northness`, `eastness` y `heat_load`**: pendiente menor a
0,5°, donde la orientación no está definida y sería el ruido de redondeo del DEM. Las cuatro
columnas fallan juntas, en los mismos 14 sitios; `elevation`, `slope`, `tpi`, `tri` y
`curvature` están completas en los 2.021.

**Contraste con el terreno** (el Excel trae `Elevation` y `Slope` medidos en campo, que quedan
en `sites.parquet`; no se sobrescriben, se pueden cruzar):

| | r | mediana de la diferencia |
|---|---:|---:|
| `elevation` DEM vs campo | 0,958 | +2,1 m (MAE 40 m; 98 sitios con \|dif\| > 200 m) |
| `slope` DEM vs campo | 0,620 | +6,3 puntos porcentuales |

**Las dos pendientes no están en la misma unidad**: la del Excel es un **porcentaje**
(0–140 %, según su propia hoja `metadata`), la del DEM son **grados**. La comparación de
arriba convierte la del DEM con `tan(slope)·100`; la `r` no depende de la unidad. Que
concuerden menos que las elevaciones es esperable: la del Excel es la pendiente de la parcela
de 250–500 m², la del DEM es la de un píxel de 900 m² derivada de su vecindad de 30 m.

La elevación es la que valida las coordenadas y el CRS, y con r = 0,958 los valida.

---

## 5. Tres advertencias de uso

### 5.1 Las observaciones caen a un tercio hacia el sur

Mediana de observaciones por sitio en la ventana de 3 años:

| banda de latitud | sitios | obs. mediana |
|---|---:|---:|
| −30° | 22 | 103 |
| −35° | 381 | 72 |
| −40° | 993 | 71 |
| −45° | 459 | 46 |
| −50° | 125 | 35 |
| −55° | 41 | 33 |

Global: mediana 63, mínimo 7, máximo 186.

No es un defecto de la extracción, es nubosidad austral, pero **cambia qué se puede modelar**.
Con 33 observaciones en 3 años —unas 11 por año, con huecos estacionales grandes— una curva
fenológica no queda determinada. Los sitios al sur de −45° hay que tratarlos como estrato
aparte, filtrando por `n_obs_mean5x5` y `max_doy_gap`, no promediarlos con el resto.

El sesgo es peor de lo que sugiere el conteo: en Chile la nubosidad se concentra al inicio de
la estación de crecimiento, así que la falta de datos **no es aleatoria respecto de la métrica
que se quiere estimar** (SOS en particular).

### 5.2 Las bandas no están recortadas; los índices sí

`band_*` llega a 1,602, que es exactamente el techo del entero C2 (65.455 × 0,0000275 − 0,2):
son píxeles saturados de nieve que pasaron la máscara, porque sólo se descarta
`snow="not_high_confidence"`. Son raros —27 a 32 filas de 114.700, 0,03 %— pero existen, y en
el sur austral conviene filtrar `band_* <= 1` antes de usar bandas crudas. Los negativos leves
(hasta −0,20) son normales en Collection 2 sobre agua y sombra.

Si aparecen valores en decenas o miles, nunca se aplicó la escala `DN·0,0000275 − 0,2` y todo
lo derivado es inválido.

### 5.3 No hay curva ajustada, y es a propósito

Acá hay observaciones con su fecha real, no una grilla ni un ajuste. Es la regla de
`docs/05` §1: adquirir a nivel de observación, decidir el pooling aguas abajo. Guardar una
grilla congelaría la elección de resolución y costaría una segunda extracción para revisarla.

Para construir la grilla o la curva, en local y sin volver al cubo:
`raw_series` e `interp_grid` en `src/biodiv/curves.py`.

---

## 6. Cómo se leen

```python
import pandas as pd

sites = pd.read_parquet("data/derived/living_trees/sites.parquet")
man   = pd.read_csv("data/derived/living_trees/manifest.csv")
kndvi = pd.read_parquet("data/derived/living_trees/series_kndvi_mean5x5.parquet")

# sitios con suficiente cobertura temporal para ajustar fenología
buenos = man.query("n_obs_mean5x5 >= 40 and max_doy_gap <= 90").site_id

serie = (kndvi[kndvi.site_id.isin(buenos) & kndvi.n_px.ge(20)]
         .merge(sites[["site_id", "year", "forest_type", "lat"]], on="site_id"))

# topografía: sufijo explícito, porque `sites` ya trae `elevation`/`slope` de terreno
topo = pd.read_parquet("data/derived/living_trees/topography/topography.parquet")
tabla = sites.merge(topo, on="site_id", suffixes=("_campo", "_dem"))
```

## 7. Procedencia

- **Origen de las parcelas:** `data/Living_Trees_Chile.xlsx`, pestaña `tree-level`.
- **Imágenes:** Landsat 5/7/8 Collection 2 Level-2 Surface Reflectance vía Data Cube Chile
  (`landsat{5,7,8}_c2l2_sr`), bucket requester-pays `usgs-landsat`.
- **Escala:** `reflectancia = DN · 0,0000275 − 0,2`; nodata = 0.
- **Máscara de nubes:** flags de `qa_pixel`, aplicados **por producto antes del concat** —
  descarta nodata, nube, sombra, cirrus y nieve de alta confianza, y dilatación de nube.
- **Grilla:** EPSG:32719, 30 m, `group_by="solar_day"`, remuestreo `nearest`.
- **DEM:** Copernicus GLO-30 (`copernicus_dem_30`) vía Data Cube Chile, remuestreo `bilinear`,
  derivadas sobre una ventana con halo de 20 px y recorte posterior al 5×5. 1.519 cargas,
  ~4,5 min sin dask.
- **Código:** `scripts/35_extract_living_trees.py`, `scripts/03_extract_topography.py`,
  `src/biodiv/cube.py`, `src/biodiv/io_living_trees.py`. Tests en
  `tests/test_living_trees.py` y `tests/test_topography.py`.
