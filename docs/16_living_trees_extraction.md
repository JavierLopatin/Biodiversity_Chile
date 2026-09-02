# Living_Trees_Chile: series de 3 años por parcela

**Para quién:** la sesión en la máquina con acceso a Data Cube Chile. Todo lo anterior a
`--dry-run` corre local.

**Qué se saca:** la serie Landsat de la **ventana causal de 3 años** anterior al censo, para
las **2.021 parcelas** de `data/Living_Trees_Chile.xlsx`, a nivel de observación, con los 5
índices y las 6 bandas escaladas, en el píxel central y como media del 5×5.

---

## 1. El conjunto, medido

La pestaña `tree-level` es **una fila por árbol** con la coordenada de la parcela repetida en
cada tallo. Lo que sigue está medido sobre el archivo entregado, no supuesto:

| | |
|---|---:|
| filas de árbol | 59.408 |
| sitios únicos `(Latitude, Longitude, Date)` | **2.021** |
| `um` (unidades de muestreo de terreno) | 1.943 |
| `um` con más de una coordenada | 78 |
| coordenadas con más de un `um` | 2 |
| sitios que comparten píxel Landsat de 30 m | **0** |
| filas sin `Date` | 46 (un sitio, `um` 17048, Aysén) |
| rango de `Date` | 2011–2020 |
| latitud | −30,64° a −54,97° |
| longitud | −74,72° a −67,34° |
| superficie de parcela (`10000/exp`) | 250 o 500 m² |

## 2. La unidad de extracción es la coordenada, no el `um`

Es la única decisión de diseño que puede arruinar el resultado sin dejar rastro. Las dos
particiones candidatas —coordenada y `um`— **no están anidadas**: 78 `um` tienen más de una
coordenada y 2 coordenadas están compartidas por más de un `um`. Agrupar por `um` promediaría
parcelas que caen en píxeles Landsat distintos, y la salida se vería perfectamente normal.

El `um` queda como atributo del sitio (`um_ids`), nunca como clave. `tests/test_living_trees.py`
fija los tres conteos: si alguno se hiciera cero, la distinción dejaría de existir y el test
lo diría.

Ningún par de sitios comparte píxel de 30 m, así que —a diferencia de Parcelas-CL, donde
`coord_shared` obligó a descartar 8 localidades— acá no hay colisiones que resolver.

## 3. La ventana

`y-2..y`, causal, igual que `scripts/01_build_subset.py`: no entra información posterior al
censo. Con `Date` entre 2011 y 2020 las ventanas van de 2009–2011 a 2018–2020, y
`cube.products_for` devuelve producto para todas. El único año delicado es 2012 —L5 terminó en
noviembre de 2011 y L8 empezó en abril de 2013—, así que las 194 parcelas de `Date`=2012
apoyan su tramo 2012 sólo en L7 SLC-off. Está registrado en el manifiesto vía
`n_obs_per_year`, que es donde hay que mirarlo.

## 4. Qué se guarda

**Observaciones con su fecha, no una curva ajustada ni una grilla.** Regla de `docs/05` §1:
adquirir a nivel de observación, decidir el pooling aguas abajo. `raw_series` / `interp_grid`
de `src/biodiv/curves.py` arman cualquier grilla en local, sin volver al cubo.

**Un `.parquet` por producto.** 11 variables × 2 lecturas espaciales = 22 tablas:

```
data/derived/living_trees/
    sites.parquet                     2.021 sitios: site_id, lat, lon, X, Y, year,
                                      win_start, win_end, pixel_id, cell, n_trees, n_um,
                                      um_ids, plot_size_m2, region, forest_type, pft, ...
    series_<var>_center.parquet       site_id, time, sensor, <var>
    series_<var>_mean5x5.parquet      site_id, time, sensor, <var>, n_px
    manifest.csv                      site_id, X, Y, lon, lat, year, win_start, win_end,
                                      cell, n_obs_center, n_obs_mean5x5, max_doy_gap,
                                      mean_doy_gap, frac_valid_px, n_obs_per_year,
                                      n_years_with_obs, patch_y, patch_x, sensors, error
    extraction.json                   los parámetros exactos de la corrida
```

`<var>` ∈ `ndvi, evi, kndvi, nbr, savi, band_blue, band_green, band_red, band_nir,
band_swir1, band_swir2`. El esquema mínimo es el mismo de
`data/derived/unlabelled/series.parquet` (`sample_id, time, kndvi`), más `sensor` —gratis, y
lo que permite auditar el salto L5→L7→L8— y, sólo en los `mean5x5`, `n_px`: cuántos de los 25
píxeles estaban despejados esa fecha. Sin `n_px` una media de 1 píxel y una de 25 son
indistinguibles.

**Por qué las dos lecturas espaciales.** La parcela real mide 250–500 m², menos que un píxel
de 900 m². El centro es la lectura estricta; la media 5×5 da la señal de rodal y amortigua el
error de GPS. En el pool no etiquetado se descartó el 5×5 porque el MAE aplanaba los 25
píxeles a 25 curvas independientes (`docs/15` §2); acá no se guardan los 25 píxeles sino su
media, que es una variable distinta y cuesta una columna.

**Las 6 bandas además de los 5 índices.** Es la misma `dc.load`, así que no cuesta lecturas de
S3. Los índices son lossy: SAVI no se reconstruye desde un NDVI guardado.

## 5. Cómo se corre

```bash
# local, sin tocar el cubo
python scripts/35_extract_living_trees.py --dry-run
python scripts/35_extract_living_trees.py --save-sites data/derived/living_trees/sites.parquet
python -m pytest tests/test_living_trees.py -q

# en la máquina del datacube
python scripts/35_extract_living_trees.py --gateway --workers 16 --resume \
    --sites data/derived/living_trees/sites.parquet --out data/derived/living_trees
```

`--resume` retoma por `(cell, year)` desde `manifest.csv` y relee las 22 tablas, así que una
corrida interrumpida no pierde trabajo. El checkpoint es cada 50 cargas (`--flush-every`).

**1.764 cargas** (celdas de 5 km × año; mediana 1 sitio por carga, bbox máximo 4,3 × 4,3 km).
A los 6,5 s/carga de la corrida del pool no etiquetado (2.898 cargas, 5,2 h, 16 workers) son
**~3,2 h**. Medido con 4 workers locales en Magallanes: ~1,2 min/carga, o sea que el cluster
de gateway no es opcional.

El orden de arranque de dask no es negociable: cluster → `configure_s3_access(client=...)` →
`Datacube`. Al revés, cada lectura del bucket requester-pays `usgs-landsat` vuelve
AccessDenied.

---

## 5 bis. La topografía (hecha, 2026-08-14)

Las 9 derivadas del DEM Copernicus GLO-30 para los mismos 2.021 sitios, con el **mismo código
y las mismas definiciones** que Parcelas-CL. No se duplicó el script: `03_extract_topography.py`
toma ahora la tabla de sitios, la columna clave y el directorio de salida por argumento.

```bash
python scripts/03_extract_topography.py --dry-run \
    --subset data/derived/living_trees/sites.parquet --id-col site_id --id-name site_id \
    --out-dir data/derived/living_trees/topography

python scripts/03_extract_topography.py \
    --subset data/derived/living_trees/sites.parquet --id-col site_id --id-name site_id \
    --out-dir data/derived/living_trees/topography --resume   # log: logs/living_trees_topography.log
```

Sirve cualquier parquet con `X`/`Y` (EPSG:32719), `lat` y una columna `cell` de agrupación.

**1.519 cargas** (celdas de 5 km; el DEM no tiene dimensión temporal, así que se agrupa sólo
por celda) en **~4,5 min sin dask**: son cajas de 2,3 km, ~77 × 77 px, y el cuello de botella
es la latencia de S3, no el cómputo. Checkpoint cada 100 celdas y `--resume` por sitio.

Salida en `data/derived/living_trees/topography/`: `topography.parquet` (2.021 × 28) y
`topography_patches.nc` (2.021 × 5 × 5 × 9). El detalle de columnas, la validación contra la
elevación de terreno y los 14 sitios planos están en el README de ese directorio, §4.

**Lo que hay que saber antes de usarla:** al montar esta extracción se encontró que `aspect`
apuntaba **cuesta arriba**, 180° girado respecto de la convención cartográfica. Está corregido
y verificado contra `gdaldem aspect`, con `tests/test_topography.py` fijando la convención.
La tabla de Living_Trees_Chile sale con la convención **correcta**; la de Parcelas-CL
(`data/derived/topography/`) sigue con la **antigua**, a la espera de un refit — aviso completo
en `docs/07_run_record.md` §3. Mientras tanto, `northness`, `eastness` y `heat_load` de los dos
conjuntos tienen signos opuestos y no deben mezclarse.

## 6. Lista de verificación al llegar

Medido en una corrida de prueba de 3 cargas (4 parcelas de Magallanes, 2015–2019):

1. **kNDVI en 0..1.** Salió 0,0002–0,762. Si sale fuera de rango falta la escala C2
   (`DN*0.0000275 - 0.2`) y hay que parar ahí: todo lo demás queda inválido.
2. **Bandas en reflectancia, entre −0,2 y 1,6.** Ése es exactamente el rango que da la escala
   C2 sobre el entero válido (1 a 65.455), no un rango físico: **las bandas no están
   recortadas**, sólo los índices lo están (`cube.to_indices`). En 586 sitios `band_nir` llegó
   a 1,602, que es el techo del entero — píxeles saturados de nieve que pasaron la máscara,
   porque `QA_KEEP` sólo descarta `snow="not_high_confidence"`. En el sur austral hay que
   filtrar por `band_nir <= 1` antes de usar las bandas crudas. Decenas o miles significan que
   nunca se aplicó la escala.
3. **`win_end - year == 0` en los 2.021 sitios.** Es la ventana causal.
4. **`n_obs` cae fuerte con la latitud, y eso limita lo que se puede ajustar.** Medido sobre
   los primeros 586 sitios, mediana de observaciones en la ventana de 3 años:

   | banda de latitud | sitios | obs. mediana |
   |---|---:|---:|
   | −35° | 19 | 66 |
   | −40° | 360 | 67 |
   | −45° | 164 | 42 |
   | −50° | 19 | **17** |
   | −55° | 24 | 30 |

   Los ~65 del norte son los de `docs/15` §9. Pero **17 observaciones en 3 años no sostienen
   una curva fenológica**: son ~6 por año, con huecos estacionales grandes. Los sitios al sur
   de −45° hay que tratarlos como un estrato aparte y filtrar por `n_obs_mean5x5` y
   `max_doy_gap` del manifiesto antes de ajustar nada, no promediarlos con el resto. Un `n_obs`
   uniforme en todo el país sería lo sospechoso.
5. **Las 22 tablas con el mismo conjunto de `site_id`.** Difieren en filas —cada variable
   tiene su propia máscara de finitud— pero no en sitios.
6. **`n_px` entre 1 y 25** en las tablas `mean5x5`, y `patch_y == patch_x == 5` en el
   manifiesto: un 3 delataría un sitio en el borde del bbox cargado.
7. **`manifest.csv` sin columna `error` poblada.** Una carga fallida no corta la corrida, se
   registra por sitio y se reintenta con `--resume`.
