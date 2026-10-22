# Series Landsat 2000–2025 para los 5.000 puntos de muestreo LandTrendr

Encargo para la sesión de Data Cube Chile. Estado: **enviado** el 2026-10-22.

## Por qué

`scripts/landtrendr_muestreo.py` sacó una muestra estratificada de 5.000 píxeles de 30 m en
bosque nativo y matorral (MapBiomas 2000), Coquimbo a Magallanes: 2.500 **con cambio**
(mayor perturbación abrupta de LandTrendr, corrida en Google Earth Engine) y 2.500 **sin
cambio** (ninguna perturbación abrupta 2001–2026), ~104 por estrato región x cobertura x estado.

De esos píxeles solo tenemos el ajuste LandTrendr (vértices y segmentos), no la serie
observada. Se necesita la serie Landsat completa, corregida y armonizada, para revisar los
cambios detectados, re-ajustar con otros métodos y validar contra fotointerpretación.

## Qué extraer

| | |
|---|---|
| Puntos | `results/tables/landtrendr_puntos_muestreo.csv`: `id`, `lon`, `lat`, `x_utm19s`, `y_utm19s`, `estado` (`cambio` / `sin cambio`), estrato y atributos LandTrendr |
| Clave | `id` (1–5000), la misma en todos los archivos de salida |
| Rango | 2000-01-01 a 2025-12-31, **todas las escenas** válidas |
| Productos | los cuatro de `cube.SENSOR_YEARS`; **Landsat 7 solo hasta 2021-12-31** (deriva orbital; igual que la corrida GEE) |
| Bandas | `blue green red nir swir1 swir2` (alias comunes a los cuatro productos) más `qa_pixel` y `qa_radsat` para enmascarar |
| Píxel | solo el **central** |
| Grilla | 30 m, EPSG:32719 |

**Grilla.** `x_utm19s, y_utm19s` son centros de píxel de la grilla GEE, con bordes en
múltiplos de 30 m en EPSG:32719. Es la misma grilla que produce `dc.load(output_crs=
"EPSG:32719", resolution=(-30, 30))`. El píxel más cercano a esas coordenadas es exactamente
el píxel LandTrendr; verificar en el piloto que la distancia al centro sea 0.

## Reutilizar, no reescribir

* `src/biodiv/cube.py`: `SENSOR_YEARS`, `BANDS`, `SR_SCALE`/`SR_OFFSET`, `clear_mask` (por
  producto, **antes** de concatenar: `cirrus` solo existe en L8/9), `group_by="solar_day"`,
  `resampling="nearest"`.
* `scripts/35_extract_living_trees.py`:
  * `crop()` para el píxel central;
  * agrupación de puntos en celdas de 5 km;
  * `--resume`, `manifest.csv` y `extraction.json`;
  * orden de arranque: cluster → `configure_s3_access(aws_unsigned=False,
    requester_pays=True, client=client)` → `datacube.Datacube()`. El bucket `usgs-landsat` es
    requester-pays.
* `scripts/03_extract_topography.py`: DEM `copernicus_dem_30` con `resampling="bilinear"`,
  `<= -1000` a NaN, y `terrain()` para pendiente y orientación. La convención es la
  corregida: orientación cuesta abajo, 0 = N, horario, la que fija `tests/test_topography.py`.
* `scripts/95_obs_center_solar.py`: `_sun()` (pysolar) para cenit y acimut por adquisición.
* `scripts/96_topo_correction_scsc.py`: `c_fe` y sus constantes.

Sugerencia: un script nuevo `scripts/170_landtrendr_series.py` que importe de esos, en vez de
copiar código.

## Cadena por observación

En este orden:

1. **Calidad.** `clear_mask` (nube, sombra, nube dilatada, cirrus, nieve) y además
   `qa_radsat == 0` (sin saturación), como en GEE.
2. **Reflectancia.** `DN * 0.0000275 - 0.2`, con DN 0 como sin dato. Descartar la observación
   entera si alguna de las seis bandas queda fuera de (0, 1).
3. **Corrección atmosférica.** Ya viene aplicada en Collection 2 Level-2 SR: LEDAPS para TM y
   ETM+, LaSRC para OLI. **No se aplica otra.** Dejarlo escrito en `extraction.json`.
4. **Armonización OLI → ETM+.** Roy et al. (2016), RMA. Son los mismos coeficientes de la
   corrida GEE (`scripts/gee_landtrendr_chile.py:_oli_to_etm`) y de
   `scripts/165_armonizacion_sensores.py:ROY`:
   `ETM = (OLI - b) / m`, solo para L8 y L9; TM y ETM+ no se tocan.
   ```
   m = [0.9785, 0.9542, 0.9825, 1.0073, 1.0171, 0.9949]   # blue..swir2
   b = [-0.0095, -0.0016, -0.0022, -0.0021, -0.0030, 0.0029]
   ```
5. **Topografía, SCS+C.**
   * Pendiente β, orientación α y elevación del píxel, desde el DEM en una ventana de 5x5
     centrada en el punto.
   * θs (cenit) y φs (acimut) por observación, con `_sun()`, la hora UTC de la escena y
     lat/lon/elevación.
   * `IL = cos θs cos β + sin θs sin β cos(φs − α)`.
   * `ρc = ρ (cos β cos θs + C) / (IL + C)`, con `C = b/m` de la regresión `ρ = b + m·IL`.
   * **C por banda y cobertura** (bosque nativo, matorral), con el estimador `fe` de
     `scripts/96` (efectos fijos de punto y de pasada orbital). Se estima **solo sobre los
     2.500 puntos sin cambio**, porque en los perturbados la caída de reflectancia contamina la
     regresión. Usar los filtros de estimación de scripts/96 (`EST_ZEN = 55`, `EST_IL = 0.5`) y
     aplicar el resultado a los 5.000.
   * **Aplicar solo** donde pendiente ≥ 0.5°, cenit ≤ 65° e IL ≥ 0.3 (`APP_ZEN`, `APP_IL`,
     `FLAT`). En el resto queda el valor sin corregir y `topo_aplicada = 0`.
   * Si C sale fuera de rango (`m <= 0` o `C > C_MAX`), no corregir esa banda x cobertura,
     marcarlo en `topo_C.csv` y avisar.
6. **Misma fecha, dos sensores.** Quedarse con el más nuevo (L9 > L8 > L7 > L5) y contar la
   colisión en el manifest. Debería ser raro.

## Esquema de salida

En el pod, `data/landtrendr_series/`:

```
cambio/                 2.500 puntos con estado = "cambio"
    blue.csv green.csv red.csv nir.csv swir1.csv swir2.csv     pasos 1-5 (corregidas)
    sin_topo/<banda>.csv                                       pasos 1-4 (sin SCS+C)
    aux/sensor.csv          5 / 7 / 8 / 9
    aux/sun_zenith.csv      grados
    aux/sun_azimuth.csv     grados desde el norte, horario
    aux/illum.csv           IL
    aux/topo_aplicada.csv   1 / 0
estable/                2.500 puntos con estado = "sin cambio", misma estructura
puntos.csv              id + columnas de landtrendr_puntos_muestreo.csv + elev, pendiente, orientacion
topo_C.csv              banda, cobertura, C, m, b, n, flag
manifest.csv            por id: observaciones por año y sensor, colisiones, fechas descartadas por QA / rango
extraction.json         parámetros, productos, cobertura real por sensor, tiempos, n_loads, nota de corrección atmosférica
```

**Formato ancho**, como lo pidió el usuario:

* primera columna `id`;
* luego una columna por fecha `YYYY-MM-DD`, la unión de las fechas del grupo, en orden
  cronológico;
* celda vacía = sin dato (nube, fuera de órbita, filtrado);
* reflectancia con 4 decimales.

Dentro de un grupo, **todas las bandas y los `aux` tienen exactamente las mismas filas
(orden por `id`) y las mismas columnas**, de modo que la celda (fila, columna) es la misma
observación en cualquier archivo.

Al final, comprimir todo en `data/landtrendr_series.zip` para bajarlo por JupyterHub. **No
subirlo a git.**

## Antes de lanzar

1. **Cobertura real.** Confirmar cuánto tiene indexado el cubo para Chile de L7 en 2000–2002 y
   de L9 en 2021 en adelante (lo mismo que pide docs/25).
2. **Piloto de 20 puntos**: 10 con cambio y 10 sin cambio, repartidos entre 30°S y 55°S, con la
   cadena completa salvo el C de SCS+C, que en el piloto puede ir con C de los 10 estables o en
   NaN. Medir:
   * tiempo por carga y extrapolación al total;
   * observaciones por año y sensor;
   * que el píxel extraído coincida con `x_utm19s, y_utm19s`.

   **Reportar al Mac por mensaje y esperar antes de lanzar los 5.000.**
3. Corrida completa con `--resume`. Si conviene partir, partir **por punto**, nunca por periodo.

## Chequeos al terminar

* 2.500 filas por archivo; ids = los de `landtrendr_puntos_muestreo.csv` según `estado`.
* Columnas idénticas entre bandas y `aux` de un mismo grupo.
* Reflectancias en (0, 1).
* Fracción de observaciones con `topo_aplicada = 1`, por cobertura.
* Coherencia con LandTrendr: en los puntos con cambio, el NBR (`(nir - swir2) / (nir + swir2)`)
  debe caer en el verano del `yod`. Revisar 10 al azar y reportar.

## Seguridad

`dask_gateway.Gateway().cluster_options()` materializa credenciales AWS STS vivas y una
contraseña de base de datos en claro. **No imprimirlo** en celdas, logs ni archivos
versionados, ni volcar el entorno al fallar (ver docs/25).

## Git

`git pull` para traer este documento y los puntos. **No descartar cambios locales** del pod;
si el pull da conflicto, reportar y parar. Ni commit ni push desde el pod salvo que el usuario
lo pida.
