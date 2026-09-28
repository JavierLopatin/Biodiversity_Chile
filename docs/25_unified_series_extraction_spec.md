# Serie óptica completa 2000–2026 para las 3.102 parcelas

Encargo para la sesión de Data Cube Chile. Estado: **redactado, sin enviar** (la sesión estaba
offline el 2026-09-28).

## Por qué

Todo lo que el repositorio tiene almacenado hoy es la **ventana causal de tres años de cada
parcela**, no el archivo. Medido sobre `data/derived/living_trees/series_ndvi_center.parquet`,
agregando por `site_id`: span temporal mediana 2,95 años, máximo 3,00, mínimo 1,56.

Eso impide la pregunta que queremos responder: si la curva fenológica y las métricas LSP
predicen diversidad, ¿cuánto varían esas mismas métricas de año en año sobre el mismo píxel,
cuando la diversidad medida no cambia? Sin serie larga no hay repetición, y sin repetición no
hay forma de separar la señal del predictor de su error de medida.

No es reprocesamiento: es una extracción nueva, porque los datos que harían falta nunca se
descargaron.

## Qué extraer

Una sola pasada, rango completo, para no volver al cubo por cada ventana.

| | |
|---|---|
| Parcelas | las 3.102 de `data/derived/plots_unified.parquet` (`lat`, `lon`), **las dos fuentes** |
| Clave | `PlotObservationID` (`PCL_*` / `LT_*`), no `site_id` |
| Rango | 2000-01-01 a 2026-12-31 |
| Productos | los cuatro de `cube.SENSOR_YEARS`: L5, L7, L8, L9 colección 2 nivel 2 |
| Bandas | `blue green red nir swir1 swir2` más `qa_pixel` |
| Píxel | `center` y `mean5x5`, misma convención que `scripts/35_extract_living_trees.py` |
| Resolución | 30 m, EPSG:32719 |
| Geometría solar | cenit y acimut por adquisición, si el producto los expone |

**Bandas crudas, no índices.** `scripts/35:24` ya lo dice: los índices son con pérdida y SAVI no
se recupera. Con las seis bandas guardadas, los cinco índices actuales y cualquier otro se
calculan después sin volver a leer S3.

## Esquema de salida

Un parquet por banda y por píxel, en `data/derived/series_unified/`:

```
series_band_<banda>_<px>.parquet
    PlotObservationID   str      PCL_32477 / LT_0027
    time                datetime64[ns]
    sensor              str      landsat5 | landsat7 | landsat8 | landsat9
    clear_frac_5x5      float32  fracción de los 25 píxeles válidos según qa_pixel
    sun_zenith          float32  grados, si el producto lo expone; NaN si no
    sun_azimuth         float32  grados desde el norte, sentido horario; NaN si no
    band_<banda>        float32
```

Doce archivos. `sensor` y `clear_frac_5x5` no son opcionales: son las dos covariables de control
del análisis, y sin ellas la extracción no sirve para lo que se pide.

**`sensor`** porque el eje temporal está confundido con el sensor. En lo ya extraído, 2012 es
100 % Landsat 7 —o sea SLC-off, con bandeado— y Landsat 8 no aparece hasta 2013. Cualquier
tendencia 2000–2026 medida sin esta columna es inatribuible entre fenología y radiometría.

**`clear_frac_5x5`** porque la densidad de observación cambia con la latitud y con el año, y una
métrica LSP ajustada sobre 12 observaciones no es comparable con una ajustada sobre 60. Hoy
`n_obs` se guarda por ajuste (`scripts/92_lsp_unified.py:109`), que llega tarde: hace falta por
observación para poder ponderar o filtrar antes de ajustar.

**`sun_zenith` / `sun_azimuth`** son un pedido nuevo, para la corrección topográfica SCS+C: la
condición de iluminación es `cos(gamma) = cos(theta_s)cos(pendiente) + sin(theta_s)sin(pendiente)cos(phi_s - orientacion)`,
y sin la geometría del sol no se calcula. Se pueden derivar analíticamente de `time` más
`lat`/`lon` con error por debajo de 0,1 grados, así que **no bloquean la corrida**: si el
producto no las expone como medida, dejarlas en NaN y seguir. Pedirlas igual sirve como
contraste contra el cálculo analítico.

Más un `extraction.json` con la misma estructura que el de Living Trees (parámetros, `n_loads`,
`n_written`, envelope) y un `manifest.csv` por parcela con el conteo de observaciones por año y
por sensor. Ese manifest es lo primero que vamos a mirar.

## Coste esperado

La extracción de Living Trees fueron 1.764 `dc.load` para 2.021 sitios con ventana de 3 años,
agrupando en celdas de 5 km. Aquí el rango es nueve veces más largo y hay 1.081 parcelas más.
Orden de magnitud: unas nueve veces aquella corrida, más la mitad de Parcelas-CL.

Volumen de salida: densidad observada ≈ 18 por año y parcela, por 27 años y 3.102 parcelas da
~1,5 millones de filas por archivo, doce archivos. Del orden de 300–400 MB en total. Cabe.

Si conviene partirlo, partir **por parcela**, nunca por periodo: el punto entero es tener la
serie continua de cada píxel.

## Antes de lanzar, dos cosas

1. **Confirmar la cobertura real de L7 en 2000–2002 y de L9 en 2021+** sobre el footprint
   chileno. `SENSOR_YEARS` declara 1999–2022 para L7 y 2021– para L9, pero eso es el rango
   nominal del producto, no lo que el cubo tiene indexado para Chile. Si 2000–2002 sale
   vacío, el rango útil empieza más tarde y conviene saberlo antes y no después.
2. **Un dry-run sobre 20 parcelas** repartidas entre 30°S y 55°S, para medir tiempo por carga y
   el conteo real de observaciones por año antes de comprometer la corrida completa.

## Seguridad

`dask_gateway.Gateway().cluster_options()` materializa credenciales AWS STS vivas y una
contraseña de base de datos en claro, tanto en su `repr` como en las variables de entorno
`DASK_GATEWAY__CLUSTER__OPTIONS__*`. **No imprimir ese objeto en una celda de notebook, en un
log ni en ningún archivo versionado.** Vale también para cualquier volcado de entorno que la
corrida pueda emitir al fallar.

## Qué NO se pide

Métricas LSP ni curvas PhenoShape. Eso se calcula después, en casa, sobre la serie ya
descargada: es un ajuste por ventana y no necesita el cubo. Separar las dos etapas es lo que
permite recortar las ventanas que queramos sin volver a pagar la extracción.
