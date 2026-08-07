# Estrategia de adquisición de datos satelitales

**Proyecto:** biodiversidad multi-faceta desde fenología satelital, Chile central
**Base:** `docs/03_cnn_architecture.md` (sustratos A y B), catálogo verificado de Data Cube Chile,
e inspección directa de Parcelas-CL (`20602096.zip`)

---

## 1. Veredicto sobre la ventana de 3 años

**La ventana `y−1, y, y+1` es la decisión correcta para Landsat. Pero no debe implementarse en la
adquisición.**

La propuesta —apilar tres años consecutivos y ajustar un único `PhenoShape` sobre el eje DOY— es la
respuesta correcta a un problema real de densidad (§2). El problema es *dónde* se aplica el
promediado.

Si la adquisición ya entrega la curva promediada de 3 años, se pierde de forma irreversible el **eje
de años**, que es exactamente el sustrato A de `docs/03` §2.1:

> "eje vertical (año) — anomalía y tendencia interanual: exactamente la señal que Lopatin (2023)
> mostró que se relaciona con las comunidades vegetales […] Esta es la única de las representaciones
> donde la segunda dimensión **no es manufacturada**."

Colapsar los tres años en una curva promedio destruye esa dimensión. Y esa dimensión es la novedad
que sostiene el paper: sin ella, el proyecto queda como "CNN sobre transformaciones manufacturadas",
que es precisamente lo que `Trait_2DCNN` ya hizo.

**Regla de diseño: adquirir al nivel de observación, decidir el pooling aguas abajo.**

La adquisición guarda la serie por píxel y por fecha, con su QA. A partir de ese único artefacto se
derivan sin volver al cubo:

| Derivado | Cómo | Alimenta |
|---|---|---|
| Curva pooled 3 años (52 pasos) | `PhenoShape` sobre las observaciones de `y−1..y+1` juntas | Sustrato B, y el baseline robusto para todas las parcelas |
| Curvas por año (K × 52) | `PhenoShape` por año, apiladas | **Sustrato A (phenocube)** |
| Métricas LSP | `PhenoLSP` sobre cualquiera de las dos | Baseline RF |
| Ventana causal `y−2..y` | mismo pipeline, otra selección de fechas | Análisis de sensibilidad (§5) |

El coste extra de guardar el nivel de observación es despreciable (§7) y convierte "3 años pooled vs
por año" en una **comparación empírica dentro de la matriz experimental** en vez de un supuesto
tomado antes de ver un dato.

---

## 2. La evidencia que justifica los 3 años

Adquisiciones por año medidas sobre tres AOI representativos del subset (conteo de escenas indexadas
que intersectan una caja de ~10 km; Cauquenes está en solape de órbitas, de ahí el conteo más alto):

| Período | Mapocho (−33,4) | Cauquenes (−36,0) | Coquimbo (−30,6) | Sensores |
|---|---|---|---|---|
| 1999–2002 | 16–29 | 22–54 | 22–45 | L5 + L7 |
| 2003–2011 | 27–43 | 36–70 | 25–42 | L5 + L7 (SLC-off desde 2003) |
| **2012** | **21** | **39** | **20** | **solo L7 SLC-off** |
| 2013–2016 | 38–43 | 69–83 | 33–44 | L7 + L8 |
| 2017–2021 | 41–48 | 78–83 | 43–49 | L7 + L8 (+L9 2021) |
| 2022–2025 | 40–49 | 78–89 | 44–51 | L8 + L9 |
| Sentinel-2 2019–2021 | 247–278 | 240–282 | 120–142 | S2 A/B |

Un año de Landsat pre-2013 entrega ~25–35 adquisiciones brutas. Descontando nubosidad invernal
—Chile mediterráneo tiene verano despejado pero el arranque de estación cae en primavera austral,
que no lo es— quedan del orden de 15–20 observaciones claras para ajustar una curva de **52 pasos**.
Es insuficiente y el reconstructor extrapola.

Con tres años se pasa a ~50–90 observaciones sobre el eje DOY. Ahí el ajuste es real.

**2012 es el peor caso del archivo** y no está en la tabla de riesgos de `docs/02`: única cobertura
L7 SLC-off, que además pierde ~22% de píxeles por franjas en cada escena. 31 parcelas fueron
censadas en 2012 y **127 parcelas (11,7%) tienen 2012 dentro de su ventana**.

---

## 3. Cómo cae la ventana sobre las parcelas reales

Sobre el subset modelable (n = 1.082: Chile central 30–38°S, año ≥1999, coordenada única):

| Situación | n | % |
|---|---|---|
| Ventana completa de 3 años | 1.033 | 95,5% |
| Ventana truncada | 49 | 4,5% |
| … toca 2012 (solo L7 SLC-off) | 127 | 11,7% |
| … toca 2019 (evento de browning) | 224 | 20,7% |
| S2 disponible en los 3 años (censo ≥2018) | 710 | 65,6% |
| S2 parcial dentro de la ventana | 32 | 3,0% |
| Solo Landsat (ventana entera <2017) | 340 | 31,4% |

Dos consecuencias que obligan a cambiar supuestos de los docs:

- **Las 49 parcelas truncadas son todas de 2026**: no existe `y+1`. La ventana debe ser adaptativa y
  el número de años efectivamente usados debe guardarse como covariable, no rellenarse.
- **Tras el filtro de coordenada única, el año de censo más antiguo del subset es 2003, no 1999.**
  Toda la cohorte de 1999 era `md001` (coordenadas a nivel de sitio). El rango temporal real es
  **2003–2026**, y el archivo requerido empieza en 2002. `docs/02` §1.1(c) debe corregirse: el
  emparejamiento retrospectivo cubre 24 años, no 28, y sin el pico de 1999.

---

## 4. El costo ecológico del pooling: no estacionariedad

Promediar `y−1..y+1` asume que la fenología es estacionaria dentro de la ventana. **La megasequía
viola ese supuesto**, y es el riesgo R6 de `docs/02` aplicado a la adquisición:

- 224 parcelas (20,7%) tienen **2019** en su ventana — el año del browning sincrónico documentado por
  Miranda et al. (2020) y por el propio paper de Parcelas-CL. Una ventana 2018–2020 mezcla un año
  anómalo con dos normales y produce una curva promedio que no describe ningún año real.
- Las ventanas de distintas parcelas caen sobre estados de sequía distintos, así que la
  heterogeneidad intra-ventana es una variable de confusión que varía entre observaciones.

**Mitigación, no eliminación:** calcular y guardar por ventana un descriptor de heterogeneidad
—dispersión entre las curvas anuales, y anomalía respecto de la climatología del píxel— y usarlo
como covariable y como criterio de exclusión sensible. Si el sustrato A resulta ganador, esta
heterogeneidad deja de ser ruido y pasa a ser la señal, que es el gancho del segundo paper.

**Problema de fuente climática:** `cr2met` termina en **2021**. 409 parcelas del subset son de
2022–2026 y quedan sin covariable climática desde esa fuente. El descriptor de anomalía debe
derivarse de la propia serie óptica (anomalía respecto de la climatología del píxel calculada sobre
todo el archivo), no de `cr2met`, o habrá que traer una fuente externa.

---

## 5. `y+1` mira al futuro — declararlo

La ventana centrada usa un año **posterior** al censo. Para un estudio retrospectivo es legítimo (no
es pronóstico, es caracterización del sitio), pero tiene dos consecuencias que hay que escribir:

1. Un referee preguntará por qué la vegetación de `y+1` informa sobre una comunidad medida en `y`. La
   respuesta —la ventana caracteriza el régimen fenológico del sitio, no el estado instantáneo— hay
   que darla explícitamente en métodos.
2. **El modelo no es desplegable para mapear el año `t` hasta que termine `t+1`.** Si el objetivo
   aplicado es mapeo operacional, la ventana debe ser causal.

Por eso la ventana causal `y−2, y−1, y` entra como **análisis de sensibilidad obligatorio**. Con
adquisición a nivel de observación no cuesta ninguna carga adicional del cubo: es otra selección de
fechas sobre el mismo artefacto. Si la diferencia de desempeño es pequeña, se reporta la causal como
principal y el problema desaparece.

---

## 6. Sentinel-2: experimento pareado, no interruptor

La intuición de "en años recientes no mezclar años" es correcta: con 240–280 adquisiciones S2 al año,
**un solo año basta** para una curva de 52 pasos, y a 10–20 m el píxel se acerca mucho más al tamaño
real de las parcelas (mediana 400 m² ≈ 4 píxeles S2 de 10 m, contra media parcela de un píxel
Landsat).

Pero cambiar de sensor según la época **confunde sensor con tiempo y con estado de sequía**: las
parcelas recientes serían S2-y-un-año y las antiguas Landsat-y-tres-años, y cualquier diferencia de
desempeño sería inatribuible.

**Estrategia:** para las 710 parcelas con S2 en los tres años (65,6% del subset), adquirir **ambos
sensores**. Eso da un diseño factorial pareado dentro de las mismas parcelas:

| | Landsat 30 m | Sentinel-2 10–20 m |
|---|---|---|
| Ventana 3 años pooled | ✓ (todas, 1.082) | ✓ (710) |
| Año único (`y`) | ✓ (donde la densidad lo permita) | ✓ (710) |
| Por año, apilado (sustrato A) | ✓ | ✓ (710) |

Con eso se responde por separado —y con las mismas parcelas— a tres preguntas que hoy están
enredadas: ¿aporta la resolución espacial?, ¿aporta la densidad temporal?, ¿aporta el pooling
multianual? Y el "sensor como estrato" de R5 deja de ser una excusa y pasa a ser un factor medido.

El coste es cargar S2 para 710 parcelas. Es asumible y es la diferencia entre un resultado
interpretable y uno confundido.

---

## 7. Geometría de extracción

- **Ventana de píxeles, no punto.** Extraer un vecindario de **5×5 píxeles** centrado en la parcela
  (150×150 m en Landsat). Cubre la augmentación 3×3 de `docs/03` §4.3 con margen, permite calcular
  la heterogeneidad local, y deja elegir el radio aguas abajo sin volver al cubo.
- **Guardar los píxeles por separado**, no el promedio. El promedio es un derivado; los píxeles
  individuales son la augmentación y también el diagnóstico de mezcla espectral.
- **Registrar cuántos píxeles cubre realmente la parcela** (`PlotSize_m2` entre 78,5 y 10.000 m²,
  mediana 400): la mayoría son **sub-píxel en Landsat**. Es una covariable de calidad y la
  cuantificación honesta de R3 en el lado del predictor.
- **Índices a guardar:** las bandas crudas escaladas, no solo los índices, para poder recalcular. Los
  índices del sustrato (NDVI, EVI, kNDVI, NBR) se derivan después.
- **Escalado obligatorio** — omitirlo invalida cualquier índice:
  - Landsat C2L2 SR: `reflectance = DN * 0.0000275 - 0.2`, vía `odc.algo.to_f32`.
  - `s2_l2a`: usar `easi_tools.load_s2l2a.load_s2l2a_with_offset`, que además maneja las
    inconsistencias de reprocesamiento PB 04.00.
- **Máscara de nubes:** Landsat con `masking.make_mask(ds.qa_pixel, **easi.qa_mask("landsat"))`; S2
  con `odc.algo.enum_to_bool` sobre `scl`. **Guardar el QA por observación**, no solo aplicar la
  máscara: `PhenoSensing.qa_to_weight` (spec `LANDSAT_C2`) usa los pesos, y el nº de observaciones
  válidas por parcela-año es la covariable de calidad que pide R4.
- **Hemisferio sur:** usar `PhenoSensing.reorder_southern_hemisphere` antes de reconstruir. La
  estación de crecimiento cruza el cambio de año calendario; sin reordenar, SOS/POS/EOS salen mal.
- **DEM:** `copernicus_dem_30` una sola vez para todo el subset, con las derivadas topográficas.

---

## 8. Plan de ejecución

Las parcelas se agrupan para minimizar llamadas al cubo. Medido sobre el subset:

- 104 celdas de 20 km contienen parcelas (mediana 5 parcelas/celda, máximo 129).
- **177 combinaciones (celda × año de censo)** → ése es el número de cargas ODC, no 1.082.

```
para cada (celda 20 km, año de censo y):
    para cada sensor en {landsat} ∪ {s2 si y ≥ 2018}:
        dc.load(bbox=celda, time=(y-1-01-01, y+1-12-31),
                measurements=bandas + QA, output_crs=UTM19S,
                resolution=30 (o 10), group_by='solar_day',
                dask_chunks={'time': 1})
        aplicar escalado; NO aplicar la máscara — guardar QA
        recortar ventana 5×5 por parcela de la celda
        escribir un artefacto por parcela
```

Reintentos y reanudación son requisito, no adorno: cada parcela escribe su propio archivo y el
driver salta lo ya hecho. Las cargas son lazy (`dask_chunks`) y solo se materializa el recorte.

**Artefacto por parcela** (NetCDF/Zarr, dimensiones `time × y × x × band`):

```
plot_id, sensor, dims (time, y=5, x=5, band)
coords: time (fecha real de adquisición), banda
vars:   reflectancia escalada por banda, qa_pixel/scl crudo
attrs:  census_year, window_years_used, n_obs_total, n_obs_clear_por_año,
        plot_size_m2, n_pixeles_cubiertos, sensor_mix, lat, lon, location,
        metadata_id, owner, coord_shared
```

Los `attrs` no son metadata decorativa: `window_years_used`, `n_obs_clear_por_año` y
`n_pixeles_cubiertos` son covariables del modelo y criterios de exclusión (R3, R4).

**Salida agregada:** `acquisition_manifest.parquet` con una fila por parcela-sensor y todas las
covariables de calidad — es lo que permite decidir exclusiones **con datos** en vez de a ojo, y lo
que se une con `targets.parquet` de la Fase 0.

---

## 9. Qué implementar

`scripts/05_acquire_timeseries.py`, CLI con `argparse`, salidas a `--out-dir` (scripts `.py`, no
notebooks):

```
--plots data/derived/plots_subset.parquet   # salida de la Fase 0
--sensor landsat|s2|both
--window centered|causal                    # y-1..y+1  |  y-2..y
--window-size 3
--patch 5                                   # 5x5 pixeles
--cell-km 20
--out-dir data/derived/timeseries
--resume
--dry-run                                   # solo cuenta cargas y volumen estimado
```

`--dry-run` primero: confirma las 177 cargas y el volumen antes de gastar horas.

**Dependencia de orden:** esto consume `plots_subset.parquet`, que produce la Fase 0 del EDA. La
decisión pendiente sobre `md001` (descartar vs agregar a 8 sitios) cambia el n de entrada, pero no
cambia nada de esta estrategia.

---

## 10. Resumen de la recomendación

1. **Sí a la ventana de 3 años** como sustrato por defecto para Landsat — la densidad pre-2013 no da
   para menos.
2. **No promediar en la adquisición.** Guardar nivel de observación; pooling aguas abajo. Es lo que
   preserva el sustrato A y convierte el supuesto en experimento.
3. **Ventana adaptativa y declarada**, con `window_years_used` como covariable (49 parcelas de 2026
   no tienen `y+1`).
4. **Ventana causal `y−2..y` como sensibilidad obligatoria**, porque `y+1` compromete el despliegue.
5. **S2 como experimento pareado sobre las 710 parcelas que lo permiten**, no como interruptor por
   época.
6. **Guardar QA y bandas crudas escaladas**, no índices ya enmascarados.
7. **Descriptor de heterogeneidad intra-ventana** derivado de la propia serie óptica, porque
   `cr2met` termina en 2021 y 409 parcelas son posteriores.
