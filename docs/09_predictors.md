# Calidad de predictores a nivel de píxel y alternativas a la fenología

Instrucciones ejecutables para la siguiente fase. El modelamiento
([`08_modelling.md`](08_modelling.md)) dejó 79 modelos corridos y el mejor Random Forest empatado
con la mejor CNN: **el margen que queda está en el predictor, no en el modelo.** Este documento
dice qué arreglar, en qué orden, y con qué criterio se decide seguir o parar.

Los mapas quedan fuera de alcance por ahora.

---

## 0. Los cuatro problemas, verificados sobre los datos

**1. Los NaN de LSP no son físicos, son numéricos.** El 99,2 % viene de un solo `return None` en
`phase.py:86-88` de PhenoSensing: cuando la fuerza estacional cae bajo `strength_thresh = 0.15`
no se aplica rotación, el pico queda estacionado en el índice 0 o 51 del arreglo, y las métricas
que necesitan una rama ascendente o descendente reciben una rebanada vacía.

| | filas | filas con ≥1 NaN |
|---|---|---|
| curva **sí rotada** (fuerza ≥ 0,15) | 4.037 | **2 (0,05 %)** |
| curva **no rotada** (caída a aseasonal) | 1.373 | **252 (18,4 %)** |

No es una señal de aridez: los sitios más áridos (Coquimbo, Las Cardas, 30–33°S) tienen **cero**
NaN y la banda húmeda de 35–37°S concentra la mayoría. Las métricas afectadas son exactamente las
que dependen de una rebanada: `los`, `ios`, `sw`, `mos` (113 filas cada una, conjuntos
idénticos), `rog` (64, ⟺ `sos == pos`) y `ros` (77, ⟺ `eos == pos`). Las otras 12 nunca fallan.

**2. El `mean5x5` esconde el problema en vez de resolverlo.** Parece libre de NaN (2 celdas), pero
`scripts/04_recompute_lsp.py:234` usa `np.nanmean` sobre los 25 píxeles. Recomputando a nivel de
píxel: **6.846 píxeles (5,1 %) fallan y el 28,2 % de las celdas (parcela, índice) tiene al menos
un píxel NaN.** Cada valor `*_mean5x5` es un promedio sobre entre 1 y 25 píxeles, y nada registra
cuántos.

**3. El 53 % de las parcelas comparte ventana de extracción.** 767 pares con solapamiento de la
ventana de 150 m; 575 parcelas en 135 componentes conexos; el mayor de 17 parcelas
(Miranda md004 2010 + Altamirano md019 2014, un recenso encima de una grilla). `kfold5_owner`
**filtra 7 de esos componentes (47 parcelas)** — mismo terreno, etiqueta de dueño distinta.
`kfold5_random` filtra el 97 %. Los esquemas de bloque geométrico no filtran nada.

**4. Nunca se probó el control obvio.** La media anual de kNDVI da Spearman −0,664 con
`pcoa1_pa`; el mejor modelo completo da R² = +0,41. Un compuesto anual sin ninguna forma
fenológica podría estar haciendo casi todo el trabajo, y el benchmark no lo sabe.

---

## 1. Fase 0 — Localizar los cubos `.nc` (30 min, máquina con datacube)

**Hacer esto antes que nada.** Decide si la fase 4 necesita el datacube y abre —o no— toda una
familia de predictores.

`scripts/02_extract_phenology.py:93-101` guardó, dentro de cada
`data/derived/phenology/{plot_id}.nc`, **la serie de tiempo cruda completa** además de la curva:

```
obs_{ndvi,evi,kndvi,nbr,savi}                 dims (time, y, x)
obs_band_{blue,green,red,nir,swir1,swir2}     reflectancia escalada
coords: time, doy, year, sensor
```

Fue deliberado; el encabezado del script lo dice: *"Observation-level data is stored alongside
the curve. Pooling is a downstream decision, not an acquisition one."*

```bash
ls data/derived/phenology/*.nc | wc -l     # se esperan 1082, ~360 MB
```

**Si existen:** cópialos a la máquina de análisis. La geomediana, las MAD y los composites por
observación se calculan directamente sobre `obs_*` **sin conexión al datacube, sin requester-pays
y sin S3**, y el resto del plan queda enteramente local.

**Si no existen:** solo entonces hace falta el datacube (§5, ruta 2).

En la máquina de análisis actual **no están**: `data/derived/phenology/` contiene únicamente
`manifest.csv`, y una búsqueda de `*.nc` en todo el sistema devuelve solo
`topography_patches.nc`.

---

## 2. Fase 1 — LSP sin NaN (3–4 días, local)

### 2.1 No hace falta el hub, ni parchear PhenoPY

`phenosensing` importa desde `/mnt/rapidita_4T/GitHub/PhenoPY` y `utils._getLSPmetrics2` corre
sobre las curvas guardadas en `phenoshape_by_index.parquet`. Su firma **sí** expone lo necesario:

```python
_getLSPmetrics2(phen, xnew, nGS, bands, phentype=1, extraction=None,
                extract_params=None, hemisphere='north', offset=None,
                strength_thresh=0.15)
```

Lo que no lo expone es el wrapper `accessor.PhenoLSP`, que reenvía solo cinco kwargs
(`accessor.py:196-204`). **La solución es saltarse el wrapper y llamar la función directamente.**
Los `.nc` no hacen falta para esto: las curvas de 52 pasos y la grilla DOY real ya están en
parquet, y son la única entrada que la función necesita.

> **Advertencia.** Una comprobación rápida durante el diseño **no reprodujo** el conteo de NaN
> almacenado (598 curvas afectadas contra ~197 celdas guardadas para ndvi), casi con seguridad por
> doble rotación: las curvas guardadas ya vienen ancladas al valle (`doy_anchor = 108`) y la
> llamada volvió a rotarlas. Por eso las cifras de mejora que aparecen abajo son **provisionales**
> hasta que pase la compuerta de reproducción.

### 2.2 `scripts/16_recompute_lsp_local.py`

Hermano local de `scripts/04_recompute_lsp.py`, sobre parquet en vez de NetCDF.

1. **Compuerta de reproducción.** Con `strength_thresh = 0.15` y el mismo manejo de rotación que
   el script 04, reproducir `lsp_all_auto.parquet` a nivel de píxel central dentro de tolerancia
   numérica y con el mismo patrón de NaN. **Si no pasa, parar.** Significa que no entendemos la
   invocación, y cualquier "mejora" sería otra cosa.
2. **Barrido del anclaje** sobre las 5.410 curvas centrales y las 135.250 de píxel:
   `strength_thresh ∈ {0.15, 0.05, 0.0}` × `offset ∈ {None, 108}`. Elegir la configuración que
   minimiza NaN **sin degradar la coherencia estacional** (`sos < pos < eos`, que hoy falla en 141
   filas). La expectativa —a confirmar— es que `strength_thresh = 0` elimine la gran mayoría, y
   que `offset = 108` (el ancla global que el proyecto ya usa) llegue casi igual de lejos.
3. **Guardas para el residuo.** Tres casos que ninguna configuración de anclaje arregla y que hoy
   no existen en PhenoPY:
   - `ampl > 0` antes de `utils.py:218` — `ratio = (phen − trough)/ampl` da 0/0 en curva plana;
   - manejo explícito de rebanada vacía en `extraction.py:28-29`, en vez de depender de que
     `argmin(nan)` devuelva 0;
   - `pos != sos` y `pos != eos` antes de `utils.py:274/277` (`rog`, `ros`).

   Van en un envoltorio propio (`src/biodiv/lsp.py`), **no** parcheando PhenoPY, para que el repo
   no dependa de un fork. Cuando una guarda dispara, la métrica queda NaN **y se marca con un
   indicador explícito**: un NaN honesto es mejor que un valor inventado.
4. **Multimodalidad — la mejora de fondo, no un parche.** Solo el **16 % de las curvas es
   unimodal**; la mediana es de 3 picos prominentes tras `rollWindow = 5`. El `argmax` global es
   inestable y es la causa raíz de que el pico aterrice en los extremos. Probar como factor
   **usar el pico armónico de `phase._harmonic_phase` en vez del `argmax`** para definir `pos`,
   que es robusto al ruido por construcción.

### 2.3 LSP por píxel y agregación explícita

Pasar de dos niveles (centro y `nanmean` de 25) a calcular **sobre los 135.250 píxeles** y
agregar con regla declarada. Salidas: `data/derived/lsp_pixels.parquet` y `lsp_all_v2.parquet`,
con por cada métrica:

- `{m}_median`, `{m}_mean`, `{m}_trimmed` (media recortada al 20 %) — mediana y recortada son
  robustas al píxel que falla, cosa que `nanmean` no es;
- `{m}_sd` entre píxeles — a la vez predictor de heterogeneidad (§4.3) y diagnóstico;
- **`n_px_valid`** — el dato que hoy no existe y que convierte un promedio opaco en uno auditable;
- `{m}_center`, conservado para la ablación de píxel de `scripts/15_pixel_ablation.py`.

**Las 7 métricas con valor de fecha (`sos, pos, eos, msp, mau, trough, mos`) deben agregarse de
forma circular**, no aritmética. Promediar DOY 5 y DOY 360 da 182, que es pleno invierno. Hoy
está mal.

### 2.4 Limpieza de features

- **`order_coherent` sale de `BLOCK_QC`.** Es algebraicamente idéntico a `rog_nan | ros_nan`
  (verificado exactamente): no aporta información independiente y diluye la importancia.
- `degenerate_px_frac` solo detecta `sos == pos`; no ve `eos == pos` ni `eos < sos`. Redefinir
  para cubrir los tres modos de fallo.
- Añadir `n_px_valid` y los indicadores de guarda como covariables de calidad.

---

## 3. Fase 2 — CV por componente de ventana (1 día, local)

`scripts/08_build_modelling_folds.py` gana `kfold5_window`: construir el grafo de solapamiento
(|dX| < 150 y |dY| < 150 sobre las coordenadas UTM), tomar sus componentes conexos y agrupar por
componente reusando `cv_groups.grouped_kfold` sin cambios. 642 grupos (507 singletons + 135
componentes), con **cero fuga por ventana compartida** por construcción.

Se conservan las 1.082 parcelas. El contraste `kfold5_owner` contra `kfold5_window` cuantifica
cuánto del R² era pseudorreplicación — es un resultado, igual que la brecha de optimismo. Se añade
también `kfold5_owner_window`, la unión de ambos agrupamientos y la partición más estricta del
proyecto.

**No se fusionan parcelas.** Fusionar a nivel de componente (1.082 → 642) era defendible: solo el
11 % de la varianza de riqueza y el 29 % de la de `pcoa1_pa` es intra-componente, y esa fracción
es por construcción inalcanzable desde una ventana de 150 m. Se deja registrado el número por si
un revisor lo pide, pero la decisión es conservar el n y manejar la pseudorreplicación en el
esquema de CV.

---

## 4. Fase 2b — Predictores no fenológicos locales (2 días, local)

La pregunta de fondo: **¿aporta la forma fenológica algo sobre un compuesto anual?**

### 4.1 Composites anuales desde las curvas guardadas

Bloque `composite` en `src/biodiv/features.py`: media, mediana, p10, p25, p75, p90, mínimo,
máximo, rango y CV temporal de cada uno de los 5 índices sobre los 52 pasos → 50 features. Sin
información de *forma* ni de *fecha*: es el control directo.

Correlaciones univariadas ya medidas contra `pcoa1_pa`: media kNDVI −0,664, media NBR −0,660,
media NDVI −0,656, y **CV temporal de NBR +0,53** — una señal distinta, que no es de forma.

### 4.2 Contrastes entre índices

Diferencias y razones entre los estadísticos anuales de los 5 índices. Medido: **media NDVI −
media SAVI da Spearman −0,693 con `pcoa1_pa`**, más fuerte que cualquier feature individual del
sistema actual.

### 4.3 Heterogeneidad espectral entre píxeles (SVH)

Bloque `svh`: sd, CV y rango entre los 25 píxeles, de la media anual de cada índice y de cada
métrica LSP (esto último sale gratis de §2.3). Es el eje A de
[`01_state_of_the_art.md`](01_state_of_the_art.md) y nunca se probó aquí.

Dos advertencias que van en el diseño, no descubiertas después:

- **La sd está confundida con la media** (corr 0,22–0,52 según índice). Reportar la correlación
  parcial controlando por la media, no la cruda: medido, la parcial cae a ≈ −0,14 sobre `lcbd_pa`
  y a ≈ −0,05 sobre `pcoa1_pa`.
- **El signo es negativo para la riqueza** en ndvi/kndvi/nbr — opuesto a la hipótesis clásica. Si
  se sostiene, es un resultado reportable, no un error a esconder.

---

## 5. Fase 4 — Geomedianas (1 día si los `.nc` están; 3–4 si no)

Una geomediana es la mediana multivariada de las bandas sobre el tiempo: un compuesto sin ninguna
noción de forma temporal. Es el contraste limpio contra la curva fenológica **siempre que se
calcule sobre exactamente los mismos píxeles, la misma ventana y la misma máscara de nubes**.

**Ruta 1 — los `.nc` existen (preferida).** `scripts/18_geomedian_from_cubes.py`: lee los cubos,
aplica `odc_algo.geomedian_with_mads` sobre el eje `time` por píxel —o una implementación propia
de Weiszfeld si `odc-algo` no está instalado— y escribe `data/derived/geomedian.parquet` y
`geomedian_pixels.parquet`. Sin datacube, sin S3. Minutos.

**Ruta 2 — no existen.** `scripts/18_extract_geomedian.py`, en la máquina con datacube. Reusa
`src/biodiv/cube.py` sin modificar (`load_window`, `clear_mask`, `products_for`) y la misma
iteración por (cell, año) de `scripts/02_extract_phenology.py`.

**No usar los productos pre-computados del catálogo.** Existen `landsat{5,7,8}_geomedian_annual`
y `s2_geomedian_{annual,seasonal}`, pero son **por sensor y por año calendario**, y las ventanas
causales de 3 años mezclan sensores (L5+L7, L7+L8, L8+L9). Usarlos confundiría sensor con época,
que es justo lo que [`05_data_acquisition.md`](05_data_acquisition.md) prohíbe.

Salida idéntica en ambas rutas, por parcela y por píxel:

- las 6 bandas de la geomediana (blue, green, red, nir, swir1, swir2) y los 5 índices derivados;
- **las tres MAD** (euclidiana, espectral, Bray-Curtis) — variabilidad intra-ventana **sin forma
  temporal**, que es precisamente el contraste que interesa;
- `count` de observaciones limpias por píxel.

### 5.1 Lo demás que se abre si los `.nc` están

Con `obs_*` en mano y sin datacube:

- **Composites sobre observaciones reales**, no sobre la curva suavizada — distinto de §4.1, que
  los calcula sobre la interpolación de 52 pasos y por tanto ya arrastra el suavizado.
- **Composites estacionales** (geomediana o mediana por trimestre austral): contraste estacional
  sin ajustar ninguna curva.
- **Predictores de reflectancia cruda**, hoy imposibles: todo el proyecto usa índices derivados, y
  las 6 bandas contienen información que ningún índice conserva.
- **Reajustar `PhenoShape` con otro suavizado.** La multimodalidad de §2.2.4 viene de
  `rollWindow = 5`; con las observaciones se puede probar otras ventanas sin descargar nada.
- **Descriptores de calidad por píxel**, más finos que los del `manifest.csv`, que son por parcela.

---

## 6. Fase 3 — Cribado con Random Forest (1 día, local)

RF es el banco de pruebas porque cuesta minutos y no tiene hiperparámetros que confundan la
comparación. **Cada bloque nuevo se criba con RF antes de que ningún modelo profundo lo toque.**

Esquema primario `kfold5_window`, secundario `kfold5_owner`, 3 semillas.

| ID | Bloque | Pregunta |
|---|---|---|
| **X00** | `composite` solo | ¿cuánto da un compuesto anual sin nada de forma? |
| X01 | `composite + topo` | el control no fenológico completo |
| **X02** | `lsp_v2 + topo` | LSP arreglada, contra la actual (`RF01`) |
| X03 | `curve + topo` | la curva sin cambios — la referencia |
| **X04** | `composite + curve + topo` | ¿la forma aporta **sobre** el compuesto? El contraste central |
| X05 | `svh + topo` | SVH sola |
| X06 | `composite + svh + topo` | ¿la heterogeneidad aporta sobre el nivel? |
| X07 | `geomedian + mads + topo` | el compuesto propio |
| X08 | `geomedian + mads + curve + topo` | fenología sobre geomediana |
| X09 | `lsp_v2 + curve + composite + svh + topo` | todo junto, techo empírico |
| X10 | mejor bloque, agregación `median` / `mean` / `trimmed` | ¿importa la regla de agregación? |
| X11 | mejor bloque, píxel como unidad muestral | §7 |

Cada fila × 5 índices donde aplique. Coste estimado: ~60 corridas de RF, ~1,5 h de CPU.

**Compuerta antes de los modelos profundos:** solo los bloques que superen a `X03` bajo
`kfold5_window` pasan a MLP, CNN 1D y CNN 2D con la infraestructura existente (`scripts/10`,
`scripts/11`, `scripts/14`). Todo `dl_runner`, los sustratos y la pérdida enmascarada se reusan
sin cambios; solo cambian los bloques de features.

**El resultado que más importa es X00 contra X03.** Si un compuesto anual iguala a la curva de 52
pasos, la contribución del proyecto se reformula: deja de ser "la fenología predice composición" y
pasa a ser una afirmación más precisa sobre qué parte de la señal temporal trabaja. Hay que estar
dispuesto a reportarlo.

---

## 7. Fase 5 — El píxel como unidad muestral (2 días, local)

`scripts/17_pixel_level_models.py`:

- Diseño de 27.050 filas (1.082 × 25) desde `lsp_pixels.parquet` y las curvas por píxel, con la
  etiqueta de la parcela repetida en sus 25 filas.
- **Los 25 píxeles de una parcela nunca cruzan folds** — ya lo garantizan `cv.pixel_rows` y el
  test de fuga existente; se añade la aserción a nivel de componente de ventana.
- En test se predice por píxel y **se promedia a nivel de parcela** antes de puntuar, para que las
  métricas sean comparables con el resto (la tabla OOF sigue con 1.082 filas por semilla).
- Se reporta el n de parcelas, nunca el de píxeles.
- Factor a probar: pesar cada píxel por su distancia al centroide. El 97 % de las parcelas es
  sub-píxel, así que los píxeles del borde de la ventana de 150 m muestrean vegetación que nadie
  censó.

---

## 8. Fase 6 — El resto de los modelos (2–3 días, 2 GPUs)

Los bloques que pasaron la compuerta, a través de MLP, CNN 1D y CNN 2D con
`scripts/14_run_matrix.py`. Se re-corre `scripts/12_model_report.py` y los tests pareados por
parcela.

---

## 9. Archivos

**Módulos nuevos**

| Archivo | Contenido |
|---|---|
| `src/biodiv/lsp.py` | Envoltorio sobre `_getLSPmetrics2` con las guardas de §2.2.3, agregación circular de las métricas de fecha, y `n_px_valid` |
| `src/biodiv/composites.py` | Estadísticos anuales por índice y contrastes entre índices |
| `src/biodiv/heterogeneity.py` | SVH entre píxeles, con la correlación parcial controlando por la media como salida de diagnóstico |

**Scripts nuevos**

| Script | Dónde | Rol |
|---|---|---|
| `16_recompute_lsp_local.py` | local | Compuerta de reproducción, barrido de anclaje, LSP por píxel |
| `17_pixel_level_models.py` | local | Modelos con el píxel como unidad muestral |
| `18_geomedian_from_cubes.py` | donde estén los `.nc` | Geomediana + MAD sin datacube |
| `18_extract_geomedian.py` | máquina con datacube | Solo si los `.nc` no aparecen |
| `19_screen_blocks.py` | local | La matriz X00–X11 con RF |

**Modificados:** `src/biodiv/features.py` (bloques `composite`, `svh`, `lsp_v2`, `geomedian`;
sacar `order_coherent`; añadir `n_px_valid`) · `scripts/08_build_modelling_folds.py`
(`kfold5_window`, `kfold5_owner_window`) · `src/biodiv/cv.py` (`SCHEME_GROUP`,
`window_components()`) · `scripts/09_run_baselines.py` (registro X00–X11) ·
`scripts/12_model_report.py` (comparación contra `X03` y contra `kfold5_owner`).

---

## 10. Verificación

1. **Compuerta de reproducción de LSP** (`tests/test_lsp.py`): con la configuración original,
   `src/biodiv/lsp.py` reproduce `lsp_all_auto.parquet` en las métricas no-NaN dentro de 1e-6 y
   con el mismo conjunto de celdas NaN. Sin esto, nada de la fase 1 es interpretable.
2. **Las guardas producen NaN, no valores inventados**: curva plana (`ampl = 0`) y curva con pico
   en el índice 0 devuelven NaN en las métricas afectadas y activan su indicador.
3. **Agregación circular**: la media circular de DOY 5 y DOY 360 cae cerca de 0/365, no de 182.
4. **`kfold5_window` no filtra**: para cada fold, ningún componente de solapamiento aparece a
   ambos lados. Va junto a los tests de fuga existentes en `tests/test_modelling.py`.
5. **Píxeles y parcelas**: los 25 píxeles de una parcela caen siempre del mismo lado en todos los
   esquemas; la tabla OOF a nivel de parcela tiene exactamente 1.082 filas por semilla.
6. **Paridad de bloques**: cada bloque nuevo devuelve el mismo orden de parcelas que `plot_ids()`
   y sin columnas duplicadas — la aserción existente, extendida.
7. **Cordura del cribado**: `X00` (compuesto anual solo) debe superar a `B01` (solo topografía).
   Si no, hay un error en la construcción del bloque, no un resultado.
