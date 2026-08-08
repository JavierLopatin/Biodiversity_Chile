# Modelamiento: diseño, matriz de modelos y decisiones

Etapa que cierra el pipeline: predictores extraídos y variables respuesta calculadas
(fase 1) → modelos. Este documento registra **qué se corre, por qué, y qué decisiones no
son obvias**. El registro de ejecución con tiempos y estado por trabajo está en
[`08_modelling_progress.md`](08_modelling_progress.md); los resultados numéricos en
`results/tables/` y `results/models/summary.csv`.

---

## 1. La pregunta

El benchmark está construido alrededor de tres contrastes anidados, en este orden:

1. **¿Aporta la curva fenológica completa sobre sus 18 resúmenes escalares?**
   `RF01` (LSP) contra `RF03` (curva de 52 semanas). Es el gap G1 de
   [`01_state_of_the_art.md`](01_state_of_the_art.md), y se responde con Random Forest antes
   de que exista ninguna red neuronal.
2. **¿Aporta la convolución sobre la curva?** El MLP tabular contra la CNN 1D, con las mismas
   features.
3. **¿Aporta una segunda dimensión, y tiene que ser real?** La CNN 1D contra las once
   imágenes 2D — nueve manufacturadas del catálogo de `Trait_2DCNN` y dos cuyo eje vertical
   no es un artefacto.

Cada contraste sólo tiene sentido si el anterior está medido. Por eso el orden de ejecución
es RF → 1D → 2D → MLP → barridos, y no al revés.

---

## 2. Decisiones que no son obvias

### 2.1 El sustrato "phenocube" no existe, y no se construye

[`03_cnn_architecture.md`](03_cnn_architecture.md) propone como sustrato principal una
imagen año × DOY. **No se construye.** La curva almacenada es un único ajuste PhenoShape
sobre una ventana causal de 3 años (`win_years = 3`, `win_end = Year` para las 1.082
parcelas), y eso es deliberado: da un perfil estacional estable a nivel de ecosistema, que es
precisamente el predictor que se quiere. Partir esa ventana en curvas por año dejaría entre 9
y 30 observaciones claras para restringir 52 pasos semanales, régimen que
[`05_data_acquisition.md`](05_data_acquisition.md) ya identifica como insuficiente.

En su lugar, dos sustratos que sí se pueden construir desde lo almacenado y cuyo eje vertical
tampoco es manufacturado:

| Sustrato | Forma | Eje vertical | Por qué es interpretable |
|---|---|---|---|
| `stack5` | (1, 5, 52) | los 5 índices de vegetación | un kernel 3×3 lee el contraste verdor-humedad *dentro de la misma semana* |
| `pxcube` | (1, 25, 52) | los 25 píxeles de la parcela, ordenados por su media | heterogeneidad fenológica intra-parcela — predictor mecanístico de beta diversidad y LCBD, que son 3 de los 9 targets |

En `pxcube` los 25 píxeles forman **una** muestra, no 25, así que no hay ninguna cuestión de
fuga entre folds: usa los píxeles como *representación*, no como augmentación.

Como control, `curve5` alimenta los mismos 5 índices como **canales** de una señal 1D. El par
`curve5` / `stack5` aísla exactamente lo que gana el eje índice al ser una dimensión espacial
en vez de un canal.

### 2.2 La normalización min-max por muestra está apagada por defecto

Toda transformación de `Trait_2DCNN` termina con `(img - min) / (max - min)` por muestra. Para
reflectancia hiperespectral es defendible. **Para fenología borra el verdor medio y la
amplitud estacional**, que son los correlatos conocidos más fuertes de productividad y
riqueza, y le entrega a la red sólo la *forma*.

El default aquí es `normalize="none"`, con estandarización global (estadísticos del fold de
entrenamiento) aplicada una sola vez. La variante por muestra se conserva como nivel explícito
de la ablación `4c/normalize`, para poder cuantificar lo que cuesta esa convención.

Excepción documentada: **GAF requiere** el reescalado a [−1, 1] porque la codificación polar
lo exige. Ahí no es una convención, es la definición.

### 2.3 El eje DOY es circular, pero no en todos los sustratos

La curva está guardada anclada al valle global (`doy_anchor = 108`, rotación aplicada por el
script 04) y el paso es de exactamente 7 días. El eje envuelve: la semana 51 es vecina de la
0. Rellenar con ceros en el borde inventa un valle que un kernel lee como evento fenológico
real.

Pero el eje **vertical** sólo es circular en las transformaciones cuyos dos ejes son DOY
(`gaf`, `mtf`, `ndi`). En `stack5` las filas son índices y en `pxcube` son píxeles: envolver
eso haría del primer índice un vecino del último, lo que no significa nada. En las imágenes
plegadas 8×8 (`reshape`, `serpentine`, `hilbert`) ambos ejes son artefactos del plegado y
ninguno envuelve. `substrates.PAD_MODE` declara el modo por sustrato y `SepConv2d` soporta
padding mixto por eje.

Además, `reshape` necesita 64 celdas y hay 52: las 12 faltantes se rellenan **envolviendo**
(`curve[:12]`), no con ceros.

### 2.4 Retransformación: el sesgo que hacía negativo el baseline trivial

Los targets se ajustan en escala Yeo-Johnson para que LCBD (escala 1e-3) y riqueza (1–50)
produzcan gradiente comparable. Pero **la inversa de una media condicional no es la media
condicional**: en un target con asimetría 2,4 cae cerca de la mediana, sistemáticamente por
debajo.

Medido: el predictor de la media entrenada devolvía R² = −0,11 en `hill_q0` bajo CV
aleatorio, donde por construcción debe ser ≈ 0.

Corrección: estimador de *smearing* de Duan (1983) — promediar la transformación inversa sobre
la distribución empírica de los residuos del propio modelo, en vez de invertir la predicción
puntual. Con dos guardas, ambas necesarias:

- **Residuos winsorizados** al 2,5–97,5 %. La inversa de Yeo-Johnson es convexa y sin cota, así
  que un solo residuo de la cola domina el promedio: sin esta guarda, RF sobre `hill_q1`
  devolvía R² = −41,8 por una única predicción explotada.
- **Recorte al rango observado en entrenamiento.** Un modelo que extrapola un número de Hill a
  10⁴ especies no está prediciendo, está reportando un artefacto numérico.

Los residuos deben venir de predicciones que el modelo **no ajustó**: out-of-bag para el
bosque, validación interna para las redes. Con residuos in-sample la dispersión es demasiado
pequeña y la corrección se queda corta.

Verificación: tras la corrección, `B00` bajo `kfold5_random` da R² entre −0,03 y +0,00 en los
nueve targets.

### 2.5 `p_lcbd_pa` y `p_lcbd_cover` no son targets

Su correlación de Spearman con el `lcbd_*` correspondiente es **−1,00 exacta**: el valor p de
permutación es una transformación de rango del propio LCBD. Ajustarlos fabricaría dos
"resultados" duplicados y contaría dos veces el LCBD en cualquier pérdida multi-tarea. Están
en una lista `DROPPED` con una aserción que impide incluirlos en una configuración.

### 2.6 Validación cruzada: dos ejes distintos, ninguno sustituye al otro

| Esquema | Unidad | Folds | Rol |
|---|---|---|---|
| `kfold5_owner` | `Owner` (12 contribuyentes) | 5 | **PRIMARIO** |
| `kfold5_block20` | bloque UTM de 20 km (97 bloques) | 5 | **COMPLEMENTO espacial** |
| `lodo_owner` | leave-one-`Owner`-out | 10 | transferibilidad dura |
| `kfold5_dataset` | `metadata_id` (23) | 5 | secundario |
| `kfold5_location` | `Location` (163) | 5 | secundario |
| `lodo_dataset` | leave-one-`metadata_id`-out | 15 | estrés |
| `kfold5_random` | ninguna | 5 | **referencia optimista declarada** |

Por qué `Owner` y no `metadata_id`: ocho proyectos de Ovalle, tres de Galleguillos y dos de
Miranda son `metadata_id` distintos de la misma persona, con el mismo protocolo de terreno y a
menudo los mismos sitios. Dejar fuera un dataset no deja fuera el protocolo; dejar fuera al
dueño sí.

Por qué además bloques geométricos: **ni `Owner` ni `metadata_id` ni `Location` son
geometría.** Medido sobre este subset, 103 pares de datasets tienen *bounding box* solapada, o
sea que dos proyectos "independientes" pueden tener parcelas a unos cientos de metros. Se
auditan bloques de 10, 20 y 25 km; 20 km da 97 bloques con el mejor compromiso tamaño/número.

La brecha `R²(kfold5_random) − R²(kfold5_owner)` se tabula por modelo y **es un resultado**, no
un diagnóstico: es el riesgo R8 de [`02_innovation_and_impact.md`](02_innovation_and_impact.md)
cuantificado.

**Validación interna anidada.** El early stopping necesita un conjunto de validación sacado del
fold de entrenamiento. Si ese corte es aleatorio, parcelas del mismo dueño o del mismo bloque
caen a ambos lados, la época de parada se elige contra una señal con fuga, y el agrupamiento
externo queda deshecho en silencio. `cv.inner_split` agrupa siempre por la **misma** columna
que el esquema externo.

**Puntuación: OOF agrupado.** La varianza de riqueza es ~80 % *entre* grupos, así que un fold
que reciba los proyectos de baja riqueza tiene casi nada de varianza y devuelve R² ≈ 0 por
buena que sea la predicción. Promediar esos cinco números no significa nada. Se puntúa una sola
vez sobre las 1.082 predicciones fuera de fold; las métricas por fold quedan como diagnóstico.
Los esquemas LODO no particionan y se reportan por fold, señalado.

### 2.7 Alfa y beta se resumen por separado

No es presentación, es una consecuencia medida en la primera corrida completa del Tier 1. Bajo
`kfold5_owner`:

| Target | Mejor R² observado | Modelo |
|---|---|---|
| `pcoa1_pa` | **+0,41** | RF sobre LSP de kNDVI |
| `pcoa2_pa` | +0,19 | RF sobre la curva de NDVI |
| `lcbd_pa` | +0,17 | RF sobre la curva de EVI |
| `hill_q0` | **−0,15** (el mejor; `B00` da −0,03) | ninguno supera a la media |
| `hill_q1` | −0,50 | ninguno supera a la media |

Es decir: **la fenología predice composición florística y unicidad composicional, y no predice
diversidad alfa**, cuando la validación deja fuera al contribuyente completo. Tiene una lectura
directa — la riqueza está confundida con el tamaño de parcela y el protocolo, que es
exactamente lo que la CV por dueño elimina; la composición no lo está.

Consecuencia operativa: `scripts/12_model_report.py` reporta `R2_alpha` y `R2_beta` como
columnas separadas y ordena por beta, y `scripts/14_run_matrix.py` **elige el índice y el
sustrato ganadores rankeando sólo por los targets de composición** (`RANK_TARGETS`). Promediar
los seis targets completos elegiría la mitad nula, es decir, ruido.

### 2.8 Riesgo R3 (área de parcela), tratado y no escondido

Spearman entre `PlotSize_m2` y `hill_q0` es **+0,33**. `log10_area` y el estrato de abundancia
entran en **todos** los diseños; dejarlos fuera no elimina la confusión, sólo la esconde.
Además:

- `B02` = RF sobre área sola, cuyo R² se reporta al lado de cada número de riqueza;
- `RF08` = ablación `--no-area` del mejor bloque;
- se encabeza con `hill_q1`/`hill_q2` (ρ con área 0,11 / 0,04) cuando el confundido domina.

Nota de la primera corrida: bajo `kfold5_owner`, `B02` da R² fuertemente negativo en riqueza.
Eso no es un fallo del modelo — es que **el tamaño de parcela está confundido con el
contribuyente** (cada proyecto usa un tamaño fijo), así que el área funciona como un
identificador de dueño y transfiere pésimo. Es el riesgo R3 hecho visible.

### 2.9 Augmentación: el jitter temporal de la documentación es 7× demasiado grueso

[`03_cnn_architecture.md`](03_cnn_architecture.md) prescribe un jitter de ±3–5 días. **Un paso
de esta curva son 7 días**, así que un `np.roll` entero desplazaría 7–35 días — suficiente para
mover el reverdecimiento a otro mes. El jitter implementado es un roll **circular fraccional**
por interpolación (`augment.frac_roll`).

Nada de flips ni rotaciones: destruyen el mapeo eje→significado del que depende todo el diseño.
Una curva fenológica reflejada horizontalmente describe senescencia antes de reverdecimiento,
que no es una planta.

**Mixup mezcla las pérdidas, no los targets.** Tres de las nueve cabezas son NaN para la mitad
de las parcelas; una combinación convexa entre un número y un NaN es NaN, así que mezclar
targets borraría las cabezas de cover en cualquier par que cruce el límite de estrato. La forma
correcta es `λ·L(pred, y_a, m_a) + (1−λ)·L(pred, y_b, m_b)`.

---

## 3. Arquitecturas

Ninguna es la de `Trait_2DCNN`. Ese proyecto alimentaba señales hiperespectrales de 1.721
bandas a backbones `timm` de 4–28 M de parámetros, con miles de muestras por fold. Aquí la
señal son 52 pasos semanales y hay 1.082 parcelas: los mismos backbones memorizan el conjunto
de entrenamiento antes de la época 5. El presupuesto es < 50 k parámetros, logrado con
convoluciones separables en profundidad y *global average pooling* en vez de `flatten`.

Conteos **medidos** (`python scripts/11_run_conv.py --count-params`), con `n_out=9`:

| Modelo | Ancho | Parámetros |
|---|---|---|
| MLP-A / B / C | (64,32) / (128,64) / (256,128,64) | 6.663 / 17.415 / 59.143 |
| Pheno1D-A / B / C | (8,16,32) / (16,32,64) / (32,64,128) | 7.017 / 15.433 / 43.785 |
| PhenoNetS-A / B / C | idem | 5.097 / 16.073 / 55.689 |
| PhenoNetS-X (control) | (128,256,512) | **787.977** |

`PhenoNetS-X` reemplaza al control negativo `efficientnet_b0` de la documentación. `timm` no
está instalado, y en cualquier caso un modelo de la misma familia cuyo único cambio es el ancho
demuestra más limpiamente que lo que falla es el *régimen de datos* y no la familia
arquitectónica.

`Pheno1D` usa kernel de 5 y no de 3: con paso de 7 días, tres taps ven 21 días, menos que el
reverdecimiento que deben detectar.

### Topografía en la CNN: cuatro opciones

| Opción | Mecanismo | Params extra | Comentario |
|---|---|---|---|
| **`late`** (default) | vector de contexto de 28 dims concatenado al embedding de 64 antes de la cabeza | +1.792 | mismo punto de fusión que MLP y 1D, lo que mantiene comparables a las tres familias |
| `none` | sin topografía | 0 | **celda obligatoria**: sin ella no se puede afirmar que la fenología aporte algo |
| `film` | `ctx → MLP → (γ,β)` por canal, aplicado tras los bloques 2 y 4 | +7.264 | la topografía *modula cómo se lee la curva*; ecológicamente la más defendible |
| `patch` | rama paralela sobre `topography_patches.nc` (9,5,5) → GAP → 32 dims | +3.072 | la única que usa heterogeneidad topográfica intra-parcela; empareja con `pxcube` y con los targets de beta |

Se descartó replicar los escalares topográficos como canales constantes de la imagen:
convolucionar sobre un mapa espacialmente constante es desperdicio y BatchNorm elimina la
constante después.

**Guarda obligatoria:** con `late` la cabeza ve la topografía directamente y puede ignorar la
imagen. Por eso `fusion=none` y `B01` (sólo topografía) se reportan al lado de cada resultado
fusionado.

---

## 4. La matriz

`scripts/14_run_matrix.py --all` la corre entera, en este orden, reanudable y consciente del
tiempo disponible.

| Etapa | Contenido | Corridas |
|---|---|---|
| `rf` | Tier 0 (B00–B03) + Tier 1 (RF01–RF08) × 5 índices, más los controles bajo `block20` y `random` | ~85 |
| `c1d` | Pheno1D sobre `curve1d` × 5 índices, más `curve5` | 6 |
| `4a` | screening de los 11 sustratos 2D sobre el índice ganador | 11 |
| `mlp` | MLP01/MLP02 × 5 índices, MLP03/04/05a/05c | 14 |
| `4b` | top-3 sustratos × los 5 índices | ~10 |
| `4c` | barridos de un factor: rotación, normalización, fusión, ancho, mixup, augmentación | 11 |
| `final` | finalistas × 5 esquemas de CV restantes, a 5 semillas | ~30 |

**Poda.** El cruce ingenuo (config × índice × sustrato × esquema × semilla) son 34.000+
corridas. Cuatro reglas lo bajan a ~170: el factor índice se criba una vez con RF (minutos) y
se re-expande sólo para el top-3 de sustratos; el factor esquema se aplica sólo a finalistas;
los barridos son de un factor a la vez, no factoriales; 3 semillas para cribado y 5 para
finalistas.

Cada etapa que depende de resultados (`4b`, `4c`, `final`) lee `results/models/summary.csv` en
el momento de construir su lista de trabajos, así que las elecciones son consecuencia de los
números y no de una suposición previa.

---

## 5. Salidas

```
results/models/
  summary.csv                       una fila por (run_id, scheme, target, seed)
  <run_id>/<scheme>/
      config.json                   configuración completa + hash git + versiones
      oof_predictions.csv           PlotObservationID, fold, seed, <t>_obs, <t>_pred
      per_seed_metrics.csv          puntaje agrupado por semilla — unidad pareada de los tests
      pooled_metrics.csv            media ± sd entre semillas, y el ensemble de semillas
      importance.csv | history.csv | model_seed*.pt
results/tables/model_comparison.csv | optimism_gap.csv | paired_tests.csv
results/interpretation/doy_attribution_<run_id>.csv
results/figures/fig08_model_comparison.pdf … fig13_doy_attribution.pdf
```

Se reportan siempre **dos** números por celda: la media ± sd entre semillas (cuán estable es un
ajuste único, que para un modelo de 16 k parámetros sobre n=1.082 es la cifra honesta) y el
puntaje del promedio de predicciones entre semillas (lo que lograría el modelo desplegado).

Métricas en **unidades originales** tras invertir la transformación: R², RMSE, nRMSE (rango
1–99 %), MAE, sesgo, ρ de Spearman.

**Tests pareados:** t pareado y Wilcoxon sobre los puntajes (target × semilla), con corrección
de Bonferroni dentro de cada familia de comparación. Un ranking sin esto es un ranking de
ruido: con 9 targets y 3–5 semillas, diferencias de 0,02 en R² son rutina.

---

## 6. Interpretabilidad

Un ranking de modelos no es un resultado ecológico. `scripts/13_interpretability.py` lo
convierte en uno:

- **Random Forest:** importancia por permutación sobre las predicciones *fuera de fold*, no por
  impureza. La impureza favorece features de alta cardinalidad, y con 52 columnas de curva
  contra 24 topográficas reportaría la curva como dominante lleve o no señal. Se agrega a nivel
  de bloque (curve / lsp / topo / area / qc).
- **CNN:** Integrated Gradients contra una línea base de la imagen media del fold de
  entrenamiento, replegado al eje de 52 semanas por el mapa inverso propio de cada sustrato
  (`substrates.unfold_to_doy`; `reshape` y `serpentine` invierten exactamente, las
  transformaciones cuadradas usan la marginal por fila). Los pasos se etiquetan con su DOY real
  desde `phenoshape_doy_grid.parquet`.

El producto es atribución contra día del año, un panel por target — la forma en que el
resultado se lee como ecología y no como métrica.

---

## 7. Resultados

Corrida completa del 2026-08-08: 79 modelos, 141 pares modelo × esquema, 1,5 h en 2 GPUs y
32 núcleos. Números en `results/tables/model_comparison.csv`; el registro por trabajo en
[`08_modelling_progress.md`](08_modelling_progress.md).

### 7.1 Qué se predice y qué no

Bajo `kfold5_owner`, promedio sobre los tres targets de composición:

| Familia | Mejor modelo | R²_beta | `pcoa1_pa` | `lcbd_pa` | `pcoa2_pa` |
|---|---|---|---|---|---|
| **RF** | `RF06` — curvas de los 5 índices | **+0,269** | +0,408 | +0,208 | +0,190 |
| **C2D** | `C2D11` — `pxcube`, kNDVI, sin fusión | +0,168 | +0,317 | +0,137 | +0,049 |
| BASE | solo coordenadas (`B03`) | +0,138 | +0,166 | +0,065 | +0,182 |
| MLP | curva de kNDVI | +0,126 | +0,259 | +0,075 | +0,045 |
| C1D | curva 1D de kNDVI | +0,093 | +0,162 | +0,100 | +0,016 |

Ningún modelo supera a la media de entrenamiento en **diversidad alfa** (§2.7). El resultado
del proyecto es composicional.

### 7.2 Los contrastes, con test pareado por parcela

El test agregado no puede resolverlos: con tres targets y tres semillas hay nueve valores
pareados, el Wilcoxon tiene un p mínimo alcanzable de 2/2⁹ = 0,0039 y Bonferroni sobre los
quince pares de cabecera lo deja en 0,059 — por encima de α, así que no puede rechazar por
grande que sea el efecto. Los contrastes se resuelven con el test pareado sobre el error
cuadrático de las **mismas 1.082 parcelas** (`results/tables/headline_plot_tests.csv`).

| Pregunta | Contraste | `lcbd_pa` | `pcoa1_pa` | `pcoa2_pa` |
|---|---|---|---|---|
| ¿Hay señal? | RF vs media de entrenamiento | **sí** p=3e-28 | **sí** p=9e-73 | **sí** p=2e-23 |
| ¿Aporta la fenología sobre la pura geografía? | RF vs solo coordenadas | **sí** p=5e-4 | **sí** p=6e-18 | ns |
| ¿Aporta la segunda dimensión? | mejor CNN 2D vs CNN 1D | **sí** p=2e-2 | **sí** p=2e-10 | ns |
| ¿La mejor CNN supera la pura geografía? | CNN 2D vs solo coordenadas | **sí** p=2e-3 | **sí** p=6e-14 | ns |
| ¿Gana el deep learning? | RF vs mejor CNN 2D | ns | ns | ns |
| ¿Aporta la convolución 1D sobre el MLP? | CNN 1D vs MLP | ns | **no**, gana el MLP p=5e-12 | ns |

Leído junto: **la segunda dimensión sí aporta, la convolución sobre la curva 1D no, y el
Random Forest empata con la mejor CNN.** El riesgo R1 de
[`02_innovation_and_impact.md`](02_innovation_and_impact.md) —"Random Forest puede ganar"— no
se materializó como derrota, pero tampoco hay victoria del deep learning: la comparación queda
en empate estadístico con RF por delante en R² puntual.

El sustrato ganador es **`pxcube`**, uno de los dos cuyo eje vertical no es manufacturado. Que
la heterogeneidad fenológica intra-parcela gane justo en los targets de beta diversidad es
coherente con el mecanismo, no un accidente de ranking.

### 7.3 La brecha de optimismo (riesgo R8)

R² medio sobre los targets de composición, mismos modelos, distintos esquemas:

| Modelo | aleatorio | bloque 20 km | por dueño | brecha aleatorio − dueño |
|---|---|---|---|---|
| `RF06` curvas 5 índices | 0,550 | 0,396 | 0,269 | **0,281** |
| `RF01` LSP kNDVI | 0,537 | 0,383 | 0,238 | 0,299 |
| solo coordenadas | 0,568 | 0,277 | 0,138 | **0,431** |
| solo topografía + área | 0,422 | 0,219 | −0,158 | **0,580** |
| solo área | 0,291 | 0,104 | −0,446 | **0,737** |

Un CV aleatorio habría reportado R² = 0,55 para el mejor modelo y **0,57 para un modelo que solo
conoce las coordenadas**. La brecha crece cuanto menos información real tiene el modelo, que es
exactamente el diagnóstico: el CV aleatorio mide autocorrelación, no capacidad predictiva. Los
bloques geométricos de 20 km quedan a mitad de camino porque controlan el espacio pero no el
protocolo de muestreo.

### 7.4 Ablaciones

| Factor | Niveles y R²_beta | Lectura |
|---|---|---|
| **Normalización de la curva** | none +0,004 · global −0,187 · **perSample −0,312** | El min-max por muestra de `Trait_2DCNN` borra amplitud y verdor medio. Cuesta 0,32 de R². |
| **Fusión topográfica** | **none +0,117** · patch +0,082 · late +0,004 · film −0,111 | La CNN funciona mejor *sin* topografía. Con la concatenación en la cabeza el tronco convolucional se degrada — la guarda del §7 del diseño detectó justo eso. |
| **Ancho** | A 4,9k −0,010 · B 15,6k +0,004 · C 54,8k +0,009 · X 784k +0,026 | Con n=1.082 el techo no es la capacidad. El control negativo sobre-parametrizado no se desploma. |
| **Rotación DOY** | trough +0,004 · calendario −0,003 | Sin efecto medible. |
| **Índice de vegetación** | kNDVI > ndvi ≈ evi ≈ nbr > savi | kNDVI supera a savi/nbr/evi con p_bonf < 0,001 en la familia RF. |

### 7.5 Píxel central contra media 5×5

Cada predictor existe a dos escalas: el píxel Landsat que contiene el centroide de la parcela,
o un resumen de la ventana 5×5 alrededor. El default era la media 5×5 y era un juicio, no una
medición. `scripts/15_pixel_ablation.py` lo mide con emparejamiento exacto — mismo modelo,
índice, folds y semilla, difiere solo el footprint — sobre RF01/RF03/RF04 y la CNN 1D × 5
índices × 3 semillas × 2 niveles.

Se corrieron dos variantes porque responden preguntas distintas:

- **`RF01p`** mueve **solo el bloque LSP**, dejando topografía y área en 5×5. Comparación
  controlada de ese bloque.
- **`--px center`** mueve **todo el diseño** a la vez (LSP, curva y topografía). Mezclar
  footprints confundiría las dos decisiones.

Ambas dan lo mismo, y en la dirección del default:

| | Δ (central − 5×5) | pareado | t | Wilcoxon |
|---|---|---|---|---|
| Solo bloque LSP, composición | **−0,0075** | n=45 | p=0,016 | p=0,016 |
| Todo el diseño, composición | **−0,0147** | n=174 | p=0,0055 | p=0,0003 |

Los cinco índices apuntan en la misma dirección, que es lo que le da peso: el efecto es chico
—0,015 de R² sobre valores de ~0,25, un 6 % relativo— pero consistente. **La media 5×5 se
queda como default.**

Desagregado por familia el resultado se parte, y esa es la parte informativa:

| | composición | alfa | gana el central |
|---|---|---|---|
| Random Forest | −0,009 (p=0,0006) | **+0,043** | 11/15 en alfa, 3/15 en composición |
| CNN 1D | −0,031 (p=0,11) | **−0,268** | 0/5 en alfa, 1/5 en composición |

**La CNN 1D es la que se hunde con el píxel único** (−0,27 en alfa, 0 de 5 modelos a favor),
mientras el bosque casi no lo nota. Es coherente con el mecanismo: promediar 25 píxeles suaviza
exactamente el ruido de alta frecuencia que un kernel de ancho 5 ajustaría, y un bosque sobre
18 métricas escalares no tiene dónde ajustarlo. Fue el motivo de incluir la CNN 1D en la
ablación y no solo bosques.

En alfa el píxel central mejora al bosque (+0,043, y +0,094 en la variante de solo-LSP), pero
las dos variantes siguen por debajo de la media de entrenamiento: es una diferencia entre dos
modelos que no sirven. Se probaron dos explicaciones —encogimiento hacia la media y calce de
footprint— y **ninguna se sostiene**: el central tiene *más* varianza de predicción, no menos,
y su ventaja crece con el tamaño de parcela (ρ = +0,17 con log-área) en vez de concentrarse en
las parcelas chicas como predeciría el argumento de footprint. Queda como medición sin
mecanismo validado.

Argumento adicional para el 5×5, independiente del R²: el diseño central arrastra **200 celdas
NaN contra 2**, porque una estación degenerada en un solo píxel deja `los/ios/sw/mos`
indefinidas en 113 parcelas.

### 7.6 Interpretabilidad

Importancia por permutación sobre las predicciones fuera de fold, agregada por bloque
(`RF06`): para `pcoa1_pa` la curva y las LSP aportan el 63 %, la topografía el 14 % y el área
el 2 %; para `hill_q0` el área sube al 11 %. Es decir, el confundido con esfuerzo de muestreo
pesa donde se esperaba y no donde está el resultado.

Integrated Gradients sobre la CNN finalista, replegado al eje de 52 semanas: el pico de
atribución cae en **DOY 293–307** para las nueve facetas — finales de octubre a inicios de
noviembre, plena primavera austral, la fase de reverdecimiento.

---

## 8. Limitaciones que se declaran, no se resuelven

- **LCBD y los ejes PCoA son cantidades relativas al conjunto de datos**, calculadas una vez
  sobre las 1.082 parcelas por el script 07. El modelo predice "el LCBD de esta parcela *dentro
  de este muestreo*", no un absoluto transferible. No es fuga de predictores y es la práctica
  estándar para LCBD, pero debe decirse.
- **El phenocube año × DOY queda fuera de alcance** (§2.1). Reconstruirlo requiere volver a los
  1.082 `.nc` del hub y re-ajustar PhenoShape por año.
- **113 parcelas tienen riqueza < 2** y una posición PCoA degenerada. La corrida principal las
  conserva; la sensibilidad `--min-richness 2` corre sobre el mismo conjunto de folds.
- **Las nueve cabezas no son independientes**: r(hill_q1, hill_q2) = 0,95, r(lcbd_pa,
  lcbd_cover) = 0,90. `hill_q2` se mantiene en la cabeza multioutput como regularizador
  multi-tarea pero se reporta como secundario.
- **Comparaciones múltiples** sobre ~170 corridas. Todas las afirmaciones salen de la matriz
  ordenada declarada arriba; Bonferroni dentro de cada familia; la separación cribado (4a) →
  confirmación (4b, sobre índices que el cribado no usó) es la guarda principal.
