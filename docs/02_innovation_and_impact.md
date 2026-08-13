# Innovación e impacto — evaluación

**Proyecto:** biodiversidad multi-faceta multitemporal desde fenología satelital, Chile central
**Fecha:** agosto 2026
**Base:** `01_state_of_the_art.md`, inspección directa de Parcelas-CL, Rasgos-CL, `PhenoSensing` y `Trait_2DCNN`

---

## Veredicto en una línea

**Defendible ante un referee:** que la curva fenológica multianual predice varias facetas de diversidad mejor que la heterogeneidad espectral de una fecha, y que predecirlas conjuntamente con una red multi-tarea es preferible a un modelo por métrica cuando n es pequeño.
**No defendible sin datos adicionales:** que esto mide "diversidad funcional" en el sentido de la literatura espectro-rasgo, y que la dark diversity estimada sobre un pool no saturado es interpretable.

---

## 1. Innovación, graduada

### 1.1 Genuinamente novedoso — sin precedente localizado

**(a) La curva fenológica completa como entrada de una CNN vía transformación señal→imagen.**
`Trait_2DCNN` implementa GAF, MTF, espectrograma y CWT: cuatro transformaciones **diseñadas para señales temporales**, aplicadas allí a un eje espectral. Devolverlas a un eje temporal real es teóricamente más limpio que su uso original, y el argumento se puede escribir así en el paper. El coste de implementación es bajo: el contrato de transformación es una sola función `transform(signal) -> (H,W[,C])` (`transforms/base.py:11-21`), y el número de bandas solo entra vía `int(np.ceil(np.sqrt(n)))` en `reshape_transform.py:18`. Cambian `n_channels` y la lista de targets; nada más fuera de `transforms/`.

Complemento directo: `PhenoSensing.PhenoShape()` ya devuelve la curva reconstruida en una grilla regular de `nGS=52` pasos por píxel, con nueve reconstructores intercambiables. La entrada del CNN está resuelta de fábrica.

**(b) Predicción conjunta multi-faceta desde un tronco compartido.**
El multi-tarea existe para rasgos (Cherif et al. 2023) y para rangos taxonómicos (Gillespie et al. 2024), no para facetas de diversidad. La justificación no es decorativa: las facetas están correlacionadas entre sí, y con n del orden de 10³ el multi-tarea actúa como regularizador —el tronco compartido tiene que aprender una representación que sirva para todas las cabezas a la vez.

Detalle operativo ya resuelto: `MaskedMSELoss` (`training/losses.py:5-15`) calcula `sum(loss * mask) / mask.sum()` con la máscara construida desde los NaN de cada muestra (`training/data_loader.py:39-40`). Esto encaja exactamente con el diseño de dos niveles de abundancia acordado: las parcelas de solo presencia tendrán NaN en las cabezas ponderadas y contribuirán igual a las cabezas binarias, sin necesidad de descartarlas ni de imputar.

**(c) Emparejamiento retrospectivo parcela–año sobre 1999–2026.**
Cada parcela aporta la fenología de *su propio año de censo*. La literatura de teledetección de biodiversidad usa casi siempre una imagen o un año único. Parcelas-CL tiene picos de muestreo en 1999 (165 parcelas), 2010 (102), 2013 (143), 2014 (109), 2019 (126), 2020 (95), 2021 (79) y 2022 (254): la variación temporal es sustancial y descartarla dejaría fuera la mitad del dataset. `get_timeseries_metrics(window_length=N)` en `PhenoSensing` ya produce métricas por ventana móvil de años.

**(d) Chile mediterráneo.**
Hotspot de biodiversidad prácticamente ausente de esta literatura. Los antecedentes locales (Ceballos et al. 2015; Lopatin et al. 2016) son de una fecha y una faceta. Sería el primer modelo multi-faceta de la región.

**(e) Dark diversity desde teledetección.**
Sin precedente localizado. Mantener como objetivo exploratorio en una sección separada, nunca como pilar del argumento —ver el riesgo R7.

### 1.2 Incremental — reconocerlo explícitamente en el paper

- **Random Forest por métrica** es el baseline estándar. No es contribución; es el control contra el cual se mide la CNN.
- **Métricas LSP + DEM como predictores** es terreno muy transitado. Lo nuevo es la respuesta multi-faceta, no el predictor.
- **Regresión sobre ejes NMDS** tiene precedente directo y cuantificado: Adams et al. (2019) explicaron 61%/49%/25% de los ejes NMDS 1–3 con Landsat OLI + terreno y Random Forest en 699 parcelas. Ese es el número a superar, y debe citarse como tal en la introducción, no evitarse.

---

## 2. Riesgos, cuantificados

| # | Riesgo | Magnitud concreta | Mitigación |
|---|---|---|---|
| **R1** | n insuficiente para CNN | Tras filtrar a Chile central y a años con cobertura satelital útil quedan del orden de 800–1.000 parcelas, contra backbones `timm` de millones de parámetros. `Trait_2DCNN` entrenó con miles de muestras por fold | CNN pequeña propia o 1D-CNN sobre la curva; **reusar el pretraining MAE-2D** de `Trait_2DCNN` (`models/mae_2d.py`, `training/pretrain_mae.py`) sobre píxeles chilenos sin etiquetar —la maquinaria ya existe y fue diseñada para 139K muestras no etiquetadas; y reportar honestamente si RF gana |
| **R2** | Diversidad funcional débil | Rasgos-CL tiene **2 rasgos continuos** (`Max_plant_height`, `Seed_mass`) y 21 categóricos; **no hay SLA/LMA, densidad de madera, LDMC ni N foliar**. Cobertura por rasgo entre 5,6% y 100%. Micorrizas y fijación de N asignadas a nivel de género. `Nfixation` es asimétrico: NA ≠ "No" | Declararlo en métodos sin rodeos; FDis/RaoQ sobre distancia de Gower mixta; evaluar complementar con TRY o GIFT para las mismas 675 especies. Los autores de Rasgos-CL **desaconsejan explícitamente** la imputación filogenética ingenua con cobertura <60%, así que no imputar |
| **R3** | Riqueza confundida con esfuerzo de muestreo | Área de parcela entre 78,5 y 10.000 m² (factor 127×), mediana 400 m², **10,8% sin dato** (160 parcelas). Los autores de Parcelas-CL advierten explícitamente que la riqueza está fuertemente confundida con la intensidad de muestreo | Área como covariable u offset obligatorio; análisis de sensibilidad excluyendo las 160 parcelas sin área; Hill numbers con rarefacción; nunca reportar riqueza cruda como target principal |
| **R4** | Curva fenológica pobre antes de 2013 | Landsat 5/7 con revisita nominal de 16 días; Landsat 7 con SLC-off desde 2003. Los años 1999 (165 parcelas) y 2010 (102) son picos de muestreo justo en la zona débil de la serie | Chile central tiene verano despejado, lo que ayuda de verdad. `upper_envelope` y Whittaker ponderado por QA (`qa_to_weight` con spec `LANDSAT_C2`) ya están en `PhenoSensing`. Añadir el **número de observaciones válidas por parcela-año** como covariable de calidad y como criterio de exclusión |
| **R5** | Armonización Landsat↔Sentinel-2 | `PhenoSensing` **no tiene compositing ni armonización cross-sensor** —verificado en el código; solo agrupa observaciones sobre el eje DOY. `rasterstats` y `shapely` están declarados como dependencias pero no se importan en ninguna parte | Usar **HLS v2** (L30+S30, 30 m, ya armonizado por NASA) para el período 2015+, y Landsat C2 SR directo para 1999–2015. Tratar el sensor como **estrato**, con un modelo o un término por estrato, no mezclar ambas fuentes en una sola serie sin corrección |
| **R6** | No estacionariedad por megasequía | Chile central en megasequía desde 2010; la relación fenología–diversidad puede derivar dentro del propio período de estudio | Convertirlo en pregunta en lugar de tratarlo como ruido: la variabilidad fenológica interanual es señal (Lopatin 2023), y la evidencia de Dronova et al. (2022) va en la misma dirección. Es el gancho del segundo paper |
| **R7** | Dark diversity sobre pool no saturado | Las curvas de acumulación de Parcelas-CL **no alcanzan asíntota** (declarado por sus autores) → el pool regional está incompletamente capturado, y `DarkDiv` estima el pool desde co-ocurrencias | Sección exploratoria separada, con la limitación declarada. No incluir en las conclusiones principales. Alternativa más segura: reportar el ratio de completitud de comunidad con intervalos, no valores puntuales |
| **R8** | Sesgo espacial extremo | Parcelas agrupadas en Valparaíso, Metropolitana y Maule; 194 localidades para 1.485 parcelas; un solo contribuyente aporta 2.929 registros de 10.821 | Block CV espacial obligatorio, nunca CV aleatorio. Reportar **ambos** para exponer la magnitud de la brecha —ese contraste es en sí un resultado útil. Considerar además CV por contribuyente (`Owner`) o por proyecto (`metadata_id`) como prueba de transferibilidad |
| **R9** | ~~Bug conocido en `get_timeseries_metrics`~~ **Resuelto** | En `accessor.py:460` el RMSE se computaba contra el cubo completo `ds`, no contra la ventana `sample`. Corregido en PhenoSensing (`ds`→`sample`, commit `ddaa3bc`) y usado para la métrica de estabilidad interanual — ver `docs/18_lsp_stability_pca.md` §1 | — |

**Riesgo de trazabilidad, no de método:** los diseños anidados existen a nivel de proyecto (fractales en `md022`, subparcelas de 1 m² en `md016`–`md018`, transectos en `md004`) pero **no están codificados en el CSV liberado**. La tabla es estrictamente a nivel de parcela. El título del dataset promete "multi-scale" pero el análisis multi-escala requeriría volver a los contribuyentes originales.

---

## 3. Impacto

### 3.1 Salidas

**Dos papers separables**, para no cargar uno solo con todo:

1. **Metodológico / benchmark.** Facetas múltiples de biodiversidad desde fenología satelital: curva completa vs métricas LSP escalares, Random Forest por métrica vs CNN multi-tarea, con validación cruzada por bloques espaciales. Objetivo de comparación explícito: superar el 61%/49%/25% de Adams et al. (2019) en ejes de ordenación usando fenología en vez de reflectancia de una fecha. Revistas candidatas: *Remote Sensing of Environment*, *Ecological Informatics*, *Methods in Ecology and Evolution*.

2. **Aplicado.** Cambio multitemporal de biodiversidad en Chile central bajo megasequía, usando el modelo del paper 1 aplicado retrospectivamente. Encaja con la línea de browning y resiliencia ya presente en el vault.

### 3.2 Por qué importa más allá del paper

- **Ejecuta un llamado explícito.** Las conclusiones de Parcelas-CL nombran la interoperabilidad con Rasgos-CL como la vía para análisis multi-faceta entre escalas. Nadie la ha ejecutado.
- **Encaje EBV / GEO BON.** Varias de las facetas propuestas están en la lista prioritaria de métricas de biodiversidad observables desde el espacio (Skidmore et al. 2021), lo que da un marco de política para la sección de implicancias.
- **Encaje de financiamiento.** ANID FONDECYT Iniciación 11241088 y FSEQ210022 ya financian Parcelas-CL, según sus agradecimientos. La continuidad temática es directa.
- **Retorno sobre repos propios.** `PhenoSensing` gana un caso de uso de biodiversidad publicado —hoy tiene notebooks de demostración pero ningún paper de aplicación. `Trait_2DCNN` gana evidencia de que las transformaciones señal→imagen generalizan fuera del dominio espectral, que es la pregunta abierta más obvia de ese manuscrito.
- **Reproducibilidad.** Ambas bases son públicas (Zenodo, GitHub) y ambos repos de método son propios. El estudio completo es reproducible de extremo a extremo sin datos restringidos, lo que es argumento de venta real en revisión.

### 3.3 Lo que este proyecto **no** va a demostrar

Conviene fijarlo ahora para no escribir un abstract que no se sostenga:

- No demuestra causalidad entre fenología y diversidad. Es predicción, no mecanismo.
- No produce un mapa continuo validado de biodiversidad para Chile central sin un análisis de extrapolación explícito: 194 localidades con fuerte agrupamiento no cubren el espacio ambiental de la región.
- No mide diversidad funcional en el sentido del espectro de economía foliar. Mide diversidad de forma de crecimiento, altura, masa de semilla, dispersión y reproducción. Es una faceta legítima, pero hay que nombrarla con precisión.
- No resuelve el problema del tamaño de parcela variable; lo controla estadísticamente.

---

## 4. Recomendación

Proceder, con tres condiciones de diseño fijadas desde el inicio:

1. **Block CV espacial desde el primer experimento**, no añadido al final. Con este grado de agrupamiento, un CV aleatorio produce R² inflados que después hay que retractar.
2. **RF como control de primera clase**, no como paja. Si RF gana, ese es el resultado y es publicable —Pettorelli et al. (2024) señalan la escasez de datos de entrenamiento como el cuello de botella real del deep learning en este dominio, y un caso bien documentado de n insuficiente aporta.
3. **Dark diversity en una sección exploratoria aislada**, con la limitación del pool no saturado declarada en el mismo párrafo donde se presenta el resultado.

El siguiente paso lógico es el EDA de Parcelas-CL: cuantificar exactamente cuántas parcelas sobreviven a cada filtro (Chile central, año no nulo, cobertura Landsat/HLS, tamaño de parcela conocido, abundancia comparable). Ese número decide si la vía CNN es viable o si el proyecto es un estudio de Random Forest con una comparación honesta de por qué el deep learning no aplica todavía.
