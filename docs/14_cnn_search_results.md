# Busqueda de la 2D-CNN: resultados

**Esquema:** `kfold5_window` unicamente. **Metrica:** R2 fuera de muestra, promediado primero dentro de cada faceta y luego entre las cinco, para que beta (3 targets por estrato) no pese el triple que diversidad oscura (1 target).

**Generado por** `scripts/33_search_report.py` desde `results/models/summary.csv` (195 corridas). No editar a mano.


## 1. Mejor corrida de cada familia

| corrida | alfa | beta p/a | beta cob. | filo | oscura | **media** |
|---|---:|---:|---:|---:|---:|---:|
| MLP06_curve_kndvi_raw100 | 0.519 | 0.529 | 0.342 | 0.164 | 0.427 | 0.396 |
| C2D02_serpentine_kndvi_raw36_ctxclim-topo-area | 0.508 | 0.509 | 0.328 | 0.175 | 0.394 | 0.383 |
| RF06_curve_all-topo-area_raw100 | 0.422 | 0.534 | 0.378 | 0.159 | 0.413 | 0.381 |
| B03_coords | 0.490 | 0.508 | 0.344 | 0.090 | 0.382 | 0.363 |
| C1D01_curve1d_kndvi_raw36_ctxclim-topo-area | 0.461 | 0.491 | 0.309 | 0.163 | 0.375 | 0.360 |

Diferencia entre el primero y el segundo: **0.013**. El ruido de semilla medido en este proyecto es ~0,011 de desviacion, asi que 2 sd = 0,022 es el umbral por debajo del cual dos corridas no se distinguen.


## 2. La serie cruda de 3 anos contra el ano compuesto

`PhenoShape` colapsa tres anos en un ano promedio. La serie cruda los mantiene en tiempo calendario, y con ellos la variacion interanual -- que en estas parcelas es **un tercio de la variacion temporal total** (sd entre anos 0,022 contra sd temporal 0,062, medido sobre kNDVI).


### Resolucion de la serie (serpentine + kNDVI + contexto)

| curva | alfa | beta p/a | beta cob. | filo | oscura | **media** |
|---|---:|---:|---:|---:|---:|---:|
| compuesto (52) | 0.464 | 0.489 | 0.308 | 0.168 | 0.380 | 0.362 |
| 24 pasos | 0.507 | 0.498 | 0.315 | 0.166 | 0.407 | 0.379 |
| 36 pasos | 0.508 | 0.509 | 0.328 | 0.175 | 0.394 | 0.383 |
| 48 pasos | 0.496 | 0.504 | 0.314 | 0.162 | 0.410 | 0.377 |
| 60 pasos | 0.495 | 0.503 | 0.320 | 0.163 | 0.396 | 0.375 |
| 68 pasos | 0.495 | 0.507 | 0.322 | 0.173 | 0.390 | 0.377 |
| 100 pasos | 0.517 | 0.494 | 0.322 | 0.169 | 0.405 | 0.381 |
| 156 pasos | 0.509 | 0.492 | 0.315 | 0.173 | 0.403 | 0.378 |
| 196 pasos | 0.498 | 0.493 | 0.315 | 0.159 | 0.396 | 0.372 |

Las 8 resoluciones se reparten **0.011**, por debajo de 2 sd de ruido de semilla: la resolucion **no importa**. Lo que importa es usar los tres anos en vez del promedio. Una version anterior de esta tabla mostraba una tendencia monotona; era un artefacto de truncamiento (`docs/13`, y `tests/test_step_cols.py`), no un resultado.


### A quien beneficia

La pregunta que decide la interpretacion: si la serie cruda sube a todas las familias por igual, entonces es **mejor informacion**; si sube preferentemente a la convolucion, entonces la imagen 2D lee estructura que una tabla de columnas no.

| corrida | familia | compuesto | mejor cruda | **ganancia** |
|---|---:|---:|---:|---:|
| C2D02_serpentine_kndvi_ctxclim-topo-area | C2D | 0.362 | 0.383 | 0.021 |
| C2D01_reshape_kndvi_ctxclim-topo-area | C2D | 0.358 | 0.378 | 0.020 |
| RF06_curve_all-topo-area | RF | 0.367 | 0.381 | 0.014 |
| MLP07_curve_all | MLP | 0.373 | 0.387 | 0.014 |
| MLP06_curve_kndvi | MLP | 0.386 | 0.396 | 0.010 |
| C1D01_curve1d_kndvi_ctxclim-topo-area | C1D | 0.359 | 0.360 | 0.001 |


## 3. Confirmacion con semillas disjuntas

La busqueda selecciono sobre las semillas {0,1,2} y reporta el maximo, lo que sobrestima. Estas corridas repiten las finalistas con semillas **{10..14}**, que no participaron en la seleccion. Se aplica tambien a los competidores: si solo se contrajera la CNN, la comparacion quedaria sesgada al reves.

| corrida | familia | seleccion {0,1,2} | **confirmacion {10..14}** | contraccion |
|---|---:|---:|---:|---:|
| MLP06_curve_kndvi_raw100 | MLP | 0.396 | 0.396 | 0.000 |
| MLP07_curve_all_raw36 | MLP | 0.387 | 0.386 | -0.000 |
| C2D02_serpentine_kndvi_raw24_ctxclim-topo-area | C2D | 0.379 | 0.383 | 0.005 |
| RF06_curve_all-topo-area_raw100 | RF | 0.381 | 0.383 | 0.002 |
| C2D02_serpentine_kndvi_raw36_ctxclim-topo-area | C2D | 0.383 | 0.383 | -0.000 |
| MLP06_curve_kndvi | MLP | 0.386 | 0.382 | -0.003 |
| C2D01_reshape_kndvi_raw100_ctxclim-topo-area | C2D | 0.378 | 0.380 | 0.002 |
| C2D07_hilbert_kndvi_raw36_ctxclim-topo-area | C2D | 0.378 | 0.379 | 0.001 |
| C2D02_serpentine_kndvi_raw100_ctxclim-topo-area | C2D | 0.381 | 0.379 | -0.003 |
| C2D09_spectrogram_kndvi_raw36_ctxclim-topo-area | C2D | 0.377 | 0.373 | -0.004 |
| MLP07_curve_all | MLP | 0.373 | 0.373 | 0.001 |
| RF06_curve_all-topo-area | RF | 0.367 | 0.367 | 0.000 |
| B03_coords | BASE | 0.363 | 0.363 | 0.000 |
| C1D01_curve1d_kndvi_raw36_ctxclim-topo-area | C1D | 0.360 | 0.362 | 0.002 |

Contraccion mediana: **+0.000**.


## 3b. El piso de ruido, medido por accidente

Al anadir los sustratos `_5idx`, el bucle por defecto corrio los cinco indices sobre un sustrato que **ignora** el indice: cinco corridas de configuracion identica, misma semilla, mismo fold, mismos datos. Dieron:

| repeticion | media |
|---|---:|
| 1 | 0.3946 |
| 2 | 0.3935 |
| 3 | 0.3907 |
| 4 | 0.3907 |
| 5 | 0.3713 |

Dispersion **0,023**, desviacion 0,010 -- a semilla fija. No es ruido de semilla: es no-determinismo de GPU (autotune de cuDNN, reducciones atomicas). Es tan grande como la distancia entre familias, y de ahi salen dos reglas que este informe respeta:

- **Ninguna corrida de una sola semilla es interpretable.** Una prueba rapida de esta misma configuracion dio 0,395 y parecia batir al MLP; con tres semillas da 0,377. Era el extremo afortunado del rango.
- **Una diferencia por debajo de 0,022 (2 sd) no distingue dos modelos.** Es el criterio que se fijo antes de mirar los resultados y no se ha movido despues.


## 4. Lo que se probo y no funciono

Se publica entero. Que la mayoria de las combinaciones no mejore es tan informativo como que una lo haga, y sin la rejilla completa el ganador no se puede interpretar.

| se probo | resultado |
|---|---|
| 10 reconstructores x 2 modos de borde | 20 corridas dentro de 0,010 -- por debajo de 1 sd de ruido de semilla. El reconstructor de la curva no importa. |
| combinar los factores positivos de la ablacion 4c | `ctx`+`noaug`+`wC` no se suman: 0,356-0,363 contra 0,362 de `ctx` solo. |
| tres arquitecturas nuevas (residual, squeeze-excitation, multiescala) | 0,342-0,359, todas por debajo de la separable simple de 17k parametros. |
| aumentacion: termino de pendiente, magnitudes escaladas, probabilidad 0,15 | 0,355-0,362. Ninguna variante supera a no aumentar. |
| los cinco indices como canales de la imagen (`_5idx`) | 0,370-0,382 contra 0,383 de un solo indice. Corregia una asimetria real -- RF06 y MLP07 leen cinco indices y cada C2D leia uno -- pero la asimetria no era lo que costaba la comparacion. |
| preentrenamiento por enmascarado (MAE) sobre 135.250 curvas de pixel | 0,359 contra 0,362 sin preentrenar. El MAE aprende (MSE de reconstruccion 0,0095 -> 0,0031) pero no transfiere: las 135.250 curvas salen de las ventanas 5x5 de las mismas 1.082 parcelas y son casi redundantes con ellas (dimension de participacion 1,3 contra 1,2). 125x mas imagenes cubriendo 1,08x mas espacio. |

## 5. Metricas por faceta, finalistas


### R2 (mayor es mejor)

| corrida | alfa | beta p/a | beta cob. | filo | oscura | **media** |
|---|---:|---:|---:|---:|---:|---:|
| MLP06_curve_kndvi_raw100 | 0.519 | 0.529 | 0.342 | 0.164 | 0.427 | 0.396 |
| MLP06_curve_kndvi_raw36 | 0.504 | 0.508 | 0.338 | 0.188 | 0.416 | 0.391 |
| MLP07_curve_all_raw36 | 0.484 | 0.520 | 0.351 | 0.162 | 0.415 | 0.387 |
| MLP06_curve_kndvi | 0.494 | 0.497 | 0.345 | 0.178 | 0.414 | 0.386 |
| C2D02_serpentine_kndvi_raw36_ctxclim-topo-area | 0.508 | 0.509 | 0.328 | 0.175 | 0.394 | 0.383 |
| C2M02_serpentine_5idx_all_raw100_ctxclim-topo-area | 0.508 | 0.506 | 0.326 | 0.173 | 0.394 | 0.382 |
| C2D02_serpentine_kndvi_raw100_ctxclim-topo-area | 0.517 | 0.494 | 0.322 | 0.169 | 0.405 | 0.381 |
| RF06_curve_all-topo-area_raw100 | 0.422 | 0.534 | 0.378 | 0.159 | 0.413 | 0.381 |
| B03_coords | 0.490 | 0.508 | 0.344 | 0.090 | 0.382 | 0.363 |
| C1D01_curve1d_kndvi_raw36_ctxclim-topo-area | 0.461 | 0.491 | 0.309 | 0.163 | 0.375 | 0.360 |

### %RMSE (menor es mejor)

| corrida | alfa | beta p/a | beta cob. | filo | oscura | **media** |
|---|---:|---:|---:|---:|---:|---:|
| MLP06_curve_kndvi_raw100 | 13.9 | 16.7 | 21.2 | 17.2 | 18.1 | 17.4 |
| MLP06_curve_kndvi_raw36 | 14.1 | 17.0 | 21.3 | 16.9 | 18.3 | 17.5 |
| MLP07_curve_all_raw36 | 14.4 | 16.8 | 21.1 | 17.2 | 18.3 | 17.6 |
| MLP06_curve_kndvi | 14.2 | 17.2 | 21.2 | 17.0 | 18.3 | 17.6 |
| C2D02_serpentine_kndvi_raw36_ctxclim-topo-area | 14.0 | 17.0 | 21.4 | 17.1 | 18.6 | 17.6 |
| C2M02_serpentine_5idx_all_raw100_ctxclim-topo-area | 14.0 | 17.1 | 21.5 | 17.1 | 18.6 | 17.7 |
| C2D02_serpentine_kndvi_raw100_ctxclim-topo-area | 13.9 | 17.3 | 21.5 | 17.1 | 18.5 | 17.7 |
| RF06_curve_all-topo-area_raw100 | 15.1 | 16.6 | 20.7 | 17.3 | 18.3 | 17.6 |
| B03_coords | 14.2 | 17.0 | 21.2 | 17.9 | 18.8 | 17.8 |
| C1D01_curve1d_kndvi_raw36_ctxclim-topo-area | 14.7 | 17.3 | 21.8 | 17.2 | 18.9 | 18.0 |

### Que dice la desagregacion

Promediando **todas** las corridas, las facetas no son igual de predecibles:

| faceta | R2 medio de todas las corridas |
|---|---:|
| beta p/a | 0.459 |
| alfa | 0.405 |
| oscura | 0.345 |
| beta cob. | 0.283 |
| filo | 0.154 |

La faceta **filo** es la mas dificil por un margen amplio (0.154 contra 0.459 de beta p/a), y eso ordena el ranking entero: las corridas que ganan lo hacen sobre todo por ahi.

