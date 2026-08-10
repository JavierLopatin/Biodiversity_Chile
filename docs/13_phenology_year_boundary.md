# La frontera del año en la curva fenológica

**Estado:** tres problemas distintos, los tres corregidos upstream
(`PhenoSensing` `eb2dff8` y `429cbe0`). Falta re-extraer y decidir si `_v2lin` pasa a
default.

**Alcance:** los defectos están en **`phenosensing`**, no en este proyecto. Cualquier uso de
`PhenoShape` los hereda.

> **Este documento se escribió una vez mal.** La primera versión trataba todo el escalón
> entre DOY 364 y DOY 1 como un defecto y proponía cerrarlo. J. Lopatin objetó que **una
> curva compuesta sobre varios años no tiene por qué cerrar**: el fin de un año se enlaza
> con el inicio del *siguiente*, y si la productividad cambia entre años los dos extremos
> difieren de verdad. Tenía razón, y perseguir la objeción destapó un tercer problema mayor
> que los dos originales. Lo que sigue es la versión corregida.

---

## 1. Tres problemas, no uno

| | qué era | cómo se detectó | estado |
|---|---|---|---|
| **A** | `PhenoShape` depende del **orden de llegada** de las observaciones | al comprobar reproducción bit a bit | `429cbe0` |
| **B** | el promediado móvil dejaba 4 de 52 pasos **sin suavizar** | al graficar en orden de calendario | `eb2dff8` |
| **C** | «arreglarlo» cerrando el año **borra la tendencia interanual** | objeción de J. Lopatin, luego medida | `429cbe0` |

## 2. Problema A — la reconstrucción no era reproducible

`_getPheno0` ordenaba por DOY con `argsort()`, que es **quicksort y no es estable**. Con
observaciones que comparten DOY, su orden relativo quedaba arbitrario. Y como `_fillNaN`
interpola sobre **posiciones de arreglo** y no sobre DOY, un orden distinto rellena los
huecos distinto y la curva cambia.

Medido reajustando las 1.082 parcelas desde sus propias observaciones guardadas:

| | parcelas | reproducen bit a bit | error mediano |
|---|---:|---:|---:|
| **sin DOY duplicados** | 690 | **690 (100 %)** | 0 |
| **con DOY duplicados** | 392 | 6 | **0,068** |

**El 36 % de las parcelas tiene empates** — inevitable con 3 años de revisita de 16 días.
Para ésas, la curva guardada **no se puede regenerar**: depende del orden de desempate que
tuvo aquella máquina, con su versión de numpy, y ese orden se perdió.

`kind="stable"` no basta: preserva el orden *de entrada* entre empates, así que permutar
las filas sigue cambiando el resultado. Comprobado. El arreglo es
`np.lexsort((y, doy))` — desempata por el valor observado, que es una propiedad del dato y
no de cómo llegó. Verificado invariante a permutaciones.

> Consecuencia para el proyecto: las curvas actuales de `phenoshape_by_index.parquet` son
> irreproducibles para 392 parcelas. Eso, por sí solo, es razón para adoptar el refit.

## 3. Problema B — las colas no se suavizaban

`_moving_average` convolucionaba en modo `"valid"` y rellenaba el hueco **copiando la
entrada cruda** en los primeros y últimos `n // 2` pasos. Con `rollWindow=5`, 4 de 52 pasos
quedaban sin suavizar mientras los otros 48 se promediaban sobre 5 vecinos — distinto nivel
de ruido en los dos extremos, que en un año son los dos lados de la misma frontera.

Es un bug puro y su arreglo es **quirúrgico**: donde la ventana completa cabe, `shrink` es
la misma convolución `"valid"` de siempre, así que **sólo cambian los pasos 0, 1, 50 y 51**.
Comprobado en 100 curvas y fijado en `test_shrink_touches_only_the_ends`.

## 4. Problema C — cerrar el año borra señal real

Si el escalón fuera tendencia interanual, la aritmética del compuesto lo predice: DOY 1
promedia observaciones ~364 días **anteriores** en tiempo real que DOY 364, así que con
pendiente `s` por año se espera `salto ≈ −s`.

Medido sobre 249 parcelas (kNDVI, tendencia estimada quitando el ciclo estacional con 2
armónicos): **la regresión da pendiente −0,945**, contra el −1 predicho, con r = −0,40.
No es casualidad: el mecanismo es real y está exactamente donde la teoría lo pone.

Explica el **16 % de la varianza** del salto y ~un tercio de su magnitud mediana
(−0,0075 de −0,0229). El resto sí es artefacto de borde. **Los dos mecanismos coexisten.**

### Los cuatro modos de borde, medidos

Sobre 199 parcelas. Se busca |salto| bajo (artefacto quitado) **y** pendiente cercana a −1
(tendencia conservada):

| modo | \|salto\| mediano | r con la pendiente | pendiente regr. | |
|---|---:|---:|---:|---|
| `legacy` | 0,03700 | −0,634 | −1,835 | el artefacto lo infla |
| `wrap` | 0,00598 | −0,364 | **−0,164** | **destruye la señal** |
| `reflect` | 0,02391 | −0,623 | −1,202 | |
| **`shrink`** | 0,02602 | **−0,648** | **−1,308** | **el default nuevo** |

`wrap` —que `eb2dff8` había puesto de default— hace lo mismo que el armónico de forma más
sutil: la ventana circular mezcla DOY 1–2 con DOY 363–364.

### El reconstructor `harmonic`

No es un retoque de las colas: está en `RECONSTRUCTORS`, al mismo nivel que `linear`, y
**reemplaza el ajuste entero**. Medido sobre los 25 píxeles de PCL0242:

| | pasos que toca | cambio en bordes | cambio en interior |
|---|---:|---:|---:|
| arreglo de `_moving_average` | **4 de 52** | 0,01015 | **exactamente 0** |
| reconstructor `harmonic` | **52 de 52** | 0,00649 | **0,01446** |

Cambia el interior más de lo que el otro arreglo cambia los bordes. Se conserva en la
librería —es la herramienta correcta cuando la periodicidad se desea— pero **no como default
de un compuesto**, y su docstring ahora lo dice con el número.

La evidencia empírica coincide: refitear todo con `harmonic` **bajó** el R² (delta medio
−0,0102) y lo bajó también en **RF03**, que ignora el orden de las columnas y no puede verse
afectado por la periodicidad. Bajó porque le quitaron información.

## 5. Por qué importa aguas abajo

Cinco sustratos rellenan `circular` en el eje temporal (`curve1d`, `curve5`, `gaf`, `mtf`,
`ndi`) y `reshape`/`serpentine`/`hilbert` rellenan envolviendo. Todos suponen que diciembre
empalma con enero.

**Pero eso no explica el desempeño de las CNN**, y está comprobado: el refit con `harmonic`
cerró el año 10× y el R² no mejoró en ninguna faceta. La hipótesis se probó y murió.

## 6. Las tres variantes de curva

| | promediado | orden | periodicidad | tendencia interanual |
|---|---|---|---|---|
| original | roto (4 pasos crudos) | irreproducible (392 parcelas) | no impuesta | presente + artefacto |
| `_v2` (harmonic + wrap) | correcto | reproducible | **impuesta dos veces** | **destruida** |
| **`_v2lin`** (linear + shrink) | correcto | reproducible | no impuesta | **preservada** |

`_v2lin` es la candidata a default. Baja el salto sólo **1,4–1,7×**, y eso es lo correcto:
lo deja en 2,1–3,1× del paso típico porque parte del escalón **es real**. `_v2` se conserva
como evidencia de qué pasa al forzar el cierre.

Se generan con `scripts/29_refit_curves_from_cubes.py`; el conmutador `BIODIV_CURVES`
apunta el pipeline entero a una variante y `runlog.make_run_id` la añade al identificador,
para que dos versiones no aterricen en el mismo directorio de resultados.

## 7. Lo que queda

- **El salto es un predictor**, no ruido: codifica la tendencia de productividad de la
  ventana causal. Bloque `trend` propuesto, sin implementar.
- **`xnew = linspace(min(x), max(x))`** abarca el rango observado y no `[1, 365]`, así que
  curvas de píxeles distintos no son comparables paso a paso. Cambio de API más invasivo,
  fuera de estos commits.
- **`tests/golden/` está obsoleto** y fallaba **antes** de todo esto: 78 % de elementos
  discrepantes, diferencia máxima 275 días. No sirvió de red de seguridad.
- **Re-extraer con la librería arreglada.** El refit desde los `.nc` ya cubre las curvas;
  las métricas LSP siguen calculadas sobre las originales.

## 8. Procedencia

- Escalón detectado el 2026-08-09 al graficar `notebooks/06_year_boundary.ipynb`, a partir
  de una recta espuria que reportó J. Lopatin (`step_to_doy` no es monótono porque las
  curvas están ancladas al valle, así que `plot` unía diciembre con enero).
- Objeción de J. Lopatin al cierre forzado, y con ella el problema C.
- Problema A encontrado al exigir reproducción bit a bit antes de creerle al refit.
- Código: `github.com/JavierLopatin/PhenoSensing`, `eb2dff8` y `429cbe0`.
