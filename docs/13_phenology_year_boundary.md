# La curva fenológica no cierra el año

**Estado:** diagnosticado y medido, **no corregido**. El arreglo correcto exige reajustar
desde las observaciones crudas (`data/derived/phenology/*.nc`), que no están en esta máquina.

**Alcance:** el defecto está en **PhenoPY / `phenosensing`**, no en este proyecto. Cualquier
uso de `PhenoShape` hereda el problema, así que este documento está escrito para servir
también de reporte upstream.

---

## 1. La observación

Al graficar las curvas en orden de calendario apareció un escalón entre **DOY 364 y DOY 1**,
que son dos días. Medido sobre las 1.082 parcelas, salto = `curva[DOY 1] − curva[DOY 364]`:

| índice | salto mediano | vs. cambio semanal típico | % negativos | % de la amplitud estacional |
|---|---:|---:|---:|---:|
| ndvi | −0,029 | **4,5×** | 67 % | — |
| savi | −0,018 | 4,2× | 71 % | — |
| evi | −0,015 | 4,0× | 69 % | — |
| kndvi | −0,016 | 3,8× | 67 % | 15 % |
| nbr | −0,027 | 3,4× | 69 % | — |

Dos cosas lo separan del ruido:

1. **La magnitud.** Es ~4× la mediana de los otros 50 saltos entre pasos consecutivos. En
   kNDVI, el 58 % de las parcelas tiene un salto mayor que 3× el típico.
2. **El signo.** Es negativo en el 67–71 % de las parcelas, en los cinco índices. Ruido
   simétrico daría 50 %. Esto es un sesgo de borde.

Y el desajuste está localizado. Ajustando una serie de Fourier —periódica por construcción—
los residuos se concentran en los extremos del año:

| armónicos | \|residuo\| en los 3 pasos de cada borde | en el centro | razón |
|---|---:|---:|---:|
| K=3 | 0,0175 | 0,0128 | 1,4× |
| K=4 | 0,0167 | 0,0104 | 1,6× |
| K=6 | 0,0140 | 0,0063 | **2,2×** |

Es la firma de un artefacto de borde, no de un modelo periódico que no le calza a los datos.

## 2. El mecanismo, en dos partes

Ambas están en `phenosensing`. Reproducir el diagnóstico: `notebooks/04_substrates_2d.ipynb`
§1b.

### 2.1 El promediado móvil deja los extremos SIN SUAVIZAR

`phenosensing/utils.py:362-364`:

```python
def _moving_average(a, n=3):
    out = np.convolve(a, np.ones(n), "valid") / n
    return np.concatenate([a[: np.int32(n / 2)], out, a[-np.int32(n / 2) :]])  # add values of tail
```

La convolución `"valid"` devuelve `len(a) − n + 1` valores, y el hueco de los bordes se
rellena **copiando los valores crudos de entrada**. Con `rollWindow=5` eso significa que
**los 2 primeros y los 2 últimos pasos de toda curva no se suavizan en absoluto**, mientras
los otros 48 se promedian sobre 5 vecinos.

Comprobado ejecutando la función:

```
entrada  [5. 1. 2. 3. ...]
salida   [5. 1. 3. 3. ...]      <- los dos primeros pasan crudos, idénticos a la entrada
out[:2] == a[:2]   ->  True
out[-2:] == a[-2:] ->  True
```

Cuatro de los 52 pasos tienen, por tanto, una varianza distinta al resto de la curva. Y son
justamente los que quedan a ambos lados de la frontera del año.

El arreglo aquí es de una línea: el vecindario tiene que ser **circular**. `np.convolve` con
la señal envuelta, o `scipy.ndimage.uniform_filter1d(a, n, mode="wrap")`.

### 2.2 Ningún reconstructor es periódico

`phenosensing/reconstruction.py:200`:

```python
RECONSTRUCTORS = {
    "linear": _linear, "RBF": _rbf, "KDE": _kde, "savgol": _savgol,
    "whittaker": _whittaker, "dlog_beck": _dlog_beck, "dlog_elmore": _dlog_elmore,
    "agauss": _agauss, "upper_envelope": _upper_envelope,
}
```

Los nueve tratan el DOY como un **intervalo abierto**. Ninguno impone `f(1) = f(365)` ni
continuidad de la derivada en la frontera. Las curvas dobles-logísticas y la gaussiana
asimétrica son además explícitamente unimodales sobre un intervalo, así que la periodicidad
no está ni siquiera disponible como opción.

Se suma que `phenosensing/utils.py:91`:

```python
xnew = np.linspace(np.min(x), np.max(x), nGS, dtype=np.int32)
```

La grilla de salida abarca el **rango observado**, no `[1, 365]`. El primer y el último nodo
caen sobre las observaciones extremas, y con `interpolType="linear"` —que es lo que usa
`scripts/02_extract_phenology.py:139`— el valor en esos nodos **es una única observación
cruda**, sin ningún promediado. Es el peor sitio posible para anclar la curva.

### 2.3 Por qué los dos efectos caen en el mismo punto

Nuestras ventanas causales agrupan 3 años de observaciones colapsadas a DOY, así que
`min(x) ≈ 1` y `max(x) ≈ 364`. Los extremos del arreglo **son** la frontera del año. Los dos
mecanismos se suman ahí: el nodo extremo es una observación única *y* además no se suaviza.

## 3. Por qué importa

Para un modelo tabular (Random Forest, MLP sobre la curva plana) el orden de las columnas es
irrelevante y el efecto se limita a que 4 de 52 columnas son algo más ruidosas.

Para **todo lo convolucional** es distinto, porque la circularidad es una suposición explícita
del diseño:

- `PAD_MODE` en `src/biodiv/substrates.py:55` declara relleno **`circular`** en el eje
  temporal de `curve1d`, `curve5`, `gaf`, `mtf`, `ndi`, y en el eje horizontal de `stack5`,
  `pxcube` y `cwt`.
- `transforms1d.wrap_pad` rellena `reshape`/`serpentine`/`hilbert` envolviendo, precisamente
  para no inventar un desplome de vegetación al final del arreglo.

Las dos decisiones son correctas —la fenología *es* cíclica— pero suponen que diciembre
empalma con enero. En estos datos no empalma. Un kernel que cruza esa frontera lee una caída
que no ocurrió, con una magnitud de 4× el cambio semanal real.

**Hipótesis, no conclusión.** Bajo `kfold5_window` el Random Forest supera a la CNN 2D en las
cinco facetas (`docs/12`, `notebooks/05_results_all_facets.ipynb` §3). Esto es una explicación
plausible: el RF no puede verse afectado y la familia convolucional sí. **No está demostrada**
y podría ser marginal frente al límite de tener 1.082 parcelas. La sección 5 dice cómo
zanjarlo.

## 4. El arreglo, en tres niveles

### Nivel 1 — upstream, en `phenosensing` (el correcto)

1. **`_moving_average` circular.** Una línea. Es un arreglo estrictamente mejor para
   cualquier usuario: no hay caso en que copiar los extremos crudos sea preferible.
2. **Una grilla de salida fija en `[1, 365]`** en vez de `[min(x), max(x)]`, para que curvas
   de distintos píxeles sean comparables paso a paso, con extrapolación circular en los
   huecos.
3. **Al menos un reconstructor periódico.** El más barato es una regresión armónica —serie
   de Fourier truncada, periódica por construcción— que además es el estándar en la
   literatura de fenología satelital (HANTS, y las series armónicas de Zhu & Woodcock).
   Encaja en el registro sin tocar nada más:

   ```python
   def _harmonic(x, y, xnew, n_harmonics=3, period=365.25, **params):
       """Regresión armónica. Periódica por construcción: f(t) = f(t + period)."""
       def design(t):
           w = 2 * np.pi * np.asarray(t) / period
           cols = [np.ones_like(w)]
           for k in range(1, n_harmonics + 1):
               cols += [np.cos(k * w), np.sin(k * w)]
           return np.column_stack(cols)
       beta, *_ = np.linalg.lstsq(design(x), y, rcond=None)
       return design(xnew) @ beta
   ```

   Con K=6 sobre nuestras curvas ya ajustadas, la reconstrucción retiene el **97,1 %** de la
   varianza y reduce el salto de frontera **5,6×** (0,031 → 0,0055). Ajustado sobre las
   observaciones crudas debería ir mejor, porque no arrastra el artefacto de borde del paso
   previo.
4. **Un test de regresión que fije la periodicidad**: para toda curva reconstruida,
   `|f(paso 0) − f(paso n−1)|` no puede exceder el cambio típico entre pasos consecutivos.
   Es la comprobación que habría atrapado esto.

### Nivel 2 — en este proyecto, al re-extraer

`scripts/02_extract_phenology.py:74` pasa `interpolType=recon, rollWindow=5, nGS=nGS`.
Cuando el nivel 1 exista, basta cambiar el reconstructor a `harmonic` y volver a correr desde
los `.nc`. **Requiere los cubos**, que están sólo en la máquina con datacube (346 MB en
`data/derived/phenology/*.nc`; aquí sólo sobrevive `manifest.csv`).

Al re-extraer conviene aprovechar y arreglar de paso lo que `docs/09_predictors.md` §A dejó
pendiente: el 5,1 % de píxeles con LSP NaN, que viene del mismo tipo de causa numérica.

### Nivel 3 — parche post-hoc, posible hoy con lo local

Las 10.820 curvas de parcela y los 135.250 píxeles están en
`phenoshape_by_index.parquet` y `phenoshape_pixels.parquet`. Proyectar cada curva de 52 pasos
sobre una base armónica de K=6 es cuestión de segundos y cierra el año.

**No es lo mismo que el nivel 2** y hay que declararlo si se reporta: sería un suavizado
periódico *sobre una curva ya ajustada*, así que hereda el sesgo de borde del ajuste original
en vez de evitarlo. Sirve para **medir si el problema importa**, no para publicarse como el
método.

## 5. Cómo zanjar si esto explica el desempeño de las CNN

El test es barato y los dos resultados son publicables:

1. Generar la variante periódica (nivel 3) como un `px`/preprocesamiento nuevo.
2. Re-correr las familias `c1d` y `4a` bajo `kfold5_window` contra la curva actual. Mismo
   modelo, mismos folds, mismas semillas: un solo factor cambia. ~40 min en los dos GPUs.
3. Leer:
   - **Sube el R²** → la discontinuidad era parte del problema. Justifica traer los `.nc` y
     hacerlo bien desde el ajuste, y justifica el reporte upstream con datos.
   - **No se mueve** → la hipótesis muere barata, y queda establecido que el mal desempeño de
     la familia convolucional es por el tamaño de muestra y no por el sustrato. Para el paper
     eso vale igual que lo contrario.

El paso 3 hay que decidirlo **antes** de mirar el resultado, o el test no distingue nada.

## 6. Procedencia

- Descubierto el 2026-08-09 al graficar `notebooks/04_substrates_2d.ipynb` §1, a partir de
  una línea recta espuria que reportó J. Lopatin. Esa recta era un bug distinto y menor
  —`step_to_doy` no es monótono porque las curvas están ancladas al valle (DOY 108), así que
  `plot` unía diciembre con enero—; el escalón apareció al corregirla.
- Mediciones y figuras: `notebooks/04_substrates_2d.ipynb` §1b.
- Código citado: `phenosensing` en `/mnt/rapidita_4T/GitHub/PhenoPY`, commit en uso al
  2026-08-09.
