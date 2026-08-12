# Extracción en la máquina del datacube: qué sacar y cómo

**Para quién:** la sesión en la máquina con acceso a Data Cube Chile. Todo lo demás del
proyecto corre local; esto es el único paso que no.

**Qué se saca:** ~17.000 series de kNDVI **sin etiquetar**, en vegetación nativa enmascarada
con MapBiomas, repartidas por todo el territorio que ocupan las parcelas, para preentrenar el
codificador de la CNN por enmascarado (MAE).

---

## 1. Por qué hace falta, y por qué lo que ya hay no sirve

El MAE ya se corrió sobre las 135.250 curvas de píxel que están en disco y **no aportó nada**
(0,359 contra 0,362 sin preentrenar, `docs/14` §4). No fue por el método: la reconstrucción
bajó de 0,0095 a 0,0031, así que aprendió. Fue porque esas curvas **no son datos nuevos**:

| | |
|---|---:|
| curvas de píxel disponibles | 135.250 |
| parcelas de las que salen | 1.082 |
| correlación entre píxeles de la misma parcela | +0,844 |
| correlación entre píxeles de parcelas distintas | +0,258 |
| dimensión de participación, 135.250 curvas de píxel | 1,3 |
| dimensión de participación, 1.082 curvas de parcela | 1,2 |

125× más imágenes cubriendo 1,08× más espacio. **Lo que falta es cobertura, y la cobertura
sólo se compra muestreando lejos de las parcelas.**

## 2. Lo que cambió respecto de la versión anterior de este documento

Dos cosas, y ninguna es un ajuste menor.

**El jitter de ±5 km se reemplazó por la máscara de MapBiomas.** Medido: sembrar con jitter
alrededor de las parcelas alcanza 19.583 km² de los ~83.000 km² de cobertura nativa —el 29 %—
y subir `n` sólo apila muestras sobre las mismas parcelas (46 por parcela a n=50.000). Es el
mismo modo de falla que ya costó el primer intento, a 5 km en vez de 150 m.

**Se dejó de guardar la ventana 5×5.** `scripts/31_pretrain_mae.py:61-64` carga el cubo
`(N, 25, 52)` y lo **aplana** a `(N*25, 1, 52)`: los 25 píxeles son 25 curvas independientes y
la imagen 2D sale del *substrate transform de la serie*, no de los vecinos espaciales. La
estructura espacial se destruye antes de que el modelo la vea, así que guardarla costaba ~180×
el disco a cambio de curvas que correlacionan a +0,844 entre sí.

## 3. Qué se guarda

**Observaciones Landsat con su fecha, no una curva ajustada ni una grilla.** Es la regla de
`docs/05` §1: *adquirir al nivel de observación, decidir el pooling aguas abajo*. `docs/14` §2
probó 8 resoluciones de grilla (24 a 196 pasos) y se repartieron 0,011 —por debajo del ruido de
semilla, o sea que la resolución no importa—, pero eso sólo fue medible porque las 8 se
derivaban sin volver al cubo. Guardar una grilla aquí congelaría esa elección.

```
data/derived/unlabelled/
    series.parquet    sample_id, time, kndvi          (~1,1 M filas, ~25 MB)
    manifest.csv      sample_id, X, Y, lon, lat, year, win_start, win_end, cell,
                      mb_code, mb_class, mb_year_used, mb_delta, n_obs
    sampling.json     los parametros exactos de la corrida
```

La grilla `s00..sNN` se construye **en local** con `biodiv.curves.interp_grid`, que es la misma
interpolación que reciben las parcelas — por eso vive en un módulo y no en un script.

**Sólo kNDVI**, que es el índice primario del proyecto. Los cinco índices salen de las mismas
seis bandas de la misma carga, así que ampliarlo cuesta ~100 MB y **cero cargas adicionales**;
queda anotado por si alguna vez pesa más volver al cubo que el disco.

## 4. La máscara de vegetación nativa

La autoridad es `src/biodiv/mapbiomas.py` y la leyenda versionada está en
`MapBiomas/legend.csv`. **Los rásters no traen leyenda embebida** y `MapBiomas/metadata.txt`
documenta la jerarquía (`1.1 Forest`, `2.1 Wetland`, …) sin decir nunca qué entero guarda el
ráster. Los enteros se anclaron contra las 1.082 parcelas, que son vegetación nativa por
diseño:

| código | clase | evidencia | nativo |
|---:|---|---|:---:|
| 3, 59, 60, 61 | Forest / Primary / Secondary / Dwarf | 2,5 % + 7,6 % + 22,9 % de las parcelas | ✅ |
| 66 | Shrubland | **48,3 % de las parcelas** | ✅ |
| 12 | Grassland | 7,5 % de las parcelas | ✅ |
| 11, 63 | Wetland, Steppe | convención MapBiomas | ✅ |
| 29 | Rocky Outcrop | sondeo alta cordillera | ❌ no es vegetación |
| 9 | Silviculture | sondeo plantación Constitución | ❌ |
| 18, 15 | Agriculture, Pasture | 3,2 % de las parcelas | ❌ |
| 24 | Infrastructure | **sondeos Santiago y Viña** | ❌ |
| 23, 25, 67 | Duna, otros no vegetados, salar | convención | ❌ |
| 33, 34 | Agua, nieve/hielo | convención | ❌ |
| 0, 79, 80 | No observado, sin asignar | — | ❌ |

Dos honestidades: **cuál de 59/60/61 es primario, secundario o achaparrado es inferencia** —
para la máscara da igual, los tres son bosque nativo, pero no se debe reportar la subclase sin
confirmar—; y **67, 79 y 80 quedan sin asignar** (juntos < 0,2 %), excluidos por defecto,
porque lo que no se reconoce no entra.

### La estabilidad es por ventana, no por colección

**Nativo en los 3 años de la ventana causal `y−2..y` de esa misma muestra.** No en 1999–2024.
Exigirlo sobre la colección entera deja que un solo año mal clasificado en 26 mate el píxel:

| regla | área | |
|---|---:|---|
| nativo en ≥1 año | 112.241 km² | ✗ no garantiza ventana homogénea |
| **nativo en la ventana causal** | **~83.000 km²** | ✅ **la que se usa** |
| nativo en los 26 años | ~41.191 km² | ✗ pierde el 39 %, sesga a núcleo de bosque denso |

La regla de ventana además hace las cifras **inmunes al estado de la colección**: medida sobre
tres ventanas distintas da 84.301 / 83.747 / 83.153 km².

⚠️ **El 10,7 % de las propias parcelas cae sobre clases que la máscara excluye** — 6,1 %
silvicultura, 3,2 % agricultura, 1,4 % urbano. Cambio de uso entre censo y mapa, error de
georreferencia o borde de plantación; la causa no está resuelta, la consecuencia sí: **el pool
se saca bajo una regla más estricta que el conjunto etiquetado**, y eso se declara.

## 5. El diseño de muestreo

Sólo hay **1.449 celdas de 10 km con cobertura nativa**, así que se visitan todas: la cobertura
viene de *qué celdas* se visitan, no de gastar una carga de datacube por punto.

1. **Celdas:** las 1.449, encontradas con una pasada decimada de cribado.
2. **Años por celda:** `--years-per-cell 2`, sorteados del histograma de años de las parcelas,
   para que el preentrenamiento vea la misma mezcla temporal que el afinado. Es el knob de
   costo: multiplica las cargas uno a uno.
3. **Puntos por (celda, año):** cuota por clase nativa según las proporciones de las parcelas,
   con **separación mínima de 1 km** medida en UTM.
4. **Pesos por banda de latitud** tomados de las parcelas, para no derivar al norte seco ni a
   la alta cordillera, que el conjunto etiquetado casi no contiene.

## 6. Qué llevar a la otra máquina

```
scripts/32_sample_unlabelled.py
src/biodiv/mapbiomas.py                  # la mascara y la leyenda
src/biodiv/cube.py                       # la especificacion operativa de Landsat
MapBiomas/legend.csv
MapBiomas/*.tif                          # 26 mapas anuales, ~3,1 GB
data/derived/plots_subset.parquet        # define el sobre a cubrir
```

Los `.tif` **no están en git** (`.gitignore:21`) y hay que copiarlos aparte. Envolvente real:
X 126.732–407.321, Y 5.800.807–6.651.332 (UTM 19S), años 2003–2026.

## 7. Cómo correrlo

```bash
python scripts/32_sample_unlabelled.py --dry-run              # sin tocar el cubo
python scripts/32_sample_unlabelled.py --dry-run --cells 40   # chequeo rapido
python scripts/32_sample_unlabelled.py --out data/derived/unlabelled
```

El `--dry-run` no requiere datacube: construye el pool completo desde los rásters locales, y es
donde se verifica la máscara antes de gastar una sola carga. Medido, con los valores por
defecto:

```
1.449 celdas de 10 km con cobertura nativa
17.146 muestras candidatas, 2.898 cargas (celda x ano)
Shrubland 10.296 | Secondary Forest 2.073 | Grassland 1.917 | Primary Forest 1.325
Wetland 1.010 | Forest 474 | Dwarf forest 27 | Steppe 24
```

Lo que hay que mirar en esa salida, en este orden:

1. **La distribución de años pegada a la de las parcelas.** Medido: 2022 21,4 % contra 23,5 %,
   2019 11,5 % contra 11,6 %, 2014 9,3 % contra 9,3 %.
2. **Cero clases no nativas.** El script lo afirma con un `assert`, no con un mensaje.
3. **El reparto de clases.** Es el que decide las ablaciones de preentrenamiento, y ninguna
   clase debería pasar del 80 %.
4. **Las muestras con mapa desplazado.** Medido: 1.795, todas de 2025–2026, que no existen en
   MapBiomas y se mapean a 2024 con su `mb_delta` registrado.

Los mapas anuales de MapBiomas cubren **1999–2024, los 26 completos**. Sólo 2025 y 2026 —donde
hay 109 parcelas— usan el año más cercano.

## 8. Verificación al llegar

```python
import pandas as pd
m = pd.read_csv("data/derived/unlabelled/manifest.csv")
s = pd.read_parquet("data/derived/unlabelled/series.parquet")
print(len(m), "muestras;", m.get("error").notna().sum() if "error" in m else 0, "errores")
print("obs por serie:", m.n_obs.describe())
print(m.groupby("mb_class").size())
print("rango kNDVI:", s.kndvi.min(), s.kndvi.max())
```

1. **`n_obs`.** Las parcelas etiquetadas tienen ~65 observaciones válidas en 3 años. Muy por
   debajo indica que la máscara QA descarta de más o que la ventana cae en un hueco.
2. **Rango de kNDVI dentro de 0..1.** Si sale en miles, falta el escalado de Collection 2.
3. **El reparto de `mb_class`**, que decide las ablaciones.

## 9. Y después

```bash
python scripts/31_pretrain_mae.py --unlabelled data/derived/unlabelled
python scripts/31_pretrain_mae.py --unlabelled data/derived/unlabelled \
    --landcover 'Shrubland,Secondary Forest,Primary Forest'
```

**Antes de gastar GPU, medir la dimensión de participación del pool nuevo contra el 1,3 de las
135.250 curvas.** Si no la supera con claridad, el problema de cobertura no se resolvió y
ninguna cantidad de preentrenamiento lo va a arreglar. Es barato y va primero.

La comparación se hace contra la corrida idéntica **sin** `--init-from`, que ya existe, así que
la diferencia aísla el preentrenamiento. Con el ruido de GPU medido en 0,010 de desviación
(`docs/14` §3b), **nada por debajo de 0,022 distingue dos modelos**: cada ablación necesita 3
semillas o no es interpretable.

**Declarar la fuga.** El preentrenamiento no ve etiquetas, así que no se filtra respuesta. Con
este muestreo tampoco ve píxeles de parcelas de test: la separación mínima es de 1 km contra
una ventana de 150 m, así que ninguna muestra cae sobre el terreno de una parcela. Eso hace
este pool **más limpio** que el de las curvas de píxel, donde sí había fuga de entrada — y por
eso `--per-fold` no aplica aquí, y el script lo rechaza.

---

## Los tres encargos para la máquina del cubo

Están dispersos en tres documentos y conviene resolverlos en una sola visita:

| encargo | dónde | estado |
|---|---|---|
| este muestreo no etiquetado | `docs/15` (aquí) | listo para correr |
| arreglo de periodicidad de la curva | `docs/13` | exige los `.nc`, sin hacer |
| SAR de Sentinel-1 y red-edge de Sentinel-2 | `docs/11` §5 | sólo si lo demás se estanca |
