# Extracción en la máquina del datacube: qué sacar y cómo

**Para quién:** la sesión en la máquina con acceso a Data Cube Chile. Todo lo demás del
proyecto corre local; esto es el único paso que no.

**Qué se saca:** un conjunto de ventanas 5×5 **sin etiquetar**, repartidas por el espacio
ambiental de las parcelas, para preentrenar el codificador de la CNN por enmascarado (MAE).

---

## 1. Por qué hace falta, y por qué lo que ya hay no sirve

El MAE ya se corrió sobre las 135.250 curvas de píxel que están en disco y **no aportó nada**
(0,359 contra 0,362 sin preentrenar, `docs/14` §4). No fue por el método: la reconstrucción
bajó de 0,0095 a 0,0031, así que aprendió. Fue porque esas curvas **no son datos nuevos**:

| | |
|---|---:|
| curvas de píxel disponibles | 135.250 |
| parcelas de las que salen | 1.082 |
| dimensión de participación, 135.250 curvas de píxel | 1,3 |
| dimensión de participación, 1.082 curvas de parcela | 1,2 |

125× más imágenes cubriendo 1,08× más espacio. El autoencoder ve unos mil objetos repetidos
veinticinco veces. **Lo que falta es cobertura, y la cobertura sólo se compra muestreando
lejos de las parcelas.** Eso es lo que hay que sacar del cubo.

## 2. La especificación del dato

La autoridad es `src/biodiv/cube.py`, que ya implementa todo esto y es lo que se usó para las
1.082 parcelas. Aquí van los números para poder verificar sin leer código:

| | |
|---|---|
| productos | `landsat5_c2l2_sr`, `landsat7_c2l2_sr`, `landsat8_c2l2_sr`, `landsat9_c2l2_sr` |
| bandas | `blue, green, red, nir, swir1, swir2, qa_pixel` (alias verificados en L5/7/8/9) |
| CRS | `EPSG:32719` (UTM 19S) |
| resolución | 30 m |
| `group_by` | `solar_day` |
| escalado SR | `reflectancia = DN * 0,0000275 − 0,2` |
| máscara QA | sin nodata, sin nube/sombra/cirrus/nieve de alta confianza, sin dilatación |
| ventana temporal | **causal**: `y−2 .. y`, 3 años que terminan en el año del censo |
| ventana espacial | **5×5 píxeles** (150×150 m) centrados en el punto |
| índices | `ndvi, evi, kndvi, nbr, savi` |
| cobertura de suelo | `landcover_chile_2014`, leída por punto y **guardada, no filtrada** |

Tres cosas que, si se omiten, invalidan todo aguas abajo y no dan error:

1. **`configure_s3_access(aws_unsigned=False, requester_pays=True)` antes de la primera
   carga.** Los buckets son requester-pays; sin esto las cargas vuelven vacías.
2. **El escalado de Collection 2.** Sin él los índices salen numéricamente plausibles y
   físicamente falsos.
3. **La máscara QA se construye por producto, antes de concatenar.** `cirrus` sólo existe en
   L8/L9; pedirlo sobre L5/L7 hace fallar `make_mask`, y si se construye tras concatenar el
   fallo depende de qué sensor llegó primero.

### Lo que se guarda, y por qué a nivel de observación

Se guarda la **serie de índices por observación**, no la curva ajustada. Es la regla de
`docs/05` §1: *adquirir al nivel de observación, decidir el pooling aguas abajo*. Con eso, la
curva compuesta, las curvas por año y la serie cruda de 3 años se derivan sin volver al cubo
— y la serie cruda resultó ser el único hallazgo real de toda la búsqueda (`docs/14` §2), que
no habría existido si la adquisición hubiera entregado la curva ya promediada.

## 3. Sobre urbano, agrícola y plantaciones

Hay dos posturas defendibles y el argumento no las resuelve: excluirlos porque el afinado es
sobre vegetación nativa, o incluirlos porque el manifold etiquetado es extraordinariamente
estrecho (dimensión de participación 1,2 de 52) y el cultivo tiene estructura que la
vegetación nativa no tiene — caídas abruptas de cosecha, dos picos al año, suelo desnudo.

**El script no decide: etiqueta.** Cada muestra lleva su clase de `landcover_chile_2014`,
resuelta en tiempo de ejecución desde los metadatos del producto y no contra enteros
hardcodeados. Con una extracción salen los tres preentrenamientos —sólo natural, todo, y
natural más una cuota de antrópico— y la elección pasa de supuesto a ablación.

`--natural-only` existe pero **conviene no usarlo**: la clase queda registrada igualmente.

## 4. Qué llevar a la otra máquina

```
scripts/32_sample_unlabelled.py
src/biodiv/cube.py                       # la especificación operativa
data/derived/plots_subset.parquet        # define el sobre ambiental a cubrir
```

`plots_subset.parquet` no es opcional: el muestreo se ancla al espacio que ocupan las
parcelas. Un muestreo uniforme sobre Chile sería en su mayoría Atacama y Patagonia, que el
conjunto etiquetado no contiene.

Envolvente real: X 126.732–407.321, Y 5.800.807–6.651.332 (UTM 19S), años 2003–2026.

## 5. Cómo correrlo

```bash
python scripts/32_sample_unlabelled.py --n 4000 --dry-run      # sin tocar el cubo
python scripts/32_sample_unlabelled.py --n 4000 --out data/derived/unlabelled
```

El `--dry-run` no requiere datacube y sirve para comprobar que `plots_subset.parquet` se lee
y que el número de cargas es razonable.

**Las cargas se agrupan por (celda, año)**, igual que `scripts/02`. Sin agrupar, n = 4000
serían ~8.000 llamadas a `dc.load`. El tamaño de celda cambia el número de llamadas y la
memoria pico, porque la celda entera se materializa de una vez:

| `--cell-km` | cargas | bbox máx. | RAM pico |
|---:|---:|---:|---:|
| 20 | 261 | 20,3 km | 2,2 GB |
| **10 (default)** | **490** | **10,2 km** | **0,6 GB** |
| 5 | 957 | 5,3 km | 0,1 GB |

Si la máquina va justa de memoria, bajar a 5. Si la latencia a S3 domina, subir a 20.

Una celda que falla no detiene la corrida: sus puntos quedan con su error en el manifiesto.

## 6. Qué produce

```
data/derived/unlabelled/
    U00000.nc ... U03999.nc     ~250-300 KB cada uno  ->  ~1,1 GB para n=4000
    manifest.csv                 una fila por muestra, con landcover y n_obs
    sampling.json                los parámetros exactos de la corrida
```

Vuelve a la máquina local **el directorio entero**; `manifest.csv` y `sampling.json` son lo
que hace la extracción auditable después.

## 7. Verificación al llegar

```python
import pandas as pd, xarray as xr
m = pd.read_csv("data/derived/unlabelled/manifest.csv")
print(len(m), "muestras;", m.get("error").notna().sum() if "error" in m else 0, "errores")
print(m.groupby(["landcover", "anthropic"]).size())
print("obs por ventana:", m.n_obs.describe())
d = xr.open_dataset("data/derived/unlabelled/U00000.nc")
print(d.sizes)          # tiene que ser y=5, x=5 y una dimension time
print(list(d.data_vars)) # ndvi, evi, kndvi, nbr, savi
```

Cuatro cosas que mirar, en este orden:

1. **`y=5, x=5`.** Si sale 10×10, la geometría no coincide con las parcelas etiquetadas y el
   preentrenamiento estaría aprendiendo de un objeto distinto al del afinado — cualquier
   resultado de transferencia quedaría confundido con un cambio de geometría. Era un defecto
   real de la primera versión de este script; corregido, pero es lo primero que hay que
   comprobar.
2. **`n_obs`.** Las parcelas etiquetadas tienen ~65 observaciones válidas en 3 años. Una
   distribución muy por debajo indica que la máscara QA está descartando de más o que la
   ventana cae en un hueco de cobertura.
3. **El reparto de `landcover`.** Es el que decide las tres ablaciones de preentrenamiento.
4. **Rango de los índices.** kNDVI y NDVI dentro de −1..1; si están en miles, falta el
   escalado de Collection 2.

## 8. Y después

Adaptar `scripts/31_pretrain_mae.py` para leer ese directorio en vez de
`phenoshape_pixels.parquet`, y preentrenar de las tres maneras. La comparación se hace contra
la corrida idéntica **sin** `--init-from`, que ya existe, así que la diferencia aísla el
preentrenamiento y nada más.

**Declarar la fuga.** El preentrenamiento no ve etiquetas, así que no se filtra respuesta.
Con este muestreo, además, tampoco ve píxeles de parcelas de test — el jitter es de ±5 km
contra una ventana de 150 m, así que ninguna muestra cae sobre el terreno de la parcela que
la sembró. Eso hace este pool **más limpio** que el de las curvas de píxel, donde sí había
fuga de entrada.
