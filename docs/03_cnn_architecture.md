# Arquitectura CNN para n ≈ 1.000

**Restricción de diseño:** ~800–1.000 parcelas, ~8 targets correlacionados, targets parcialmente faltantes.
**Se mantiene:** la idea de transformación señal→imagen para regresión.
**Se descarta:** el pipeline de backbones `timm` a 224×224 de `Trait_2DCNN`.

---

## 1. Por qué no se puede heredar `Trait_2DCNN` tal cual

`Trait_2DCNN` reescala todo a 224×224 (`training/config.py:86`, `output_size=224`) por una razón puramente instrumental: los backbones de `timm` lo exigen. Con 1.721 bandas espectrales el `reshape` a 42×42 y el zoom a 224×224 tenía sentido de escala.

Aquí la señal de entrada es una curva fenológica de **52 pasos** (`PhenoShape(nGS=52)`). Consecuencias:

| | Trait_2DCNN | Este proyecto |
|---|---|---|
| Longitud de señal | 1.721 | 52 |
| Lado natural de la imagen | ⌈√1721⌉ = 42 | ⌈√52⌉ = **8** |
| Tamaño usado | 224×224 (zoom ×5,3) | 8×8 — **sin zoom** |
| n de entrenamiento | miles por fold | ~800–1.000 |
| Parámetros del modelo | 4–25 M (`efficientnet_b0`…`convnext_tiny`) | objetivo **< 50 k** |

Inflar 52 valores a 50.176 píxeles no añade información; añade parámetros. Con n=1.000 y ~25 M de parámetros la red memoriza el conjunto de entrenamiento antes de la época 5.

**Regla de diseño:** el tamaño de la imagen lo fija la señal, no el backbone.

---

## 2. Dos sustratos de imagen, ambos benchmarkeables

La contribución sigue siendo el benchmark de transformaciones, igual que en `Trait_2DCNN`. Cambia el catálogo.

### 2.1 Sustrato A — *phenocube*: imagen natural (año × DOY) ← recomendado

Para cada parcela censada en el año *t*, apilar las curvas fenológicas reconstruidas de los años *t−K … t*:

```
entrada: (C, Y, D)
  C = índices de vegetación (NDVI, EVI, kNDVI, NBR)   → 2–4 canales
  Y = años de la ventana retrospectiva                → 6–10 filas
  D = pasos de la curva intra-anual (nGS)             → 52 columnas
ejemplo típico: (4, 8, 52)
```

Los dos ejes tienen significado físico distinto y una convolución 2D los explota de forma interpretable:

- **eje horizontal (DOY)** — forma intra-anual: verdor, senescencia, amplitud, asimetría.
- **eje vertical (año)** — anomalía y tendencia interanual: exactamente la señal que Lopatin (2023) mostró que se relaciona con las comunidades vegetales, y que Dronova et al. (2022) mostraron que la diversidad amortigua.
- **kernel 3×3** = "cómo cambia la forma estacional entre años vecinos". Eso no se puede leer en una curva 1D ni en un GAF.

Esta es la única de las representaciones donde la segunda dimensión **no es manufacturada**. Ventaja de escritura: se defiende sin apelar a analogías.

Producción del sustrato: `PhenoSensing.PhenoShape()` por año, o `get_timeseries_metrics(window_length=1)` — con la corrección previa del bug de RMSE en `accessor.py:460`.

### 2.2 Sustrato B — curva única, transformación manufacturada

Cuando solo hay un año útil (parcelas pre-2003, series con pocas observaciones válidas), la entrada es una curva 1D de 52 puntos y se aplica el catálogo de `Trait_2DCNN`, **a su tamaño nativo, sin zoom**:

| Transformación | Salida | Canales | Comentario |
|---|---|---|---|
| `reshape` | 8×8 | 1 | ganadora en Trait_2DCNN; padding 52→64 |
| `serpentine` | 8×8 | 1 | continuidad temporal en los bordes de fila |
| `gaf` (GASF+GADF) | 52×52 | 2 | diseñada para series temporales; aquí en su dominio |
| `mtf` | 52×52 | 1 | ídem |
| `cwt` (morlet) | 52×52 | 1 | escala × tiempo; natural para fenología multiescala |
| `spectrogram` | ~26×26 | 3 | STFT sobre 52 puntos queda muy corta — probablemente descartable |
| `hilbert` | 8×8 | 1 | requiere remuestreo a potencia de 4; distorsiona con 52 puntos |

GAF, MTF, CWT y espectrograma **fueron diseñadas para señales temporales** y en `Trait_2DCNN` se aplican, contra su origen, a un eje espectral. Aquí operan en su dominio nativo. Ese es el argumento del paper.

### 2.3 Baselines obligatorios

Sin estos tres, el benchmark no prueba nada:

1. **1D-CNN** sobre la curva de 52 puntos (control directo: ¿aporta la 2ª dimensión?).
2. **Random Forest** sobre las 18 métricas LSP de `PhenoLSP()` + topografía (control: ¿aporta la curva sobre sus resúmenes escalares?).
3. **RF sobre la curva aplanada** (52 × C features) — control de que la ganancia es de la convolución y no del acceso a la curva.

---

## 3. `PhenoNet-S` — la arquitectura

Tronco compartido diminuto + cabezas multi-target. Convoluciones separables en profundidad y *global average pooling* en vez de `flatten`: ahí es donde se ahorran dos órdenes de magnitud de parámetros.

```python
import torch, torch.nn as nn

class SepConvBlock(nn.Module):
    """Depthwise-separable conv + BN + GELU. ~8x menos parametros que una conv normal."""
    def __init__(self, c_in, c_out, k=3, stride=1, p_drop=0.0):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, stride=stride, padding=k // 2,
                            groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.pw(self.dw(x)))))


class PhenoNetS(nn.Module):
    """
    Tronco compartido pequeno para regresion multi-faceta desde imagenes fenologicas.

    Entrada esperada: (B, C, H, W)
      sustrato A (phenocube):  H = anios de la ventana,  W = nGS
      sustrato B (transform):  H, W = lado nativo de la transformacion (8x8 o 52x52)

    Salida: (B, n_targets). Los targets faltantes se manejan con mascara en la perdida.
    """
    def __init__(self, in_channels=4, n_targets=8, width=(16, 32, 64),
                 p_drop_conv=0.1, p_drop_head=0.3):
        super().__init__()
        w1, w2, w3 = width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, 3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            SepConvBlock(w1, w2, p_drop=p_drop_conv),
            SepConvBlock(w2, w2, p_drop=p_drop_conv),
            SepConvBlock(w2, w3, stride=2, p_drop=p_drop_conv),
            SepConvBlock(w3, w3, p_drop=p_drop_conv),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # (B, w3, 1, 1) — sin flatten
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_drop_head),
            nn.Linear(w3, w3),
            nn.GELU(),
            nn.Dropout(p_drop_head),
            nn.Linear(w3, n_targets),
        )

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))))

    def embed(self, x):
        """Representacion compartida de w3 dims — para inspeccion y para linear probing."""
        return self.pool(self.blocks(self.stem(x))).flatten(1)
```

**Presupuesto de parámetros** — medido ejecutando el código con `in_channels=4`, `n_targets=8`:

| `width` | Parámetros | Uso |
|---|---|---|
| `(8, 16, 32)` | **4.384** | n < 500, o sustrato 8×8 |
| `(16, 32, 64)` | **14.648** | **por defecto** para n ≈ 1.000 |
| `(32, 64, 128)` | **52.840** | solo con pretraining self-supervised o augmentación por píxel |

Desglose de la configuración por defecto: stem 608, bloques 9.360, cabeza 4.680. Embedding compartido de 64 dims.

El mismo modelo acepta los tres sustratos sin ningún cambio, gracias al `AdaptiveAvgPool2d(1)`: verificado con entradas `(4,8,52)` phenocube, `(4,8,8)` reshape y `(4,52,52)` GAF — las tres devuelven `(B, 8)`.

Contraste: `efficientnet_b0` son ~4 M, `convnext_tiny` ~28 M. Se baja **2–3 órdenes de magnitud**.

**Decisiones y por qué:**

- **Separable en profundidad** — reduce ~8× los parámetros de convolución con pérdida mínima de capacidad a esta escala.
- **`AdaptiveAvgPool2d(1)` en vez de `flatten`** — con 52×52 y 64 canales, un `flatten` daría 173.056 features y una cabeza lineal de 11 M de parámetros. El GAP la deja en 64. Este es el ahorro más grande de todo el diseño.
- **Un solo `stride=2`** — con entradas de 8 o 52 píxeles de lado no hay margen para la pirámide de 5 niveles de un backbone estándar.
- **BatchNorm y no LayerNorm** — batches de 32 sobre n=1.000 son estables; BN aporta regularización por ruido de batch, que aquí interesa.
- **Cabeza única de `n_targets` salidas**, no cabezas separadas por faceta. Cabezas separadas multiplican parámetros y rompen el efecto regularizador del tronco compartido. Si alguna faceta se comporta mal, ahí sí separarla.
- **Sin pooling adaptativo a 224, sin resize** — el tamaño nativo es el tamaño.

---

## 4. Régimen de entrenamiento — es tan importante como la arquitectura

Con n=1.000 la arquitectura es la mitad del problema. La otra mitad:

### 4.1 Pérdida y escalado de targets — reusar de `Trait_2DCNN`

- `MaskedMSELoss` (`training/losses.py:5-15`): `sum(loss * mask) / mask.sum()`, máscara desde los NaN por muestra (`training/data_loader.py:39-40`). Encaja directamente con el diseño de dos niveles de abundancia: las 224 parcelas de solo presencia llevan NaN en las cabezas ponderadas y contribuyen normalmente a las binarias. **No se descarta ninguna parcela y no se imputa nada.**
- `PowerTransformer(yeo-johnson, standardize=True)` por target (`training/data_loader.py:305-327`), ajustado **solo con el fold de entrenamiento**. Pone todas las facetas en varianza unitaria, que es el mecanismo de ponderación implícito entre tareas.
- Considerar `MaskedHuberLoss` (ya implementada, `losses.py:18-35`) si LCBD o los ejes NMDS traen outliers.

### 4.2 Augmentación — específica de fenología, no de visión

Nada de flips ni rotaciones: destruyen el mapeo eje→significado.

| Augmentación | Qué hace | Justificación |
|---|---|---|
| **Jitter temporal** | desplazar la curva ±3–5 DOY (roll circular) | incertidumbre real en el anclaje de fase |
| **Escalado de amplitud** | multiplicar por U(0,95, 1,05) | variación de calibración y de fracción de cobertura |
| **Desplazamiento de línea base** | sumar U(−0,02, 0,02) | efectos de suelo y de atmósfera residual |
| **Ruido gaussiano** | N(0, 0,01) sobre la imagen | ya implementado en `data_loader.py:29-30` |
| **Year dropout** (sustrato A) | enmascarar 1–2 filas de año al azar | fuerza robustez a años faltantes, que es el caso real |
| **Mixup** entre parcelas | interpolar entradas y targets con λ~Beta(0,2 ; 0,2) | el más efectivo con n pequeño en regresión; requiere targets continuos, que es el caso |

### 4.3 Augmentación por píxel — multiplica n por ~9

Cada parcela ocupa entre 1 y 11 píxeles Landsat (78,5–10.000 m² contra 900 m²/píxel). Tomar la ventana 3×3 alrededor del centroide da ~9 muestras por parcela con la misma etiqueta.

- Sube el n efectivo de ~1.000 a ~9.000, con ruido de etiqueta correlacionado.
- **Condición innegociable:** los 9 píxeles de una parcela van siempre al mismo fold. El block CV espacial ya lo garantiza; un CV aleatorio produciría fuga directa y R² fantasma.
- Reportar el n de parcelas, no el de píxeles, en el abstract.

### 4.4 Pretraining self-supervised — la palanca más grande

`Trait_2DCNN` ya trae MAE-2D completo: `models/mae_2d.py` (`MAE2D:32`, `random_masking:127`, `MAEEncoder:263`), `training/pretrain_mae.py`, `training/finetune_mae.py` con modos `finetune` y `linear_probe`. Fue diseñado para 139 K muestras sin etiqueta.

Aquí el corpus no etiquetado es enorme y gratis: **todos los píxeles de vegetación de Chile central con curva fenológica reconstruible**, del orden de 10⁶–10⁷. Reconstruir píxeles enmascarados de la phenocube enseña la estructura estacional e interanual sin usar una sola etiqueta de campo.

El encoder MAE hay que reducirlo — el ViT-Tiny actual sigue siendo grande para 8×52. Un autoencoder convolucional con el mismo tronco `PhenoNetS` es lo apropiado a esta escala.

Orden de prioridad si el tiempo es limitado: **(1) block CV espacial correcto → (2) mixup + augmentación fenológica → (3) pretraining self-supervised → (4) ajuste fino de arquitectura.** El punto 4 es el de menor retorno.

### 4.5 Validación

- **Block CV espacial obligatorio desde el primer experimento.** Bloques de 10–25 km, o agrupamiento por `Location` (194 niveles). Con este grado de agrupamiento un CV aleatorio da R² inflados que después hay que retractar.
- Reportar **también** el CV aleatorio, explícitamente etiquetado como optimista. La brecha entre ambos es un resultado en sí.
- CV por `Owner` o `metadata_id` (12 contribuyentes, 25 proyectos) como prueba de transferibilidad entre protocolos de muestreo.
- **Ensemble de 5 semillas** — con 11 k parámetros el coste es trivial y la reducción de varianza es real. Reportar media ± desviación entre semillas, no una corrida única.
- Early stopping con `patience=15` sobre `val_loss` (ya configurado en `train_2d.py:60-65`).

---

## 5. Matriz experimental mínima

| Modelo | Entrada | Params | Qué prueba |
|---|---|---|---|
| RF | 18 métricas LSP + DEM | — | baseline del campo |
| RF | curva aplanada (52×C) + DEM | — | ¿aporta la curva sobre sus resúmenes? |
| 1D-CNN | curva 52×C | ~5 k | ¿aporta la convolución? |
| **PhenoNet-S** | phenocube (C, Y, 52) | ~11 k | ¿aporta la 2ª dimensión *natural*? |
| **PhenoNet-S** | `reshape` 8×8 | ~11 k | ¿aporta la 2ª dimensión *manufacturada*? |
| **PhenoNet-S** | `gaf` / `mtf` / `cwt` 52×52 | ~11 k | catálogo de transformaciones en su dominio nativo |
| PhenoNet-S + MAE | phenocube | ~11 k | ¿cuánto aporta el pretraining? |
| `efficientnet_b0` | phenocube escalada a 224 | 4 M | control negativo: documentar el sobreajuste |

El último es deliberado. Que un backbone estándar sobreajuste con n=1.000 es un resultado reportable, y responde por adelantado a la pregunta obvia del referee de por qué no se usó una arquitectura conocida.

---

## 6. Efecto sobre la evaluación de innovación

La contribución no se debilita al bajar la escala del modelo; se **desplaza y se afila**:

- **Antes:** "aplicamos transformaciones señal→imagen y un backbone de visión a datos fenológicos".
- **Ahora:** "mostramos que una CNN de 11 k parámetros sobre una imagen fenológica año×DOY predice varias facetas de diversidad conjuntamente, y que los backbones de visión estándar fracasan en este régimen de datos".

El segundo enunciado es más específico, más falsable y más útil para la comunidad, que sistemáticamente trabaja con cientos a miles de parcelas y no con millones de imágenes. Pettorelli et al. (2024) identifican la escasez de datos de entrenamiento como el cuello de botella real del deep learning en biodiversidad: un resultado positivo en régimen de datos pequeños ataca ese cuello de botella directamente.

**El riesgo R1 de `02_innovation_and_impact.md` queda mitigado, no eliminado.** Sigue en pie que Random Forest puede ganar. Con esta arquitectura la comparación pasa a ser justa en vez de un hombre de paja.
