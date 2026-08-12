"""Patrón de muestreo: las dos distribuciones de distancia que STeMP declara obligatorias.

`Sampling pattern` es campo obligatorio de STeMP (Linnenbrink et al. 2026, Tabla A1), y su
app lo infiere comparando distancias al vecino más próximo. No es burocracia: **es lo que
decide si la validación elegida es la correcta**. Con muestras agrupadas respecto del dominio
de predicción, una CV aleatoria mide interpolación a corta distancia y llama a eso exactitud
del mapa.

Dos distribuciones, siguiendo Milà et al. (2022) y Linnenbrink et al. (2024):

``Gj``
    de cada muestra a la muestra más próxima. Es la distancia que un fold de CV *deja* entre
    entrenamiento y test si no se bloquea nada.
``Gij``
    de cada punto del dominio de predicción a la muestra más próxima. Es la distancia a la
    que el modelo va a tener que predecir de verdad.

**El diagnóstico es la comparación, no cada una por separado.** Si `Gij` está desplazada muy
a la derecha de `Gj`, el mapa predice sistemáticamente más lejos de lo que la CV pone a
prueba, y la CV es optimista por construcción. Si las dos coinciden, la CV aleatoria es
defendible. Es exactamente el criterio con el que `kNNDM` construye sus folds.

Se reporta además el índice de Clark-Evans (1954) `R = d_obs / d_esperada`, con la esperanza
bajo aleatoriedad espacial completa `0,5 / sqrt(n/A)`: `R < 1` agrupado, `≈ 1` aleatorio,
`> 1` regular. Es interpretable de un vistazo y clásico en ecología, pero **no sustituye** a
la comparación de arriba: es intrínseco a las muestras y no sabe nada del dominio donde se
va a predecir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class SamplingPattern:
    """El resultado, con el estadístico que respalda la etiqueta y no sólo la etiqueta."""

    label: str                  # "clustered" | "random" | "regular"
    clark_evans: float          # R = d_obs / d_CSR
    gj: np.ndarray              # muestra -> muestra mas proxima (m)
    gij: np.ndarray             # dominio -> muestra mas proxima (m)
    ks_stat: float              # separacion entre las dos ECDF
    ks_p: float
    area_m2: float
    n_samples: int
    n_domain: int

    def summary(self) -> dict:
        q = [5, 25, 50, 75, 95]
        return {
            "label": self.label,
            "clark_evans_R": round(self.clark_evans, 3),
            "n_samples": self.n_samples,
            "n_domain": self.n_domain,
            "area_km2": round(self.area_m2 / 1e6, 1),
            **{f"Gj_p{p}_km": round(float(np.percentile(self.gj, p)) / 1000, 3)
               for p in q},
            **{f"Gij_p{p}_km": round(float(np.percentile(self.gij, p)) / 1000, 3)
               for p in q},
            "median_ratio_Gij_Gj": round(float(np.median(self.gij) / np.median(self.gj)), 2),
            "ks_stat": round(self.ks_stat, 3),
            "ks_p": float(self.ks_p),
        }


def nnd(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Distancia de cada fila de ``a`` al punto más próximo de ``b`` (o de ``a`` sin sí mismo).

    Con ``b=None`` se pide el **segundo** vecino, porque el primero es el propio punto a
    distancia cero. Los duplicados exactos —dos parcelas en el mismo píxel— dan 0 y eso es
    correcto: la distancia entre ellas *es* cero.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"se esperaba (n, 2), llego {a.shape}")
    if b is None:
        if len(a) < 2:
            raise ValueError("hacen falta al menos 2 puntos para Gj")
        d, _ = cKDTree(a).query(a, k=2)
        return d[:, 1]
    b = np.asarray(b, dtype=float)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"se esperaba (m, 2), llego {b.shape}")
    d, _ = cKDTree(b).query(a, k=1)
    return d


def clark_evans(xy: np.ndarray, area_m2: float) -> float:
    """``R = d_observada / d_esperada`` bajo aleatoriedad espacial completa.

    Sin corrección de borde: con 1.082 parcelas repartidas en decenas de miles de km² el
    sesgo de borde es de segundo orden frente al agrupamiento que se está midiendo, y la
    corrección exigiría el polígono del dominio y no sólo su área.
    """
    d = nnd(xy)
    n = len(xy)
    expected = 0.5 / np.sqrt(n / area_m2)
    return float(d.mean() / expected)


def sampling_pattern(sample_xy: np.ndarray, domain_xy: np.ndarray,
                     area_m2: float | None = None,
                     clustered_ratio: float = 1.5,
                     regular_R: float = 1.2) -> SamplingPattern:
    """Clasifica el patrón y devuelve las dos distribuciones que STeMP pide.

    ``area_m2`` por omisión es la caja envolvente del **dominio**, no la de las muestras: el
    denominador de Clark-Evans tiene que ser el área donde se va a predecir, o unas muestras
    concentradas en una esquina de un dominio grande saldrían "aleatorias" respecto de su
    propia esquina.

    Los dos umbrales son convención declarada, no ley: ``clustered_ratio`` es cuántas veces
    más lejos predice el mapa que lo que la CV pone a prueba (mediana de `Gij` sobre mediana
    de `Gj`), y ``regular_R`` es el Clark-Evans por encima del cual el patrón se llama
    regular. Se exponen como argumentos para que el juicio quede en la llamada y no
    escondido aquí.
    """
    sample_xy = np.asarray(sample_xy, dtype=float)
    domain_xy = np.asarray(domain_xy, dtype=float)
    if area_m2 is None:
        span = domain_xy.max(axis=0) - domain_xy.min(axis=0)
        area_m2 = float(span[0] * span[1])

    gj = nnd(sample_xy)
    gij = nnd(domain_xy, sample_xy)
    R = clark_evans(sample_xy, area_m2)
    ks = ks_2samp(gj, gij)
    ratio = float(np.median(gij) / np.median(gj)) if np.median(gj) > 0 else np.inf

    if ratio >= clustered_ratio:
        label = "clustered"
    elif R >= regular_R:
        label = "regular"
    else:
        label = "random"

    return SamplingPattern(label=label, clark_evans=R, gj=gj, gij=gij,
                           ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                           area_m2=area_m2, n_samples=len(sample_xy),
                           n_domain=len(domain_xy))
