"""Área de aplicabilidad (Meyer & Pebesma 2021): dónde el modelo tiene derecho a predecir.

Un mapa asigna un valor a cada píxel, tenga o no el modelo información para hacerlo. El AOA
delimita dónde la tiene: **el subconjunto del dominio cuya combinación de predictores se
parece a alguna que el modelo vio al entrenar**. Fuera de ahí la predicción es extrapolación
y su error no está acotado por ninguna validación cruzada.

Hace falta aquí por una razón medida, no por formalismo: las parcelas de este proyecto están
extremadamente agrupadas (Clark-Evans R = 0,14) y el dominio nativo dista una mediana de
13,2 km de la parcela más próxima (`results/tables/sampling_pattern.json`). Un mapa sobre
todo el territorio va a predecir mayoritariamente lejos de cualquier dato.

**El índice.** Para cada punto nuevo `k`:

    DI_k = d_k / d̄            d_k = distancia minima a un punto de entrenamiento
                               d̄   = media de las distancias por pares del entrenamiento

en el espacio de predictores centrado, escalado y **ponderado por importancia de variable**.
La ponderación importa: sin ella, veinte columnas de fenología irrelevantes pesan lo mismo
que la elevación y el índice mide parecido en variables que el modelo ignora.

**El umbral.** No es un percentil del DI del dominio -- eso lo haría depender de dónde se
predice-- sino del DI del **entrenamiento consigo mismo, calculado entre folds de validación
cruzada**: para cada parcela, la distancia al vecino de entrenamiento más próximo *que no
esté en su fold*. El corte es `Q3 + 1,5·IQR` de esa distribución, el "máximo sin atípicos".
Usar los folds del proyecto y no una partición aleatoria es lo que hace que el umbral herede
el mismo bloqueo espacial que el modelo.

Referencia: Meyer, H. & Pebesma, E. (2021), *Predicting into unknown space? Estimating the
area of applicability of spatial prediction models*, Methods in Ecology and Evolution 12(9),
1620-1633. doi:10.1111/2041-210X.13650
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AOA:
    di: np.ndarray              # indice de disimilitud de cada punto nuevo
    inside: np.ndarray          # di <= threshold
    threshold: float
    di_train: np.ndarray        # DI del entrenamiento entre folds, del que sale el umbral
    mean_pairwise: float        # d̄, el normalizador

    @property
    def fraction_inside(self) -> float:
        return float(self.inside.mean())


def _weighted_scale(train: np.ndarray, new: np.ndarray, weights: np.ndarray | None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Centra y escala con la media y sd del **entrenamiento**, y aplica los pesos.

    Con los estadísticos del conjunto nuevo, un dominio con más varianza que el
    entrenamiento se vería artificialmente parecido a él: el escalado tiene que venir del
    lado que define el espacio conocido.
    """
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0                      # una columna constante no aporta distancia
    t = (train - mu) / sd
    n = (new - mu) / sd
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (train.shape[1],):
            raise ValueError(f"weights debe tener {train.shape[1]} elementos, tiene {w.shape}")
        if np.any(w < 0):
            raise ValueError("las importancias no pueden ser negativas")
        s = w.sum()
        if s <= 0:
            raise ValueError("todas las importancias son cero")
        w = w / s * len(w)                 # media 1, para que d̄ siga siendo comparable
        t, n = t * w, n * w
    return t, n


def _mean_pairwise(t: np.ndarray, max_n: int = 4000, seed: int = 0) -> float:
    """Media de las distancias por pares del entrenamiento, submuestreando si hace falta.

    Es O(n²) y con 1.082 parcelas es exacto; el submuestreo sólo actúa si algún día se
    entrena con decenas de miles.
    """
    if len(t) > max_n:
        idx = np.random.default_rng(seed).choice(len(t), max_n, replace=False)
        t = t[idx]
    d = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    iu = np.triu_indices(len(t), k=1)
    return float(d[iu].mean())


def train_di(train: np.ndarray, folds: list[np.ndarray], weights: np.ndarray | None = None,
             ) -> tuple[np.ndarray, float, float]:
    """DI del entrenamiento entre folds, más el umbral y el normalizador.

    ``folds`` es una lista de arreglos de índices, uno por fold: para cada punto, el vecino
    más próximo se busca **fuera** de su propio fold. Con una partición aleatoria el umbral
    saldría demasiado bajo, porque casi siempre habría un vecino contiguo en otro fold.
    """
    t, _ = _weighted_scale(train, train[:1], weights)
    dbar = _mean_pairwise(t)
    di = np.full(len(t), np.nan)
    for f in folds:
        f = np.asarray(f)
        mask = np.ones(len(t), bool)
        mask[f] = False
        if not mask.any() or not len(f):
            continue
        d, _ = cKDTree(t[mask]).query(t[f], k=1)
        di[f] = d / dbar
    if np.isnan(di).any():
        raise ValueError("hay puntos de entrenamiento sin fold asignado")
    q1, q3 = np.percentile(di, [25, 75])
    return di, float(q3 + 1.5 * (q3 - q1)), dbar


def aoa(train: np.ndarray, new: np.ndarray, folds: list[np.ndarray],
        weights: np.ndarray | None = None) -> AOA:
    """Índice de disimilitud y máscara de aplicabilidad para ``new``."""
    train = np.asarray(train, dtype=float)
    new = np.asarray(new, dtype=float)
    if train.ndim != 2 or new.ndim != 2 or train.shape[1] != new.shape[1]:
        raise ValueError(f"shapes incompatibles: {train.shape} vs {new.shape}")
    t, n = _weighted_scale(train, new, weights)
    di_tr, thr, dbar = train_di(train, folds, weights)
    d, _ = cKDTree(t).query(n, k=1)
    di = d / dbar
    return AOA(di=di, inside=di <= thr, threshold=thr, di_train=di_tr, mean_pairwise=dbar)
