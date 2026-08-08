"""Masked regression losses.

Vendored from ``Trait_2DCNN/training/losses.py`` (lines 5-35), unchanged in semantics.

The mask is what lets the cover tier exist. 536 of the 1,082 plots have no comparable
abundance measurement, so `lcbd_cover`, `pcoa1_cover` and `pcoa2_cover` are NaN for them by
design. Multiplying the elementwise loss by a 0/1 mask and dividing by ``mask.sum()`` means
those plots contribute normally to the six complete heads and not at all to the three masked
ones — no plot is dropped and nothing is imputed.

Note what the normalisation does and does not do: it averages over *observed entries pooled
across samples and targets*, so a head with twice the observations contributes twice the
gradient. That is intentional here — the alternative, per-target reweighting, would let the
546-plot cover heads pull as hard as the 1,082-plot ones and would make the joint fit worse
for both. The Yeo-Johnson scaling already equalises the *scale* of the heads, which is the
part that matters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum()
        if denom == 0:
            return (pred * 0.0).sum()          # keeps the graph alive on a fully masked batch
        return (((pred - target) ** 2) * mask).sum() / denom


class MaskedHuberLoss(nn.Module):
    """Default for this project: LCBD and the PCoA axes carry genuine outliers.

    A plot with almost no compositional information sits far out on a PCoA axis; squared
    error would let a handful of those dominate the gradient for every head that shares the
    trunk.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum()
        if denom == 0:
            return (pred * 0.0).sum()
        err = pred - target
        a = err.abs()
        quad = torch.clamp(a, max=self.delta)
        loss = 0.5 * quad**2 + self.delta * (a - quad)
        return (loss * mask).sum() / denom


def make_loss(name: str = "huber", delta: float = 1.0) -> nn.Module:
    return {"huber": lambda: MaskedHuberLoss(delta), "mse": MaskedMSELoss}[name]()
