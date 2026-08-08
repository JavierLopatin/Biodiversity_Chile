"""Phenology-specific augmentation, applied to the 1-D curve before the substrate map.

Operating on the curve rather than on the image is what lets one implementation serve the
MLP, the 1-D CNN and every 2-D substrate: the transform is recomputed per batch, which costs
about 0.5 ms for a 52x52 GAF and nothing at all for `reshape` or `stack5`.

**No flips, no rotations, no crops.** Standard vision augmentation destroys the axis-to-
meaning mapping that the whole design rests on: a horizontally flipped phenological curve
describes senescence before green-up, which is not a plant.

One detail that is easy to get wrong. `docs/03_cnn_architecture.md` prescribes a temporal
jitter of +/-3-5 days. One step of this curve is **7 days**, so an integer ``np.roll`` would
shift by 7-35 days — five to ten times the intended perturbation, and enough to move green-up
into a different month. The jitter here is therefore a *fractional* circular roll, done by
interpolation.
"""

from __future__ import annotations

import numpy as np


def frac_roll(x: np.ndarray, shift_steps: float) -> np.ndarray:
    """Circular roll by a fractional number of steps, along the last axis.

    Linear interpolation between the two neighbouring steps. Circular because the DOY axis
    wraps: rolling week 51 forward lands on week 0.
    """
    n = x.shape[-1]
    idx = (np.arange(n) - shift_steps) % n
    lo = np.floor(idx).astype(int) % n
    hi = (lo + 1) % n
    w = idx - np.floor(idx)
    return x[..., lo] * (1.0 - w) + x[..., hi] * w


def augment_curves(curves: np.ndarray, rng: np.random.Generator,
                   jitter_sd: float = 0.5, amp: float = 0.05,
                   baseline: float = 0.02, noise: float = 0.01) -> np.ndarray:
    """Apply the four per-sample curve augmentations. ``curves`` is (B, C, 52).

    - **jitter** ``N(0, jitter_sd)`` steps: real uncertainty in the phase anchoring, whose
      within-plot circular sd is 14 days at the median (docs/07).
    - **amplitude** ``U(1-amp, 1+amp)``: calibration drift and fractional-cover variation.
    - **baseline** ``U(-baseline, +baseline)``: residual soil and atmospheric effects.
    - **noise** ``N(0, noise)``: generic regularisation.

    All four are applied per sample and shared across that sample's channels, because a
    calibration or phase error affects every vegetation index of the same plot together.
    """
    b = curves.shape[0]
    out = np.empty_like(curves)
    shifts = rng.normal(0.0, jitter_sd, size=b)
    scales = rng.uniform(1.0 - amp, 1.0 + amp, size=b)
    offsets = rng.uniform(-baseline, baseline, size=b)
    for i in range(b):
        out[i] = frac_roll(curves[i], shifts[i]) * scales[i] + offsets[i]
    if noise > 0:
        out += rng.normal(0.0, noise, size=out.shape).astype(out.dtype)
    return out


def mixup_lambda(rng: np.random.Generator, alpha: float = 0.2) -> float:
    return float(rng.beta(alpha, alpha))


def mixup_pair(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Return the shuffled batch and lambda.

    The targets are deliberately **not** mixed. Three of the nine heads are NaN for half the
    plots, and a convex combination of a number and a NaN is a NaN — mixing targets would
    silently delete the cover heads for any pair that crosses the stratum boundary. The
    correct form mixes the two *masked losses* instead:

        lam * L(pred, y_a, m_a) + (1 - lam) * L(pred, y_b, m_b)

    which is what :func:`biodiv.trainer.train_one_fold` does.
    """
    perm = rng.permutation(x.shape[0])
    return perm, mixup_lambda(rng)


def pixel_sampler(pixel_cube: np.ndarray, anchor_sd: np.ndarray,
                  fallback: np.ndarray, max_anchor_sd: float = 30.0):
    """Draw one of the 25 pixels of each plot per epoch; fall back where anchoring is unstable.

    ``anchor_sd`` is the within-plot circular standard deviation of the phase anchor, per
    plot. Where it exceeds 30 days the 25 pixels are anchored to different phenological years
    and their curves are not mutually comparable — sampling among them would inject label
    noise that looks like signal (docs/07, section on phase-anchor incoherence: >30 d in 21%
    of plot-index pairs). Those plots use the 5x5 mean instead, and the excluded fraction is
    reported.

    Returns a closure so the draw is fresh every epoch.
    """
    unstable = np.asarray(anchor_sd) > max_anchor_sd

    def draw(rng: np.random.Generator, rows: np.ndarray) -> np.ndarray:
        k = rng.integers(0, pixel_cube.shape[1], size=len(rows))
        out = pixel_cube[rows, k]
        out[unstable[rows]] = fallback[rows][unstable[rows], 0]
        return out

    draw.excluded_fraction = float(unstable.mean())          # type: ignore[attr-defined]
    return draw
