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
                   baseline: float = 0.02, noise: float = 0.01,
                   slope: float = 0.0, prob: float = 1.0) -> np.ndarray:
    """Apply the per-sample curve augmentations. ``curves`` is (B, C, n).

    - **jitter** ``N(0, jitter_sd)`` steps: real uncertainty in the phase anchoring, whose
      within-plot circular sd is 14 days at the median (docs/07).
    - **amplitude** ``U(1-amp, 1+amp)``: calibration drift and fractional-cover variation.
    - **baseline** ``U(-baseline, +baseline)``: residual soil and atmospheric effects.
    - **noise** ``N(0, noise)``: generic regularisation.
    - **slope** ``U(-slope, +slope)``: a linear tilt across the year, off by default.
    - **prob**: fraction of samples augmented at all; 1.0 augments every one.

    All are applied per sample and shared across that sample's channels, because a
    calibration or phase error affects every vegetation index of the same plot together.

    **On the slope term, and why it is off by default.** It is the one augmentation the
    Trait_2DCNN pipeline has that this project lacked (`dataaugment`, from
    Deep-Chemometrics), and in that paper the choice of augmentation moved the score by as
    much as the 1D-to-2D transform itself. But a tilt is not a neutral nuisance here: the
    curve is a composite over three years, and the interannual trend inside it is **real
    signal** -- regressing the year-boundary step on that trend gives a slope of -0.945
    against a predicted -1 (`docs/13_phenology_year_boundary.md`). Augmenting with random
    tilts teaches the model to ignore exactly that. It is exposed so the question can be
    answered by measurement rather than assumed either way.

    **On `prob`.** Trait_2DCNN augments 15% of samples; this project has always augmented
    100%. Since the measured ablation has `no-augment` *beating* the default, the amount of
    augmentation is a live question, not a settled one.
    """
    b = curves.shape[0]
    out = curves.copy()
    shifts = rng.normal(0.0, jitter_sd, size=b)
    scales = rng.uniform(1.0 - amp, 1.0 + amp, size=b)
    offsets = rng.uniform(-baseline, baseline, size=b)
    tilts = rng.uniform(-slope, slope, size=b) if slope > 0 else np.zeros(b)
    hit = rng.random(b) < prob if prob < 1.0 else np.ones(b, dtype=bool)
    # a ramp from -0.5 to +0.5 across the cycle, so a tilt pivots about the mean and does
    # not also shift the level -- that is what `baseline` is for
    ramp = np.linspace(-0.5, 0.5, curves.shape[-1], dtype=curves.dtype)
    for i in range(b):
        if not hit[i]:
            continue
        out[i] = frac_roll(curves[i], shifts[i]) * scales[i] + offsets[i]
        if tilts[i]:
            out[i] = out[i] + tilts[i] * ramp
    if noise > 0:
        out[hit] += rng.normal(0.0, noise, size=out[hit].shape).astype(out.dtype)
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
