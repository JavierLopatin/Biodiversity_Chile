# Seasonal phase: LSP anchoring and the 2D transform

Two decisions that look like one. They are not, and they pull in opposite directions.

Written in English because it is destined for the methods section.

---

## 1. The measurement

Audited over the extracted plots (`scripts/04_recompute_lsp.py`, NDVI, centre pixel and
5x5 window), central Chile, 2003–2026:

| Quantity | Value |
|---|---|
| Plots whose harmonic peak lies within 60 d of the 1-Jan cut | **81 %** (91/113) |
| Plot-level peak in Jan–Feb | 76 / 113 |
| Plot-level peak in Nov–Dec | 15 / 113 |
| Plot-level peak in **May–Aug** (austral winter) | 14 / 113 |
| Circular concentration of the peak, plot level | R = 0.68 |
| Circular concentration, pixel level | R = 0.42 |
| Trough concentrated in | Jul–Aug |

Two facts follow, and they are in tension:

1. **There is a dominant regional phase** (R = 0.68): most plots green up around the
   austral summer, peaking near the calendar-year boundary.
2. **It is not the only phase.** 14 plots (12 %) peak in austral winter — the matorral
   pattern of greening on the winter rains and senescing through the summer drought. A
   single global rotation serves the majority and mis-anchors this minority.

---

## 2. LSP anchoring — per pixel (`hemisphere="auto"`)

**Decision: `auto`.** The reason is ecological, not metric. Mis-anchoring the
winter-peaking plots is not random error: it is bias *correlated with community type*,
which is the very thing the model is being asked to predict. An aggregate score would not
reveal it, because the aggregate is dominated by the majority phase:

| anchoring | degenerate pixels (`sos == pos`) | median LOS | IQR LOS |
|---|---|---|---|
| `north` (no rotation) | 13.9 % | 129 | 42 |
| `south` (global austral) | 0.3 % | 185 | 27 |
| `auto` (per pixel) | 9.2 % | 150 | 71 |

`south` wins on every aggregate number and is still the wrong choice: it buys that score by
forcing one phase on vegetation that has two.

### Known limitation, quantified — `auto` under-rotates here

`season_phase` flags a pixel `aseasonal` when first-harmonic strength falls below
`strength_thresh` (default **0.15**), and `resolve_anchor` then applies **no rotation** to
it. In this dataset:

- median per-pixel seasonality strength = **0.108**, below the threshold;
- only **15 %** of pixels clear it;
- averaging the 25 pixels of a plot first does **not** help (median strength 0.097, 13 %
  clear) — the weak seasonality is real, not pixel noise.

So `auto` currently behaves like `north` for roughly six pixels in seven, and only rotates
the strongly seasonal minority. That is *better* than a forced global rotation, but it does
not yet deliver the intended per-community anchoring.

`PhenoLSP` does not expose `strength_thresh`, so lowering it requires calling
`season_phase` directly and rotating explicitly. **Open item**, not resolved here.

### Within-plot coherence — monitored, not assumed

Per-pixel anchoring can assign different phenological years to neighbouring pixels of the
same plot, which would break the 5x5 augmentation (the 25 samples would not be mutually
comparable). Measured circular sd of the anchor across each plot's 25 pixels:

| circular sd | plots |
|---|---|
| > 15 d | 60 % |
| > 30 d | 35 % |
| > 60 d | 7 % |

`scripts/04_recompute_lsp.py` writes `anchor_circ_sd` per plot. **Flag plots above the
chosen threshold before using their window for augmentation.**

---

## 3. The 2D transform — a global rotation, and never a per-plot one

This is a **separate** decision from LSP anchoring, and the reasoning inverts.

### The problem is real and measured

81 % of plots peak within 60 days of the calendar boundary. On a 52-step calendar-DOY
vector the growing season is therefore split across the two free ends of the array:

- **`reshape` to 8x8**: the season occupies the last row and the first row — adjacent in
  time, maximally distant in the image. No 3x3 kernel can see across the break.
- **Substrate A (year x DOY)**: same discontinuity along the horizontal axis, where the
  kernel is supposed to read "how the seasonal shape changes between neighbouring years".
- **GAF / MTF / CWT / spectrogram**: all sensitive to endpoint effects on a sequence that
  is in truth periodic.

The DOY axis is a **circular** coordinate that the array represents as if it had two ends.
That mismatch is the defect, not the absence of centring per se.

### Global rotation, identical for every plot

Rotate the DOY axis so the array boundary falls at the **regional trough (~1 July)**,
placing the productive season in the middle of the array.

Crucially the shift must be **the same constant for all plots**. A global shift is a
relabelling of the axis: it preserves the phase *differences between* plots, and those
differences are signal — winter-peaking matorral versus summer-peaking sclerophyll is
exactly the community distinction the model should exploit.

**Per-plot centring would be a mistake.** Aligning every plot's own peak to the array
centre deletes phase from the input and moves it into a discarded offset. It is the right
move for LSP (each pixel's metrics belong in its own frame) and the wrong move for a
CNN input.

### Complementary and more principled: circular padding

Use circular (wrap) padding along the DOY axis in the convolutions instead of zero padding.
This treats the coordinate as periodic and removes the boundary artefact *regardless of
rotation*. Rotation and circular padding address the same defect and can be combined.

### Random Forest is unaffected

RF treats the 52 curve values as unordered features; any fixed permutation of the axis
gives identical splits and identical results. **The rotation question applies only to the
CNN and the 2D transforms.** It cannot explain any RF-vs-CNN difference, which is worth
stating explicitly since RF is the first-class baseline.

---

## 4. What is stored, and what remains open

Curves are stored on the **calendar-DOY axis, unrotated** — the canonical, unambiguous
representation. Rotation belongs to the transform stage, where it must be an explicit,
logged step rather than silent preprocessing.

`phenoshape` and the raw observations are anchoring-independent, so every option below can
be tested without returning to the datacube.

**Carry into the experimental matrix as factors, not as assumptions:**

| Factor | Levels | Applies to |
|---|---|---|
| DOY rotation | none / global to austral trough | CNN, 2D transforms |
| Convolution padding along DOY | zero / circular | CNN |
| LSP anchoring | `auto` / `south` / `auto` with lowered `strength_thresh` | LSP metrics, RF baseline |

**Open items:**

1. `auto` under-rotates: 85 % of pixels fall below `strength_thresh = 0.15`. Requires
   calling `season_phase` directly to lower it. Unresolved.
2. Within-plot anchor incoherence exceeds 30 d in 35 % of plots. Threshold for exclusion
   not yet set.
3. The phase audit so far covers 35.8–37.9°S. **Re-run over the full 30–38°S subset once
   extraction finishes** — the northern, more arid plots may carry a different phase
   distribution, and the winter-peaking fraction could be larger there.
