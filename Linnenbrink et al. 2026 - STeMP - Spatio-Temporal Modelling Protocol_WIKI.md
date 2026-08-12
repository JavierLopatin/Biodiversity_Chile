---
type: paper
title: "Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol"
created: 2026-08-11
updated: 2026-08-11
tags:
  - paper
status: developing
year: 2026
authors:
  - "Jan Linnenbrink"
  - "Jakub Nowosad"
  - "Marvin Ludwig"
  - "Anna Frederike Jablotschkin"
  - "Fabian Schumacher"
  - "Teja Kattenborn"
  - "Hanna Meyer"
venue: "Preprint (not yet peer-reviewed)"
doi: ""
url: "https://github.com/LOEK-RS/STeMP"
bibkey: "linnenbrink2026stemp"
aliases:
  - "Linnenbrink 2026"
  - "Linnenbrink et al. 2026"
  - "linnenbrink-2026-stemp"
  - "STeMP paper"

# --- Domain-specific (remote sensing / ecology) ---
sensor: ""
spatial_resolution: ""
temporal_scope: ""
study_area: "worked example: South America (plant species richness prediction)"
ecosystem: ""

# --- Methods & reproducibility ---
methods:
  - "reporting-protocol design (Overview/Model/Prediction sections)"
  - "Shiny web application (golem framework)"
  - "automated pitfall-warning system (rule-based)"
  - "worked example: random forest regression"
software: "R (Shiny, golem, testthat, shinytest); caret/tidymodels/mlr3 RDS model ingestion"
open_data: true
open_code: true

# --- Findings ---
key_claim: "STeMP is a standardized reporting protocol + accompanying R/Shiny software for spatio-temporal machine-learning models that both documents modelling decisions and automatically flags common spatial-ML pitfalls (clustered-sample overoptimism, data leakage, missing extrapolation uncertainty, unjustified spatial-proxy use)."
key_metrics: "Worked example (RF, plant species richness, South America): R²=0.737, RMSE=22.330 — flagged by STeMP itself as likely overoptimistic."

# --- Personal evaluation ---
my_rating:
my_use_case: "Candidate reporting checklist for any spatio-temporal ML mapping product in the vault (e.g. refugia/functional-diversity RS models); its auto-detected pitfall list overlaps directly with [[Spatial Block Cross-Validation]]."

# --- Wiki connections ---
methodology: "reporting-protocol / methodology-transparency framework, not a predictive method itself"
contradicts: []
supports:
  - "[[Spatial Block Cross-Validation]]"
related:
  - "[[Spatial Block Cross-Validation]]"
sources: []
---

# Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol

## Citation
> Linnenbrink, J., Nowosad, J., Ludwig, M., Jablotschkin, A.F., Schumacher, F., Kattenborn, T., Meyer, H. *STeMP: Spatio-Temporal Modelling Protocol.* Preprint, July 2026. Software: https://github.com/LOEK-RS/STeMP (Zenodo archive v2026.07.00, https://doi.org/10.5281/zenodo.21494506).

## Key Claim
Spatio-temporal ML models in the geosciences lack a standardized reporting protocol (unlike species distribution models with ODMAP, or agent-based models with ODD); STeMP fills this gap with a three-section protocol plus a Shiny app that auto-fills fields from uploaded R model objects and raises warnings for known spatial-ML pitfalls.

## Abstract / TL;DR
Machine-learning models are highly sensitive to training-data characteristics and methodological choices (cross-validation strategy, tuning), yet spatio-temporal ML studies routinely under-report these decisions. STeMP proposes a protocol (Overview / Model / Prediction sections) and an accompanying R package with a web app that auto-extracts metadata from trained models and spatial datasets, and warns authors/reviewers about pitfalls like clustered-sample evaluation bias and data leakage.

## Methodology
- **Design**: Protocol structured in 3 top-level sections — Overview (metadata), Model (7 sub-sections: response, predictors, evaluation/selection, learning method, interpretation, uncertainty), and optional Prediction (map evaluation, prediction domain, post-processing). Mandatory vs. optional fields enumerated in Table A1 (see paper).
- **Sensor / Data**: N/A — this is a reporting-protocol/software paper, not a predictive-mapping study.
- **Study area**: Worked example uses South America (plant species richness).
- **Sample / N**: Worked example trains on sPlotOpen plot data (Sabatini et al. 2021) + elevation (Jarvis et al. 2008) + WorldClim climate predictors (Fick & Hijmans 2017).
- **Analysis**: Random forest regression in the worked example; the protocol itself is evaluated qualitatively (does it surface known pitfalls?), not benchmarked against alternative protocols.

## Findings
1. **Protocol structure**: Overview → Model (7 sub-sections) → optional Prediction, aligned with the typical spatio-temporal ML workflow from data acquisition to post-processed prediction (Fig. 1, Table A1).
2. **Automated pitfall detection** (rule-based, from uploaded model + spatial data): (a) evaluation strategy mismatched to a clustered sampling design (e.g., random CV with spatially clustered training points); (b) data leakage between training and evaluation; (c) absence of uncertainty quantification when extrapolation is likely; (d) use of spatial proxies as predictors without a documented predictor-selection strategy.
3. **Worked example demonstrates the pitfalls in practice**: a random forest model of plant species richness reports R²=0.737 / RMSE=22.330 under random cross-validation; STeMP's own warning system flags this as likely overoptimistic once it detects the training points are geographically clustered relative to the prediction domain, that spatial-proxy predictors were used without selection, and that no extrapolation/uncertainty method was applied.
4. **Implementation**: modular Shiny app built on the `golem` framework, extending the ODMAP web application (Zurell et al. 2020); auto-fills fields from `.RDS` model objects trained via `caret`, `tidymodels`, or `mlr3`, and from uploaded training-point/prediction-domain `.gpkg` files (auto-infers sampling pattern from nearest-neighbour distances, per Meyer et al. 2026 / Linnenbrink et al. 2024 / Baddeley et al. 2015).

## Quantitative Results
- Worked example: RF regression, plant species richness (South America) — **R² = 0.737, RMSE = 22.330** under random CV (text-reported, not chart-extracted).
- Predictor resolution in the example: 0.0833° (≈9.3 km at the equator) — used by the authors to argue the resulting map is suited to continent-scale, not local-planning, applications.

> [!note] Two figures in this source carry `VLM-CHART, VERIFY` warnings (a "heatmap" of the predicted richness map and an "area" plot of nearest-neighbour distance densities). Per vault policy, no numeric values from those blocks are used above — only the R²/RMSE/resolution figures that appear in the source's plain-text paragraphs.

## Strengths
- Directly operationalizes a known, high-consequence problem in the vault (see [[Spatial Block Cross-Validation]]): clustered-sample + random-CV overoptimism, already documented empirically at up to ~28 percentage points inflation for CNN vegetation segmentation ([[Kattenborn et al. 2022]], cited in this paper as Nowosad et al. 2026 co-authorship overlap).
- Automation (auto-fill from R model objects + spatial data) lowers the reporting-burden objection that typically kills protocol adoption.
- Explicitly targets three distinct audiences (developers, evaluators/reviewers, end-users) with protocol fields tuned to each.
- Open governance (GitHub, GPL license) + versioned Zenodo archive of the software.

## Limitations
- **R-only auto-extraction**: only `caret`/`tidymodels`/`mlr3` RDS objects are currently supported; no Python/scikit-learn/PyTorch ingestion path, which limits adoption outside the R geospatial-ML community. Authors flag ONNX support as future work.
- **Still spatially weighted**: despite "spatio-temporal" framing, the authors explicitly acknowledge the protocol remains "mainly focused on spatial machine-learning," with temporal-dimension fields less developed.
- **Rule-based warnings, not adjudication**: the pitfall checks are heuristics operating on uploaded metadata (e.g., inferred clustering from nearest-neighbour distances); they can miss pitfalls not covered by the four checked categories, and a clean "no warnings" result could be mistaken for a validated model.
- **No adoption/impact evaluation**: the paper does not test whether completing the protocol changes reviewer scrutiny or measurably reduces over-optimistic reporting in practice — validation is by worked example, not by a study of protocol efficacy.
- **Governance/versioning risk**: protocol definition lives on GitHub and is community-editable; without strict field-versioning discipline, cross-study comparability (the protocol's own stated goal) could erode over time.

## Reproducibility
- **Data availability**: Worked-example code and data available via Zenodo (https://doi.org/10.5281/zenodo.21493718).
- **Code availability**: R package + Shiny app on GitHub (https://github.com/LOEK-RS/STeMP), GPL license; software version in this manuscript (v2026.07.00) archived on Zenodo (https://doi.org/10.5281/zenodo.21494506).
- **My replication notes**: Protocol + R package are directly reusable as a documentation checklist for any spatio-temporal RS/ML mapping product; would need to confirm current CRAN/GitHub install status and RDS-object compatibility with whatever R modelling framework is in use before relying on auto-fill.

## Connections
- **Builds on**: ODMAP protocol (Zurell et al. 2020 — cited, not yet ingested), ODD protocol (Grimm et al. 2006 — cited, not yet ingested), Model Cards (Mitchell et al. 2019 — cited, not yet ingested).
- **Contradicts**: —
- **Concepts introduced**: [[STeMP (Spatio-Temporal Modelling Protocol)]]
- **Concepts used**: [[Spatial Block Cross-Validation]] (same pitfall class: clustered-sample + random-CV overoptimism)
- **Authors**: [[Jan Linnenbrink]], [[Hanna Meyer]], [[Teja Kattenborn]]

## Quotes
> "Failure to account for these effects can lead to substantial pitfalls like over-optimistic performance estimates, which may result in false confidence in the models and may ultimately even harm the trust in spatial predictive modelling in general." (Introduction)

> "While detected issues should not automatically be interpreted as errors – as they may result from conscious and well-motivated modelling decisions – they still warrant careful consideration or justification." (§3.1)

## Raw Source
`[[.raw/pdfs/Machine-Learning/Linnenbrink et al. 2026 - STeMP - Spatio-Temporal Modelling Protocol.pdf]]`
