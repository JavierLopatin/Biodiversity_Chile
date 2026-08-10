# Resultados — `kfold5_window`

99 corridas x 15 targets. Generado por `scripts/28_results_tables.py`; no editar a mano.

`%RMSE` es el RMSE dividido por el rango 1–99 del target, y `sesgo` la media del residuo con el mismo denominador. Los dos son porcentajes de rango, así que se pueden comparar entre facetas; el R² y el Spearman no necesitan normalización.

## El mejor modelo de cada target

| facet | target | run_id | family | features | index | substrate | n | seeds | R2 | R2_sd | spearman | %RMSE | sesgo % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha | hill_q0 | B03_coords | BASE | coords |  |  | 1082 | 3 | 0.663 | 0.003 | 0.788 | 12.5 | -0.2 |
| alpha | hill_q1 | MLP06_curve_kndvi | MLP | curve | kndvi | tabular | 1082 | 5 | 0.461 | 0.016 | 0.656 | 14.9 | -0.1 |
| alpha | hill_q2 | MLP06_curve_kndvi | MLP | curve | kndvi | tabular | 1082 | 5 | 0.408 | 0.022 | 0.584 | 14.5 | -0.2 |
| beta_cover | lcbd_cover | B03_coords | BASE | coords |  |  | 546 | 3 | 0.411 | 0.007 | 0.616 | 20.4 | -0.8 |
| beta_cover | pcoa1_cover | RF04_lsp-curve-topo-area_nbr | RF | lsp+curve+topo+area | nbr |  | 546 | 3 | 0.393 | 0.010 | 0.663 | 22.0 | 2.8 |
| beta_cover | pcoa2_cover | RF06_curve_all-topo-area | RF | curve_all+topo+area |  |  | 546 | 3 | 0.340 | 0.010 | 0.610 | 19.4 | 0.3 |
| beta_pa | lcbd_pa | B03_coords | BASE | coords |  |  | 1082 | 3 | 0.507 | 0.004 | 0.700 | 17.6 | 0.1 |
| beta_pa | pcoa1_pa | RF03_curve-topo-area_evi | RF | curve+topo+area | evi |  | 1082 | 3 | 0.574 | 0.005 | 0.770 | 16.0 | -0.1 |
| beta_pa | pcoa2_pa | B03_coords | BASE | coords |  |  | 1082 | 3 | 0.504 | 0.004 | 0.733 | 16.4 | 0.3 |
| dark | dark_n | MLP06_curve_kndvi | MLP | curve | kndvi | tabular | 1082 | 5 | 0.414 | 0.016 | 0.665 | 18.3 | -0.7 |
| phylo | mntd | C1D01_curve1d_kndvi_ctxclim-topo-area | C1D | clim+topo+area | kndvi | curve1d | 969 | 5 | 0.211 | 0.011 | 0.438 | 18.6 | 0.1 |
| phylo | mpd | RF04_lsp-curve-topo-area_ndvi | RF | lsp+curve+topo+area | ndvi |  | 969 | 3 | 0.159 | 0.008 | 0.441 | 16.1 | 0.4 |
| phylo | ses_mntd | RF01_lsp-topo-area_nbr | RF | lsp+topo+area | nbr |  | 969 | 3 | 0.173 | 0.001 | 0.350 | 17.6 | 0.5 |
| phylo | ses_mpd | MLP06_curve_kndvi | MLP | curve | kndvi | tabular | 969 | 5 | 0.210 | 0.013 | 0.372 | 16.2 | -0.1 |
| phylo | ses_pd | RF04_lsp-curve-topo-area_kndvi | RF | lsp+curve+topo+area | kndvi |  | 969 | 3 | 0.200 | 0.003 | 0.338 | 16.1 | -0.1 |

## Por faceta — las 10 mejores corridas de cada una

### `alpha`

| run_id | family | index | substrate | n_targets | R2 | R2_min | R2_max | %RMSE | |sesgo| | spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP06_curve_kndvi | MLP | kndvi | tabular | 3 | 0.494 | 0.408 | 0.613 | 14.2 | 0.3 | 0.672 |
| B03_coords | BASE |  |  | 3 | 0.490 | 0.373 | 0.663 | 14.2 | 0.5 | 0.662 |
| C1D01_curve1d_kndvi_ctxclim-topo-area | C1D | kndvi | curve1d | 3 | 0.465 | 0.368 | 0.594 | 14.6 | 0.7 | 0.646 |
| C2D02_serpentine_kndvi_ctxclim-topo-area | C2D | kndvi | serpentine | 3 | 0.464 | 0.370 | 0.590 | 14.7 | 0.5 | 0.651 |
| MLP07_curve_all | MLP |  | tabular | 3 | 0.462 | 0.382 | 0.557 | 14.7 | 0.7 | 0.666 |
| C2D01_reshape_kndvi_ctxclim-topo-area | C2D | kndvi | reshape | 3 | 0.455 | 0.363 | 0.580 | 14.8 | 0.5 | 0.645 |
| RF03_curve-topo-area_kndvi | RF | kndvi |  | 3 | 0.447 | 0.332 | 0.594 | 14.9 | 1.4 | 0.650 |
| MLP05c_curve_kndvi | MLP | kndvi | tabular | 3 | 0.442 | 0.347 | 0.574 | 15.0 | 0.2 | 0.640 |
| RF03_curve-topo-area_evi | RF | evi |  | 3 | 0.441 | 0.306 | 0.629 | 14.9 | 1.6 | 0.635 |
| RF04_lsp-curve-topo-area_kndvi | RF | kndvi |  | 3 | 0.441 | 0.322 | 0.595 | 14.9 | 1.5 | 0.647 |

### `beta_pa`

| run_id | family | index | substrate | n_targets | R2 | R2_min | R2_max | %RMSE | |sesgo| | spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| RF06_curve_all-topo-area | RF |  |  | 3 | 0.516 | 0.490 | 0.563 | 16.9 | 0.2 | 0.719 |
| RF01_lsp-topo-area_kndvi | RF | kndvi |  | 3 | 0.510 | 0.481 | 0.564 | 17.0 | 0.1 | 0.716 |
| RF02_lsp-qc-topo-area_kndvi | RF | kndvi |  | 3 | 0.509 | 0.479 | 0.562 | 17.0 | 0.1 | 0.717 |
| RF04_lsp-curve-topo-area_kndvi | RF | kndvi |  | 3 | 0.508 | 0.484 | 0.553 | 17.0 | 0.3 | 0.715 |
| B03_coords | BASE |  |  | 3 | 0.508 | 0.504 | 0.513 | 17.0 | 0.2 | 0.719 |
| RF03_curve-topo-area_evi | RF | evi |  | 3 | 0.508 | 0.462 | 0.574 | 17.0 | 0.1 | 0.715 |
| RF03_curve-topo-area_kndvi | RF | kndvi |  | 3 | 0.507 | 0.478 | 0.558 | 17.0 | 0.3 | 0.714 |
| RF04_lsp-curve-topo-area_evi | RF | evi |  | 3 | 0.507 | 0.457 | 0.571 | 17.0 | 0.1 | 0.715 |
| RF04_lsp-curve-topo-area_savi | RF | savi |  | 3 | 0.504 | 0.453 | 0.574 | 17.1 | 0.1 | 0.711 |
| RF03_curve-topo-area_savi | RF | savi |  | 3 | 0.503 | 0.455 | 0.573 | 17.1 | 0.1 | 0.710 |

### `beta_cover`

| run_id | family | index | substrate | n_targets | R2 | R2_min | R2_max | %RMSE | |sesgo| | spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP06_curve_kndvi | MLP | kndvi | tabular | 3 | 0.345 | 0.279 | 0.386 | 21.2 | 1.3 | 0.608 |
| RF06_curve_all-topo-area | RF |  |  | 3 | 0.344 | 0.321 | 0.372 | 21.2 | 1.8 | 0.601 |
| B03_coords | BASE |  |  | 3 | 0.344 | 0.247 | 0.411 | 21.2 | 1.6 | 0.591 |
| RF05_lsp_all-topo-area | RF |  |  | 3 | 0.335 | 0.313 | 0.370 | 21.4 | 1.7 | 0.599 |
| RF04_lsp-curve-topo-area_evi | RF | evi |  | 3 | 0.333 | 0.300 | 0.389 | 21.4 | 1.9 | 0.587 |
| MLP07_curve_all | MLP |  | tabular | 3 | 0.331 | 0.261 | 0.376 | 21.4 | 1.3 | 0.606 |
| RF04_lsp-curve-topo-area_nbr | RF | nbr |  | 3 | 0.329 | 0.284 | 0.393 | 21.4 | 1.5 | 0.594 |
| RF04_lsp-curve-topo-area_savi | RF | savi |  | 3 | 0.328 | 0.294 | 0.388 | 21.5 | 1.9 | 0.579 |
| RF02_lsp-qc-topo-area_kndvi | RF | kndvi |  | 3 | 0.328 | 0.293 | 0.361 | 21.5 | 1.8 | 0.584 |
| RF03_curve-topo-area_kndvi | RF | kndvi |  | 3 | 0.327 | 0.288 | 0.367 | 21.5 | 1.8 | 0.578 |

### `phylo`

| run_id | family | index | substrate | n_targets | R2 | R2_min | R2_max | %RMSE | |sesgo| | spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| RF02_lsp-qc-topo-area_kndvi | RF | kndvi |  | 5 | 0.180 | 0.152 | 0.202 | 17.0 | 0.2 | 0.370 |
| RF04_lsp-curve-topo-area_kndvi | RF | kndvi |  | 5 | 0.178 | 0.151 | 0.209 | 17.1 | 0.2 | 0.371 |
| MLP06_curve_kndvi | MLP | kndvi | tabular | 5 | 0.178 | 0.143 | 0.210 | 17.0 | 0.2 | 0.391 |
| RF04_lsp-curve-topo-area_ndvi | RF | ndvi |  | 5 | 0.178 | 0.159 | 0.199 | 17.1 | 0.3 | 0.371 |
| C1D01_curve1d_kndvi_ctxclim-topo-area | C1D | kndvi | curve1d | 5 | 0.177 | 0.137 | 0.211 | 17.1 | 0.2 | 0.393 |
| MLP01_lsp_nbr | MLP | nbr | tabular | 5 | 0.176 | 0.132 | 0.205 | 17.1 | 0.2 | 0.381 |
| C2D01_reshape_nbr | C2D | nbr | reshape | 5 | 0.175 | 0.142 | 0.197 | 17.1 | 0.3 | 0.361 |
| RF01_lsp-topo-area_kndvi | RF | kndvi |  | 5 | 0.174 | 0.144 | 0.196 | 17.1 | 0.2 | 0.367 |
| RF03_curve-topo-area_kndvi | RF | kndvi |  | 5 | 0.174 | 0.129 | 0.205 | 17.1 | 0.2 | 0.372 |
| RF03_curve-topo-area_ndvi | RF | ndvi |  | 5 | 0.173 | 0.131 | 0.204 | 17.1 | 0.3 | 0.364 |

### `dark`

| run_id | family | index | substrate | n_targets | R2 | R2_min | R2_max | %RMSE | |sesgo| | spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP06_curve_kndvi | MLP | kndvi | tabular | 1 | 0.414 | 0.414 | 0.414 | 18.3 | 0.7 | 0.665 |
| MLP07_curve_all | MLP |  | tabular | 1 | 0.394 | 0.394 | 0.394 | 18.6 | 0.7 | 0.654 |
| RF06_curve_all-topo-area | RF |  |  | 1 | 0.394 | 0.394 | 0.394 | 18.7 | 0.6 | 0.641 |
| RF02_lsp-qc-topo-area_ndvi | RF | ndvi |  | 1 | 0.391 | 0.391 | 0.391 | 18.7 | 0.7 | 0.643 |
| RF01_lsp-topo-area_ndvi | RF | ndvi |  | 1 | 0.391 | 0.391 | 0.391 | 18.7 | 0.6 | 0.642 |
| RF05_lsp_all-topo-area | RF |  |  | 1 | 0.390 | 0.390 | 0.390 | 18.7 | 0.7 | 0.641 |
| RF01_lsp-topo-area_kndvi | RF | kndvi |  | 1 | 0.388 | 0.388 | 0.388 | 18.7 | 0.6 | 0.640 |
| RF02_lsp-qc-topo-area_kndvi | RF | kndvi |  | 1 | 0.388 | 0.388 | 0.388 | 18.7 | 0.7 | 0.641 |
| RF04_lsp-curve-topo-area_ndvi | RF | ndvi |  | 1 | 0.385 | 0.385 | 0.385 | 18.8 | 0.7 | 0.640 |
| MLP05c_curve_kndvi | MLP | kndvi | tabular | 1 | 0.383 | 0.383 | 0.383 | 18.8 | 1.1 | 0.640 |

