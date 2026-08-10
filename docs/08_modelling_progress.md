# Registro de ejecución del modelamiento

Generado por `scripts/14_run_matrix.py`. Una sección por etapa, en el orden en que se corrieron. El estado `skipped-deadline` significa que el trabajo no alcanzó a lanzarse dentro del presupuesto de tiempo y queda pendiente para la próxima invocación (el script es reanudable).


## Etapa `rf+c1d` — 2026-08-08 07:09

29/29 trabajos completados, 2.3 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `rf/RF04@kfold5_owner` | ok | 13.9 |
| `rf/RF03@kfold5_owner` | ok | 13.0 |
| `rf/RF08@kfold5_owner` | ok | 12.7 |
| `rf/RF02@kfold5_owner` | ok | 11.9 |
| `rf/RF03@kfold5_block20` | ok | 11.8 |
| `rf/RF01@kfold5_owner` | ok | 11.4 |
| `rf/RF01@kfold5_block20` | ok | 11.2 |
| `rf/RF01@kfold5_random` | ok | 9.9 |
| `rf/RF03@kfold5_random` | ok | 9.7 |
| `rf/RF06@kfold5_owner` | ok | 3.9 |
| `rf/RF05@kfold5_owner` | ok | 3.5 |
| `rf/B01@kfold5_block20` | ok | 2.4 |
| `rf/B01@kfold5_random` | ok | 2.2 |
| `rf/B03@kfold5_random` | ok | 2.1 |
| `rf/B03@kfold5_block20` | ok | 2.0 |
| `rf/B02@kfold5_block20` | ok | 1.9 |
| `rf/B02@kfold5_random` | ok | 1.8 |
| `rf/B03@kfold5_owner` | ok | 1.8 |
| `rf/B01@kfold5_owner` | ok | 1.7 |
| `rf/B02@kfold5_owner` | ok | 1.5 |
| `c1d/nbr` | ok | 1.0 |
| `c1d/evi` | ok | 1.0 |
| `c1d/ndvi` | ok | 0.9 |
| `c1d/kndvi` | ok | 0.8 |
| `c1d/curve5` | ok | 0.8 |
| `c1d/savi` | ok | 0.7 |
| `rf/B00@kfold5_block20` | ok | 0.1 |
| `rf/B00@kfold5_random` | ok | 0.1 |
| `rf/B00@kfold5_owner` | ok | 0.1 |


## Etapa `4a` — 2026-08-08 07:20

11/11 trabajos completados, 0.3 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `4a/cwt` | ok | 5.5 |
| `4a/mtf` | ok | 3.0 |
| `4a/gaf` | ok | 2.7 |
| `4a/ndi` | ok | 2.2 |
| `4a/pxcube` | ok | 1.5 |
| `4a/cos2d` | ok | 1.5 |
| `4a/spectrogram` | ok | 1.3 |
| `4a/reshape` | ok | 0.8 |
| `4a/hilbert` | ok | 0.8 |
| `4a/stack5` | ok | 0.7 |
| `4a/serpentine` | ok | 0.7 |


## Etapa `mlp` — 2026-08-08 07:22

14/14 trabajos completados, 0.1 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `mlp/MLP05a/kndvi` | ok | 0.3 |
| `mlp/MLP04` | ok | 0.3 |
| `mlp/MLP05c/kndvi` | ok | 0.3 |
| `mlp/MLP02/nbr` | ok | 0.3 |
| `mlp/MLP02/evi` | ok | 0.3 |
| `mlp/MLP02/ndvi` | ok | 0.3 |
| `mlp/MLP01/ndvi` | ok | 0.3 |
| `mlp/MLP01/kndvi` | ok | 0.3 |
| `mlp/MLP03/kndvi` | ok | 0.2 |
| `mlp/MLP02/savi` | ok | 0.2 |
| `mlp/MLP01/evi` | ok | 0.2 |
| `mlp/MLP02/kndvi` | ok | 0.2 |
| `mlp/MLP01/savi` | ok | 0.2 |
| `mlp/MLP01/nbr` | ok | 0.2 |


## Etapa `4b` — 2026-08-08 07:26

10/10 trabajos completados, 0.2 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `4b/pxcube/nbr` | ok | 1.8 |
| `4b/pxcube/savi` | ok | 1.7 |
| `4b/pxcube/evi` | ok | 1.6 |
| `4b/pxcube/ndvi` | ok | 1.5 |
| `4b/serpentine/nbr` | ok | 0.8 |
| `4b/serpentine/ndvi` | ok | 0.8 |
| `4b/serpentine/evi` | ok | 0.8 |
| `4b/serpentine/savi` | ok | 0.7 |
| `4b/serpentine/kndvi` | ok | 0.0 |
| `4b/pxcube/kndvi` | ok | 0.0 |


## Etapa `4c` — 2026-08-08 07:43

11/11 trabajos completados, 0.4 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `4c/width-X` | ok | 10.8 |
| `4c/width-C` | ok | 2.9 |
| `4c/fusion-patch` | ok | 2.0 |
| `4c/fusion-none` | ok | 1.8 |
| `4c/fusion-film` | ok | 1.7 |
| `4c/rotation-calendar` | ok | 1.7 |
| `4c/normalize-perSample` | ok | 1.6 |
| `4c/normalize-global` | ok | 1.1 |
| `4c/width-A` | ok | 1.0 |
| `4c/mixup` | ok | 0.0 |
| `4c/no-augment` | ok | 0.0 |


## Etapa `final` — 2026-08-08 08:10

25/25 trabajos completados, 1.6 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `final/C2D11_pxcube_kndvi_none@kfold5_random` | ok | 7.2 |
| `final/C2D11_pxcube_kndvi_none@lodo_owner` | ok | 7.2 |
| `final/RF06_curve_all-topo-area@lodo_owner` | ok | 6.9 |
| `final/RF06_curve_all-topo-area@kfold5_dataset` | ok | 6.9 |
| `final/RF06_curve_all-topo-area@kfold5_random` | ok | 6.8 |
| `final/RF06_curve_all-topo-area@kfold5_location` | ok | 6.8 |
| `final/C2D11_pxcube_kndvi_none@kfold5_block20` | ok | 6.4 |
| `final/RF06_curve_all-topo-area@kfold5_block20` | ok | 6.0 |
| `final/C2D11_pxcube_kndvi_none@kfold5_location` | ok | 5.5 |
| `final/C2D11_pxcube_kndvi_none@kfold5_dataset` | ok | 4.4 |
| `final/C2D10_stack5@kfold5_random` | ok | 3.7 |
| `final/C1D01_curve1d_kndvi@lodo_owner` | ok | 3.2 |
| `final/C2D10_stack5@lodo_owner` | ok | 2.9 |
| `final/C2D10_stack5@kfold5_location` | ok | 2.9 |
| `final/C2D10_stack5@kfold5_block20` | ok | 2.8 |
| `final/C1D01_curve1d_kndvi@kfold5_random` | ok | 2.6 |
| `final/C1D01_curve1d_kndvi@kfold5_block20` | ok | 2.3 |
| `final/C1D01_curve1d_kndvi@kfold5_location` | ok | 2.2 |
| `final/C2D10_stack5@kfold5_dataset` | ok | 1.5 |
| `final/C1D01_curve1d_kndvi@kfold5_dataset` | ok | 1.4 |
| `final/MLP02_curve_kndvi@lodo_owner` | ok | 1.0 |
| `final/MLP02_curve_kndvi@kfold5_block20` | ok | 0.9 |
| `final/MLP02_curve_kndvi@kfold5_random` | ok | 0.9 |
| `final/MLP02_curve_kndvi@kfold5_location` | ok | 0.7 |
| `final/MLP02_curve_kndvi@kfold5_dataset` | ok | 0.5 |

---

## Salida del reporte final

`scripts/12_model_report.py`:

```
79 runs, 1188 pooled rows
-> results/tables/model_comparison.csv

top 15 by mean R2 over the beta targets — lcbd_pa, pcoa1_pa, pcoa2_pa (kfold5_owner).
Alpha is shown beside it because the two behave differently and a single average reports neither:
                        run_id family substrate index  R2_beta  R2_alpha  R2_main
      RF06_curve_all-topo-area     RF       NaN   NaN   +0.269    -0.604   -0.168
    RF03_curve-topo-area_kndvi     RF       NaN kndvi   +0.254    -0.553   -0.150
RF04_lsp-curve-topo-area_kndvi     RF       NaN kndvi   +0.251    -0.531   -0.140
     RF08_lsp-curve-topo_kndvi     RF       NaN kndvi   +0.250    -0.447   -0.098
     RF03_curve-topo-area_ndvi     RF       NaN  ndvi   +0.241    -0.557   -0.158
      RF03_curve-topo-area_evi     RF       NaN   evi   +0.239    -0.658   -0.210
      RF01_lsp-topo-area_kndvi     RF       NaN kndvi   +0.238    -0.428   -0.095
   RF02_lsp-qc-topo-area_kndvi     RF       NaN kndvi   +0.238    -0.413   -0.088
 RF04_lsp-curve-topo-area_ndvi     RF       NaN  ndvi   +0.232    -0.522   -0.145
      RF08_lsp-curve-topo_ndvi     RF       NaN  ndvi   +0.229    -0.417   -0.094
      RF03_curve-topo-area_nbr     RF       NaN   nbr   +0.228    -0.642   -0.207
     RF03_curve-topo-area_savi     RF       NaN  savi   +0.227    -0.611   -0.192
  RF04_lsp-curve-topo-area_evi     RF       NaN   evi   +0.227    -0.606   -0.190
       RF08_lsp-curve-topo_nbr     RF       NaN   nbr   +0.225    -0.445   -0.110
  RF04_lsp-curve-topo-area_nbr     RF       NaN   nbr   +0.225    -0.585   -0.180

-> results/tables/optimism_gap.csv

-> results/tables/paired_tests.csv  (1747 pairs in 4 comparison families, 0 significant)

-> results/tables/headline_plot_tests.csv  (27/45 significant, per-plot paired squared error, n=1082)
  target             model_a                  model_b                   better  delta_mse  p_ttest_bonf
 lcbd_pa C1D01_curve1d_kndvi  C2D11_pxcube_kndvi_none  C2D11_pxcube_kndvi_none  3.831e-10       0.02203
 lcbd_pa          B03_coords  C2D11_pxcube_kndvi_none  C2D11_pxcube_kndvi_none  1.031e-09      0.002065
 lcbd_pa          B03_coords RF06_curve_all-topo-area RF06_curve_all-topo-area  1.039e-09      0.000452
 lcbd_pa            B00_area               B03_coords               B03_coords  1.621e-09     1.162e-06
 lcbd_pa            B00_area      C1D01_curve1d_kndvi      C1D01_curve1d_kndvi  2.269e-09     2.331e-31
 lcbd_pa            B00_area        MLP02_curve_kndvi        MLP02_curve_kndvi  2.397e-09      7.16e-31
 lcbd_pa            B00_area  C2D11_pxcube_kndvi_none  C2D11_pxcube_kndvi_none  2.652e-09     4.964e-33
 lcbd_pa            B00_area RF06_curve_all-topo-area RF06_curve_all-topo-area   2.66e-09     2.773e-28
pcoa2_pa   MLP02_curve_kndvi RF06_curve_all-topo-area RF06_curve_all-topo-area   0.003554        0.0148
pcoa1_pa   MLP02_curve_kndvi RF06_curve_all-topo-area RF06_curve_all-topo-area   0.004737      0.004484
pcoa1_pa C1D01_curve1d_kndvi        MLP02_curve_kndvi        MLP02_curve_kndvi   0.005503     4.933e-12
pcoa1_pa C1D01_curve1d_kndvi  C2D11_pxcube_kndvi_none  C2D11_pxcube_kndvi_none   0.008002     1.628e-10
-> figures in results/figures
```

`scripts/13_interpretability.py --auto`:

```
-> rf_block_importance_RF02_lsp-qc-topo-area_kndvi.csv
     target  area   lsp    qc  topo
    hill_q0 0.170 0.505 0.040 0.285
    hill_q1 0.107 0.478 0.050 0.365
    hill_q2 0.102 0.457 0.053 0.389
 lcbd_cover 0.076 0.464 0.103 0.356
    lcbd_pa 0.059 0.633 0.050 0.258
pcoa1_cover 0.011 0.467 0.066 0.455
   pcoa1_pa 0.036 0.640 0.041 0.283
pcoa2_cover 0.025 0.480 0.059 0.436
   pcoa2_pa 0.046 0.591 0.062 0.301
-> doy_attribution_C2D11_pxcube_kndvi_wA.csv and fig13_doy_attribution.pdf

peak attribution week per target:
     target  step   doy  attribution
    hill_q0     2 122.0     0.026433
    hill_q1     2 122.0     0.027826
    hill_q2     2 122.0     0.028376
 lcbd_cover    49  86.0     0.027983
    lcbd_pa    49  86.0     0.024344
pcoa1_cover    49  86.0     0.025060
   pcoa1_pa    49  86.0     0.022928
pcoa2_cover     2 122.0     0.030726
   pcoa2_pa     2 122.0     0.028901
```


## Etapa `rf+c1d` — 2026-08-09 12:09

17/17 trabajos completados, 3.1 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `rf/RF04@kfold5_window` | ok | 31.7 |
| `rf/RF03@kfold5_window` | ok | 31.3 |
| `rf/RF02@kfold5_window` | ok | 29.6 |
| `rf/RF01@kfold5_window` | ok | 27.9 |
| `rf/RF08@kfold5_window` | ok | 23.6 |
| `rf/RF06@kfold5_window` | ok | 9.0 |
| `rf/RF05@kfold5_window` | ok | 6.4 |
| `rf/B01@kfold5_window` | ok | 3.2 |
| `rf/B03@kfold5_window` | ok | 3.1 |
| `rf/B02@kfold5_window` | ok | 2.7 |
| `c1d/evi` | ok | 2.6 |
| `c1d/kndvi` | ok | 2.5 |
| `c1d/ndvi` | ok | 2.5 |
| `c1d/nbr` | ok | 2.4 |
| `c1d/savi` | ok | 2.4 |
| `c1d/curve5` | ok | 2.4 |
| `rf/B00@kfold5_window` | ok | 0.1 |


## Etapa `4a` — 2026-08-09 12:34

11/11 trabajos completados, 0.8 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `4a/cwt` | ok | 13.5 |
| `4a/mtf` | ok | 8.6 |
| `4a/gaf` | ok | 6.0 |
| `4a/pxcube` | ok | 3.8 |
| `4a/spectrogram` | ok | 3.4 |
| `4a/cos2d` | ok | 3.1 |
| `4a/ndi` | ok | 2.5 |
| `4a/stack5` | ok | 2.4 |
| `4a/hilbert` | ok | 2.0 |
| `4a/serpentine` | ok | 1.9 |
| `4a/reshape` | ok | 1.8 |


## Etapa `mlp` — 2026-08-09 12:37

14/14 trabajos completados, 0.1 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `mlp/MLP04` | ok | 0.6 |
| `mlp/MLP05a/kndvi` | ok | 0.5 |
| `mlp/MLP02/evi` | ok | 0.5 |
| `mlp/MLP05c/kndvi` | ok | 0.5 |
| `mlp/MLP02/kndvi` | ok | 0.5 |
| `mlp/MLP02/ndvi` | ok | 0.5 |
| `mlp/MLP02/savi` | ok | 0.5 |
| `mlp/MLP02/nbr` | ok | 0.5 |
| `mlp/MLP01/evi` | ok | 0.5 |
| `mlp/MLP01/nbr` | ok | 0.4 |
| `mlp/MLP03/kndvi` | ok | 0.4 |
| `mlp/MLP01/kndvi` | ok | 0.4 |
| `mlp/MLP01/savi` | ok | 0.4 |
| `mlp/MLP01/ndvi` | ok | 0.4 |


## Etapa `4b` — 2026-08-09 12:48

15/15 trabajos completados, 0.4 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `4b/hilbert/ndvi` | ok | 2.0 |
| `4b/hilbert/evi` | ok | 1.9 |
| `4b/hilbert/savi` | ok | 1.9 |
| `4b/hilbert/nbr` | ok | 1.8 |
| `4b/serpentine/savi` | ok | 1.8 |
| `4b/reshape/nbr` | ok | 1.8 |
| `4b/serpentine/ndvi` | ok | 1.7 |
| `4b/reshape/ndvi` | ok | 1.7 |
| `4b/serpentine/evi` | ok | 1.7 |
| `4b/reshape/savi` | ok | 1.7 |
| `4b/serpentine/nbr` | ok | 1.7 |
| `4b/reshape/evi` | ok | 1.6 |
| `4b/reshape/kndvi` | ok | 0.0 |
| `4b/hilbert/kndvi` | ok | 0.0 |
| `4b/serpentine/kndvi` | ok | 0.0 |


## Etapa `4c` — 2026-08-09 12:56

11/11 trabajos completados, 0.3 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `4c/width-A` | ok | 2.3 |
| `4c/rotation-calendar` | ok | 2.2 |
| `4c/normalize-perSample` | ok | 2.1 |
| `4c/fusion-none` | ok | 2.1 |
| `4c/fusion-patch` | ok | 1.7 |
| `4c/width-X` | ok | 1.7 |
| `4c/width-C` | ok | 1.5 |
| `4c/fusion-film` | ok | 1.4 |
| `4c/normalize-global` | ok | 0.6 |
| `4c/mixup` | ok | 0.0 |
| `4c/no-augment` | ok | 0.0 |


## Etapa `clim` — 2026-08-09 12:57

5/5 trabajos completados, 0.0 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `clim/MLP07` | ok | 1.0 |
| `clim/MLP06` | ok | 0.8 |
| `clim/curve1d` | ok | 0.0 |
| `clim/serpentine` | ok | 0.0 |
| `clim/reshape` | ok | 0.0 |


## Etapa `final` — 2026-08-09 12:57

0/0 trabajos completados, 0.0 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|


## Etapa `clim` — 2026-08-09 20:58

5/5 trabajos completados, 0.2 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `clim/reshape` | ok | 4.1 |
| `clim/serpentine` | ok | 3.9 |
| `clim/curve1d` | ok | 3.5 |
| `clim/MLP06` | ok | 0.0 |
| `clim/MLP07` | ok | 0.0 |


## Etapa `4c` — 2026-08-09 21:02

11/11 trabajos completados, 0.4 h de cómputo acumulado.


| trabajo | estado | minutos |
|---|---|---|
| `4c/normalize-perSample` | ok | 3.1 |
| `4c/rotation-calendar` | ok | 2.9 |
| `4c/width-A` | ok | 2.5 |
| `4c/mixup` | ok | 2.5 |
| `4c/fusion-none` | ok | 2.5 |
| `4c/fusion-patch` | ok | 2.0 |
| `4c/fusion-film` | ok | 1.7 |
| `4c/width-C` | ok | 1.7 |
| `4c/width-X` | ok | 1.6 |
| `4c/no-augment` | ok | 1.0 |
| `4c/normalize-global` | ok | 0.7 |
