#!/usr/bin/env Rscript
# PCA sobre 18 metricas LSP (5 indices) + 4 metricas de estabilidad interanual (Lopatin
# 2023, scripts/47_interannual_stability.py), con las 15 facetas de diversidad proyectadas
# como variables SUPLEMENTARIAS (no entran al ajuste del PCA).
#
# Convencion del proyecto: R + ape/vegan para ordinacion (scripts/07_compute_taxonomic_
# beta_responses.R ya usa ape::pcoa para las facetas de composicion). factoextra/FactoMineR
# NO estan instalados en este entorno -- confirmado -- asi que la proyeccion suplementaria
# no usa quanti.sup ni fviz_pca_biplot.
#
# Como se proyecta la variable suplementaria: NO es un "predict() sobre filas nuevas" --
# las facetas de diversidad son columnas EXTRA sobre las MISMAS parcelas ya usadas para
# ajustar el PCA, no observaciones nuevas. La proyeccion estandar de una variable
# suplementaria continua en PCA es su correlacion con los scores de cada eje (equivalente,
# salvo escala, a lo que FactoMineR hace con quanti.sup) -- eso es lo que se calcula aqui.
#
# Uso:
#   Rscript scripts/48_lsp_stability_pca.R

suppressPackageStartupMessages({
  library(arrow)
})

ROOT <- getwd()
DERIVED <- file.path(ROOT, "data", "derived")

# --- 1. LSP: largo (plot_id, index) -> ancho (18 metricas x 5 indices = 90 columnas) -----
lsp <- as.data.frame(arrow::read_parquet(file.path(DERIVED, "lsp_all_auto.parquet")))
METRICS <- c("sos", "pos", "eos", "vsos", "vpos", "veos", "los", "msp", "mau",
            "vmsp", "vmau", "ampl", "ios", "rog", "ros", "sw", "trough", "mos")
mean5x5_cols <- paste0(METRICS, "_mean5x5")

plot_ids <- sort(unique(lsp$plot_id))
lsp_wide <- data.frame(plot_id = plot_ids)
for (ix in sort(unique(lsp$index))) {
  sub <- lsp[lsp$index == ix, c("plot_id", mean5x5_cols)]
  names(sub)[-1] <- paste0(ix, "_", METRICS)
  lsp_wide <- merge(lsp_wide, sub, by = "plot_id", all.x = TRUE)
}
cat(sprintf("LSP ancho: %d parcelas x %d columnas (18 metricas x %d indices)\n",
           nrow(lsp_wide), ncol(lsp_wide) - 1, length(unique(lsp$index))))

# --- 2. Estabilidad interanual (scripts/47) ----------------------------------------------
stab <- as.data.frame(arrow::read_parquet(
  file.path(DERIVED, "interannual_stability.parquet")))
names(stab)[names(stab) == "PlotObservationID"] <- "plot_id"

pred <- merge(lsp_wide, stab, by = "plot_id", all.x = TRUE)
predictor_cols <- setdiff(names(pred), "plot_id")
cat(sprintf("predictores totales (LSP + estabilidad): %d\n", length(predictor_cols)))

# --- 3. Facetas de diversidad (variables suplementarias, no entran al ajuste) ------------
targets <- as.data.frame(arrow::read_parquet(
  file.path(DERIVED, "biodiversity_responses.parquet")))
phylo <- as.data.frame(arrow::read_parquet(file.path(DERIVED, "phylo_responses.parquet")))
dark <- as.data.frame(arrow::read_parquet(file.path(DERIVED, "dark_diversity.parquet")))

FACETS <- c("hill_q0", "hill_q1", "hill_q2", "lcbd_pa", "pcoa1_pa", "pcoa2_pa",
           "lcbd_cover", "pcoa1_cover", "pcoa2_cover",
           "mpd", "mntd", "ses_pd", "ses_mpd", "ses_mntd", "dark_n")
supp <- Reduce(function(a, b) merge(a, b, by = "PlotObservationID", all = TRUE),
              list(targets[, c("PlotObservationID", intersect(names(targets), FACETS))],
                  phylo[, c("PlotObservationID", intersect(names(phylo), FACETS))],
                  dark[, c("PlotObservationID", intersect(names(dark), FACETS))]))
names(supp)[names(supp) == "PlotObservationID"] <- "plot_id"

# --- 4. PCA sobre casos completos de LSP+estabilidad -------------------------------------
ok <- complete.cases(pred[, predictor_cols])
cat(sprintf("\ncasos completos para el PCA: %d de %d parcelas (%d excluidas por NA)\n",
           sum(ok), nrow(pred), sum(!ok)))

X <- pred[ok, predictor_cols]
pca <- prcomp(X, scale. = TRUE, center = TRUE)

var_exp <- pca$sdev^2 / sum(pca$sdev^2)
cum2 <- sum(var_exp[1:2])
cat(sprintf("\nvarianza explicada: PC1=%.1f%% PC2=%.1f%% PC3=%.1f%%  (PC1+PC2=%.1f%%)\n",
           100 * var_exp[1], 100 * var_exp[2], 100 * var_exp[3], 100 * cum2))
if (cum2 < 0.2) {
  warning(sprintf(
    "PC1+PC2 explican solo %.1f%% -- por debajo del umbral de 20%% que scripts/07 usa "
    , "para PCoA. La ordinacion en 2 ejes puede no ser representativa.", 100 * cum2))
}

# --- 5. Proyeccion de cada faceta como variable suplementaria (correlacion con los ejes) -
plot_ids_pca <- pred$plot_id[ok]
scores <- as.data.frame(pca$x[, 1:min(5, ncol(pca$x))])
scores$plot_id <- plot_ids_pca

supp_rows <- list()
for (f in FACETS) {
  if (!f %in% names(supp)) next
  d <- merge(scores, supp[, c("plot_id", f)], by = "plot_id")
  d <- d[!is.na(d[[f]]), ]
  n <- nrow(d)
  if (n < 10) {
    cat(sprintf("  %-14s PENDIENTE (n=%d, muy pocos casos)\n", f, n))
    next
  }
  cors <- sapply(names(scores)[names(scores) != "plot_id"],
                 function(pc) cor(d[[f]], d[[pc]], method = "pearson"))
  supp_rows[[f]] <- c(facet = f, n = n, cors)
  cat(sprintf("  %-14s n=%4d  cor(PC1)=%+.3f  cor(PC2)=%+.3f\n",
             f, n, cors["PC1"], cors["PC2"]))
}
supp_out <- as.data.frame(do.call(rbind, supp_rows), stringsAsFactors = FALSE)
for (col in setdiff(names(supp_out), "facet")) supp_out[[col]] <- as.numeric(supp_out[[col]])

# --- 6. Escribir para que el biplot se arme en Python (scripts/49) ----------------------
out_dir <- file.path(ROOT, "results", "tables")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

loadings <- as.data.frame(pca$rotation[, 1:min(5, ncol(pca$rotation))])
loadings$variable <- rownames(loadings)

arrow::write_parquet(scores, file.path(out_dir, "pca_lsp_stability_scores.parquet"))
arrow::write_parquet(loadings, file.path(out_dir, "pca_lsp_stability_loadings.parquet"))
arrow::write_parquet(supp_out, file.path(out_dir, "pca_lsp_stability_supp_facets.parquet"))
writeLines(sprintf("PC%d,%.6f", seq_along(var_exp), var_exp),
          file.path(out_dir, "pca_lsp_stability_variance.csv"))

cat(sprintf("\nescrito: %s/pca_lsp_stability_{scores,loadings,supp_facets}.parquet\n",
           out_dir))
