#!/usr/bin/env Rscript
# Facetas de diversidad sobre la base unificada (Parcelas-CL + Living Trees Chile,
# scripts/51_build_unified_dataset.R): hill_q0_unified (riqueza), beta_pa_unified
# (binario, toda parcela con >=1 especie registrada), beta_freq_unified (relativizado
# por fila,
# generaliza el "cover" de scripts/07_compute_taxonomic_beta_responses.R a cualquier
# moneda de abundancia real -- cobertura, conteo, area basal).
#
# El estrato "presencia" (Abundance_parameter=="NA", Value fijo en 1 para toda especie
# de la parcela) se EXCLUYE de beta_freq: relativizar un vector de puros 1 da 1/riqueza
# por especie -- un artefacto de evenness plano, no una frecuencia medida. Es el mismo
# problema ya documentado para hill_q1/q2 sobre parcelas de solo presencia.
#
# Misma receta que script 07: hillR, adespatial::beta.div, vegan::vegdist +
# ape::pcoa(correction="cailliez"). Nombres con sufijo _unified -- no tocan
# data/derived/biodiversity_responses.parquet (target de modelado actual, intacto).
#
# Uso:
#   Rscript scripts/52_unified_diversity_facets.R

suppressPackageStartupMessages({
  library(arrow)
})

ROOT <- getwd()
DERIVED <- file.path(ROOT, "data", "derived")
K_AXES <- 2

occ <- as.data.frame(arrow::read_parquet(file.path(DERIVED, "occurrences_unified.parquet")))
plots <- as.data.frame(arrow::read_parquet(file.path(DERIVED, "plots_unified.parquet")))
plot_ids <- sort(unique(plots$PlotObservationID))

cat(sprintf("parcelas: %d, filas de ocurrencia: %d, especies: %d\n",
            length(plot_ids), nrow(occ), length(unique(occ$species))))
cat("Abundance_parameter en occ:\n")
print(table(occ$Abundance_parameter, useNA = "always"))

build_comm <- function(df, ids) {
  m <- as.data.frame.matrix(xtabs(Value ~ PlotObservationID + species, data = df))
  missing_ids <- setdiff(ids, rownames(m))
  if (length(missing_ids) > 0) {
    pad <- matrix(0, nrow = length(missing_ids), ncol = ncol(m),
                   dimnames = list(missing_ids, colnames(m)))
    m <- rbind(m, pad)
  }
  out <- as.matrix(m[ids, , drop = FALSE])
  storage.mode(out) <- "double"
  out
}

comm_full <- build_comm(occ, plot_ids)

# --- 1. riqueza (hill q0, no depende de la moneda de abundancia) -------------------------
hill_q0 <- hillR::hill_taxa(comm_full, q = 0, MARGIN = 1)

log_step <- function(fmt, ...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(fmt, ...)))
  flush(stdout())
}

run_facet <- function(comm_beta, comm_nmds, method_beta, dist_method, label) {
  keep_sp <- colSums(comm_beta) > 0
  comm_beta <- comm_beta[, keep_sp, drop = FALSE]
  comm_nmds <- comm_nmds[, keep_sp, drop = FALSE]
  log_step("[%s] %d parcelas x %d especies", label, nrow(comm_beta), ncol(comm_beta))

  log_step("[%s] beta.div (method=%s, nperm=999) ...", label, method_beta)
  bd <- adespatial::beta.div(comm_beta, method = method_beta, nperm = 999)
  log_step("[%s] beta.div listo", label)

  log_step("[%s] PCoA (distance=%s) ...", label, dist_method)
  d <- vegan::vegdist(comm_nmds, method = dist_method, binary = (dist_method == "jaccard"))
  pc <- ape::pcoa(d, correction = "cailliez")
  log_step("[%s] PCoA listo", label)

  vecs <- if (!is.null(pc$vectors.cor)) pc$vectors.cor else pc$vectors
  k <- min(K_AXES, ncol(vecs))
  sc <- as.data.frame(vecs[, seq_len(k), drop = FALSE])
  colnames(sc) <- paste0("pcoa", seq_len(k))

  rel_eig <- if ("Rel_corr_eig" %in% colnames(pc$values)) pc$values$Rel_corr_eig
             else pc$values$Relative_eig
  var_explained <- sum(rel_eig[seq_len(k)])
  cat(sprintf("[%s] PCoA varianza explicada por %d ejes: %.1f%%\n", label, k, 100 * var_explained))
  if (var_explained < 0.2) {
    warning(sprintf("[%s] PCoA explica poca varianza (%.1f%% en %d ejes)",
                     label, 100 * var_explained, k))
  }

  data.frame(
    PlotObservationID = rownames(comm_beta),
    lcbd = as.numeric(bd$LCBD),
    p_lcbd = as.numeric(bd$p.LCBD),
    sc,
    row.names = NULL
  )
}

# --- 2. beta presencia/ausencia, toda parcela con al menos 1 especie registrada ----------
# 8 parcelas de Living Trees quedan con riqueza 0 (todos sus arboles sin D, ver script 50)
# -- fila de puros ceros. Jaccard binario no esta definido entre dos comunidades vacias y
# rompe la descomposicion espectral del PCoA (NA/Inf en eigen()). Se excluyen aqui, NO de
# hill_q0 (0 es una riqueza real y valida ahi, no un hueco que esconder).
has_data <- rowSums(comm_full) > 0
pa_ids <- plot_ids[has_data]
cat(sprintf("parcelas excluidas de beta_pa por no tener ninguna especie registrada: %d de %d\n",
            sum(!has_data), length(plot_ids)))
comm_pa <- (comm_full[pa_ids, , drop = FALSE] > 0) * 1
storage.mode(comm_pa) <- "double"
pa_res <- run_facet(comm_pa, comm_pa, "jaccard", "jaccard", "presence-absence")
names(pa_res)[-1] <- paste0(names(pa_res)[-1], "_pa_unified")

# --- 3. beta frecuencia, toda parcela con abundancia real (fuera del estrato presencia) --
ap_by_plot <- occ[!duplicated(occ$PlotObservationID),
                  c("PlotObservationID", "Abundance_parameter")]
freq_ids <- ap_by_plot$PlotObservationID[
  !is.na(ap_by_plot$Abundance_parameter) & ap_by_plot$Abundance_parameter != "NA"
]
freq_ids <- intersect(plot_ids, freq_ids)
cat(sprintf("parcelas con abundancia real (fuera del estrato 'presencia'): %d de %d\n",
            length(freq_ids), length(plot_ids)))

comm_freq <- comm_full[freq_ids, , drop = FALSE]
comm_freq_rel <- vegan::decostand(comm_freq, method = "total")
freq_res <- run_facet(comm_freq, comm_freq_rel, "hellinger", "bray", "frequency")
names(freq_res)[-1] <- paste0(names(freq_res)[-1], "_freq_unified")

# --- 4. unir y escribir -------------------------------------------------------------------
resp <- data.frame(
  PlotObservationID = plot_ids,
  hill_q0_unified = as.numeric(hill_q0[plot_ids]),
  row.names = NULL
)
resp <- merge(resp, pa_res, by = "PlotObservationID", all.x = TRUE)
resp <- merge(resp, freq_res, by = "PlotObservationID", all.x = TRUE)
resp <- resp[match(plot_ids, resp$PlotObservationID), ]
rownames(resp) <- NULL

richness_ref <- plots$richness[match(resp$PlotObservationID, plots$PlotObservationID)]
cat(sprintf("cor(hill_q0_unified, riqueza reportada) = %.4f (chequeo de sanidad, debiera ser ~1)\n",
            cor(resp$hill_q0_unified, richness_ref, use = "complete.obs")))

n_na_freq <- sum(is.na(resp$lcbd_freq_unified))
cat(sprintf("filas: %d, NA en *_freq_unified: %d (esperado = fuera de abundancia real = %d)\n",
            nrow(resp), n_na_freq, length(plot_ids) - length(freq_ids)))

out_path <- file.path(DERIVED, "unified_diversity_responses.parquet")
arrow::write_parquet(resp, out_path)
cat(sprintf("\nescrito: %s\n", out_path))
