#!/usr/bin/env Rscript
# Curva de particion de Hill (alfa/beta/gamma, q=0,1,2) por remuestreo creciente. Ver
# scripts/lib/beta_freq.R para el porque del diseno (row-relativizacion antes de pooler).
#
# Substrato (decision del autor, 2026-09-23, docs/24 paso 6): las parcelas que llevan
# lcbd_count_sorensen (scripts/62), con su conteo real de individuos de
# occurrences_unified_counts.parquet. Antes salia del pool de frecuencia maximizado
# (Parcelas-CL completo con cepas continuas + Living Trees completo), que describe un pool
# que el modelo nunca ve; ahora la curva describe el recambio que LCBD reparte.
#
# `--woody`: occurrences_unified_counts_woody.parquet y lcbd_count_sorensen_woody.parquet
# (scripts/83), salida unified_beta_freq_curve_woody.csv.
#
# Uso:
#   Rscript scripts/59_unified_beta_freq_curve.R
#   Rscript scripts/59_unified_beta_freq_curve.R --woody --reps 200

suppressWarnings(suppressMessages({
  library(arrow); library(hillR)
}))
source("scripts/lib/beta_freq.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
SFX        <- if ("--woody" %in% args) "_woody" else ""
OCC        <- sprintf("data/derived/occurrences_unified_counts%s.parquet", SFX)
LCBD       <- sprintf("data/derived/lcbd_count_sorensen%s.parquet", SFX)
OUT        <- sprintf("data/derived/unified_beta_freq_curve%s.csv", SFX)
Q          <- c(0, 1, 2)
REPS       <- getarg("--reps", 100)
N_SIZES    <- getarg("--n-sizes", 12)
SEED       <- 42

message("== pool de conteo (parcelas con LCBD) ==")
occ <- as.data.frame(read_parquet(OCC))
ids <- as.data.frame(read_parquet(LCBD))
ids <- ids$PlotObservationID[!is.na(ids$lcbd_count_sorensen)]
occ <- occ[occ$PlotObservationID %in% ids, ]
comm <- as.matrix(as.data.frame.matrix(xtabs(Value ~ PlotObservationID + species, data = occ)))
comm <- comm[rowSums(comm) > 0, colSums(comm) > 0, drop = FALSE]
stopifnot(nrow(comm) == length(ids))
message(sprintf("  %d parcelas x %d especies (%s)", nrow(comm), ncol(comm), basename(LCBD)))

# Tres curvas: por base y sobre el agrupado (columna `source`, docs/24).
SOURCES <- c(parcelas_cl = "^PCL_", living_trees = "^LT_", pooled = ".")
t0 <- Sys.time()
curve <- do.call(rbind, lapply(names(SOURCES), function(k) {
  sub <- comm[grepl(SOURCES[[k]], rownames(comm)), , drop = FALSE]
  sub <- sub[, colSums(sub) > 0, drop = FALSE]
  N <- nrow(sub)
  sizes <- unique(round(exp(seq(log(10), log(N), length.out = N_SIZES))))
  message(sprintf("\n== [%s] %d parcelas x %d especies, reps=%d; tamanos: %s ==",
                  k, N, ncol(sub), REPS, paste(sizes, collapse = ", ")))
  cbind(source = k, beta_freq_curve(sub, sizes, q = Q, reps = REPS, seed = SEED),
        stringsAsFactors = FALSE)
}))
message(sprintf("  %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

write.csv(curve, OUT, row.names = FALSE)
message(sprintf("-> %s (%d filas)", OUT, nrow(curve)))

message("\n== resumen (n chico -> n grande, por q) ==")
for (k in names(SOURCES)) for (qi in Q) {
  d <- curve[curve$source == k & curve$q == qi, ]
  d <- d[order(d$n), ]
  lo <- d[1, ]; hi <- d[nrow(d), ]
  message(sprintf("  %-12s q=%d  n=%4d: alfa=%.2f beta=%.2f gamma=%.1f  ->  n=%4d: alfa=%.2f beta=%.2f gamma=%.1f",
                  k, qi, lo$n, lo$alpha_mean, lo$beta_mean, lo$gamma_mean,
                  hi$n, hi$alpha_mean, hi$beta_mean, hi$gamma_mean))
}

message("\nlisto.")
