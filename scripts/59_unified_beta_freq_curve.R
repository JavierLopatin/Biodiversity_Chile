#!/usr/bin/env Rscript
# Curva de particion de Hill (alfa/beta/gamma, q=0,1,2) por remuestreo creciente, sobre el
# pool de frecuencia maximizado (Parcelas-CL completo, cepas continuas + Living Trees
# completo). Ver scripts/lib/beta_freq.R para el porque del diseno (row-relativizacion
# resuelve el problema de unidades inconmensurables) y scripts/58_recover_zamorano_cover.py
# para el hallazgo que la motiva (cobertura real de 84 parcelas recuperada desde
# data/Parcelas_CL_RAW/).
#
# Uso:
#   Rscript scripts/59_unified_beta_freq_curve.R
#   Rscript scripts/59_unified_beta_freq_curve.R --reps 200

suppressWarnings(suppressMessages({
  library(arrow); library(hillR)
}))
source("scripts/lib/parcelas_comm.R")
source("scripts/lib/beta_freq.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP        <- "data/20602096.zip"
LT_LONG    <- "data/derived/living_trees_long.parquet"
ZAM_FIX    <- "data/derived/zamorano_cover_corrected.parquet"
OUT        <- "data/derived/unified_beta_freq_curve.csv"
Q          <- c(0, 1, 2)
REPS       <- getarg("--reps", 100)
N_SIZES    <- getarg("--n-sizes", 12)
SEED       <- 42

message("== pool de frecuencia ==")
pc <- parcelas_species(ZIP, quiet = TRUE)
pc$raw <- apply_zamorano_correction(pc$raw, ZAM_FIX)
comm <- build_freq_pool(pc$raw, LT_LONG)

N <- nrow(comm)
sizes <- unique(round(exp(seq(log(10), log(N), length.out = N_SIZES))))
message(sprintf("  tamanos: %s", paste(sizes, collapse = ", ")))

message(sprintf("\n== curva (remuestreo, reps=%d) ==", REPS))
t0 <- Sys.time()
curve <- beta_freq_curve(comm, sizes, q = Q, reps = REPS, seed = SEED)
message(sprintf("  %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

write.csv(curve, OUT, row.names = FALSE)
message(sprintf("-> %s (%d filas)", OUT, nrow(curve)))

message("\n== resumen (n chico -> n grande, por q) ==")
for (qi in Q) {
  d <- curve[curve$q == qi, ]
  d <- d[order(d$n), ]
  lo <- d[1, ]; hi <- d[nrow(d), ]
  message(sprintf("  q=%d  n=%4d: alfa=%.2f beta=%.2f gamma=%.1f  ->  n=%4d: alfa=%.2f beta=%.2f gamma=%.1f",
                  qi, lo$n, lo$alpha_mean, lo$beta_mean, lo$gamma_mean,
                  hi$n, hi$alpha_mean, hi$beta_mean, hi$gamma_mean))
}

message("\nlisto.")
