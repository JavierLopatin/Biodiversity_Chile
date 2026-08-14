#!/usr/bin/env Rscript
# Curva empirica de estabilidad de dark diversity por tamano de pool de co-ocurrencia.
#
# Dark diversity no tiene una teoria de rarefaccion/extrapolacion establecida (no es una
# cantidad que se "acumule" con el tamano de muestra como un numero de Hill) -- a diferencia
# de las curvas de scripts/56 y scripts/59, esta no es una prediccion validada contra una
# formula de referencia, es exploratoria: cuantifica CUANTO SE MUEVE la estimacion de
# dark_n (DarkDiv::DarkDiv, method="Hypergeometric", umbral 0,9 -- mismo metodo que
# scripts/27 y scripts/55) segun el tamano del pool de co-ocurrencia usado para ajustarlo.
#
# Por que importa: scripts/27 y scripts/55 ya usan el pool COMPLETO (no el subset de
# modelado) precisamente porque un pool chico contamina dark_n con esfuerzo de muestreo
# (rho con log-area sube de +0,085 a +0,315 con pool reducido, medido en scripts/27). Esta
# curva hace ese argumento cuantitativo explicito en vez de solo cualitativo: cuanto se
# angosta la banda, y en que tamano deja de moverse la media, a medida que el pool crece
# hacia el completo (~3.497 parcelas Parcelas-CL+Living Trees).
#
# Uso:
#   Rscript scripts/57_dark_diversity_curve.R
#   Rscript scripts/57_dark_diversity_curve.R --reps 100 --n-sizes 12

suppressWarnings(suppressMessages({
  library(ape); library(arrow); library(DarkDiv)
  library(V.PhyloMaker2)   # solo por tips.info.TPL, que usa unified_species()
}))
source("scripts/lib/unified_comm.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP     <- "data/20602096.zip"
LT_LONG <- "data/derived/living_trees_long.parquet"
PLOTS   <- "data/derived/plots_unified.parquet"
TREE    <- "data/derived/phylo_tree_unified.tre"
OUT     <- "data/derived/dark_diversity_curve.csv"
THR     <- getarg("--thr", 0.9)
REPS    <- getarg("--reps", 100)
N_SIZES <- getarg("--n-sizes", 12)
MIN_N   <- getarg("--min-n", 100)
SEED    <- 42

# --------------------------------------------------------------------------------------
# 1. pool -- mismo que scripts/55 (comunidad restringida al arbol unificado, pool completo)
# --------------------------------------------------------------------------------------

message("== pool ==")
tree <- read.tree(TREE)
us <- unified_species(ZIP, LT_LONG, quiet = TRUE)
plots_uni <- read_parquet(PLOTS)
cmm <- unified_comm(us$pc_raw, LT_LONG, tree, plots_uni$PlotObservationID, quiet = TRUE)
comm <- cmm$full
N <- nrow(comm)
message(sprintf("  %d parcelas x %d especies", N, ncol(comm)))

sizes <- unique(round(exp(seq(log(MIN_N), log(N), length.out = N_SIZES))))
message(sprintf("  tamanos: %s", paste(sizes, collapse = ", ")))

# --------------------------------------------------------------------------------------
# 2. curva -- por tamano, `reps` submuestras sin reemplazo; por replica, DarkDiv sobre
#    la submuestra (como pool autocontenido) y el dark_n medio/mediano de esa submuestra
# --------------------------------------------------------------------------------------

message(sprintf("\n== curva (remuestreo, reps=%d) ==", REPS))
set.seed(SEED)
t_all <- Sys.time()
rows <- list()
for (n in sizes) {
  reps_n <- if (n == N) 1 else REPS
  means <- medians <- numeric(reps_n)
  for (i in seq_len(reps_n)) {
    idx <- if (n == N) seq_len(N) else sample.int(N, n)
    sub <- comm[idx, , drop = FALSE]
    sub <- sub[, colSums(sub) > 0, drop = FALSE]
    dd <- DarkDiv(sub, method = "Hypergeometric")
    dark_n <- rowSums(!is.na(dd$Dark) & dd$Dark > THR)
    means[i] <- mean(dark_n)
    medians[i] <- median(dark_n)
  }
  alpha_lo <- 0.025
  rows[[length(rows) + 1]] <- data.frame(
    n = n, reps = reps_n,
    dark_n_mean = mean(means),
    dark_n_mean_lo = quantile(means, alpha_lo, names = FALSE),
    dark_n_mean_hi = quantile(means, 1 - alpha_lo, names = FALSE),
    dark_n_median = mean(medians),
    dark_n_median_lo = quantile(medians, alpha_lo, names = FALSE),
    dark_n_median_hi = quantile(medians, 1 - alpha_lo, names = FALSE))
  message(sprintf("  n=%5d (reps=%3d): dark_n medio = %.2f [%.2f, %.2f]",
                  n, reps_n, mean(means), rows[[length(rows)]]$dark_n_mean_lo,
                  rows[[length(rows)]]$dark_n_mean_hi))
}
curve <- do.call(rbind, rows)
message(sprintf("\n  total: %.1f min", as.numeric(difftime(Sys.time(), t_all, units = "mins"))))

write.csv(curve, OUT, row.names = FALSE)
message(sprintf("-> %s (%d filas)", OUT, nrow(curve)))

message("\nlisto.")
