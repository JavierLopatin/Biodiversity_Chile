#!/usr/bin/env Rscript
# Dark diversity unificada (Parcelas-CL + Living Trees Chile). Mismo metodo que
# scripts/27_compute_dark_diversity.R (DarkDiv::DarkDiv method="Hypergeometric", umbral
# 0.9 sobre la probabilidad de pertenencia), extendido al pool de co-ocurrencia unificado
# (scripts/lib/unified_comm.R). NO toca dark_diversity.parquet -- ese sigue siendo el
# target del pipeline de modelado actual sobre Parcelas-CL solo.
#
# Mismo criterio de pool que script 27: se usa el dataset COMPLETO de cada fuente
# (Parcelas-CL 1.485 + Living Trees 2.020) para estimar el pool de co-ocurrencia, aunque
# solo se reporten las ~3.102 parcelas del set unificado de modelado -- con el pool
# reducido la dark diversity se contamina de esfuerzo de muestreo (medido en script 27:
# rho con log-area sube de +0,085 a +0,315 si se usa solo el subset).
#
# Uso:
#   Rscript scripts/55_unified_dark_diversity.R
#   Rscript scripts/55_unified_dark_diversity.R --thr 0.95

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
OUT_DIR <- "data/derived"
THR     <- getarg("--thr", 0.9)

# --------------------------------------------------------------------------------------
# 1. comunidad -- pool completo, no el subset reportado
# --------------------------------------------------------------------------------------

message("== comunidad ==")
tree <- read.tree(TREE)
us <- unified_species(ZIP, LT_LONG, quiet = TRUE)
plots_uni <- read_parquet(PLOTS)
cmm <- unified_comm(us$pc_raw, LT_LONG, tree, plots_uni$PlotObservationID)
comm <- cmm$full                      # pool: TODO Parcelas-CL + TODO Living Trees
obs <- rowSums(comm)

# --------------------------------------------------------------------------------------
# 2. estimacion
# --------------------------------------------------------------------------------------

message("\n== DarkDiv ==")
t0 <- Sys.time()
dd <- DarkDiv(comm, method = "Hypergeometric")
message(sprintf("  %.1f s", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

pr <- dd$Dark                                    # NA donde la especie SI esta presente
dark_bin <- !is.na(pr) & pr > THR
message(sprintf("  probabilidades de pertenencia: mediana %.3f, p90 %.3f",
                median(pr, na.rm = TRUE), quantile(pr, 0.9, na.rm = TRUE)))

message(paste("\n  sensibilidad al umbral (rho contra la riqueza observada) --",
              "confirmar que 0,9 sigue en la meseta, no asumirlo transferido:"))
for (t in c(0.5, 0.7, 0.8, 0.9, 0.95, 0.99)) {
  d <- rowSums(!is.na(pr) & pr > t)
  message(sprintf("    thr %.2f -> mediana %5.1f especies oscuras   rho(obs) = %+.3f%s",
                  t, median(d), cor(d, obs, method = "spearman"),
                  if (isTRUE(all.equal(t, THR))) "   <- elegido" else ""))
}

# --------------------------------------------------------------------------------------
# 3. dark diversity filogenetica (con el arbol unificado, scripts/54)
# --------------------------------------------------------------------------------------

message("\n== dark diversity filogenetica ==")
tips <- colnames(comm)
tr <- keep.tip(tree, tips)
cph <- cophenetic(tr)
dark_mpd <- vapply(seq_len(nrow(comm)), function(i) {
  k <- which(dark_bin[i, ])
  if (length(k) < 2) return(NA_real_)
  sp <- colnames(comm)[k]
  mean(cph[sp, sp][lower.tri(diag(length(sp)))])
}, numeric(1))
message(sprintf("  MPD del conjunto oscuro: mediana %.0f Ma", median(dark_mpd, na.rm = TRUE)))

# --------------------------------------------------------------------------------------
# 4. tabla de respuestas -- solo el set unificado reportado, NA en el resto del pool
# --------------------------------------------------------------------------------------

dark <- data.frame(
  PlotObservationID    = rownames(comm),
  n_obs_unified        = obs,
  dark_n_unified       = rowSums(dark_bin),                        # <- el target
  pool_n_unified       = obs + rowSums(dark_bin),
  dark_mpd_unified     = dark_mpd,
  completeness_unified = log(obs / rowSums(pr, na.rm = TRUE)),     # descriptor, NO target
  stringsAsFactors = FALSE
)

out <- merge(data.frame(PlotObservationID = plots_uni$PlotObservationID),
             dark, by = "PlotObservationID", all.x = TRUE)
write_parquet(out, file.path(OUT_DIR, "unified_dark_diversity.parquet"))
message(sprintf("\n  -> unified_dark_diversity.parquet  %d filas, %d sin cobertura",
                nrow(out), sum(is.na(out$dark_n_unified))))

# --------------------------------------------------------------------------------------
# 5. que sirve como target -- mismo criterio de script 27
# --------------------------------------------------------------------------------------

message("\n== correlacion con riqueza (criterio de seleccion de script 27) ==")
hill <- read_parquet(file.path(OUT_DIR, "unified_diversity_responses.parquet"))
m <- merge(out, hill, by = "PlotObservationID", all.x = TRUE)
rho <- function(a, b) cor(a, b, method = "spearman", use = "complete.obs")
for (v in c("dark_n_unified", "pool_n_unified", "dark_mpd_unified", "completeness_unified")) {
  message(sprintf("  %-20s rho(hill_q0_unified) = %+.3f", v, rho(m[[v]], m$hill_q0_unified)))
}

message("\nlisto.")
