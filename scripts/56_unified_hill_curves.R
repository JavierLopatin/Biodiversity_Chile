#!/usr/bin/env Rscript
# Curvas de Hill (q=0,1,2: riqueza, tipo Shannon, tipo Simpson) taxonomicas y
# filogeneticas, sobre Parcelas-CL solo y sobre el pool unificado completo (Parcelas-CL +
# Living Trees Chile). Mismo metodo que scripts/26_rarefaction_inext.R (iNEXT.3D para
# taxonomica, la reimplementacion de scripts/lib/pd_inext.R para filogenetica), extendido
# de q=0 solo a los 3 ordenes de Hill -- ver ese archivo para de donde salen las formulas
# de q=1,2 y como se validaron contra iNEXT.3D (tests/test_phylo.R).
#
# "con todos los datos": el ensamble "unificado completo" usa TODO Parcelas-CL (1.485) +
# TODO Living Trees (2.020), no solo el set de 3.102 usado para modelar -- mismo criterio
# que ya usa scripts/27 para el pool de dark diversity (mas parcelas de co-ocurrencia =
# mejor estimacion, aunque no todas entren al modelo).
#
# El arbol es el unificado de scripts/54 (phylo_tree_unified.tre, 610 tips) para LAS DOS
# curvas filogeneticas -- asi las dos quedan en pie de igualdad metodologica (mismo arbol),
# aunque eso significa que la curva "Parcelas-CL completo" de aqui no es bit-a-bit igual a
# la de fig15 (esa usa el arbol de 601 tips de scripts/25, no el de 610 de scripts/54) --
# la diferencia esperada es chica (9 especies nuevas, en su mayoria en generos ya
# representados) pero real, y se documenta en vez de mezclarla en silencio con fig15.
#
# RIESGO DE COMPUTO: la rarefaccion en q=1,2 hace un loop O(t) por rama por tamano -- no
# hay atajo, es lo que hace `RPD` de verdad. Mas cara que el atajo O(1) de q=0. Se corre en
# background y se reportan los tiempos reales, no se asumen.
#
# Uso:
#   Rscript scripts/56_unified_hill_curves.R
#   Rscript scripts/56_unified_hill_curves.R --nboot 200 --reps 200 --knots 40

suppressWarnings(suppressMessages({
  library(ape); library(arrow); library(iNEXT.3D); library(V.PhyloMaker2)
}))
source("scripts/lib/parcelas_comm.R")
source("scripts/lib/unified_comm.R")
source("scripts/lib/pd_inext.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP       <- "data/20602096.zip"
LT_LONG   <- "data/derived/living_trees_long.parquet"
PLOTS_UNI <- "data/derived/plots_unified.parquet"
TREE_UNI  <- "data/derived/phylo_tree_unified.tre"
OUT_DIR   <- "data/derived"
Q         <- c(0, 1, 2)
NBOOT     <- getarg("--nboot", 50)
REPS      <- getarg("--reps", 200)
KNOTS     <- getarg("--knots", 40)
SEED      <- 42

# --------------------------------------------------------------------------------------
# 1. comunidades
# --------------------------------------------------------------------------------------

message("== comunidades ==")
tree <- read.tree(TREE_UNI)

pc <- parcelas_species(ZIP, quiet = TRUE)
cm_pcl <- parcelas_comm(pc$raw, tree, pc$raw$PlotObservationID, quiet = TRUE)$full
message(sprintf("  Parcelas-CL completo: %d parcelas x %d especies",
                nrow(cm_pcl), ncol(cm_pcl)))

us <- unified_species(ZIP, LT_LONG, quiet = TRUE)
plots_uni <- read_parquet(PLOTS_UNI)
cm_uni <- unified_comm(us$pc_raw, LT_LONG, tree, plots_uni$PlotObservationID,
                       quiet = TRUE)$full
message(sprintf("  unificado completo: %d parcelas x %d especies", nrow(cm_uni), ncol(cm_uni)))

assemblages <- list("Parcelas-CL completo" = cm_pcl, "unificado completo" = cm_uni)

# --------------------------------------------------------------------------------------
# 2. curvas
# --------------------------------------------------------------------------------------

run_one <- function(comm, label) {
  n <- nrow(comm)
  message(sprintf("  %s: n=%d, endpoint=%d", label, n, 2 * n))

  t0 <- Sys.time()
  set.seed(SEED)
  td <- iNEXT.3D::iNEXT3D(setNames(list(t(comm)), label), diversity = "TD", q = Q,
                          datatype = "incidence_raw", endpoint = 2 * n,
                          knots = KNOTS, nboot = NBOOT)
  s <- td$TDiNextEst$size_based
  eff <- intersect(c("nt", "mT", "m"), names(s))[1]
  qcol <- intersect(c("Order.q", "order", "q"), names(s))[1]
  message(sprintf("    taxonomica: %.1f min",
                  as.numeric(difftime(Sys.time(), t0, units = "mins"))))

  sizes <- sort(unique(c(s[[eff]], n)))

  t1 <- Sys.time()
  pc_curve <- pd_curve(comm, tree, sizes = sizes, q = Q, knots = KNOTS)
  message(sprintf("    filogenetica: %.1f min",
                  as.numeric(difftime(Sys.time(), t1, units = "mins"))))

  t2 <- Sys.time()
  band <- subsample_band(comm, tree, sizes, q = Q, reps = REPS, seed = SEED)
  message(sprintf("    banda: %.1f min",
                  as.numeric(difftime(Sys.time(), t2, units = "mins"))))

  curves <- lapply(Q, function(qi) {
    td_q <- s[s[[qcol]] == qi, ]
    pd_q <- pc_curve[pc_curve$q == qi, ]
    bi_td <- match(td_q[[eff]], band$n)
    bi_pd <- match(pd_q$n, band$n)
    rbind(
      data.frame(dataset = label, metric = "taxonomica", q = qi, n = td_q[[eff]],
                method = td_q$Method, value = td_q$qTD,
                lo = band[[paste0("sr_lo_q", qi)]][bi_td],
                hi = band[[paste0("sr_hi_q", qi)]][bi_td], stringsAsFactors = FALSE),
      data.frame(dataset = label, metric = "filogenetica_meanPD", q = qi, n = pd_q$n,
                method = pd_q$method, value = pd_q$meanpd,
                lo = band[[paste0("meanpd_lo_q", qi)]][bi_pd],
                hi = band[[paste0("meanpd_hi_q", qi)]][bi_pd], stringsAsFactors = FALSE)
    )
  })
  do.call(rbind, curves)
}

t_all <- Sys.time()
runs <- lapply(names(assemblages), function(k) run_one(assemblages[[k]], k))
curves <- do.call(rbind, runs)
message(sprintf("\n  total: %.1f min", as.numeric(difftime(Sys.time(), t_all, units = "mins"))))

write.csv(curves, file.path(OUT_DIR, "unified_hill_curves.csv"), row.names = FALSE)
message(sprintf("  -> unified_hill_curves.csv  (%d filas)", nrow(curves)))

# --------------------------------------------------------------------------------------
# 3. resumen
# --------------------------------------------------------------------------------------

message("\n== resumen (observado -> 2n) ==")
for (qi in Q) for (m in unique(curves$metric)) for (d in unique(curves$dataset)) {
  z <- curves[curves$metric == m & curves$dataset == d & curves$q == qi, ]
  if (!nrow(z)) next
  obs <- z[z$method == "Observed", ][1, ]
  ext <- z[nrow(z), ]
  message(sprintf("  q=%d  %-20s %-22s obs(n=%4.0f) %8.2f -> 2n=%4.0f %8.2f (+%.1f%%)",
                  qi, m, d, obs$n, obs$value, ext$n, ext$value,
                  100 * (ext$value / obs$value - 1)))
}

message("\nlisto.")
