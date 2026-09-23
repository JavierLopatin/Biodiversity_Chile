#!/usr/bin/env Rscript
# Curvas de Hill (q=0,1,2: riqueza, tipo Shannon, tipo Simpson) taxonomicas y
# filogeneticas, sobre Parcelas-CL solo y sobre el pool unificado completo (Parcelas-CL +
# Living Trees Chile). Mismo metodo que scripts/26_rarefaction_inext.R (iNEXT.3D para
# taxonomica, la reimplementacion de scripts/lib/pd_inext.R para filogenetica), extendido
# de q=0 solo a los 3 ordenes de Hill -- ver ese archivo para de donde salen las formulas
# de q=1,2 y como se validaron contra iNEXT.3D (tests/test_phylo.R).
#
# Substrato (decision del autor, 2026-09-23, docs/24 paso 6): las curvas salen de las mismas
# parcelas sobre las que se ajusta y valida cada faceta, no del zip completo de Parcelas-CL
# mas Living Trees. La figura describe las variables respuesta del modelo; un pool que el
# modelo nunca ve (593 especies, todo el estrato de cobertura) se leia como si fuera lo
# que se predice. Por eso:
#   TD -> la tabla de conteos, restringida a las parcelas con td_inext_q0 (scripts/64)
#   PD -> la tabla de conteos, restringida a las parcelas con pd_inext_q0 (scripts/63) y a
#         las especies con punta en el arbol
# Las dos en incidencia (presencia/ausencia por parcela), como antes.
#
# `--woody` lee occurrences_unified_counts_woody.parquet y las facetas _woody (scripts/83)
# y escribe unified_hill_curves_woody.csv; sin la bandera, la variante con hierbas.
#
# El arbol es el unificado de scripts/54 (phylo_tree_unified.tre, 610 tips). No hace falta
# podarlo para la variante lenosa: pd_curve y subsample_band (lib/pd_inext.R) ya lo podan a
# las columnas de la comunidad, y la profundidad de referencia sale de las puntas observadas.
#
# RIESGO DE COMPUTO: la rarefaccion en q=1,2 hace un loop O(t) por rama por tamano -- no
# hay atajo, es lo que hace `RPD` de verdad. Mas cara que el atajo O(1) de q=0. Se corre en
# background y se reportan los tiempos reales, no se asumen.
#
# Uso:
#   Rscript scripts/56_unified_hill_curves.R
#   Rscript scripts/56_unified_hill_curves.R --woody
#   Rscript scripts/56_unified_hill_curves.R --nboot 200 --reps 200 --knots 40

suppressWarnings(suppressMessages({
  library(ape); library(arrow); library(iNEXT.3D)
}))
source("scripts/lib/pd_inext.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
SFX       <- if ("--woody" %in% args) "_woody" else ""
OCC       <- sprintf("data/derived/occurrences_unified_counts%s.parquet", SFX)
TD_FACET  <- sprintf("data/derived/td_inext_coverage%s.parquet", SFX)
PD_FACET  <- sprintf("data/derived/pd_inext_coverage%s.parquet", SFX)
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
occ <- as.data.frame(read_parquet(OCC))
occ$tip <- gsub(" ", "_", occ$species)

facet_ids <- function(path, col) {
  f <- as.data.frame(read_parquet(path))
  f$PlotObservationID[!is.na(f[[col]])]
}
incidence <- function(d, ids, col) {
  d <- d[d$PlotObservationID %in% ids, ]
  m <- unclass(table(d$PlotObservationID, d[[col]]) > 0) * 1
  m[, colSums(m) > 0, drop = FALSE]
}

td_ids <- facet_ids(TD_FACET, "td_inext_q0")
pd_ids <- facet_ids(PD_FACET, "pd_inext_q0")
comm_td <- incidence(occ, td_ids, "species")
comm_pd <- incidence(occ[occ$tip %in% tree$tip.label, ], pd_ids, "tip")
stopifnot(nrow(comm_td) == length(td_ids), nrow(comm_pd) == length(pd_ids))
message(sprintf("  TD: %d parcelas x %d especies (%s)", nrow(comm_td), ncol(comm_td), basename(TD_FACET)))
message(sprintf("  PD: %d parcelas x %d especies con punta en el arbol (%s)",
                nrow(comm_pd), ncol(comm_pd), basename(PD_FACET)))
LABEL <- "unificado analisis"

# --------------------------------------------------------------------------------------
# 2. curvas
# --------------------------------------------------------------------------------------

# Banda de submuestreo de la riqueza (q = 0, 1, 2) sobre TODAS las especies de la
# comunidad TD. `subsample_band` la calcula solo sobre especies con punta en el arbol, que
# para TD dejaria afuera las que no la tienen.
sr_band <- function(comm, sizes, q, reps, seed, conf = 0.95) {
  Tn <- nrow(comm)
  sizes <- sort(unique(sizes[sizes >= 1 & sizes <= Tn]))
  a <- (1 - conf) / 2
  set.seed(seed)
  do.call(rbind, lapply(sizes, function(t_size) {
    m <- matrix(vapply(seq_len(reps), function(r) {
      Y <- colSums(comm[sample.int(Tn, t_size), , drop = FALSE])
      hill_from_incidence(Y, rep(1, length(Y)), t_size, q)
    }, numeric(length(q))), nrow = length(q))
    row <- data.frame(n = t_size)
    for (j in seq_along(q)) {
      row[[paste0("sr_lo_q", q[j])]] <- quantile(m[j, ], a, names = FALSE)
      row[[paste0("sr_hi_q", q[j])]] <- quantile(m[j, ], 1 - a, names = FALSE)
    }
    row
  }))
}

run_td <- function(comm, label) {
  n <- nrow(comm)
  message(sprintf("  TD %s: n=%d, endpoint=%d", label, n, 2 * n))
  t0 <- Sys.time()
  set.seed(SEED)
  td <- iNEXT.3D::iNEXT3D(setNames(list(t(comm)), label), diversity = "TD", q = Q,
                          datatype = "incidence_raw", endpoint = 2 * n,
                          knots = KNOTS, nboot = NBOOT)
  s <- td$TDiNextEst$size_based
  eff <- intersect(c("nt", "mT", "m"), names(s))[1]
  qcol <- intersect(c("Order.q", "order", "q"), names(s))[1]
  band <- sr_band(comm, s[[eff]], Q, REPS, SEED)
  message(sprintf("    %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  do.call(rbind, lapply(Q, function(qi) {
    td_q <- s[s[[qcol]] == qi, ]
    bi <- match(td_q[[eff]], band$n)
    data.frame(dataset = label, metric = "taxonomica", q = qi, n = td_q[[eff]],
               method = td_q$Method, value = td_q$qTD,
               lo = band[[paste0("sr_lo_q", qi)]][bi],
               hi = band[[paste0("sr_hi_q", qi)]][bi], stringsAsFactors = FALSE)
  }))
}

run_pd <- function(comm, label) {
  n <- nrow(comm)
  message(sprintf("  PD %s: n=%d, endpoint=%d", label, n, 2 * n))
  t0 <- Sys.time()
  sizes <- sort(unique(c(round(seq(1, 2 * n, length.out = KNOTS)), n)))
  pc_curve <- pd_curve(comm, tree, sizes = sizes, q = Q, knots = KNOTS)
  band <- subsample_band(comm, tree, sizes, q = Q, reps = REPS, seed = SEED)
  message(sprintf("    %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  do.call(rbind, lapply(Q, function(qi) {
    pd_q <- pc_curve[pc_curve$q == qi, ]
    bi <- match(pd_q$n, band$n)
    data.frame(dataset = label, metric = "filogenetica_meanPD", q = qi, n = pd_q$n,
               method = pd_q$method, value = pd_q$meanpd,
               lo = band[[paste0("meanpd_lo_q", qi)]][bi],
               hi = band[[paste0("meanpd_hi_q", qi)]][bi], stringsAsFactors = FALSE)
  }))
}

t_all <- Sys.time()
curves <- rbind(run_td(comm_td, LABEL), run_pd(comm_pd, LABEL))
message(sprintf("\n  total: %.1f min", as.numeric(difftime(Sys.time(), t_all, units = "mins"))))

OUT <- file.path(OUT_DIR, sprintf("unified_hill_curves%s.csv", SFX))
write.csv(curves, OUT, row.names = FALSE)
message(sprintf("  -> %s  (%d filas)", OUT, nrow(curves)))

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
