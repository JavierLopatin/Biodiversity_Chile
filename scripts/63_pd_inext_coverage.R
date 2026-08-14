#!/usr/bin/env Rscript
# PD alpha coverage-standardized con iNEXT.3D, misma receta que Perez-Giraldo et al. 2025
# (Ecography): iNEXT.3D::estimate3D(diversity="PD", datatype="abundance",
# base="coverage", level=NULL, nboot=1, PDtree=...). q=c(0,1,2), para equiparar con el
# tratamiento alfa/beta/gamma taxonomico ya usado en el proyecto (Hill q=0,1,2). Sobre el
# subset de conteo real (data/derived/occurrences_unified_counts.parquet, scripts/61)
# restringido a especies presentes en el arbol filogenetico unificado de 610 tips
# (data/derived/phylo_tree_unified.tre, scripts/54_unified_phylo_responses.R).
#
# estimate3D(base="coverage") exige >=5 especies observadas por unidad (verificado
# empiricamente esta sesion, mismo gate que ObsAsy3D) -- se loopea parcela por parcela con
# try/catch, no batch, y se reporta explicito cuantas quedan afuera.
#
# Facets nuevas `pd_inext_q0/q1/q2`, aditivas -- no reemplazan pd_faith_unified/ses.pd_unified.

suppressWarnings(suppressMessages({
  library(arrow); library(ape); library(iNEXT.3D)
}))

DERIVED <- "data/derived"
t_start <- Sys.time()

tree <- read.tree(file.path(DERIVED, "phylo_tree_unified.tre"))
occ  <- read_parquet(file.path(DERIVED, "occurrences_unified_counts.parquet"))

occ$species_ <- gsub(" ", "_", occ$species)
occ_tree <- occ[occ$species_ %in% tree$tip.label, ]
cat(sprintf("conteo real restringido a especies del arbol: %d de %d filas (%d/%d especies)\n",
            nrow(occ_tree), nrow(occ), length(unique(occ_tree$species_)), length(unique(occ$species_))))

comm <- xtabs(Value ~ PlotObservationID + species_, data = occ_tree)
comm <- comm[rowSums(comm) > 0, ]
sp_per_plot <- rowSums(comm > 0)
cat(sprintf("parcelas con >=1 especie del arbol: %d\n", nrow(comm)))
cat(sprintf("parcelas con >=5 especies del arbol: %d de %d (%.1f%%)\n",
            sum(sp_per_plot >= 5), nrow(comm), 100 * mean(sp_per_plot >= 5)))

ids_ok <- names(sp_per_plot)[sp_per_plot >= 5]
Q <- c(0, 1, 2)

pd_cov <- matrix(NA_real_, length(ids_ok), length(Q), dimnames = list(NULL, paste0("q", Q)))
sc_obs <- numeric(length(ids_ok)); n_ind <- numeric(length(ids_ok))
ok <- logical(length(ids_ok))
err_msgs <- character(0)

for (i in seq_along(ids_ok)) {
  id <- ids_ok[i]
  v <- comm[id, ]
  v <- v[v > 0]
  sub_tree <- tryCatch(keep.tip(tree, names(v)), error = function(e) NULL)
  if (is.null(sub_tree)) { ok[i] <- FALSE; next }
  r <- tryCatch(
    estimate3D(data = list(x = setNames(as.integer(v), names(v))), diversity = "PD", q = Q,
               datatype = "abundance", base = "coverage", level = NULL,
               nboot = 1, PDtree = sub_tree),
    error = function(e) { err_msgs[[length(err_msgs) + 1]] <<- conditionMessage(e); e }
  )
  if (inherits(r, "error")) { ok[i] <- FALSE; next }
  ok[i]        <- TRUE
  pd_cov[i, ]  <- r$qPD[match(Q, r$Order.q)]
  sc_obs[i]    <- r$SC[1]
  n_ind[i]     <- sum(v)
}

cat(sprintf("\ncorrieron ok: %d de %d parcelas con >=5 especies (%.1f%%)\n",
            sum(ok), length(ids_ok), 100 * mean(ok)))
if (length(err_msgs)) {
  cat("errores (primeros 5 distintos):\n")
  for (m in head(unique(err_msgs), 5)) cat("  -", m, "\n")
}

out <- data.frame(
  PlotObservationID = ids_ok[ok],
  pd_inext_q0 = pd_cov[ok, "q0"],
  pd_inext_q1 = pd_cov[ok, "q1"],
  pd_inext_q2 = pd_cov[ok, "q2"],
  sc_pd_inext = sc_obs[ok],
  n_ind_pd_inext = n_ind[ok]
)
write_parquet(out, file.path(DERIVED, "pd_inext_coverage.parquet"))
cat(sprintf("-> %s (%d parcelas)\n", file.path(DERIVED, "pd_inext_coverage.parquet"), nrow(out)))

# --- comparacion contra pd_faith_unified ya existente (q0, comparable a Faith's PD) -------
resp <- read_parquet(file.path(DERIVED, "unified_phylo_responses.parquet"))
cmp <- merge(out, resp[, c("PlotObservationID", "pd_faith_unified", "n_sp_tree_unified")],
             by = "PlotObservationID")
cat(sprintf("\nparcelas en comun con pd_faith_unified: %d\n", nrow(cmp)))
cat(sprintf("Spearman(pd_inext_q0, pd_faith_unified) = %.3f\n",
            cor(cmp$pd_inext_q0, cmp$pd_faith_unified, method = "spearman")))
cat(sprintf("Spearman(pd_inext_q0, pd_inext_q1) = %.3f   Spearman(pd_inext_q1, pd_inext_q2) = %.3f\n",
            cor(cmp$pd_inext_q0, cmp$pd_inext_q1, method = "spearman"),
            cor(cmp$pd_inext_q1, cmp$pd_inext_q2, method = "spearman")))
cat(sprintf("cobertura observada (SC) -- min/mediana/max: %.3f / %.3f / %.3f\n",
            min(cmp$sc_pd_inext), median(cmp$sc_pd_inext), max(cmp$sc_pd_inext)))

cat(sprintf("\ntiempo total: %.1f min\n", as.numeric(Sys.time() - t_start, units = "mins")))
