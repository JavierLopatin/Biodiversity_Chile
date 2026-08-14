#!/usr/bin/env Rscript
# Alpha taxonomica coverage-standardized con iNEXT.3D, misma receta que Perez-Giraldo et al.
# 2025 (Ecography): iNEXT.3D::estimate3D(diversity="TD", datatype="abundance",
# base="coverage", level=NULL, nboot=1). q=c(0,1,2). A diferencia de PD (scripts/63), TD
# no necesita arbol filogenetico -- corre sobre TODAS las especies del conteo real
# (data/derived/occurrences_unified_counts.parquet, scripts/61), sin la restriccion a
# especies del arbol de 610 tips que le baja el N a PD (888 vs 2499 candidatas -- ver
# `probar con incluir todo`, este script es esa prueba).
#
# estimate3D(base="coverage") exige >=5 especies observadas por unidad (mismo gate ya
# encontrado en script 63) -- se loopea parcela por parcela con try/catch.
#
# Facets nuevas `td_inext_q0/q1/q2`, aditivas -- no reemplazan hill_q0_unified.

suppressWarnings(suppressMessages({
  library(arrow); library(iNEXT.3D)
}))

DERIVED <- "data/derived"
t_start <- Sys.time()

occ <- read_parquet(file.path(DERIVED, "occurrences_unified_counts.parquet"))
comm <- xtabs(Value ~ PlotObservationID + species, data = occ)
comm <- comm[rowSums(comm) > 0, ]
sp_per_plot <- rowSums(comm > 0)
cat(sprintf("parcelas con conteo real: %d\n", nrow(comm)))
cat(sprintf("parcelas con >=5 especies (SIN restringir al arbol): %d de %d (%.1f%%)\n",
            sum(sp_per_plot >= 5), nrow(comm), 100 * mean(sp_per_plot >= 5)))

ids_ok <- names(sp_per_plot)[sp_per_plot >= 5]
Q <- c(0, 1, 2)

td_cov <- matrix(NA_real_, length(ids_ok), length(Q), dimnames = list(NULL, paste0("q", Q)))
sc_obs <- numeric(length(ids_ok)); n_ind <- numeric(length(ids_ok))
ok <- logical(length(ids_ok))
err_msgs <- character(0)

for (i in seq_along(ids_ok)) {
  id <- ids_ok[i]
  v <- comm[id, ]
  v <- v[v > 0]
  r <- tryCatch(
    estimate3D(data = list(x = setNames(as.integer(v), names(v))), diversity = "TD", q = Q,
               datatype = "abundance", base = "coverage", level = NULL, nboot = 1),
    error = function(e) { err_msgs[[length(err_msgs) + 1]] <<- conditionMessage(e); e }
  )
  if (inherits(r, "error")) { ok[i] <- FALSE; next }
  ok[i]       <- TRUE
  td_cov[i, ] <- r$qTD[match(Q, r$Order.q)]
  sc_obs[i]   <- r$SC[1]
  n_ind[i]    <- sum(v)
}

cat(sprintf("\ncorrieron ok: %d de %d parcelas con >=5 especies (%.1f%%)\n",
            sum(ok), length(ids_ok), 100 * mean(ok)))
if (length(err_msgs)) {
  cat("errores (primeros 5 distintos):\n")
  for (m in head(unique(err_msgs), 5)) cat("  -", m, "\n")
}

out <- data.frame(
  PlotObservationID = ids_ok[ok],
  td_inext_q0 = td_cov[ok, "q0"],
  td_inext_q1 = td_cov[ok, "q1"],
  td_inext_q2 = td_cov[ok, "q2"],
  sc_td_inext = sc_obs[ok],
  n_ind_td_inext = n_ind[ok]
)
write_parquet(out, file.path(DERIVED, "td_inext_coverage.parquet"))
cat(sprintf("-> %s (%d parcelas)\n", file.path(DERIVED, "td_inext_coverage.parquet"), nrow(out)))

# --- comparacion contra hill_q0_unified ya existente (q0, riqueza observada) --------------
resp <- read_parquet(file.path(DERIVED, "unified_diversity_responses.parquet"))
cmp <- merge(out, resp[, c("PlotObservationID", "hill_q0_unified")], by = "PlotObservationID")
cat(sprintf("\nparcelas en comun con hill_q0_unified: %d\n", nrow(cmp)))
cat(sprintf("Spearman(td_inext_q0, hill_q0_unified) = %.3f\n",
            cor(cmp$td_inext_q0, cmp$hill_q0_unified, method = "spearman")))
cat(sprintf("Spearman(td_inext_q0, td_inext_q1) = %.3f   Spearman(td_inext_q1, td_inext_q2) = %.3f\n",
            cor(cmp$td_inext_q0, cmp$td_inext_q1, method = "spearman"),
            cor(cmp$td_inext_q1, cmp$td_inext_q2, method = "spearman")))
cat(sprintf("cobertura observada (SC) -- min/mediana/max: %.3f / %.3f / %.3f\n",
            min(cmp$sc_td_inext), median(cmp$sc_td_inext), max(cmp$sc_td_inext)))

cat(sprintf("\ntiempo total: %.1f min\n", as.numeric(Sys.time() - t_start, units = "mins")))
