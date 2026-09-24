#!/usr/bin/env Rscript
# LCBD leñoso recalculado DENTRO de cada estrato de vegetación (propuesta de modelos por
# tipo, Fassnacht et al. 2021). Estratos: forest_type de CONAF para Living Trees, y
# Parcelas-CL como un estrato propio (es una fuente, no un tipo de vegetación).
#
# Mismo estimador que scripts/62 (beta.div.comp(coef="S", quant=TRUE) -> LCBD.comp(sqrt.D)),
# pero sobre la matriz de cada estrato sola. Con el LCBD contra el pool de las 2.499 el
# target seguiría codificando la distancia de cada parcela a los otros estratos: en
# Parcelas-CL eso era dos tercios de la ventaja aparente de la fenología (54a1ca0).
#
# El LCBD suma 1 dentro de cada estrato, así que su escala depende del n del estrato: los
# valores NO son comparables entre estratos ni se fusionan en un mapa. Se reporta R² por
# estrato.
#
# Uso:
#   Rscript scripts/87_lcbd_by_stratum.R            # estratos con >= MIN_N parcelas
suppressMessages({ library(arrow); library(adespatial) })

DERIVED <- "data/derived"
MIN_N <- 50

occ <- as.data.frame(read_parquet(file.path(DERIVED, "occurrences_unified_counts_woody.parquet")))
plots <- as.data.frame(read_parquet(file.path(DERIVED, "plots_unified.parquet")))
plots$estrato <- ifelse(plots$source == "living_trees", plots$forest_type, "Parcelas-CL")
occ$estrato <- plots$estrato[match(occ$PlotObservationID, plots$PlotObservationID)]

n_by <- tapply(occ$PlotObservationID, occ$estrato, function(x) length(unique(x)))
keep <- names(n_by)[n_by >= MIN_N]
cat(sprintf("estratos con >= %d parcelas: %d de %d\n", MIN_N, length(keep), length(n_by)))

out <- list()
for (e in keep) {
  d <- occ[occ$estrato == e, ]
  comm <- xtabs(Value ~ PlotObservationID + species, data = d)
  comm <- comm[rowSums(comm) > 0, colSums(comm) > 0, drop = FALSE]
  comp <- beta.div.comp(comm, coef = "S", quant = TRUE)
  lcbd <- LCBD.comp(comp$D, sqrt.D = TRUE)
  cat(sprintf("  %-26s %4d parcelas x %3d especies  BDtotal=%.4f\n",
              e, nrow(comm), ncol(comm), lcbd$beta["BDtotal"]))
  out[[e]] <- data.frame(PlotObservationID = rownames(comm), estrato = e,
                         lcbd_count_sorensen = as.numeric(lcbd$LCBD))
}
out <- do.call(rbind, out)
f <- file.path(DERIVED, "lcbd_count_sorensen_woody_by_stratum.parquet")
write_parquet(out, f)
cat(sprintf("-> %s (%d filas)\n", f, nrow(out)))
