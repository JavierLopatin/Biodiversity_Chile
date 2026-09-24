#!/usr/bin/env Rscript
# LCBD taxonomica con la metrica exacta de Perez-Giraldo et al. 2025 (Ecography):
# adespatial::beta.div.comp(coef="S", quant=TRUE) -> LCBD.comp(sqrt.D=TRUE), sobre
# conteo real de individuos (data/derived/occurrences_unified_counts.parquet, ver
# scripts/61_build_unified_counts.py). Distinto del beta.div(method="hellinger"/"jaccard")
# ya usado en scripts/52_unified_diversity_facets.R (misma familia adespatial, coeficiente
# de disimilitud distinto -- ver docs/19_unified_facets_methodology.md).
#
# Facet nueva `lcbd_count_sorensen`, aditiva -- no reemplaza lcbd_pa_unified/lcbd_freq_unified.

suppressMessages({
  library(arrow)
  library(adespatial)
})

DERIVED <- "data/derived"
# `--woody`: lee occurrences_unified_counts_woody.parquet (scripts/83, docs/24) y escribe
# con sufijo _woody; sin la bandera, los nombres de siempre (variante con hierbas).
SFX <- if ("--woody" %in% commandArgs(trailingOnly = TRUE)) "_woody" else ""
# `--pcl`: la matriz de comunidad solo con las parcelas de Parcelas-CL. El LCBD pasa a medir
# unicidad dentro de esa flora y no contra un pool dominado por Living Trees. Salida con
# sufijo adicional _pcl (lcbd_count_sorensen_woody_pcl.parquet).
PCL_ONLY <- "--pcl" %in% commandArgs(trailingOnly = TRUE)

occ <- read_parquet(file.path(DERIVED, paste0("occurrences_unified_counts", SFX, ".parquet")))
if (PCL_ONLY) occ <- occ[startsWith(occ$PlotObservationID, "PCL_"), ]
comm <- xtabs(Value ~ PlotObservationID + species, data = occ)
comm <- comm[, colSums(comm) > 0]
comm <- comm[rowSums(comm) > 0, ]
cat(sprintf("pool de conteo real: %d parcelas x %d especies\n", nrow(comm), ncol(comm)))

cat("\n[Sorensen cuantitativo] beta.div.comp(coef='S', quant=TRUE) ...\n")
t0 <- Sys.time()
comp <- beta.div.comp(comm, coef = "S", quant = TRUE)
cat(sprintf("  listo en %.1f s\n", as.numeric(Sys.time() - t0, units = "secs")))

lcbd <- LCBD.comp(comp$D, sqrt.D = TRUE)
cat(sprintf("  SStotal=%.4f  BDtotal=%.4f\n", lcbd$beta["SStotal"], lcbd$beta["BDtotal"]))

out <- data.frame(
  PlotObservationID = rownames(comm),
  lcbd_count_sorensen = as.numeric(lcbd$LCBD)
)

write_parquet(out, file.path(DERIVED, paste0("lcbd_count_sorensen", SFX, if (PCL_ONLY) "_pcl" else "", ".parquet")))
cat(sprintf("\n-> %s (%d parcelas)\n", file.path(DERIVED, paste0("lcbd_count_sorensen", SFX, if (PCL_ONLY) "_pcl" else "", ".parquet")), nrow(out)))

# --- comparacion contra las LCBD ya existentes (pool completo, otro coeficiente) ---------
resp <- read_parquet(file.path(DERIVED, "unified_diversity_responses.parquet"))
cmp <- merge(out, resp[, c("PlotObservationID", "lcbd_pa_unified", "lcbd_freq_unified")],
             by = "PlotObservationID")
cat(sprintf("\nparcelas en comun con facetas existentes: %d de %d\n", nrow(cmp), nrow(out)))
cat(sprintf("Spearman(lcbd_count_sorensen, lcbd_pa_unified)   = %.3f  (n=%d completos)\n",
            cor(cmp$lcbd_count_sorensen, cmp$lcbd_pa_unified, method = "spearman", use = "complete.obs"),
            sum(complete.cases(cmp[, c("lcbd_count_sorensen", "lcbd_pa_unified")]))))
cat(sprintf("Spearman(lcbd_count_sorensen, lcbd_freq_unified) = %.3f  (n=%d completos)\n",
            cor(cmp$lcbd_count_sorensen, cmp$lcbd_freq_unified, method = "spearman", use = "complete.obs"),
            sum(complete.cases(cmp[, c("lcbd_count_sorensen", "lcbd_freq_unified")]))))
