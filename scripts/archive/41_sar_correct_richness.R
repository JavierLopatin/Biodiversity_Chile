#!/usr/bin/env Rscript
# Correccion especie-area (SAR) para hill_q0/q1/q2.
#
# `kfold5_owner` colapsa a R2 negativo incluso en los modelos ganadores actuales, y de las
# facetas riqueza (hill_q0/q1/q2) es la que peor lo hace. Candidato mecanico: PlotSize_m2
# varia 128x (78.5 a 10.000 m2) y esta casi 1:1 confundido con Owner -- 9 de 11
# contribuyentes usan un unico tamano fijo. Relacion especie-area clasica: parcela mas
# grande, mas especies detectadas, sin que sea senal ecologica real.
#
# Rarefaccion tipo iNEXT (Chao-Jost) NO es aplicable aca, no por debilidad sino por
# estructura del dato (verificado contra Parcelas-CL_metadata.txt):
#   - `Abundance_parameter` es heterogeneo e intransferible entre contribuyentes ("Cover",
#     "Basal_area", "Abundance" sin definir si es conteo de individuos) -- otra vez casi
#     1:1 con Owner.
#   - 98/1082 parcelas (~9%) tienen Abundance=0: solo lista de presencia, sin ningun dato
#     de abundancia -- no hay insumo para rarefactar por ningun metodo.
#   - Mediana observada: 5 especies/parcela -- muy poco para que un estimador de cobertura
#     (Chao1/ACE) sea estable incluso donde si hay abundancia utilizable.
# La correccion SAR (log-area) es la unica que aplica pareja a las 1082 parcelas sin
# depender del metodo de abundancia de cada contribuyente, porque PlotSize_m2 esta
# reportado y es comparable para todas.
#
# Nota metodologica, no escondida: con area casi colineal con Owner en 9/11 contribuyentes,
# esta regresion no separa limpiamente "efecto area fisico" de "efecto protocolo/
# observador correlacionado con area" -- y eso es intencional aca, no un defecto: el
# objetivo es remover exactamente esa varianza confundida, cualquiera sea su origen fisico
# o de protocolo.
#
# Reescala a 400 m2 (el tamano mas comun entre contribuyentes de este dataset, y tambien
# el usado en Miranda et al. 2023 -- coincidencia util, no el motivo de la eleccion).
#
# Uso:
#   Rscript scripts/41_sar_correct_richness.R
#   Rscript scripts/41_sar_correct_richness.R --ref-area 400

suppressPackageStartupMessages({
  library(optparse)
  library(arrow)
})

option_list <- list(
  make_option("--responses", type = "character",
             default = "data/derived/biodiversity_responses.parquet"),
  make_option("--plots", type = "character", default = "data/derived/plots_subset.parquet"),
  make_option("--out", type = "character",
             default = "data/derived/biodiversity_responses_sar.parquet"),
  make_option("--ref-area", type = "double", default = 400, dest = "ref_area",
             help = "area de referencia en m2 [default %default]")
)
opt <- parse_args(OptionParser(option_list = option_list))

resp <- as.data.frame(arrow::read_parquet(opt$responses))
plots <- as.data.frame(arrow::read_parquet(opt$plots))

d <- merge(resp[, c("PlotObservationID", "hill_q0", "hill_q1", "hill_q2")],
          plots[, c("PlotObservationID", "PlotSize_m2")],
          by = "PlotObservationID")
d <- d[is.finite(d$PlotSize_m2) & d$PlotSize_m2 > 0, ]
cat(sprintf("parcelas con PlotSize_m2 valido: %d de %d\n", nrow(d), nrow(resp)))

log_area <- log(d$PlotSize_m2)
log_ref <- log(opt$ref_area)

cat(sprintf("\nAjuste log(hill_q_i) ~ log(PlotSize_m2), area de referencia %.0f m2:\n\n",
           opt$ref_area))
cat(sprintf("  %-8s %10s %10s %10s %8s\n", "orden", "pendiente", "ee", "R2", "p"))

fit_and_correct <- function(y, log_area, log_ref) {
  fit <- lm(log(y) ~ log_area)
  s <- summary(fit)
  slope <- coef(fit)[["log_area"]]
  se <- s$coefficients["log_area", "Std. Error"]
  p <- s$coefficients["log_area", "Pr(>|t|)"]
  r2 <- s$r.squared
  corrected <- exp(log(y) - slope * (log_area - log_ref))
  list(slope = slope, se = se, r2 = r2, p = p, corrected = corrected)
}

out <- data.frame(PlotObservationID = d$PlotObservationID)
for (q in c("hill_q0", "hill_q1", "hill_q2")) {
  y <- d[[q]]
  if (any(y <= 0)) {
    stop(sprintf("%s tiene valores <= 0 -- log() invalido, revisar antes de continuar", q))
  }
  fit <- fit_and_correct(y, log_area, log_ref)
  cat(sprintf("  %-8s %10.4f %10.4f %10.4f %8.2e\n",
             q, fit$slope, fit$se, fit$r2, fit$p))
  out[[paste0(q, "_sar")]] <- fit$corrected
}

cat("\nChequeo de sanidad -- la correccion debe: (a) seguir correlacionada con el crudo\n",
   "(es un reescalado, no una metrica nueva), (b) tener correlacion ~0 con log(area)\n",
   "(eso es lo que confirma que el efecto de area se removio):\n\n")
for (q in c("hill_q0", "hill_q1", "hill_q2")) {
  raw <- d[[q]]
  corrected <- out[[paste0(q, "_sar")]]
  cat(sprintf("  %-8s  cor(crudo, corregido)=%.3f   cor(corregido, log_area)=%+.3f\n",
             q, cor(raw, corrected), cor(corrected, log_area)))
}

arrow::write_parquet(out, opt$out)
cat(sprintf("\nescrito %s (%d parcelas, %d columnas)\n", opt$out, nrow(out), ncol(out) - 1))
