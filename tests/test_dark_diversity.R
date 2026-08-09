#!/usr/bin/env Rscript
# Compuertas de la dark diversity.
#
#   Rscript tests/test_dark_diversity.R
#
# Dos de estos tests fijan RESULTADOS NEGATIVOS a proposito (la completitud es riqueza
# reetiquetada; la PD oscura es el conteo de especies oscuras otra vez). Estan aqui para que
# nadie los "arregle" mas adelante creyendo que son un bug: son propiedades del estimador
# sobre estos datos y la decision de no usarlos como target depende de ellas.

suppressWarnings(suppressMessages({
  library(ape); library(arrow); library(DarkDiv); library(V.PhyloMaker2)
}))
source("scripts/lib/parcelas_comm.R")

ok <- 0L; failed <- character()
check <- function(label, expr) {
  res <- tryCatch(isTRUE(expr), error = function(e) {
    message("    error: ", conditionMessage(e)); FALSE })
  if (res) { ok <<- ok + 1L; message(sprintf("  ok    %s", label)) }
  else { failed <<- c(failed, label); message(sprintf("  FALLA %s", label)) }
}
rho <- function(a, b) cor(a, b, method = "spearman", use = "complete.obs")

tree <- read.tree("data/derived/phylo_tree.tre")
res <- parcelas_species("data/20602096.zip", quiet = TRUE)
plots_sub <- read_parquet("data/derived/plots_subset.parquet")
cm <- parcelas_comm(res$raw, tree, plots_sub$PlotObservationID, quiet = TRUE)
comm <- cm$full
obs <- rowSums(comm)

dd <- DarkDiv(comm, method = "Hypergeometric")
pr <- dd$Dark
dark_bin <- !is.na(pr) & pr > 0.9
dark_n <- rowSums(dark_bin)

# --------------------------------------------------------------------------------------
message("\n== consistencia estructural ==")

# una especie presente no puede estar en la sombra: si esto falla, se estan contando
# presencias como ausencias y todo lo demas es ruido
check("ninguna especie presente aparece como oscura", !any(dark_bin & comm > 0))
check("el pool nunca excede el total de especies", all(obs + dark_n <= ncol(comm)))
check("toda parcela tiene alguna especie oscura", all(dark_n > 0))

# --------------------------------------------------------------------------------------
message("\n== lo que hace a dark_n un target usable ==")

# El criterio del proyecto: una faceta nueva tiene que ser casi ortogonal a la riqueza, o no
# aporta nada que hill_q0 no diga ya.
r <- rho(dark_n, obs)
message(sprintf("    rho(dark_n, riqueza) = %+.3f", r))
check("dark_n es casi ortogonal a la riqueza observada", abs(r) < 0.30)

# NEGATIVO FIJADO: la completitud es riqueza. No es un bug del calculo -- la riqueza esta en
# el numerador y la dark diversity apenas varia, asi que el cociente es monotono en riqueza.
compl <- log(obs / rowSums(pr, na.rm = TRUE))
rc <- rho(compl, obs)
message(sprintf("    rho(completitud, riqueza) = %+.3f  (se espera > 0,9)", rc))
check("la completitud SIGUE siendo riqueza reetiquetada", rc > 0.9)

# --------------------------------------------------------------------------------------
message("\n== el umbral importa, la suma de probabilidades no sirve ==")

check("sumar probabilidades da un pool inverosimil (> 200 especies para parcelas de 5)",
      median(rowSums(pr, na.rm = TRUE)) > 200)
check("con umbral el conteo baja a un orden defendible", median(dark_n) < 100)
# por encima de 0,8 la conclusion no depende del umbral elegido
rs <- vapply(c(0.8, 0.9, 0.95), function(t) rho(rowSums(!is.na(pr) & pr > t), obs), numeric(1))
check("la ortogonalidad es estable entre umbrales 0,8-0,95", all(abs(rs) < 0.25))

# --------------------------------------------------------------------------------------
message("\n== coherencia espacial contra linea base ==")

# La validacion que decide si el pool es local o una lista nacional. El numero absoluto no
# sirve: hay que compararlo con una ausente cualquiera.
meta <- unique(res$raw[, c("PlotObservationID", "X", "Y")])
meta <- meta[match(rownames(comm), meta$PlotObservationID), ]
xy <- as.matrix(meta[, c("X", "Y")])
set.seed(1)
idx <- sample.int(nrow(comm), 300)      # muestra, para que el test tarde segundos
r2 <- vapply(idx, function(i) {
  d <- sqrt((xy[, 1] - xy[i, 1])^2 + (xy[, 2] - xy[i, 2])^2)
  lp <- colSums(comm[d <= 50000, , drop = FALSE]) > 0
  dk <- which(dark_bin[i, ])
  if (!length(dk)) return(c(NA_real_, NA_real_))
  c(mean(lp[dk]), mean(lp[comm[i, ] == 0]))
}, numeric(2))
ratio <- median(r2[1, ], na.rm = TRUE) / median(r2[2, ], na.rm = TRUE)
message(sprintf("    a 50 km: oscuras %.1f%% vs ausentes %.1f%%  ->  %.1fx",
                100 * median(r2[1, ], na.rm = TRUE), 100 * median(r2[2, ], na.rm = TRUE), ratio))
check("las especies oscuras estan mas cerca que una ausente cualquiera", ratio > 2)

# --------------------------------------------------------------------------------------
message(sprintf("\n%d pasaron, %d fallaron", ok, length(failed)))
if (length(failed)) { message("  ", paste(failed, collapse = "\n  ")); quit(status = 1) }
