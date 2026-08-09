#!/usr/bin/env Rscript
# Compuertas de la parte filogenetica. Cada una guarda un modo de fallo que produce un
# numero plausible en vez de un error.
#
#   Rscript tests/test_phylo.R
#
# El test 1 tarda ~1,5 min porque corre iNEXT.3D como referencia, que es justamente lo
# lento que motiva la reimplementacion.

suppressWarnings(suppressMessages({
  library(ape); library(arrow); library(iNEXT.3D); library(V.PhyloMaker2)
}))
source("scripts/lib/parcelas_comm.R")
source("scripts/lib/pd_inext.R")

ok <- 0L; failed <- character()
check <- function(label, expr) {
  res <- tryCatch(isTRUE(expr), error = function(e) {
    message("    error: ", conditionMessage(e)); FALSE })
  if (res) { ok <<- ok + 1L; message(sprintf("  ok    %s", label)) }
  else { failed <<- c(failed, label); message(sprintf("  FALLA %s", label)) }
}

tree <- read.tree("data/derived/phylo_tree.tre")
res <- parcelas_species("data/20602096.zip", quiet = TRUE)
plots_sub <- read_parquet("data/derived/plots_subset.parquet")
cm <- parcelas_comm(res$raw, tree, plots_sub$PlotObservationID, quiet = TRUE)

# --------------------------------------------------------------------------------------
message("\n== reproduccion de Parcelas-CL ==")

# El paper reporta 675 "especies", pero eso cuenta 54 registros a rango de genero y 7 a
# rango de familia. A rango de especie o inferior, colapsando al binomio, hay 601. Si este
# numero cambia, o el dataset se actualizo o la resolucion de sinonimos se rompio.
check("601 binomios unicos", nrow(res$sp) + 0L == 601L)
check("1485 parcelas en el dataset completo", nrow(cm$full) == 1485L)
check("1082 parcelas en el subset", nrow(cm$sub) == 1082L)
check("el arbol cubre >= 97% de los registros",
      mean(res$raw$tip %in% tree$tip.label) >= 0.97)
check("ninguna especie del subset falta en el arbol",
      all(colnames(cm$sub) %in% tree$tip.label))

# --------------------------------------------------------------------------------------
message("\n== rarefaccion de PD contra iNEXT.3D ==")

# La compuerta que autoriza a usar `pd_curve()` en vez de iNEXT.3D. Sin esto la ganancia de
# velocidad no vale nada: seria una curva rapida y posiblemente equivocada.
v <- validate_pd_inext(cm$full, tree, n_units = 100)
message(sprintf("    error relativo maximo = %.2e sobre %d tamanos",
                v$max_rel_error, nrow(v$table)))
check("pd_curve reproduce iNEXT.3D a precision de maquina", v$max_rel_error < 1e-8)

# --------------------------------------------------------------------------------------
message("\n== coherencia interna de la curva ==")

cur <- pd_curve(cm$full, tree)
obs <- cur[cur$method == "Observed", ]
check("la curva es monotona creciente", all(diff(cur$pd) >= -1e-9))
check("el punto observado es la PD total del ensamble",
      abs(obs$pd - sum(branch_incidence(cm$full, tree)$L)) < 1e-6)
check("meanPD = PD / profundidad del arbol",
      abs(obs$meanpd - obs$pd / reference_time(tree, colnames(cm$full))) < 1e-9)
# con una sola parcela la PD esperada tiene que ser mucho menor que la total: si salieran
# parecidas es que la interpolacion esta devolviendo el total en todos los tamanos
check("PD(t=1) << PD(t=T)", cur$pd[cur$n == 1] < 0.25 * obs$pd)
check("la extrapolacion supera lo observado",
      max(cur$pd[cur$method == "Extrapolation"]) > obs$pd)

# --------------------------------------------------------------------------------------
message("\n== banda de submuestreo ==")

# El fallo que costo dos intentos: bandas que no contenian su propia linea. El bootstrap de
# parcelas con reemplazo y el parametrico con Y/T fallan los dos aqui.
band <- subsample_band(cm$full, tree, cur$n, reps = 60)
i <- match(band$n, cur$n)
check("la banda de PD contiene la curva en todos los puntos",
      all(band$pd_lo[-1] <= cur$pd[i][-1] + 1e-6 & cur$pd[i][-1] <= band$pd_hi[-1] + 1e-6))
check("la banda se estrecha al crecer el esfuerzo",
      (band$pd_hi - band$pd_lo)[nrow(band)] < (band$pd_hi - band$pd_lo)[2])

# --------------------------------------------------------------------------------------
message(sprintf("\n%d pasaron, %d fallaron", ok, length(failed)))
if (length(failed)) { message("  ", paste(failed, collapse = "\n  ")); quit(status = 1) }
