#!/usr/bin/env Rscript
# Compuertas de la curva de particion de Hill (alfa/beta/gamma) por remuestreo.
#
# Sin formula que portar ni validar contra una referencia externa (a diferencia de
# tests/test_phylo.R): `hillR::hill_taxa_parti` ya es un paquete instalado y en uso en el
# proyecto, no una reimplementacion. Estas compuertas chequean coherencia matematica y del
# remuestreo, mismo espiritu que las de `subsample_band()` en test_phylo.R.
#
#   Rscript tests/test_beta.R

suppressWarnings(suppressMessages({
  library(arrow); library(hillR)
}))
source("scripts/lib/parcelas_comm.R")
source("scripts/lib/beta_freq.R")

ok <- 0L; failed <- character()
check <- function(label, expr) {
  res <- tryCatch(isTRUE(expr), error = function(e) {
    message("    error: ", conditionMessage(e)); FALSE })
  if (res) { ok <<- ok + 1L; message(sprintf("  ok    %s", label)) }
  else { failed <<- c(failed, label); message(sprintf("  FALLA %s", label)) }
}

# --------------------------------------------------------------------------------------
message("\n== correccion Zamorano ==")

pc <- parcelas_species("data/20602096.zip", quiet = TRUE)
zam_before <- pc$raw[pc$raw$Owner == "Zamorano-Elgueta, C.", "Value"]
check("antes de corregir, Value de Zamorano es degenerado (siempre 1.0)",
      length(unique(zam_before)) == 1 && unique(zam_before) == 1.0)

pc$raw <- apply_zamorano_correction(pc$raw, "data/derived/zamorano_cover_corrected.parquet",
                                    quiet = TRUE)
zam_after <- pc$raw[pc$raw$Owner == "Zamorano-Elgueta, C.", "Value"]
# el crudo usa clases de cobertura (multiplos de 10 mayormente), no un continuo -- unos 10
# valores distintos es lo esperable, no >50 (eso hubiera sido pedir algo que el dato de
# origen ni siquiera tiene)
check("tras corregir, Zamorano recupera un gradiente real de cobertura (no ya un unico valor)",
      length(unique(zam_after)) >= 8)
check("al menos 90% de los pares parcela-especie de Zamorano dejaron de ser 1.0",
      mean(zam_after != 1.0) >= 0.9)
check("la correccion no cambio el total de filas (10821)", nrow(pc$raw) == 10821)

pc_fresh <- parcelas_species("data/20602096.zip", quiet = TRUE)
own <- pc$raw$Owner != "Zamorano-Elgueta, C."
check("Value de los demas duenos no se toco",
      isTRUE(all.equal(pc$raw$Value[own], pc_fresh$raw$Value[own])))

# --------------------------------------------------------------------------------------
message("\n== pool de frecuencia ==")

comm <- build_freq_pool(pc$raw, "data/derived/living_trees_long.parquet", quiet = TRUE)
check("toda fila del pool relativiza (ninguna fila de solo ceros)",
      all(rowSums(comm) > 0))
check("ninguna columna de solo ceros (serian especies fantasma)",
      all(colSums(comm) > 0))
check("el pool mezcla Parcelas-CL (prefijo PCL_) y Living Trees (prefijo LT_)",
      any(grepl("^PCL_", rownames(comm))) && any(grepl("^LT_", rownames(comm))))

# --------------------------------------------------------------------------------------
message("\n== particion de Hill (todo el pool, sin submuestreo) ==")

full0 <- hill_parti_q(comm, q = 0)
full1 <- hill_parti_q(comm, q = 1)
full2 <- hill_parti_q(comm, q = 2)
check("gamma(q=0) es la riqueza total del pool",
      abs(full0$gamma - ncol(comm)) < 1e-6)
check("beta = gamma / alfa (definicion multiplicativa)",
      abs(full0$beta - full0$gamma / full0$alpha) < 1e-6 &&
      abs(full1$beta - full1$gamma / full1$alpha) < 1e-6 &&
      abs(full2$beta - full2$gamma / full2$alpha) < 1e-6)
check("Hill de alfa es monotono decreciente en q (q0 >= q1 >= q2)",
      full0$alpha >= full1$alpha - 1e-6 && full1$alpha >= full2$alpha - 1e-6)
check("Hill de gamma es monotono decreciente en q (q0 >= q1 >= q2)",
      full0$gamma >= full1$gamma - 1e-6 && full1$gamma >= full2$gamma - 1e-6)
check("beta esta acotado entre 1 (sin turnover) y N (turnover total)",
      full0$beta >= 1 - 1e-6 && full0$beta <= nrow(comm) + 1e-6)

# --------------------------------------------------------------------------------------
message("\n== curva de remuestreo ==")

set.seed(1)
sizes <- c(20, 50, 150, 400)
curve <- beta_freq_curve(comm, sizes, q = c(0, 1, 2), reps = 40, seed = 7)

check("la curva trae los 4 tamanos x 3 ordenes de q pedidos",
      nrow(curve) == length(sizes) * 3)
d0 <- curve[curve$q == 0, ]; d0 <- d0[order(d0$n), ]
check("gamma crece con n (mas parcelas pooleadas acumulan mas riqueza)",
      all(diff(d0$gamma_mean) >= -1e-6))
check("beta crece con n (mas turnover capturado al crecer el pool)",
      all(diff(d0$beta_mean) >= -1e-6))
check("la media cae dentro de su propia banda [lo, hi] en todo tamano",
      all(d0$beta_mean >= d0$beta_lo - 1e-6 & d0$beta_mean <= d0$beta_hi + 1e-6))
# el ancho ABSOLUTO de la banda puede crecer con n (beta mismo sigue creciendo rapido en
# este rango, lejos de la asintota del pool completo -- a diferencia de una curva de
# riqueza/PD que sí se achata) -- lo que se angosta es el ancho RELATIVO (banda / media),
# igual que el coeficiente de variacion de cualquier estimador que se estabiliza.
rel_width <- (d0$beta_hi - d0$beta_lo) / d0$beta_mean
check("el ancho relativo de la banda de beta se angosta al crecer n",
      rel_width[nrow(d0)] < rel_width[1])

full_size <- nrow(comm)
curve_full <- beta_freq_curve(comm, full_size, q = 0, reps = 5, seed = 7)
check("al tamano N=pool completo, la curva reproduce la particion sin submuestreo",
      abs(curve_full$beta_mean - full0$beta) < 1e-6 &&
      abs(curve_full$gamma_mean - full0$gamma) < 1e-6)

# --------------------------------------------------------------------------------------
message(sprintf("\n%d pasaron, %d fallaron", ok, length(failed)))
if (length(failed)) { message("  ", paste(failed, collapse = "\n  ")); quit(status = 1) }
