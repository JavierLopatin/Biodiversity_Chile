# Curva de particion multiplicativa de Hill (alfa/beta/gamma, Chao/Chiu/Jost 2014) sobre el
# pool de FRECUENCIA maximizado (Parcelas-CL + Living Trees Chile).
#
# Por que "frecuencia" y no "conteo de individuos": Parcelas-CL trae 3 unidades de
# abundancia inconmensurables entre parcelas (cobertura %, conteo, area basal) -- ver el
# comentario en `parcelas_comm()` (scripts/lib/parcelas_comm.R) sobre por que eso bloqueo
# usar abundancia ahi. La resolucion: `hillR::hill_taxa_parti(rel_then_pool=TRUE)`
# RELATIVIZA CADA PARCELA POR FILA ANTES DE COMBINAR (confirmado leyendo su fuente real,
# `comm_alpha <- sweep(comm,1,rowSums(comm),"/")`) -- dentro de una misma parcela todas las
# especies se midieron con el MISMO protocolo (Abundance_parameter es por parcela, no por
# especie), asi que su proporcion relativa dentro de la fila es comparable aunque la escala
# absoluta entre parcelas no lo sea. Eso es exactamente lo que ya hace `beta_freq`
# (scripts/52) para las facetas LCBD/PCoA -- esto extiende el mismo principio a los numeros
# de Hill alfa/beta/gamma.
#
# Elegibles: toda cepa continua de Parcelas-CL (Cover, Cover_st, Abundance, Basal_area --
# se excluye solo Abundance_parameter=="NA", presencia pura sin magnitud) + Living Trees
# completo (area basal real, por arbol). Pool COMPLETO (1.485 Parcelas-CL, sin restringir
# al subset de modelado), mismo criterio que ya usan scripts/25,27,54,55,56 para dark
# diversity y las curvas de Hill taxonomicas/filogeneticas.
#
# Sin formula que portar ni validar contra una referencia externa (a diferencia de
# `pd_inext.R`): `hill_taxa_parti` ya esta instalado y en uso en el proyecto. El "curva" es
# una banda de remuestreo (mismo patron que `subsample_band()`), no una rarefaccion
# analitica -- ilustra como se estabiliza la particion alfa/beta/gamma a medida que crece el
# pool de parcelas, no una prediccion validada como las de PD.

suppressWarnings(suppressMessages(library(hillR)))


#' Corrige el `Value` degenerado de Zamorano-Elgueta, C. (ver scripts/58_recover_zamorano_cover.py).
apply_zamorano_correction <- function(raw, corrected_path, quiet = FALSE) {
  if (!file.exists(corrected_path)) return(raw)
  fix <- as.data.frame(arrow::read_parquet(corrected_path))
  m <- match(paste(raw$PlotObservationID, raw$Accepted_species),
            paste(fix$PlotObservationID, fix$Accepted_species))
  n_fixed <- sum(!is.na(m))
  raw$Value[!is.na(m)] <- fix$Value_corrected[m[!is.na(m)]]
  if (!quiet) message(sprintf("  correccion Zamorano aplicada: %d pares parcela-especie",
                              n_fixed))
  raw
}


#' Matriz parcela x especie de VALORES de frecuencia (no 0/1), pool completo unificado.
#'
#' `pc_raw`: `parcelas_species(zip)$raw`, YA con la correccion de Zamorano aplicada por el
#' llamador (evita leer el zip dos veces si el caller ya tiene `pc_raw` de otro paso).
#' `lt_long_path`: `data/derived/living_trees_long.parquet` (ya en area basal, prefijo
#' "LT_").
build_freq_pool <- function(pc_raw, lt_long_path, quiet = FALSE) {
  say <- function(...) if (!quiet) message(sprintf(...))

  pc <- pc_raw[pc_raw$Abundance_parameter != "NA", ]
  say("  Parcelas-CL con cepa continua (no presencia-pura): %d parcelas de %d",
      length(unique(pc$PlotObservationID)), length(unique(pc_raw$PlotObservationID)))
  pc_agg <- aggregate(Value ~ PlotObservationID + binom, data = pc, FUN = sum)
  pc_agg$PlotObservationID <- paste0("PCL_", pc_agg$PlotObservationID)
  names(pc_agg) <- c("PlotObservationID", "species", "Value")

  lt <- as.data.frame(arrow::read_parquet(lt_long_path))
  lt_agg <- aggregate(Value ~ PlotObservationID + species, data = lt, FUN = sum)

  combined <- rbind(pc_agg, lt_agg)
  comm <- as.data.frame.matrix(
    xtabs(Value ~ PlotObservationID + species, data = combined))
  comm <- as.matrix(comm)

  n_empty <- sum(rowSums(comm) == 0)
  if (n_empty > 0) {
    say("  parcelas con fila de ceros (excluidas -- no relativizan): %d", n_empty)
    comm <- comm[rowSums(comm) > 0, , drop = FALSE]
  }
  say("  pool de frecuencia: %d parcelas x %d especies (%d de Parcelas-CL + %d de LT)",
      nrow(comm), ncol(comm), length(unique(pc_agg$PlotObservationID)),
      length(unique(lt_agg$PlotObservationID)))
  comm
}


#' Particion alfa/beta/gamma para un q dado, sobre TODO `comm` (sin submuestreo).
hill_parti_q <- function(comm, q) {
  r <- hillR::hill_taxa_parti(comm, q = q, show_warning = FALSE)
  data.frame(q = q, n = nrow(comm), alpha = r$TD_alpha, beta = r$TD_beta,
            gamma = r$TD_gamma)
}


#' Curva empirica de particion de Hill por remuestreo creciente (analoga a
#' `subsample_band()` en pd_inext.R, pero para alfa/beta/gamma en vez de PD).
#'
#' Para cada tamano de `sizes`, `reps` submuestras sin reemplazo de esa cantidad de
#' parcelas; en cada una se corre `hill_taxa_parti` para cada q de `q`. Se reporta
#' media y percentiles (banda `conf`) de alfa/beta/gamma por tamano y orden.
beta_freq_curve <- function(comm, sizes, q = c(0, 1, 2), reps = 100, conf = 0.95,
                            seed = 42) {
  set.seed(seed)
  N <- nrow(comm)
  sizes <- sort(unique(pmin(sizes, N)))
  sizes <- sizes[sizes >= 2]  # beta no esta definido con 1 sola parcela

  alpha_lo <- (1 - conf) / 2
  rows <- list()
  for (n in sizes) {
    reps_n <- if (n == N) 1 else reps  # a n=N toda submuestra es la misma parcela set
    draws <- vector("list", reps_n)
    for (i in seq_len(reps_n)) {
      idx <- if (n == N) seq_len(N) else sample.int(N, n)
      sub <- comm[idx, , drop = FALSE]
      sub <- sub[, colSums(sub) > 0, drop = FALSE]
      draws[[i]] <- do.call(rbind, lapply(q, hill_parti_q, comm = sub))
    }
    d <- do.call(rbind, draws)
    for (qi in q) {
      dq <- d[d$q == qi, ]
      rows[[length(rows) + 1]] <- data.frame(
        n = n, q = qi, reps = reps_n,
        alpha_mean = mean(dq$alpha), alpha_lo = quantile(dq$alpha, alpha_lo, names = FALSE),
        alpha_hi = quantile(dq$alpha, 1 - alpha_lo, names = FALSE),
        beta_mean = mean(dq$beta), beta_lo = quantile(dq$beta, alpha_lo, names = FALSE),
        beta_hi = quantile(dq$beta, 1 - alpha_lo, names = FALSE),
        gamma_mean = mean(dq$gamma), gamma_lo = quantile(dq$gamma, alpha_lo, names = FALSE),
        gamma_hi = quantile(dq$gamma, 1 - alpha_lo, names = FALSE))
    }
  }
  do.call(rbind, rows)
}
