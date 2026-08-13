# Resolucion taxonomica y matrices de incidencia para la base UNIFICADA
# (Parcelas-CL + Living Trees Chile). Espejo de scripts/lib/parcelas_comm.R -- no lo
# reemplaza, lo extiende: reusa `parcelas_species()`/`FAMILY_MANUAL`/`SYNONYMS` de ahi,
# nunca duplica esas constantes.
#
# Convencion de IDs: `parcelas_species()$raw$PlotObservationID` viene SIN prefijo
# (numerico, tal cual el CSV de Parcelas-CL). El resto del proyecto unificado
# (scripts 50-53) prefija todo con "PCL_"/"LT_" para blindar contra colision -- este
# archivo prefija Parcelas-CL con "PCL_" al vuelo, para no tener que tocar
# `parcelas_species()` ni `parcelas_comm()`, que siguen siendo el pipeline vigente para
# las facetas filogenicas/dark de Parcelas-CL solo.
source("scripts/lib/parcelas_comm.R")


#' Lista de especies unificada (species/genus/family), lista para `phylo.maker()`.
#'
#' Parte de `parcelas_species()` (601 binomios de Parcelas-CL, sin restringir al subset
#' de modelado -- el pool de dark diversity tambien necesita el dataset completo) y le
#' suma los binomios de Living Trees que no eran ya el mismo binomio exacto. Las especies
#' nuevas se resuelven con la MISMA receta que `parcelas_species()`: tabla de familias de
#' `tips.info.TPL`, y si no esta ahi, `FAMILY_MANUAL`.
unified_species <- function(zip, lt_long_path, quiet = FALSE) {
  say <- function(...) if (!quiet) message(sprintf(...))

  pc <- parcelas_species(zip, quiet = quiet)
  lt <- as.data.frame(arrow::read_parquet(lt_long_path))

  lt_sp_all <- unique(lt$species)
  new_species <- setdiff(lt_sp_all, pc$sp$species)
  say("  Living Trees: %d especies, %d ya en Parcelas-CL (mismo binomio exacto), %d nuevas",
      length(lt_sp_all), length(lt_sp_all) - length(new_species), length(new_species))

  if (length(new_species) == 0) {
    return(list(pc_raw = pc$raw, sp = pc$sp))
  }

  new_sp <- data.frame(species = new_species,
                       genus = sub("\\s.*$", "", new_species),
                       stringsAsFactors = FALSE)

  data(tips.info.TPL, package = "V.PhyloMaker2", envir = environment())
  g2f <- unique(tips.info.TPL[, c("genus", "family")])
  g2f <- g2f[!duplicated(g2f$genus), ]
  new_sp$family <- g2f$family[match(new_sp$genus, g2f$genus)]

  need <- is.na(new_sp$family)
  new_sp$family[need] <- unname(FAMILY_MANUAL[new_sp$genus[need]])
  new_sp$family_manual <- need & !is.na(new_sp$family)
  say("  especies nuevas -- familia desde el megaarbol: %d | desde FAMILY_MANUAL: %d",
      sum(!need), sum(new_sp$family_manual))

  lost <- new_sp[is.na(new_sp$family), ]
  if (nrow(lost) && !quiet) {
    message(sprintf("  DESCARTADAS por falta de familia: %s",
                    paste(sort(unique(lost$genus)), collapse = ", ")))
  }
  new_sp <- new_sp[!is.na(new_sp$family), c("species", "genus", "family")]

  sp_union <- rbind(pc$sp[, c("species", "genus", "family")], new_sp)
  say("  lista unificada: %d binomios (%d Parcelas-CL + %d Living Trees nuevos)",
      nrow(sp_union), nrow(pc$sp), nrow(new_sp))

  list(pc_raw = pc$raw, sp = sp_union)
}


#' Matrices de incidencia unificadas (parcela x especie, 0/1).
#'
#' `full`: pool de co-ocurrencia completo -- Parcelas-CL SIN restringir (1.485, prefijo
#' "PCL_") + Living Trees completo (prefijo "LT_" ya aplicado en `lt_long`). Mismo
#' criterio que `scripts/27_compute_dark_diversity.R` ya usa para Parcelas-CL solo: mas
#' parcelas de co-ocurrencia dan mejor estimacion de pool, aunque no todas se reporten.
#' `sub`: solo `plot_ids_report` (el set unificado de modelado, ~3.102 parcelas).
unified_comm <- function(pc_raw, lt_long_path, tree, plot_ids_report, quiet = FALSE) {
  say <- function(...) if (!quiet) message(sprintf(...))

  lt <- as.data.frame(arrow::read_parquet(lt_long_path))
  combined <- rbind(
    data.frame(PlotObservationID = paste0("PCL_", pc_raw$PlotObservationID),
              tip = pc_raw$tip, stringsAsFactors = FALSE),
    data.frame(PlotObservationID = lt$PlotObservationID,
              tip = gsub(" ", "_", lt$species), stringsAsFactors = FALSE)
  )

  in_tree <- combined$tip %in% tree$tip.label
  say("  registros con tip en el arbol: %d de %d (%.1f%%)",
      sum(in_tree), nrow(combined), 100 * mean(in_tree))

  full <- (table(combined$PlotObservationID[in_tree], combined$tip[in_tree]) > 0) * 1
  sub <- full[rownames(full) %in% plot_ids_report, , drop = FALSE]
  sub <- sub[, colSums(sub) > 0, drop = FALSE]
  say("  pool completo: %d parcelas x %d especies", nrow(full), ncol(full))
  say("  set unificado reportado: %d parcelas x %d especies (de %d pedidas)",
      nrow(sub), ncol(sub), length(plot_ids_report))

  empty <- rowSums(sub) == 0
  if (any(empty)) {
    say(paste("  AVISO: %d parcelas del set reportado quedan con 0 especies en el arbol",
             "-- no van a resolver mpd/mntd/PCoA (mismo problema que beta_pa_unified)"),
        sum(empty))
  }

  list(full = full, sub = sub)
}
