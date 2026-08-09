# Resolucion taxonomica de Parcelas-CL y construccion de las matrices de incidencia.
#
# Vive aparte porque lo usan dos scripts (25_compute_phylo_responses.R construye el arbol y
# las respuestas por parcela; 26_rarefaction_inext.R las curvas de acumulacion) y duplicar
# la resolucion de sinonimos y familias garantizaria que en algun momento diverjan.
#
# La cifra que reconcilia con el paper: Parcelas-CL declara 675 "especies", pero eso cuenta
# 54 registros determinados solo a genero y 7 solo a familia. A rango de especie hay 597
# taxones, y colapsando subespecies y variedades al binomio quedan 601 binomios unicos --
# ese es el numero que un arbol puede representar. Un registro a rango de genero no tiene
# posicion en el arbol y darle un tip inventaria un linaje que nadie observo.

# Sinonimos: Parcelas-CL usa WFO/WCVP actuales, el megaarbol usa The Plant List (2013).
# Sin este mapeo estos generos se pierden enteros aunque su linaje SI este en el arbol.
SYNONYMS <- c(
  Neltuma        = "Prosopis",
  Leucostele     = "Echinopsis",
  Temu           = "Blepharocalyx",
  Hesperocyparis = "Cupressus",
  Jarava         = "Stipa"
)

# Familias de los generos que NO estan en la tabla de ningun megaarbol de V.PhyloMaker2.
# Sin familia, `phylo.maker` no tiene donde colgarlos y los descarta en silencio -- y son
# justo los endemismos chilenos, es decir el material mas distintivo de esta flora. Con
# familia, S3 los injerta en el nodo basal de su familia: la posicion es gruesa, pero el
# linaje entra.
#
# Comprobado: ninguno aparece en tips.info.TPL, .WP ni .LCVP. Archidasyphyllum, Phycella y
# Podagrostis SI son tips de GBOTB.extended.WP, pero ese arbol no es el que usamos y su
# tabla de familias tampoco los trae, asi que igual necesitan asignacion manual.
#
# ATENCION: asignadas a mano y NO verificadas contra una autoridad taxonomica. El script
# las imprime al correr para que un botanico las revise. Un error aqui no rompe nada
# visible -- solo cuelga un linaje del clado equivocado.
FAMILY_MANUAL <- c(
  Anisomeria       = "Phytolaccaceae",
  Archidasyphyllum = "Asteraceae",       # segregado de Dasyphyllum (Barnadesioideae)
  Aristeguietia    = "Asteraceae",
  Bridgesia        = "Sapindaceae",
  Corynabutilon    = "Malvaceae",
  Crocosmia        = "Iridaceae",
  Diostea          = "Verbenaceae",
  Flourensia       = "Asteraceae",
  Gayella          = "Sapotaceae",       # el basonimo en el CSV es Pouteria splendens
  Glenniea         = "Sapindaceae",
  Hemionitis       = "Pteridaceae",      # helecho; el megaarbol trae 520 tips de helechos
  Lapageria        = "Philesiaceae",
  Laureliopsis     = "Atherospermataceae",
  Lithraea         = "Anacardiaceae",    # SI esta en el arbol, pero falta en tips.info.TPL
  Llagunoa         = "Sapindaceae",
  Microphyes       = "Caryophyllaceae",
  Nierembergia     = "Solanaceae",
  Ophryosporus     = "Asteraceae",
  Ovidia           = "Thymelaeaceae",
  Peyritschia      = "Poaceae",
  Phycella         = "Amaryllidaceae",
  Pleocarphus      = "Asteraceae",
  Podagrostis      = "Poaceae",          # segregado de Agrostis
  Podanthus        = "Asteraceae",
  Relchela         = "Poaceae",
  Retanilla        = "Rhamnaceae",
  Rumex            = "Polygonaceae",     # ausente del megaarbol pese a ser cosmopolita
  Spinoliva        = "Asteraceae",
  Synammia         = "Polypodiaceae",    # helecho
  Trevoa           = "Rhamnaceae"
)


#' Lee el CSV de Parcelas-CL y resuelve binomio, genero y familia.
#'
#' Devuelve una lista con `raw` (los registros, con columnas `binom`, `genus`, `tip`) y
#' `sp` (la tabla species/genus/family lista para `phylo.maker`, sin los que quedaron sin
#' familia).
parcelas_species <- function(zip, quiet = FALSE) {
  say <- function(...) if (!quiet) message(sprintf(...))

  raw <- read.csv(unz(zip, "Parcelas_CL.csv"), stringsAsFactors = FALSE)
  raw$binom <- sub("^(\\S+\\s+\\S+).*$", "\\1", raw$Accepted_species)
  raw$genus <- sub("\\s.*$", "", raw$binom)

  # el sinonimo se aplica al genero Y al binomio, para que el epiteto sobreviva.
  # `sub()` NO vectoriza `replacement`: usa solo el primer elemento y renombra todo al
  # mismo genero en silencio. Hay que sustituir el genero como cadena, no con una regex.
  hit <- raw$genus %in% names(SYNONYMS)
  if (any(hit)) {
    new_genus <- unname(SYNONYMS[raw$genus[hit]])
    epithet <- sub("^\\S+\\s+", "", raw$binom[hit])
    raw$binom[hit] <- paste(new_genus, epithet)
    raw$genus[hit] <- new_genus
    say("  sinonimos aplicados a %d registros de %d generos: %s",
        sum(hit), length(unique(new_genus)),
        paste(sort(unique(names(SYNONYMS)[match(new_genus, SYNONYMS)])), collapse = ", "))
  }
  raw$tip <- gsub(" ", "_", raw$binom)

  # solo taxones a rango de especie o inferior: un registro a rango de familia o genero no
  # tiene posicion en el arbol y contarlo como un tip inventaria un linaje
  ok_rank <- raw$Accepted_name_rank %in% c("species", "variety", "subspecies")
  sp <- unique(data.frame(species = raw$binom[ok_rank], genus = raw$genus[ok_rank],
                          stringsAsFactors = FALSE))
  sp <- sp[grepl("\\s", sp$species), ]
  say("  %d binomios unicos (de %d taxones en Parcelas-CL)",
      nrow(sp), length(unique(raw$Accepted_species)))

  data(tips.info.TPL, package = "V.PhyloMaker2", envir = environment())
  g2f <- unique(tips.info.TPL[, c("genus", "family")])
  g2f <- g2f[!duplicated(g2f$genus), ]
  sp$family <- g2f$family[match(sp$genus, g2f$genus)]

  need <- is.na(sp$family)
  sp$family[need] <- unname(FAMILY_MANUAL[sp$genus[need]])
  sp$family_manual <- need & !is.na(sp$family)
  say("  familia desde el megaarbol: %d | desde la lista manual: %d generos",
      sum(!need), length(unique(sp$genus[sp$family_manual])))
  if (any(sp$family_manual) && !quiet) {
    fm <- unique(sp[sp$family_manual, c("genus", "family")])
    message("  >> REVISAR estas asignaciones manuales de familia:")
    message("     ", paste(sprintf("%s=%s", fm$genus, fm$family), collapse = "  "))
  }

  lost <- sp[is.na(sp$family), ]
  if (nrow(lost) && !quiet) {
    message(sprintf("  DESCARTADAS por falta de familia: %d especies en %d generos",
                    nrow(lost), length(unique(lost$genus))))
    message("    ", paste(sort(unique(lost$genus)), collapse = ", "))
    message("    (anadelos a FAMILY_MANUAL si su linaje debe entrar al arbol)")
  }

  list(raw = raw, sp = sp[!is.na(sp$family), ])
}


#' Matrices de incidencia (parcela x especie, 0/1) para el dataset completo y el subset.
#'
#' Presencia/ausencia y no abundancia porque en Parcelas-CL la abundancia viene en tres
#' unidades inconmensurables -- cobertura en %, conteo de individuos y area basal -- mas un
#' 25% de parcelas sin ninguna. Ponderar mezclaria "40% de cobertura" con "40 individuos" y
#' convertiria la metrica en un artefacto del protocolo del contribuyente.
parcelas_comm <- function(raw, tree, plot_ids_subset, quiet = FALSE) {
  say <- function(...) if (!quiet) message(sprintf(...))

  in_tree <- raw$tip %in% tree$tip.label
  say("  registros con tip en el arbol: %d de %d (%.1f%%)",
      sum(in_tree), nrow(raw), 100 * mean(in_tree))

  full <- (table(raw$PlotObservationID[in_tree], raw$tip[in_tree]) > 0) * 1
  sub <- full[rownames(full) %in% plot_ids_subset, , drop = FALSE]
  sub <- sub[, colSums(sub) > 0, drop = FALSE]
  say("  Parcelas-CL completo: %d parcelas x %d especies", nrow(full), ncol(full))
  say("  subset del proyecto:  %d parcelas x %d especies (de %d en plots_subset)",
      nrow(sub), ncol(sub), length(plot_ids_subset))

  list(full = full, sub = sub)
}
