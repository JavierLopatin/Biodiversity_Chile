#!/usr/bin/env Rscript
# Diversidad filogenetica: arbol, respuestas por parcela y curvas de acumulacion.
#
# Cierra la faceta que `07_compute_taxonomic_beta_responses.R` dejo pendiente por no haber
# "ninguna filogenia referenciada en este repo". El paper de Parcelas-CL reporta diversidad
# filogenetica en su Fig. 4c pero no documenta el arbol ni el metodo, asi que se construye
# uno aqui y se deja explicito de donde sale.
#
# El arbol: megaarbol `GBOTB.extended.TPL` de V.PhyloMaker2 (74.529 tips, backbone GBOTB de
# Smith & Brown 2018 extendido con Zanne et al. 2014, nomenclatura The Plant List). Las
# especies ausentes se injertan con el escenario S3 -- punto medio entre el nodo basal y la
# corona del genero.
#
# Cuanto de esto es informacion real, medido sobre nuestras especies: de los 501 binomios a
# rango de especie, 271 (54%) estan en el megaarbol con posicion propia y el resto se injerta
# por genero. Suena mal, pero la incertidumbre del injerto resulta irrelevante a esta escala:
# con 10 replicas del escenario estocastico S2 la correlacion entre replicas es 1,0000 para
# PD, MPD y MNTD, y el coeficiente de variacion por parcela es 0,0000 en la mediana (p95 de
# 0,0017 para MNTD). Dentro de un genero las ramas son cortas frente a las distancias entre
# familias, y con mediana de 5 especies por parcela lo que domina es que linajes profundos
# estan presentes, no donde cae exactamente cada especie dentro de su genero.
#
# El punto debil real son 27 generos ausentes de LOS TRES megaarboles, en buena parte
# endemismos chilenos (Bridgesia, Lapageria, Retanilla, Trevoa, Llagunoa, Podanthus,
# Laureliopsis, Archidasyphyllum): un arbol global armado desde GenBank esta mal muestreado
# justo donde esta flora es mas distintiva. Parte se recupera mapeando sinonimos, porque
# Parcelas-CL estandarizo con WFO/WCVP actuales y The Plant List esta congelado en 2013
# (Neltuma->Prosopis, Leucostele->Echinopsis, Temu->Blepharocalyx, Hesperocyparis->Cupressus,
# Jarava->Stipa). El script aplica ese mapeo y reporta lo que queda fuera.
#
# Que targets salen de aqui, y cual NO:
#   PD de Faith  -- correlaciona 0,966 con la riqueza, asi que hereda entera su patologia:
#                   bajo el esquema primario `kfold5_window` la alfa alcanza R2 = +0,569 pero
#                   el 88% de eso es acertar el nivel del contribuyente, y solo +0,063 es
#                   intra-contribuyente (docs/10_findings.md seccion 1b). Se calcula y se
#                   guarda, pero como descriptor, no como target.
#   MPD          -- correlaciona 0,076 con la riqueza. Casi ortogonal: informacion nueva.
#   MNTD         -- correlaciona -0,465. Parcialmente independiente.
#   SES de las tres -- lo que separa "hay muchas especies" de "hay muchos linajes distintos".
#
# Las curvas de acumulacion replican la Fig. 4b,c del paper de Parcelas-CL y anaden la
# comparacion que ahi no existe: el subset de 1.082 parcelas de este proyecto contra las
# 1.485 del dataset completo. Responde si filtrar por Chile central, ano >= 1999 y coordenada
# unica costo representatividad taxonomica o filogenetica.
#
# Uso:
#   Rscript scripts/25_compute_phylo_responses.R
#   Rscript scripts/25_compute_phylo_responses.R --runs 199 --reps 50   # mas rapido

suppressWarnings(suppressMessages({
  library(ape); library(picante); library(arrow)
  # se ADJUNTA, no se usa `::`: los megaarboles son argumentos por defecto de phylo.maker()
  # y solo se resuelven con el paquete en el search path
  library(V.PhyloMaker2)
}))

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP     <- "data/20602096.zip"
PLOTS   <- "data/derived/plots_subset.parquet"
OUT_DIR <- "data/derived"
N_RUNS  <- getarg("--runs", 499)   # aleatorizaciones del modelo nulo para los SES
N_REPS  <- getarg("--reps", 100)   # replicas por punto de la curva de acumulacion
SEED    <- 42

# Sinonimos: Parcelas-CL usa WFO/WCVP actuales, el megaarbol usa The Plant List (2013).
# Sin este mapeo estos generos se pierden enteros aunque su linaje SI este en el arbol.
SYNONYMS <- c(
  Neltuma        = "Prosopis",
  Leucostele     = "Echinopsis",
  Temu           = "Blepharocalyx",
  Hesperocyparis = "Cupressus",
  Jarava         = "Stipa"
)

# Familias de los generos que NO estan en ningun megaarbol de V.PhyloMaker2. Sin familia,
# `phylo.maker` no tiene donde colgarlos y los descarta en silencio -- y son justo los
# endemismos chilenos, es decir el material mas distintivo de esta flora. Con familia, S3
# los injerta en el nodo basal de su familia: la posicion es gruesa, pero el linaje entra.
#
# Comprobado que los 27 estan genuinamente ausentes de TPL, WP y LCVP (WP solo recupera
# Archidasyphyllum, Phycella y Podagrostis, que por eso no aparecen aqui).
#
# ATENCION: asignadas a mano y NO verificadas contra una autoridad taxonomica. El script
# las imprime al correr para que un botanico las revise. Un error aqui no rompe nada
# visible -- solo cuelga un linaje del clado equivocado.
FAMILY_MANUAL <- c(
  Anisomeria     = "Phytolaccaceae",
  Aristeguietia  = "Asteraceae",
  Bridgesia      = "Sapindaceae",
  Corynabutilon  = "Malvaceae",
  Crocosmia      = "Iridaceae",
  Diostea        = "Verbenaceae",
  Flourensia     = "Asteraceae",
  Glenniea       = "Sapindaceae",
  Hemionitis     = "Pteridaceae",       # helecho; el megaarbol trae 520 tips de helechos
  Lapageria      = "Philesiaceae",
  Laureliopsis   = "Atherospermataceae",
  Lithraea       = "Anacardiaceae",     # SI esta en el arbol, pero falta en tips.info.TPL
  Llagunoa       = "Sapindaceae",
  Microphyes     = "Caryophyllaceae",
  Nierembergia   = "Solanaceae",
  Ophryosporus   = "Asteraceae",
  Ovidia         = "Thymelaeaceae",
  Peyritschia    = "Poaceae",
  Pleocarphus    = "Asteraceae",
  Podanthus      = "Asteraceae",
  Relchela       = "Poaceae",
  Retanilla      = "Rhamnaceae",
  Rumex          = "Polygonaceae",      # ausente del megaarbol pese a ser cosmopolita
  Spinoliva      = "Asteraceae",
  Synammia       = "Polypodiaceae",     # helecho
  Trevoa         = "Rhamnaceae"
)

# --------------------------------------------------------------------------------------
# 1. lista de especies
# --------------------------------------------------------------------------------------

message("== lista de especies ==")
raw <- read.csv(unz(ZIP, "Parcelas_CL.csv"), stringsAsFactors = FALSE)
plots_sub <- read_parquet(PLOTS)

raw$binom <- sub("^(\\S+\\s+\\S+).*$", "\\1", raw$Accepted_species)
raw$genus <- sub("\\s.*$", "", raw$binom)
# el sinonimo se aplica al genero Y al binomio, para que el epiteto sobreviva.
# `sub()` NO vectoriza `replacement`: usa solo el primer elemento y renombra todo al mismo
# genero en silencio. Hay que sustituir el genero como cadena, no con una regex.
hit <- raw$genus %in% names(SYNONYMS)
if (any(hit)) {
  new_genus <- unname(SYNONYMS[raw$genus[hit]])
  epithet <- sub("^\\S+\\s+", "", raw$binom[hit])
  raw$binom[hit] <- paste(new_genus, epithet)
  raw$genus[hit] <- new_genus
  message(sprintf("  sinonimos aplicados a %d registros de %d generos: %s",
                  sum(hit), length(unique(new_genus)),
                  paste(sort(unique(names(SYNONYMS)[match(new_genus, SYNONYMS)])),
                        collapse = ", ")))
}

# solo taxones a rango de especie o inferior: un registro a rango de familia o genero no
# tiene posicion en el arbol y contarlo como un tip inventaria un linaje
ok_rank <- raw$Accepted_name_rank %in% c("species", "variety", "subspecies")
sp <- unique(data.frame(species = raw$binom[ok_rank], genus = raw$genus[ok_rank],
                        stringsAsFactors = FALSE))
sp <- sp[grepl("\\s", sp$species), ]
message(sprintf("  %d binomios unicos (de %d taxones en Parcelas-CL)",
                nrow(sp), length(unique(raw$Accepted_species))))

data(tips.info.TPL, package = "V.PhyloMaker2", envir = environment())
g2f <- unique(tips.info.TPL[, c("genus", "family")])
g2f <- g2f[!duplicated(g2f$genus), ]
sp$family <- g2f$family[match(sp$genus, g2f$genus)]

# los que la tabla del megaarbol no cubre, desde la lista curada
need <- is.na(sp$family)
sp$family[need] <- unname(FAMILY_MANUAL[sp$genus[need]])
sp$family_manual <- need & !is.na(sp$family)
message(sprintf("  familia desde el megaarbol: %d | desde la lista manual: %d generos",
                sum(!need), length(unique(sp$genus[sp$family_manual]))))
if (any(sp$family_manual)) {
  fm <- unique(sp[sp$family_manual, c("genus", "family")])
  message("  >> REVISAR estas asignaciones manuales de familia:")
  message("     ", paste(sprintf("%s=%s", fm$genus, fm$family), collapse = "  "))
}

lost <- sp[is.na(sp$family), ]
if (nrow(lost)) {
  message(sprintf("  DESCARTADAS por falta de familia: %d especies en %d generos",
                  nrow(lost), length(unique(lost$genus))))
  message("    ", paste(sort(unique(lost$genus)), collapse = ", "))
  message("    (anadelos a FAMILY_MANUAL si su linaje debe entrar al arbol)")
}
sp <- sp[!is.na(sp$family), ]

# --------------------------------------------------------------------------------------
# 2. arbol
# --------------------------------------------------------------------------------------

message("\n== arbol (V.PhyloMaker2, GBOTB.extended.TPL, escenario S3) ==")
set.seed(SEED)
t0 <- Sys.time()
built <- phylo.maker(sp.list = sp[, c("species", "genus", "family")], scenarios = "S3")
tree <- built$scenario.3
message(sprintf("  %d tips en %.1f min", Ntip(tree),
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))
st <- table(built$species.list$status)
message(sprintf("  posicion propia del megaarbol: %d   injertadas por genero: %d (%.0f%%)",
                st[["prune"]], st[["bind"]], 100 * st[["bind"]] / sum(st)))
write.tree(tree, file.path(OUT_DIR, "phylo_tree.tre"))
write.csv(built$species.list, file.path(OUT_DIR, "phylo_species_status.csv"),
          row.names = FALSE)

# --------------------------------------------------------------------------------------
# 3. matrices de comunidad
# --------------------------------------------------------------------------------------

raw$tip <- gsub(" ", "_", raw$binom)
in_tree <- raw$tip %in% tree$tip.label
message(sprintf("\n== comunidad ==\n  registros con tip en el arbol: %d de %d (%.1f%%)",
                sum(in_tree), nrow(raw), 100 * mean(in_tree)))

comm_full <- (table(raw$PlotObservationID[in_tree], raw$tip[in_tree]) > 0) * 1
keep <- rownames(comm_full) %in% plots_sub$PlotObservationID
comm_sub <- comm_full[keep, , drop = FALSE]
comm_sub <- comm_sub[, colSums(comm_sub) > 0, drop = FALSE]
message(sprintf("  Parcelas-CL completo: %d parcelas x %d especies",
                nrow(comm_full), ncol(comm_full)))
message(sprintf("  subset del proyecto:  %d parcelas x %d especies (de %d en plots_subset)",
                nrow(comm_sub), ncol(comm_sub), nrow(plots_sub)))

# --------------------------------------------------------------------------------------
# 4. respuestas por parcela
# --------------------------------------------------------------------------------------

message("\n== respuestas por parcela ==")
cph <- cophenetic(tree)

# `include.root = FALSE` no es cosmetico. Con TRUE, picante::pd() llama a node.age() sobre el
# subarbol podado a las especies de cada comunidad, y ese subarbol puede quedar sin raiz
# aunque el arbol completo si la tenga -- basta una comunidad cuya poda deje la raiz con un
# solo hijo. Con 199 aleatorizaciones eso ocurre casi seguro y aborta ses.pd(). Con FALSE se
# suma solo el largo de rama que conecta a las especies presentes, sin el camino hasta la
# raiz, que es ademas la definicion mas comun para comparar comunidades dentro de un arbol.
# Observado y nulo usan el mismo criterio, que es lo que hace que el SES signifique algo.
PD_ROOT <- FALSE
pdv <- picante::pd(comm_sub, tree, include.root = PD_ROOT)

# SES: "taxa.labels" baraja las etiquetas del arbol, que es el nulo que pregunta si los
# linajes de la parcela estan mas o menos emparentados de lo esperado A IGUAL RIQUEZA.
# Es justamente lo que descuenta la correlacion 0,97 entre PD y riqueza.
set.seed(SEED)
message(sprintf("  SES con %d aleatorizaciones (modelo nulo taxa.labels)...", N_RUNS))
ses_pd   <- picante::ses.pd(comm_sub, tree, null.model = "taxa.labels", runs = N_RUNS,
                            include.root = PD_ROOT)
ses_mpd  <- picante::ses.mpd(comm_sub, cph, null.model = "taxa.labels", runs = N_RUNS)
ses_mntd <- picante::ses.mntd(comm_sub, cph, null.model = "taxa.labels", runs = N_RUNS)

resp <- data.frame(
  PlotObservationID = rownames(comm_sub),
  pd_faith  = pdv$PD,
  n_sp_tree = pdv$SR,
  mpd       = picante::mpd(comm_sub, cph),
  mntd      = picante::mntd(comm_sub, cph),
  ses_pd    = ses_pd$pd.obs.z,
  ses_mpd   = ses_mpd$mpd.obs.z,
  ses_mntd  = ses_mntd$mntd.obs.z,
  p_ses_pd  = ses_pd$pd.obs.p,
  p_ses_mpd = ses_mpd$mpd.obs.p,
  p_ses_mntd= ses_mntd$mntd.obs.p,
  stringsAsFactors = FALSE
)
# las parcelas del subset sin ninguna especie en el arbol quedan NA, nunca se imputan ni se
# descartan: es el mismo criterio que usa el script 07 con el estrato de cobertura
resp <- merge(data.frame(PlotObservationID = plots_sub$PlotObservationID),
              resp, by = "PlotObservationID", all.x = TRUE)
write_parquet(resp, file.path(OUT_DIR, "phylo_responses.parquet"))
message(sprintf("  -> phylo_responses.parquet  %d filas, %d sin cobertura filogenetica",
                nrow(resp), sum(is.na(resp$pd_faith))))

message("\n  correlacion con la riqueza (el criterio para elegir targets):")
for (v in c("pd_faith", "mpd", "mntd", "ses_pd", "ses_mpd", "ses_mntd")) {
  message(sprintf("    %-9s r = %+.3f", v,
                  cor(resp[[v]], resp$n_sp_tree, use = "complete.obs")))
}

# --------------------------------------------------------------------------------------
# 5. curvas de acumulacion
# --------------------------------------------------------------------------------------

# Remuestreo Monte Carlo de PARCELAS, que es lo que mide el esfuerzo de muestreo de un
# dataset compilado, y es lo que hace la Fig. 4b,c de Parcelas-CL. No es rarefaccion por
# individuos: aqui la unidad de esfuerzo es la parcela.
accum <- function(cm, tr, reps, label, seed = SEED) {
  n <- nrow(cm)
  ms <- unique(round(exp(seq(log(1), log(n), length.out = 40))))
  set.seed(seed)
  out <- do.call(rbind, lapply(ms, function(m) {
    v <- replicate(reps, {
      idx <- sample.int(n, m)
      pool <- colnames(cm)[colSums(cm[idx, , drop = FALSE]) > 0]
      c(length(pool),
        # mismo criterio de raiz que las respuestas por parcela: solo el largo de rama
        # que conecta a las especies presentes
        if (length(pool) > 1) sum(keep.tip(tr, pool)$edge.length) else NA_real_)
    })
    data.frame(m = m,
               sr_mean = mean(v[1, ]), sr_lo = quantile(v[1, ], .025),
               sr_hi = quantile(v[1, ], .975),
               pd_mean = mean(v[2, ], na.rm = TRUE),
               pd_lo = quantile(v[2, ], .025, na.rm = TRUE),
               pd_hi = quantile(v[2, ], .975, na.rm = TRUE))
  }))
  out$dataset <- label
  out
}

message(sprintf("\n== curvas de acumulacion (%d replicas por punto) ==", N_REPS))
curves <- rbind(
  accum(comm_full, tree, N_REPS, "Parcelas-CL completo"),
  accum(comm_sub,  tree, N_REPS, "subset del proyecto")
)
write.csv(curves, file.path(OUT_DIR, "rarefaction_curves.csv"), row.names = FALSE)

for (d in unique(curves$dataset)) {
  z <- curves[curves$dataset == d, ]
  last <- z[nrow(z), ]
  # pendiente en el ultimo tramo, en escala log-log: 0 seria saturacion completa
  z2 <- tail(z, 5)
  sl_sr <- coef(lm(log(sr_mean) ~ log(m), z2))[2]
  sl_pd <- coef(lm(log(pd_mean) ~ log(m), z2))[2]
  message(sprintf("  %-22s n=%4d  especies=%3.0f  PD=%7.0f  pendiente final log-log: SR %.3f  PD %.3f",
                  d, last$m, last$sr_mean, last$pd_mean, sl_sr, sl_pd))
}
message(sprintf("\n  -> rarefaction_curves.csv"))
message("\nlisto.")
