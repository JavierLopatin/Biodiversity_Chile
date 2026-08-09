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
# Cuanto de esto es informacion real, medido sobre nuestras especies: de los 601 binomios a
# rango de especie, 336 (56%) estan en el megaarbol con posicion propia y el resto se injerta
# por genero. Suena mal, pero la incertidumbre del injerto resulta irrelevante a esta escala:
# con 10 replicas del escenario estocastico S2 la correlacion entre replicas es 1,0000 para
# PD, MPD y MNTD, y el coeficiente de variacion por parcela es 0,0000 en la mediana (p95 de
# 0,0017 para MNTD). Dentro de un genero las ramas son cortas frente a las distancias entre
# familias, y con mediana de 5 especies por parcela lo que domina es que linajes profundos
# estan presentes, no donde cae exactamente cada especie dentro de su genero.
#
# El punto debil real son 30 generos ausentes de la tabla de familias de LOS TRES megaarboles,
# en buena parte endemismos chilenos (Bridgesia, Lapageria, Retanilla, Trevoa, Llagunoa,
# Podanthus, Laureliopsis, Archidasyphyllum): un arbol global armado desde GenBank esta mal
# muestreado justo donde esta flora es mas distintiva. Sin familia `phylo.maker` los descarta
# EN SILENCIO, asi que se les asigna a mano en scripts/lib/parcelas_comm.R -- 126 especies que
# de otro modo se perderian, y con ellas 41 parcelas enteras. Esas asignaciones no estan
# verificadas contra una autoridad taxonomica y el script las imprime al correr.
#
# Que targets salen de aqui, y cual NO:
#   PD de Faith  -- correlaciona 0,948 con la riqueza, asi que hereda entera su patologia:
#                   bajo el esquema primario `kfold5_window` la alfa alcanza R2 = +0,569 pero
#                   el 88% de eso es acertar el nivel del contribuyente, y solo +0,063 es
#                   intra-contribuyente (docs/10_findings.md seccion 1b). Se calcula y se
#                   guarda, pero como descriptor, no como target.
#   MPD          -- correlaciona +0,123 con la riqueza. Casi ortogonal: informacion nueva.
#   MNTD         -- correlaciona -0,503. Parcialmente independiente.
#   SES de las tres -- lo que separa "hay muchas especies" de "hay muchos linajes distintos".
#                   Los tres quedan por debajo de |0,07| contra la riqueza, que es lo que se
#                   les pide.
#
# Las curvas de acumulacion NO estan aqui: scripts/26_rarefaction_inext.R.
#
# Uso:
#   Rscript scripts/25_compute_phylo_responses.R
#   Rscript scripts/25_compute_phylo_responses.R --runs 199   # mas rapido

suppressWarnings(suppressMessages({
  library(ape); library(picante); library(arrow)
  # se ADJUNTA, no se usa `::`: los megaarboles son argumentos por defecto de phylo.maker()
  # y solo se resuelven con el paquete en el search path
  library(V.PhyloMaker2)
}))
source("scripts/lib/parcelas_comm.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP     <- "data/20602096.zip"
PLOTS   <- "data/derived/plots_subset.parquet"
OUT_DIR <- "data/derived"
N_RUNS  <- getarg("--runs", 499)   # aleatorizaciones del modelo nulo para los SES
SEED    <- 42

# La resolucion taxonomica (sinonimos, familias manuales) y la construccion de las matrices
# de incidencia viven en scripts/lib/parcelas_comm.R, porque las comparte el script 26.

message("== lista de especies ==")
pc <- parcelas_species(ZIP)
raw <- pc$raw
sp <- pc$sp
plots_sub <- read_parquet(PLOTS)

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

cmm <- parcelas_comm(raw, tree, plots_sub$PlotObservationID)
comm_full <- cmm$full
comm_sub <- cmm$sub

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

# Las curvas de acumulacion NO se calculan aqui. Estaban, con un remuestreo Monte Carlo que
# solo podia interpolar, y las reemplaza scripts/26_rarefaction_inext.R con rarefaccion y
# extrapolacion de iNEXT -- que es el metodo de la Fig. 4b,c del paper y ademas estima
# cuanto de la flora falta por muestrear, cosa que un remuestreo no puede.

message("\nlisto.  (las curvas de acumulacion: Rscript scripts/26_rarefaction_inext.R)")
