#!/usr/bin/env Rscript
# Diversidad filogenetica unificada (Parcelas-CL + Living Trees Chile). Misma receta que
# scripts/25_compute_phylo_responses.R (V.PhyloMaker2 GBOTB.extended.TPL, escenario S3;
# picante::pd/mpd/mntd/ses.* con null model "taxa.labels", include.root=FALSE), extendida
# a la lista de especies unificada (scripts/lib/unified_comm.R). NO toca phylo_tree.tre ni
# phylo_responses.parquet -- esos siguen siendo el target del pipeline de modelado actual
# sobre Parcelas-CL solo.
#
# Uso:
#   Rscript scripts/54_unified_phylo_responses.R
#   Rscript scripts/54_unified_phylo_responses.R --runs 199   # mas rapido

suppressWarnings(suppressMessages({
  library(ape); library(picante); library(arrow)
  library(V.PhyloMaker2)
}))
source("scripts/lib/unified_comm.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP     <- "data/20602096.zip"
LT_LONG <- "data/derived/living_trees_long.parquet"
PLOTS   <- "data/derived/plots_unified.parquet"
OUT_DIR <- "data/derived"
N_RUNS  <- getarg("--runs", 499)
SEED    <- 42

message("== lista de especies unificada ==")
us <- unified_species(ZIP, LT_LONG)
pc_raw <- us$pc_raw
sp <- us$sp
plots_uni <- read_parquet(PLOTS)

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
write.tree(tree, file.path(OUT_DIR, "phylo_tree_unified.tre"))
write.csv(built$species.list, file.path(OUT_DIR, "phylo_species_status_unified.csv"),
          row.names = FALSE)

# --------------------------------------------------------------------------------------
# 3. matriz de comunidad
# --------------------------------------------------------------------------------------

message("\n== matriz de comunidad ==")
cmm <- unified_comm(pc_raw, LT_LONG, tree, plots_uni$PlotObservationID)
comm_sub <- cmm$sub

# --------------------------------------------------------------------------------------
# 4. respuestas por parcela
# --------------------------------------------------------------------------------------

message("\n== respuestas por parcela ==")
n_lt2 <- sum(rowSums(comm_sub) < 2)
message(sprintf(paste("  parcelas con <2 especies en el arbol: %d de %d (mpd/mntd dan NA",
                      "ahi, picante lo maneja solo -- no se filtra la matriz)"),
                n_lt2, nrow(comm_sub)))

cph <- cophenetic(tree)

# `include.root=FALSE`: con TRUE, node.age() puede abortar si podar el arbol a las
# especies de una comunidad deja la raiz con un solo hijo -- casi seguro con N_RUNS
# aleatorizaciones. Mismo criterio que script 25.
PD_ROOT <- FALSE
pdv <- picante::pd(comm_sub, tree, include.root = PD_ROOT)

set.seed(SEED)
message(sprintf("  SES con %d aleatorizaciones (modelo nulo taxa.labels)...", N_RUNS))
ses_pd   <- picante::ses.pd(comm_sub, tree, null.model = "taxa.labels", runs = N_RUNS,
                            include.root = PD_ROOT)
ses_mpd  <- picante::ses.mpd(comm_sub, cph, null.model = "taxa.labels", runs = N_RUNS)
ses_mntd <- picante::ses.mntd(comm_sub, cph, null.model = "taxa.labels", runs = N_RUNS)

resp <- data.frame(
  PlotObservationID  = rownames(comm_sub),
  pd_faith_unified   = pdv$PD,
  n_sp_tree_unified  = pdv$SR,
  mpd_unified        = picante::mpd(comm_sub, cph),
  mntd_unified       = picante::mntd(comm_sub, cph),
  ses_pd_unified     = ses_pd$pd.obs.z,
  ses_mpd_unified    = ses_mpd$mpd.obs.z,
  ses_mntd_unified   = ses_mntd$mntd.obs.z,
  p_ses_pd_unified   = ses_pd$pd.obs.p,
  p_ses_mpd_unified  = ses_mpd$mpd.obs.p,
  p_ses_mntd_unified = ses_mntd$mntd.obs.p,
  stringsAsFactors = FALSE
)
# las parcelas sin ninguna especie en el arbol quedan NA, nunca se imputan ni se
# descartan -- mismo criterio que script 25/07
resp <- merge(data.frame(PlotObservationID = plots_uni$PlotObservationID),
              resp, by = "PlotObservationID", all.x = TRUE)
write_parquet(resp, file.path(OUT_DIR, "unified_phylo_responses.parquet"))
message(sprintf("  -> unified_phylo_responses.parquet  %d filas, %d sin cobertura filogenetica",
                nrow(resp), sum(is.na(resp$pd_faith_unified))))

message("\n  correlacion con la riqueza (el criterio de script 25 para elegir targets):")
for (v in c("pd_faith_unified", "mpd_unified", "mntd_unified",
            "ses_pd_unified", "ses_mpd_unified", "ses_mntd_unified")) {
  message(sprintf("    %-18s r = %+.3f", v,
                  cor(resp[[v]], resp$n_sp_tree_unified, use = "complete.obs")))
}

message("\nlisto.")
