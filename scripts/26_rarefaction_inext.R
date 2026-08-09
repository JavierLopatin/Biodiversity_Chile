#!/usr/bin/env Rscript
# Curvas de rarefaccion/extrapolacion basadas en muestras, replicando la Fig. 4b,c de
# Cerda-Paredes et al. (2026) con el metodo que esa figura usa.
#
# Por que iNEXT y no el remuestreo Monte Carlo del script 25. La figura del paper tiene tres
# rasgos que la identifican: tramo solido hasta el n observado con el punto marcado, tramo
# punteado extrapolado hasta ~2n, y banda de confianza. Eso es la firma de iNEXT (Chao,
# Hsieh y colaboradores), no de un remuestreo. Y hay una diferencia de fondo, no de estilo:
#
#   - El remuestreo solo puede INTERPOLAR. Submuestrear las 1.485 parcelas nunca dice cuanto
#     falta por descubrir; la curva termina exactamente en lo observado, por construccion.
#   - iNEXT interpola con la formula analitica de rarefaccion (sin varianza de Monte Carlo)
#     y ademas EXTRAPOLA con el estimador de Chao, que es lo que responde la pregunta que
#     motiva la figura: cuanto de la flora quedo fuera.
#
# El eje y del panel c del paper llega a ~60, no a las decenas de miles de una PD sumada en
# millones de anos. Eso es `PDtype = "meanPD"`: la PD dividida por la profundidad del arbol
# (390,7 Myr en el nuestro), que se lee como "numero efectivo de linajes" y es comparable
# con el eje del panel b. Se guardan las dos versiones; la figura usa meanPD para poder
# compararla con la publicada.
#
# La parte taxonomica la calcula iNEXT.3D directamente (0,7 s). La filogenetica NO: su ruta
# de PD crece como O(n^2) y las 1.485 parcelas costarian horas, asi que se usa la
# reimplementacion de `scripts/lib/pd_inext.R`, que es el mismo estimador y esta comprobada
# contra iNEXT.3D a precision de maquina (ver tests/test_phylo.R).
#
# Presencia/ausencia, como en el script 25 y por el mismo motivo: la abundancia de
# Parcelas-CL viene en tres unidades inconmensurables. Ver `scripts/lib/parcelas_comm.R`.
#
# Uso:
#   Rscript scripts/26_rarefaction_inext.R
#   Rscript scripts/26_rarefaction_inext.R --nboot 200    # banda mas fina

suppressWarnings(suppressMessages({
  library(ape); library(arrow); library(iNEXT.3D); library(ggplot2)
  library(V.PhyloMaker2)   # solo por tips.info.TPL, que usa el helper
}))
source("scripts/lib/parcelas_comm.R")
source("scripts/lib/pd_inext.R")

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP     <- "data/20602096.zip"
PLOTS   <- "data/derived/plots_subset.parquet"
TREE    <- "data/derived/phylo_tree.tre"
OUT_DIR <- "data/derived"
FIG_DIR <- "results/figures"
NBOOT   <- getarg("--nboot", 50)    # solo para el e.e. de la asintota de iNEXT
REPS    <- getarg("--reps", 200)    # submuestreos por punto para la banda
KNOTS   <- getarg("--knots", 40)
SEED    <- 42

dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
stopifnot(file.exists(TREE))   # el arbol lo construye el script 25; aqui no se reconstruye

# --------------------------------------------------------------------------------------
# 1. comunidad
# --------------------------------------------------------------------------------------

message("== comunidad ==")
tree <- read.tree(TREE)
res <- parcelas_species(ZIP)
plots_sub <- read_parquet(PLOTS)
cm <- parcelas_comm(res$raw, tree, plots_sub$PlotObservationID)
assemblages <- list("Parcelas-CL completo" = cm$full, "subset del proyecto" = cm$sub)

# --------------------------------------------------------------------------------------
# 2. curvas
# --------------------------------------------------------------------------------------

# Cada conjunto extrapola hasta 2x SU PROPIO n. Con un endpoint comun el subset (n=1.082)
# quedaria extrapolado a 2,7n, fuera del rango donde el estimador de Chao es fiable.
run_one <- function(comm, label) {
  n <- nrow(comm)
  message(sprintf("  %s: n=%d, endpoint=%d", label, n, 2 * n))

  # -- taxonomica, con iNEXT.3D tal cual (0,7 s). La lista TIENE que ir nombrada: sin
  # nombre la ruta de PD falla y la de TD inventa "Assemblage_1".
  set.seed(SEED)
  td <- iNEXT.3D::iNEXT3D(setNames(list(t(comm)), label), diversity = "TD", q = 0,
                          datatype = "incidence_raw", endpoint = 2 * n,
                          knots = KNOTS, nboot = NBOOT)
  s <- td$TDiNextEst$size_based
  eff <- intersect(c("nt", "mT", "m"), names(s))[1]   # el nombre cambia entre versiones
  sizes <- sort(unique(c(s[[eff]], n)))

  # -- filogenetica, con la reimplementacion, en los MISMOS tamanos que la taxonomica
  pc <- pd_curve(comm, tree, sizes = sizes)

  # -- banda comun a los dos paneles. Se usa la de submuestreo tambien para la taxonomica,
  # aunque iNEXT trae la suya, para que las dos bandas signifiquen lo mismo: mezclar el
  # bootstrap de iNEXT arriba con submuestreo abajo invita a leer la diferencia de anchura
  # entre paneles como una propiedad de los datos cuando seria del metodo.
  band <- subsample_band(comm, tree, sizes, reps = REPS, seed = SEED)
  bi_td <- match(s[[eff]], band$n)
  bi_pd <- match(pc$n, band$n)

  curves <- rbind(
    data.frame(dataset = label, metric = "taxonomica", n = s[[eff]], method = s$Method,
               value = s$qTD, lo = band$sr_lo[bi_td], hi = band$sr_hi[bi_td],
               stringsAsFactors = FALSE),
    data.frame(dataset = label, metric = "filogenetica_meanPD", n = pc$n,
               method = pc$method, value = pc$meanpd,
               lo = band$meanpd_lo[bi_pd], hi = band$meanpd_hi[bi_pd]),
    data.frame(dataset = label, metric = "filogenetica_PD", n = pc$n,
               method = pc$method, value = pc$pd,
               lo = band$pd_lo[bi_pd], hi = band$pd_hi[bi_pd])
  )
  list(curves = curves, asy = td$TDAsyEst)
}

t0 <- Sys.time()
runs <- lapply(names(assemblages), function(k) run_one(assemblages[[k]], k))
curves <- do.call(rbind, lapply(runs, `[[`, "curves"))
message(sprintf("  %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

write.csv(curves, file.path(OUT_DIR, "rarefaction_inext.csv"), row.names = FALSE)
message(sprintf("  -> rarefaction_inext.csv  (%d filas)", nrow(curves)))

# --------------------------------------------------------------------------------------
# 3. lo que dicen las curvas
# --------------------------------------------------------------------------------------

message("\n== resumen ==")
for (m in unique(curves$metric)) {
  for (d in unique(curves$dataset)) {
    z <- curves[curves$metric == m & curves$dataset == d, ]
    obs <- z[z$method == "Observed", ][1, ]
    ext <- z[nrow(z), ]
    message(sprintf("  %-20s %-22s observado(n=%4.0f) %8.1f  ->  2n=%4.0f %8.1f  (+%.1f%%)",
                    m, d, obs$n, obs$value, ext$n, ext$value,
                    100 * (ext$value / obs$value - 1)))
  }
}
message("\n  asintota de riqueza taxonomica (Chao2):")
asy <- do.call(rbind, lapply(runs, function(r) r$asy[r$asy$qTD == "Species richness", ]))
write.csv(asy, file.path(OUT_DIR, "rarefaction_asymptote.csv"), row.names = FALSE)
for (i in seq_len(nrow(asy))) {
  a <- asy[i, ]
  message(sprintf("    %-22s observado %5.1f  ->  asintota %6.1f +- %.1f  (falta el %.0f%% de la flora)",
                  a$Assemblage, a$TD_obs, a$TD_asy, a$s.e., 100 * (1 - a$TD_obs / a$TD_asy)))
}

# --------------------------------------------------------------------------------------
# 4. figura
# --------------------------------------------------------------------------------------

# solido hasta lo observado, punteado despues: la misma convencion de la Fig. 4 del paper,
# que separa lo que los datos muestran de lo que el estimador infiere. La banda cubre solo
# el tramo solido a proposito; ver `subsample_band()` para por que el tramo extrapolado se
# queda sin ella.
plot_panel <- function(metric, ylab, tag) {
  z <- curves[curves$metric == metric, ]
  z$phase <- ifelse(z$method == "Extrapolation", "extrapolado", "observado")
  pts <- z[z$method == "Observed", ]
  bridge <- transform(pts, phase = "extrapolado")   # el punto pertenece a los dos tramos
  ggplot(rbind(z, bridge), aes(n, value, colour = dataset, fill = dataset)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.18, colour = NA) +
    geom_line(aes(linetype = phase), linewidth = 0.8) +
    geom_point(data = pts, size = 2.6) +
    scale_linetype_manual(values = c(observado = "solid", extrapolado = "22"),
                          guide = "none") +
    scale_colour_manual(values = c("Parcelas-CL completo" = "#2f6f7f",
                                   "subset del proyecto" = "#c1553b")) +
    scale_fill_manual(values = c("Parcelas-CL completo" = "#2f6f7f",
                                 "subset del proyecto" = "#c1553b")) +
    expand_limits(y = 0) +
    labs(x = "Unidades de muestreo (parcelas)", y = ylab, tag = tag,
         colour = NULL, fill = NULL) +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          legend.position = "inside", legend.position.inside = c(0.98, 0.05),
          legend.justification = c(1, 0),
          legend.background = element_rect(fill = alpha("white", 0.75), colour = NA),
          plot.tag = element_text(face = "bold"))
}

fig <- patchwork::wrap_plots(
  plot_panel("taxonomica", "Riqueza taxonomica", "a"),
  plot_panel("filogenetica_meanPD", "Riqueza filogenetica (meanPD)", "b") +
    theme(legend.position = "none"),
  ncol = 1)
stem <- file.path(FIG_DIR, "fig15_rarefaction_parcelas_cl")
ggsave(paste0(stem, ".pdf"), fig, width = 6.5, height = 7.5)
ggsave(paste0(stem, ".png"), fig, width = 6.5, height = 7.5, dpi = 200)
message("\n  -> results/figures/fig15_rarefaction_parcelas_cl.{pdf,png}")
message("\nlisto.")
