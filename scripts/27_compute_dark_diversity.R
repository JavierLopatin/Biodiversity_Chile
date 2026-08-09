#!/usr/bin/env Rscript
# Dark diversity: las especies que PODRIAN estar en una parcela y no estan.
#
# Concepto de Partel, Szava-Kovats y Zobel (2011). El pool de especies de un sitio se parte
# en lo observado y lo oscuro; el cociente entre ambos, la "completitud de la comunidad", se
# propuso justamente como una medida de diversidad independiente del tamano del pool. Aqui
# se estima con co-ocurrencia usando el paquete `DarkDiv` (Carmona y Partel 2021, GEB 30:
# 316-326), metodo hipergeometrico.
#
# TRES DECISIONES DE DISENO, cada una medida y no supuesta:
#
# 1. EL POOL SE ESTIMA CON LAS 1.485 PARCELAS, no con las 1.082 del subset, aunque solo se
#    reporten las del subset. Con el pool reducido la dark diversity se contamina de
#    esfuerzo de muestreo: la correlacion con el log del area de parcela sube de +0,085 a
#    +0,315 y la varianza explicada por el contribuyente de 0,22 a 0,41. Las dos versiones
#    correlacionan 0,958 entre si, asi que es la misma senal, solo que mas limpia. Mas
#    parcelas de co-ocurrencia = mejor estimacion, aunque esten fuera del area de estudio.
#
# 2. SE CUENTA CON UMBRAL, no sumando probabilidades. `DarkDiv` devuelve una probabilidad de
#    pertenencia por especie ausente, y la practica de sumarlas todas da aqui una mediana de
#    253 especies oscuras para parcelas con mediana de 5 observadas -- afirmar que una
#    parcela de 400 m2 en Chile central podria albergar 253 lenosas mas no es un resultado,
#    es un artefacto de sumar 596 numeros pequenos. La mediana de esas probabilidades es
#    0,38. Contando solo las que superan 0,9 quedan ~63 especies, y el estimador deja de ser
#    riqueza disfrazada (ver punto 3).
#
# 3. LA COMPLETITUD NO SIRVE COMO TARGET, y esto es un resultado negativo que hay que
#    reportar, no esconder: log(observadas/oscuras) correlaciona +0,988 con la riqueza
#    observada y +0,987 con hill_q0. Es riqueza reetiquetada, exactamente igual que la PD de
#    Faith. La razon es estructural -- la riqueza esta en el numerador y la dark diversity
#    varia poco entre parcelas, asi que el cociente es una funcion monotona de la riqueza.
#    Se calcula y se guarda como descriptor. El target util es `dark_n`.
#
# Uso:
#   Rscript scripts/27_compute_dark_diversity.R
#   Rscript scripts/27_compute_dark_diversity.R --thr 0.95

suppressWarnings(suppressMessages({
  library(ape); library(arrow); library(DarkDiv); library(ggplot2)
  library(V.PhyloMaker2)   # solo por tips.info.TPL, que usa el helper
}))
source("scripts/lib/parcelas_comm.R")
source("scripts/lib/pd_inext.R")   # branch_incidence(), para la dark diversity filogenetica

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default) {
  i <- match(flag, args); if (is.na(i)) default else as.numeric(args[i + 1])
}
ZIP     <- "data/20602096.zip"
PLOTS   <- "data/derived/plots_subset.parquet"
TREE    <- "data/derived/phylo_tree.tre"
OUT_DIR <- "data/derived"
FIG_DIR <- "results/figures"
THR     <- getarg("--thr", 0.9)     # probabilidad minima para contar una especie como oscura
NEAR_KM <- getarg("--near", 50)     # radio de la comprobacion de coherencia espacial

dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------------------------
# 1. comunidad
# --------------------------------------------------------------------------------------

message("== comunidad ==")
tree <- read.tree(TREE)
res <- parcelas_species(ZIP, quiet = TRUE)
plots_sub <- read_parquet(PLOTS)
cm <- parcelas_comm(res$raw, tree, plots_sub$PlotObservationID)
comm <- cm$full                       # fuente de co-ocurrencia: TODO Parcelas-CL
obs <- rowSums(comm)

meta <- unique(res$raw[, c("PlotObservationID", "Owner", "PlotSize_m2", "X", "Y")])
meta$PlotSize_m2 <- suppressWarnings(as.numeric(meta$PlotSize_m2))
meta <- meta[match(rownames(comm), meta$PlotObservationID), ]

# --------------------------------------------------------------------------------------
# 2. estimacion
# --------------------------------------------------------------------------------------

message("\n== DarkDiv ==")
t0 <- Sys.time()
dd <- DarkDiv(comm, method = "Hypergeometric")
# Favorability corrige por prevalencia. Sirve de control de que el resultado no depende del
# metodo -- pero OJO con como se compara: las dos escalas no son intercambiables. El 11% de
# las celdas hipergeometricas supera 0,9 contra el 2,6% de las de favorability, asi que
# aplicar el mismo umbral a las dos y comparar conteos da rho = +0,12 y parece que los
# metodos se contradicen. No se contradicen: la correlacion de RANGOS entre las dos matrices
# completas es +0,71 sobre 881.905 celdas. Lo que no es robusto es el umbral absoluto, no el
# orden de las especies candidatas. Por eso el control se hace a conteo igualado.
fav <- DarkDiv(comm, method = "Favorability")
message(sprintf("  %.1f s", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

pr <- dd$Dark                                    # NA donde la especie SI esta presente
dark_bin <- !is.na(pr) & pr > THR
message(sprintf("  probabilidades de pertenencia: mediana %.3f, p90 %.3f",
                median(pr, na.rm = TRUE), quantile(pr, 0.9, na.rm = TRUE)))

# sensibilidad al umbral: si la eleccion cambiara la conclusion, habria que decirlo
message("\n  sensibilidad al umbral (rho contra la riqueza observada):")
for (t in c(0.5, 0.7, 0.8, 0.9, 0.95, 0.99)) {
  d <- rowSums(!is.na(pr) & pr > t)
  message(sprintf("    thr %.2f -> mediana %5.1f especies oscuras   rho(obs) = %+.3f%s",
                  t, median(d), cor(d, obs, method = "spearman"),
                  if (isTRUE(all.equal(t, THR))) "   <- elegido" else ""))
}

# --------------------------------------------------------------------------------------
# 3. dark diversity filogenetica
# --------------------------------------------------------------------------------------

# Cuanto LINAJE falta, no cuantas especies. Dos parcelas pueden tener 60 especies oscuras
# cada una y que en una sean 60 congeneres y en la otra 60 familias distintas.
message("\n== dark diversity filogenetica ==")
tips <- colnames(comm)
tr <- keep.tip(tree, tips)
ntip <- Ntip(tr)
parts <- prop.part(tr)
memb <- matrix(0L, nrow = ntip, ncol = ntip + tr$Nnode)
memb[cbind(seq_len(ntip), seq_len(ntip))] <- 1L
for (i in seq_along(parts)) memb[parts[[i]], ntip + i] <- 1L
rownames(memb) <- tr$tip.label
L <- numeric(ncol(memb)); L[tr$edge[, 2]] <- tr$edge.length
Tdepth <- reference_time(tr)

pd_of <- function(mat) {                    # PD de cada fila de una matriz de incidencia
  as.vector((mat[, rownames(memb), drop = FALSE] %*% memb > 0) %*% L)
}
dark_pd <- pd_of(dark_bin * 1L)
obs_pd <- pd_of(comm)
message(sprintf("  PD oscura: mediana %.0f Ma (%.1f linajes efectivos) contra %.0f observada",
                median(dark_pd), median(dark_pd) / Tdepth, median(obs_pd)))

# La PD del conjunto oscuro resulta ser contar especies oscuras otra vez: rho +0,96 con
# dark_n, la misma patologia que la PD de Faith con la riqueza. Lo que si es otra cosa es la
# distancia media entre las especies oscuras -- responde si lo que falta son 60 congeneres o
# 60 familias distintas, y eso no lo dice el conteo.
cph <- cophenetic(tr)
dark_mpd <- vapply(seq_len(nrow(comm)), function(i) {
  k <- which(dark_bin[i, ])
  if (length(k) < 2) return(NA_real_)
  sp <- colnames(comm)[k]
  mean(cph[sp, sp][lower.tri(diag(length(sp)))])
}, numeric(1))
message(sprintf("  MPD del conjunto oscuro: mediana %.0f Ma   rho con dark_n = %+.3f",
                median(dark_mpd, na.rm = TRUE),
                cor(dark_mpd, rowSums(dark_bin), method = "spearman", use = "complete.obs")))

# --------------------------------------------------------------------------------------
# 4. coherencia espacial: el pool no puede ser una lista nacional
# --------------------------------------------------------------------------------------

# La objecion obvia a estimar dark diversity por co-ocurrencia sobre 30-55 grados S es que a
# una parcela de Coquimbo se le adjudiquen especies patagonicas. El metodo deberia filtrarlo
# solo -- una especie que nunca co-ocurre con las tuyas recibe indicacion baja -- pero eso
# hay que comprobarlo, no suponerlo.
#
# Se mide que fraccion de las especies oscuras de cada parcela se ha observado de verdad
# cerca, Y LA MISMA FRACCION PARA TODAS LAS AUSENTES, que es la linea base. El numero
# absoluto no dice nada por si solo: que el 46% de las oscuras se haya visto a <50 km suena
# mal hasta que se ve que para una ausente cualquiera es el 17%. Lo que valida el metodo es
# la razon entre las dos.
message("\n== coherencia espacial ==")
xy <- as.matrix(meta[, c("X", "Y")])
spatial <- do.call(rbind, lapply(c(25, 50, 100, 200), function(km) {
  r <- vapply(seq_len(nrow(comm)), function(i) {
    d <- sqrt((xy[, 1] - xy[i, 1])^2 + (xy[, 2] - xy[i, 2])^2)
    local_pool <- colSums(comm[d <= km * 1000, , drop = FALSE]) > 0
    dk <- which(dark_bin[i, ])
    if (!length(dk)) return(c(NA_real_, NA_real_))
    c(mean(local_pool[dk]), mean(local_pool[comm[i, ] == 0]))
  }, numeric(2))
  data.frame(km = km, oscuras = median(r[1, ], na.rm = TRUE),
             base = median(r[2, ], na.rm = TRUE))
}))
spatial$razon <- spatial$oscuras / spatial$base
write.csv(spatial, file.path(OUT_DIR, "dark_diversity_spatial.csv"), row.names = FALSE)
for (i in seq_len(nrow(spatial))) {
  x <- spatial[i, ]
  message(sprintf("  %3.0f km: de las oscuras se ha visto cerca el %.1f%%, de una ausente cualquiera el %.1f%%  ->  %.1fx",
                  x$km, 100 * x$oscuras, 100 * x$base, x$razon))
}
# se guarda por parcela el radio de referencia, para poder cribar parcelas dudosas
near_frac <- vapply(seq_len(nrow(comm)), function(i) {
  d <- sqrt((xy[, 1] - xy[i, 1])^2 + (xy[, 2] - xy[i, 2])^2)
  local_pool <- colSums(comm[d <= NEAR_KM * 1000, , drop = FALSE]) > 0
  dk <- which(dark_bin[i, ])
  if (!length(dk)) NA_real_ else mean(local_pool[dk])
}, numeric(1))

# --------------------------------------------------------------------------------------
# 5. tabla de respuestas
# --------------------------------------------------------------------------------------

dark <- data.frame(
  PlotObservationID = rownames(comm),
  n_obs        = obs,
  dark_n       = rowSums(dark_bin),                       # <- el target
  dark_prob    = rowSums(pr, na.rm = TRUE),               # suma de probabilidades
  dark_n_fav   = NA_real_,   # se rellena abajo, a conteo igualado
  pool_n       = obs + rowSums(dark_bin),
  dark_pd      = dark_pd,
  dark_lin     = dark_pd / Tdepth,   # linajes efectivos: reescalado de dark_pd, no info nueva
  dark_mpd     = dark_mpd,
  completeness = log(obs / rowSums(pr, na.rm = TRUE)),    # descriptor, NO target
  near_frac    = near_frac,
  stringsAsFactors = FALSE
)
# control por metodo: para cada parcela se toman las k especies mejor rankeadas por
# favorability, con k = las que el hipergeometrico declaro oscuras. Asi la comparacion mide
# acuerdo en QUE especies, que es lo que importa, y no acuerdo en donde cae un umbral.
fv <- fav$Dark; fv[is.na(fv)] <- -Inf
jac <- vapply(seq_len(nrow(comm)), function(i) {
  k <- sum(dark_bin[i, ]);  if (!k) return(NA_real_)
  top <- order(fv[i, ], decreasing = TRUE)[seq_len(k)]
  length(intersect(top, which(dark_bin[i, ]))) / length(union(top, which(dark_bin[i, ])))
}, numeric(1))
dark$dark_n_fav <- NULL
dark$jaccard_fav <- jac
message(sprintf("\n  acuerdo con Favorability a conteo igualado: Jaccard mediana %.3f", median(jac, na.rm = TRUE)))
message(sprintf("  correlacion de rangos de las matrices completas: %+.3f",
                cor(as.vector(pr), as.vector(fav$Dark), method = "spearman", use = "complete.obs")))

# se reportan solo las parcelas del subset, con NA donde no hay cobertura -- el mismo
# criterio del script 25 y del 07
out <- merge(data.frame(PlotObservationID = plots_sub$PlotObservationID),
             dark, by = "PlotObservationID", all.x = TRUE)
write_parquet(out, file.path(OUT_DIR, "dark_diversity.parquet"))
message(sprintf("\n  -> dark_diversity.parquet  %d filas, %d sin cobertura",
                nrow(out), sum(is.na(out$dark_n))))

# --------------------------------------------------------------------------------------
# 6. que sirve como target
# --------------------------------------------------------------------------------------

message("\n== criterio de seleccion de targets ==")
tax <- read_parquet(file.path(OUT_DIR, "biodiversity_responses.parquet"))
m <- merge(merge(dark, meta, by = "PlotObservationID"), tax,
           by = "PlotObservationID", all.x = TRUE)
ok_a <- !is.na(m$PlotSize_m2)
rho <- function(a, b) cor(a, b, method = "spearman", use = "complete.obs")
message("  metrica      rho(riqueza) rho(logArea) R2_Owner | hill_q0  pcoa1_pa  lcbd_pa")
for (v in c("dark_n", "dark_prob", "dark_pd", "dark_mpd", "pool_n", "completeness")) {
  message(sprintf("  %-12s %+11.3f %+12.3f %8.2f | %+7.3f %+9.3f %+8.3f",
                  v, rho(m[[v]], m$n_obs), rho(log(m$PlotSize_m2[ok_a]), m[[v]][ok_a]),
                  summary(lm(m[[v]] ~ factor(m$Owner)))$r.squared,
                  rho(m[[v]], m$hill_q0), rho(m[[v]], m$pcoa1_pa), rho(m[[v]], m$lcbd_pa)))
}


# --------------------------------------------------------------------------------------
# 7. figura
# --------------------------------------------------------------------------------------

sub <- m[m$PlotObservationID %in% plots_sub$PlotObservationID, ]
pan <- function(x, y, xl, yl, tag) {
  ggplot(sub, aes(.data[[x]], .data[[y]])) +
    geom_point(size = 0.7, alpha = 0.3, colour = "#2f6f7f") +
    geom_smooth(method = "loess", formula = y ~ x, se = FALSE,
                colour = "#c1553b", linewidth = 0.8) +
    labs(x = xl, y = yl, tag = tag,
         subtitle = sprintf("rho = %+.3f", rho(sub[[x]], sub[[y]]))) +
    theme_bw(base_size = 10) +
    theme(panel.grid.minor = element_blank(), plot.tag = element_text(face = "bold"),
          plot.subtitle = element_text(size = 8, colour = "grey30"))
}
fig <- patchwork::wrap_plots(
  pan("n_obs", "dark_n", "Especies observadas", "Especies oscuras (p > 0,9)", "a"),
  pan("n_obs", "completeness", "Especies observadas", "Completitud  log(obs/oscuras)", "b"),
  pan("dark_n", "dark_mpd", "Especies oscuras", "MPD del conjunto oscuro (Ma)", "c"),
  pan("n_obs", "near_frac", "Especies observadas",
      sprintf("Fraccion de oscuras vista a <%.0f km", NEAR_KM), "d"),
  ncol = 2)
stem <- file.path(FIG_DIR, "fig16_dark_diversity")
ggsave(paste0(stem, ".pdf"), fig, width = 8, height = 6.5)
ggsave(paste0(stem, ".png"), fig, width = 8, height = 6.5, dpi = 200)
message(sprintf("\n  -> %s.{pdf,png}", stem))
message("\nlisto.")
