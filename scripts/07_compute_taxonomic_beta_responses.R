#!/usr/bin/env Rscript
# Taxonomic diversity (Hill numbers), LCBD and PCoA for Biodiversity_Chile.
#
# Fase 1 de las variables respuesta (ver docs/02_innovation_and_impact.md, tabla de
# riesgos): diversidad filogenetica, funcional y dark diversity quedan para una
# segunda pasada -- cada una depende de un insumo aun no resuelto (ninguna filogenia
# referenciada en este repo, Rasgos-CL solo aporta 2 rasgos continuos, y la curva de
# acumulacion de especies de Parcelas-CL no satura).
#
# Diseno de dos niveles por abundancia (README "Diseno acordado"):
#   - presencia/ausencia sobre las 1.082 parcelas del subset (Jaccard)
#   - ponderado por abundancia, restringido al estrato "cover" (Hellinger/Bray-Curtis)
#     Las parcelas fuera de ese estrato quedan NA en las columnas *_cover -- nunca se
#     descartan ni se imputan.
#
# Composicion via PCoA (no NMDS): con mediana de riqueza = 5 especies/parcela, NMDS
# (vegan::metaMDS) produce soluciones inestables -- parcelas con muy poca informacion
# quedan mal restringidas y el optimizador iterativo las dispara a valores absurdos en
# un eje. PCoA (ape::pcoa, con correccion de Cailliez para autovalores negativos) es
# una descomposicion espectral deterministica, sin reinicios aleatorios, que no tiene
# esa patologia.
#
# Uso:
#   Rscript scripts/07_compute_taxonomic_beta_responses.R \
#       --zip data/20602096.zip --plots data/derived/plots_subset.parquet \
#       --out data/derived/biodiversity_responses.parquet

suppressPackageStartupMessages({
  library(optparse)
  library(arrow)
  library(vegan)
  library(adespatial)
  library(hillR)
  library(ape)
})

option_list <- list(
  make_option("--zip", type = "character", default = "data/20602096.zip"),
  make_option("--plots", type = "character", default = "data/derived/plots_subset.parquet"),
  make_option("--out", type = "character", default = "data/derived/biodiversity_responses.parquet"),
  make_option("--k-axes", type = "integer", default = 2, dest = "k_axes",
              help = "numero de ejes PCoA a extraer [default %default]"),
  make_option("--seed", type = "integer", default = 42)
)
opt <- parse_args(OptionParser(option_list = option_list))

set.seed(opt$seed)

# --- 1. subset de parcelas ya filtrado (congelado por 01_build_subset.py) --
plots <- as.data.frame(arrow::read_parquet(opt$plots))
plot_ids <- as.character(plots$PlotObservationID)
stratum <- setNames(as.character(plots$stratum), plot_ids)

cat(sprintf("plots_subset: %d parcelas\n", length(plot_ids)))
flush(stdout())

# --- 2. Parcelas-CL en formato largo, directo desde el zip -----------------
# Espejo de src/biodiv/io_parcelas.py::load_long -- mismas columnas, misma unidad
# taxonomica (Accepted_species, que puede resolver a genero/familia/variedad/
# subespecie; richness ya cuenta esos rangos como taxones distintos, y aqui se
# mantiene la misma definicion para que hill_q0 sea comparable con richness).
long <- read.csv(unz(opt$zip, "Parcelas_CL.csv"),
                  colClasses = "character", na.strings = character(0))
long$Value <- suppressWarnings(as.numeric(long$Value))
long <- long[long$PlotObservationID %in% plot_ids, ]

cat(sprintf("registros del subset: %d filas, %d taxones\n",
            nrow(long), length(unique(long$Accepted_species))))
flush(stdout())

# Un punado de pares (parcela, especie) tiene 2 registros (distintos estratos/formas
# de crecimiento del mismo taxon; 3 de 10.818 combinaciones en el CSV publicado) --
# se suma Value, agregacion estandar en ecologia de vegetacion.
agg <- aggregate(Value ~ PlotObservationID + Accepted_species, data = long, FUN = sum)

build_comm <- function(df, ids) {
  m <- as.data.frame.matrix(xtabs(Value ~ PlotObservationID + Accepted_species, data = df))
  missing_ids <- setdiff(ids, rownames(m))
  if (length(missing_ids) > 0) {
    pad <- matrix(0, nrow = length(missing_ids), ncol = ncol(m),
                   dimnames = list(missing_ids, colnames(m)))
    m <- rbind(m, pad)
  }
  out <- as.matrix(m[ids, , drop = FALSE])
  storage.mode(out) <- "double"
  out
}

comm_full <- build_comm(agg, plot_ids)

# --- 3. Diversidad taxonomica (Hill numbers) --------------------------------
hill_q0 <- hillR::hill_taxa(comm_full, q = 0, MARGIN = 1)
hill_q1 <- hillR::hill_taxa(comm_full, q = 1, MARGIN = 1)
hill_q2 <- hillR::hill_taxa(comm_full, q = 2, MARGIN = 1)

cat(sprintf("cor(hill_q0, richness) = %.4f (chequeo de sanidad, debiera ser ~1)\n",
            cor(hill_q0[plot_ids], plots$richness)))

# --- 4. LCBD + PCoA, dos niveles --------------------------------------------

log_step <- function(fmt, ...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(fmt, ...)))
  flush(stdout())
}

run_facet <- function(comm_beta, comm_nmds, method_beta, dist_method, label) {
  keep_sp <- colSums(comm_beta) > 0
  comm_beta <- comm_beta[, keep_sp, drop = FALSE]
  comm_nmds <- comm_nmds[, keep_sp, drop = FALSE]
  log_step("[%s] %d parcelas x %d especies", label, nrow(comm_beta), ncol(comm_beta))

  log_step("[%s] beta.div (method=%s, nperm=999) ...", label, method_beta)
  bd <- adespatial::beta.div(comm_beta, method = method_beta, nperm = 999)
  log_step("[%s] beta.div listo", label)

  log_step("[%s] PCoA (distance=%s) ...", label, dist_method)
  d <- vegan::vegdist(comm_nmds, method = dist_method, binary = (dist_method == "jaccard"))
  # Jaccard/Bray no son euclidianas -- pueden dar autovalores negativos; correccion de
  # Cailliez (1983) los desplaza a una escala donde la matriz vuelve a ser embebible.
  pc <- ape::pcoa(d, correction = "cailliez")
  log_step("[%s] PCoA listo", label)

  vecs <- if (!is.null(pc$vectors.cor)) pc$vectors.cor else pc$vectors
  k <- min(opt$k_axes, ncol(vecs))
  sc <- as.data.frame(vecs[, seq_len(k), drop = FALSE])
  colnames(sc) <- paste0("pcoa", seq_len(k))

  rel_eig <- if ("Rel_corr_eig" %in% colnames(pc$values)) pc$values$Rel_corr_eig
             else pc$values$Relative_eig
  var_explained <- sum(rel_eig[seq_len(k)])
  cat(sprintf("[%s] PCoA varianza explicada por %d ejes: %.1f%%\n", label, k, 100 * var_explained))
  if (var_explained < 0.2) {
    warning(sprintf("[%s] PCoA explica poca varianza (%.1f%% en %d ejes)",
                     label, 100 * var_explained, k))
  }

  data.frame(
    PlotObservationID = rownames(comm_beta),
    lcbd = as.numeric(bd$LCBD),
    p_lcbd = as.numeric(bd$p.LCBD),
    sc,
    row.names = NULL
  )
}

# Presencia/ausencia sobre las 1.082 parcelas
comm_pa <- (comm_full > 0) * 1
storage.mode(comm_pa) <- "double"
pa_res <- run_facet(comm_pa, comm_pa, "jaccard", "jaccard", "presence-absence")
names(pa_res)[-1] <- paste0(names(pa_res)[-1], "_pa")

# Ponderado por abundancia, solo estrato "cover" (Cover + Cover_st, ~546 parcelas)
# Cover_st usa una escala distinta de Cover (hasta ~4700 vs. ~100); se relativiza por
# fila (decostand "total") antes de Bray-Curtis para que la mezcla de escalas no
# domine la disimilitud -- beta.div(method="hellinger") ya relativiza internamente,
# por lo que ahi se usa comm_cover sin transformar.
cover_ids <- plot_ids[stratum[plot_ids] == "cover"]
comm_cover <- comm_full[cover_ids, , drop = FALSE]
comm_cover_rel <- vegan::decostand(comm_cover, method = "total")
cover_res <- run_facet(comm_cover, comm_cover_rel, "hellinger", "bray", "cover")
names(cover_res)[-1] <- paste0(names(cover_res)[-1], "_cover")

# --- 5. Unir y escribir ------------------------------------------------------
resp <- data.frame(
  PlotObservationID = plot_ids,
  hill_q0 = as.numeric(hill_q0[plot_ids]),
  hill_q1 = as.numeric(hill_q1[plot_ids]),
  hill_q2 = as.numeric(hill_q2[plot_ids]),
  row.names = NULL
)
resp <- merge(resp, pa_res, by = "PlotObservationID", all.x = TRUE)
resp <- merge(resp, cover_res, by = "PlotObservationID", all.x = TRUE)
resp <- resp[match(plot_ids, resp$PlotObservationID), ]
rownames(resp) <- NULL

n_na_cover <- sum(is.na(resp$lcbd_cover))
cat(sprintf("filas: %d, NA en columnas *_cover: %d (esperado = parcelas fuera de stratum 'cover' = %d)\n",
            nrow(resp), n_na_cover, sum(stratum[plot_ids] != "cover")))

# Aviso de outliers: aunque PCoA no tiene la inestabilidad iterativa de NMDS, una
# parcela con muy poca informacion (ej. 1 especie no compartida con casi ninguna otra)
# igual puede dominar un eje en la descomposicion espectral. Se deja el valor tal cual
# (no se recorta ni se imputa), pero se avisa para que quede documentado.
pcoa_cols <- grep("^pcoa", names(resp), value = TRUE)
for (col in pcoa_cols) {
  x <- resp[[col]]
  ok <- !is.na(x)
  if (sum(ok) < 8) next
  z <- abs(x - median(x[ok])) / mad(x[ok])
  extreme <- which(ok & z > 10)
  if (length(extreme) > 0) {
    cat(sprintf("AVISO: %d parcela(s) con %s extremo (parcela con poca informacion composicional): %s\n",
                length(extreme), col, paste(resp$PlotObservationID[extreme], collapse = ", ")))
  }
}

arrow::write_parquet(resp, opt$out)
cat(sprintf("escrito %s (%d filas, %d columnas)\n", opt$out, nrow(resp), ncol(resp)))
