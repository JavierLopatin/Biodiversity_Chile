#!/usr/bin/env Rscript
# Curva de Hill taxonomica Y filogenetica (q=0,1,2), abundancia real, replicando el
# mecanismo de rarefaccion/extrapolacion analitica (interpolacion + extrapolacion a 2x,
# linea solida/punteada) -- NO el remuestreo empirico de scripts/65, que esta curva
# reemplaza en el notebook.
#
# Taxonomica: `iNEXT::iNEXT(x, q, datatype="abundance")` -- el mecanismo EXACTO que usa
# Perez-Giraldo et al. 2025 (Ecography, Taxonomic_diversity.R) para su parte taxonomica.
# Diferencia con su script: ellos parten sus datos en 5 grupos fijos ("Fractal", bloques
# espaciales de replicas) y corren una curva por grupo -- nosotros no tenemos un
# equivalente (pedido explicito: "a todos los datos", sin grupos). Un solo assemblage =
# TODOS los individuos de las 2.499 parcelas de conteo real
# (occurrences_unified_counts.parquet, scripts/61) sumados en un vector.
#
# Filogenetica: SIN equivalente en el script de Perez-Giraldo (el de ellos solo tiene un
# punto vía `iNEXT.3D::estimate3D`, ya replicado en scripts/63) -- esta es una extension
# nuestra, mismo mecanismo analitico pero para PD: `iNEXT.3D::iNEXT3D(diversity="PD",
# datatype="abundance", PDtree=...)`, restringido a especies del arbol unificado de 610
# tips (237 de las 261 del pool de conteo). `Type="meanPD"` (PD/Tdepth) ya viene asi de
# la funcion, sin necesidad de normalizar a mano (a diferencia del bug de hillR::hill_phylo
# encontrado y corregido en scripts/65).
#
# `nboot=1` (la convencion del resto de esta sesion, matchea el punto de Perez-Giraldo)
# ROMPE el motor de curva de PD internamente (`if (ans==Inf) ...` sobre un NA -- bug real
# del paquete, verificado, distinto del de `estimate3D` que si tolera nboot=1). Usamos
# `nboot=0` en su lugar: sin banda de confianza (igual que nboot=1 hubiera dado, SE=NA),
# pero sin el error. La taxonomica (paquete `iNEXT` clasico, no `iNEXT.3D`) no tiene este
# problema y corre con su default `nboot=50` (banda de confianza real).
#
# Uso:
#   Rscript scripts/66_hill_curve_inext_abundance.R

suppressWarnings(suppressMessages({
  library(arrow); library(ape); library(iNEXT); library(iNEXT.3D)
}))

DERIVED <- "data/derived"
Q <- c(0, 1, 2)

occ  <- read_parquet(file.path(DERIVED, "occurrences_unified_counts.parquet"))
comm <- as.matrix(as.data.frame.matrix(xtabs(Value ~ PlotObservationID + species, data = occ)))
comm <- comm[rowSums(comm) > 0, , drop = FALSE]
pooled_taxo <- colSums(comm)
pooled_taxo <- pooled_taxo[pooled_taxo > 0]
cat(sprintf("pool taxonomico: %d parcelas, %d individuos, %d especies\n",
            nrow(comm), sum(pooled_taxo), length(pooled_taxo)))

tree <- read.tree(file.path(DERIVED, "phylo_tree_unified.tre"))
sp_tree <- gsub(" ", "_", colnames(comm)) %in% tree$tip.label
comm_tree <- comm[, sp_tree, drop = FALSE]
colnames(comm_tree) <- gsub(" ", "_", colnames(comm_tree))
pooled_phylo <- colSums(comm_tree)
pooled_phylo <- pooled_phylo[pooled_phylo > 0]
sub_tree <- keep.tip(tree, names(pooled_phylo))
cat(sprintf("pool filogenetico (restringido al arbol): %d individuos, %d especies\n",
            sum(pooled_phylo), length(pooled_phylo)))

# --- taxonomica: iNEXT clasico, igual a Perez-Giraldo -------------------------------------
t0 <- Sys.time()
out_taxo <- iNEXT(list(all_plots = pooled_taxo), q = Q, datatype = "abundance",
                  endpoint = 2 * sum(pooled_taxo), knots = 40, se = TRUE, nboot = 50)
cat(sprintf("taxonomica (iNEXT) lista en %.1f s\n", as.numeric(Sys.time() - t0, units = "secs")))

curve_taxo <- out_taxo$iNextEst$size_based[, c("Order.q", "m", "Method", "qD", "qD.LCL", "qD.UCL")]
names(curve_taxo) <- c("q", "n_individuals", "method", "value", "lo", "hi")
curve_taxo$metric <- "taxonomic"

# --- filogenetica: iNEXT.3D, extension propia, nboot=0 (ver docstring) --------------------
t0 <- Sys.time()
out_phylo <- iNEXT3D(data = list(all_plots = pooled_phylo), diversity = "PD", q = Q,
                     datatype = "abundance", PDtree = sub_tree,
                     endpoint = 2 * sum(pooled_phylo), knots = 40, nboot = 0)
cat(sprintf("filogenetica (iNEXT.3D PD) lista en %.1f s\n", as.numeric(Sys.time() - t0, units = "secs")))

curve_phylo <- out_phylo$PDiNextEst$size_based[, c("Order.q", "m", "Method", "qPD", "qPD.LCL", "qPD.UCL")]
names(curve_phylo) <- c("q", "n_individuals", "method", "value", "lo", "hi")
curve_phylo$metric <- "phylogenetic"

curve <- rbind(curve_taxo, curve_phylo)
write.csv(curve, file.path(DERIVED, "unified_hill_curve_inext_abundance.csv"), row.names = FALSE)
cat(sprintf("\n-> %s (%d filas)\n", file.path(DERIVED, "unified_hill_curve_inext_abundance.csv"),
            nrow(curve)))

cat("\nvalor en el punto observado, por q y metrica:\n")
obs <- curve[curve$method == "Observed", ]
print(obs[, c("metric", "q", "n_individuals", "value")])
