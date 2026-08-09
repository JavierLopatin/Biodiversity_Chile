# Rarefaccion y extrapolacion de diversidad filogenetica basada en muestras (q = 0).
#
# POR QUE NO SE USA iNEXT.3D AQUI. Su ruta de PD es la funcion correcta pero no escala:
# medido en esta maquina con 6 nudos y sin bootstrap, 89 s con 100 unidades de muestreo y
# 315 s con 200 -- crecimiento cuadratico. Extrapolando, las 1.485 parcelas costarian ~5 h
# por conjunto y por tipo de PD, y con nboot=50 seria del orden de una semana. La parte
# taxonomica de iNEXT.3D, en cambio, tarda 0,7 s sobre la misma matriz y se usa tal cual.
#
# Lo que sigue es el MISMO estimador (Chao, Chiu, Hsieh, Davis, Nipperess y Faith 2015,
# "Rarefaction and extrapolation of phylogenetic diversity", Methods Ecol Evol 6:380-388),
# implementado con algebra matricial en vez de bucles. `validate_pd_inext()` comprueba que
# reproduce la salida de iNEXT.3D en un submuestreo donde esa referencia si se puede correr;
# esa comprobacion es la que autoriza a usar esta version, y esta en tests/test_phylo.R.
#
# La idea del estimador: una rama del arbol es a la diversidad filogenetica lo que una
# especie es a la riqueza. Se cuenta en cuantas unidades de muestreo aparece cada rama
# (Y_b), se rarefaccionan las ramas igual que se rarefaccionarian especies, y cada una se
# pondera por su largo L_b. Con Y_b = 1 para todas las ramas terminales y todas de largo 1
# se recupera exactamente la rarefaccion de riqueza.


#' Largo e incidencia de cada rama del arbol dado un conjunto de unidades de muestreo.
#'
#' `comm` es parcelas x especies, 0/1. Devuelve un data.frame con una fila por rama:
#' `L` (largo) e `Y` (en cuantas parcelas aparece algun descendiente de esa rama).
#'
#' Se incluye la rama que va del ancestro comun a la raiz del arbol de referencia, porque
#' esa es la convencion de Chao et al. y la que hace que meanPD = PD / T se lea como
#' "numero efectivo de linajes" comparable con el eje de riqueza.
branch_incidence <- function(comm, tree) {
  tips <- intersect(colnames(comm), tree$tip.label)
  stopifnot(length(tips) > 1)
  comm <- comm[, tips, drop = FALSE]
  tr <- if (length(tips) < ape::Ntip(tree)) ape::keep.tip(tree, tips) else tree

  # matriz de pertenencia tip x nodo: 1 si el tip desciende del nodo. `prop.part` da las
  # particiones de los nodos internos en un solo recorrido; los tips son su propia hoja.
  ntip <- ape::Ntip(tr)
  parts <- ape::prop.part(tr)                 # lista indexada por nodo interno
  memb <- matrix(0L, nrow = ntip, ncol = ntip + tr$Nnode)
  memb[cbind(seq_len(ntip), seq_len(ntip))] <- 1L
  for (i in seq_along(parts)) memb[parts[[i]], ntip + i] <- 1L
  rownames(memb) <- tr$tip.label

  # una parcela "tiene" un nodo si tiene al menos un descendiente suyo
  hit <- (comm[, rownames(memb), drop = FALSE] %*% memb) > 0
  Y_node <- colSums(hit)

  # cada arista se identifica por su nodo hijo
  child <- tr$edge[, 2]
  out <- data.frame(L = tr$edge.length, Y = Y_node[child])

  # la rama basal: del nodo raiz hacia atras hasta la profundidad de referencia. En un
  # arbol ultrametrico esa profundidad es la distancia raiz-tip, y la rama esta presente
  # en toda parcela que tenga alguna especie.
  out <- rbind(out, data.frame(L = 0, Y = max(Y_node)))
  out[out$Y > 0, , drop = FALSE]
}


#' Profundidad de referencia: distancia de la raiz a los tips.
reference_time <- function(tree, tips = NULL) {
  tr <- if (is.null(tips)) tree else ape::keep.tip(tree, intersect(tips, tree$tip.label))
  max(ape::node.depth.edgelength(tr))
}


#' Curva de rarefaccion/extrapolacion de PD.
#'
#' @param comm parcelas x especies, 0/1
#' @param tree arbol filogenetico
#' @param sizes numero de unidades de muestreo en que evaluar la curva
#' @return data.frame con n, pd, meanpd y `method`
pd_curve <- function(comm, tree, sizes = NULL, knots = 40) {
  Tn <- nrow(comm)
  br <- branch_incidence(comm, tree)
  Tdepth <- reference_time(tree, colnames(comm)[colSums(comm) > 0])
  if (is.null(sizes)) {
    sizes <- unique(round(seq(1, 2 * Tn, length.out = knots)))
    sizes <- sort(unique(c(sizes, Tn)))
  }

  pd_obs <- sum(br$L)

  # --- interpolacion: largo esperado de rama en t unidades tomadas al azar sin reemplazo.
  # Una rama contribuye si al menos una de las t unidades la contiene, con probabilidad
  # 1 - C(T-Y, t)/C(T, t). Se calcula con lgamma para que no desborde con T = 1.485.
  lchoose_ratio <- function(Y, t) {
    ok <- (Tn - Y) >= t
    r <- rep(0, length(Y))
    r[ok] <- exp(lchoose(Tn - Y[ok], t) - lchoose(Tn, t))
    r
  }
  rare <- vapply(sizes[sizes <= Tn], function(t) sum(br$L * (1 - lchoose_ratio(br$Y, t))),
                 numeric(1))

  # --- extrapolacion: estimador tipo Chao2 aplicado al largo de rama no detectado.
  # g1 y g2 son el largo TOTAL de las ramas vistas en exactamente 1 y 2 unidades; ocupan
  # el lugar de f1 y f2 (numero de especies unicas y duplicadas) de la riqueza.
  g1 <- sum(br$L[br$Y == 1])
  g2 <- sum(br$L[br$Y == 2])
  G0 <- if (g2 > 0) ((Tn - 1) / Tn) * g1^2 / (2 * g2) else ((Tn - 1) / Tn) * g1 * (g1 - 1) / 2
  G0 <- max(G0, 0)
  ext <- vapply(sizes[sizes > Tn], function(t) {
    tstar <- t - Tn
    if (G0 <= 0 || g1 <= 0) return(pd_obs)
    pd_obs + G0 * (1 - (1 - g1 / (Tn * G0 + g1))^tstar)
  }, numeric(1))

  n <- c(sizes[sizes <= Tn], sizes[sizes > Tn])
  pd <- c(rare, ext)
  data.frame(n = n, pd = pd, meanpd = pd / Tdepth,
             method = ifelse(n < Tn, "Rarefaction",
                             ifelse(n == Tn, "Observed", "Extrapolation")),
             stringsAsFactors = FALSE)
}


#' Banda de dispersion por submuestreo SIN reemplazo de las unidades de muestreo.
#'
#' Devuelve, para cada tamano t <= T, los cuantiles de la riqueza taxonomica y de la PD
#' obtenidas al tomar t parcelas al azar. Es la dispersion exacta de la cantidad que la
#' curva de rarefaccion estima como media, asi que la banda esta centrada por construccion
#' -- es la misma convencion que `vegan::specaccum`.
#'
#' POR QUE NO EL BOOTSTRAP DE iNEXT. Se probaron los dos alternativos y los dos fallan de
#' forma medible:
#'   1. Remuestrear parcelas CON reemplazo: una muestra de T parcelas contiene solo ~63% de
#'      parcelas distintas y las copias no aportan especies, asi que la curva queda
#'      sistematicamente por debajo. La banda no llegaba a tocar su propia linea.
#'   2. Bootstrap parametrico con Y_i/T por especie: cada especie singleton desaparece con
#'      probabilidad (1-1/T)^T = 0,37, asi que cada replica pierde ~37% de los singletons y
#'      con ellos su largo de rama. iNEXT compensa exactamente eso simulando ademas las f0
#'      especies no detectadas -- que en un arbol no se pueden anadir sin inventar donde
#'      cuelgan. Medido: la banda quedaba en meanPD 48,5-50,6 contra una curva en 52,8.
#'
#' Lo que esta banda NO es: un intervalo sobre el ensamble subyacente. Solo cubre el tramo
#' interpolado; para el extrapolado no hay analogo por submuestreo y se deja sin banda, que
#' es la lectura honesta -- la incertidumbre de la extrapolacion vive justamente en las
#' especies que no se han visto.
subsample_band <- function(comm, tree, sizes, reps = 100, conf = 0.95, seed = 42) {
  Tn <- nrow(comm)
  sizes <- sort(unique(sizes[sizes >= 1 & sizes <= Tn]))
  Tdepth <- reference_time(tree, colnames(comm)[colSums(comm) > 0])

  # el mapa parcela -> nodos del arbol se construye UNA vez; despues cada submuestreo es
  # un colSums sobre las filas elegidas, no un keep.tip
  tips <- intersect(colnames(comm), tree$tip.label)
  tr <- if (length(tips) < ape::Ntip(tree)) ape::keep.tip(tree, tips) else tree
  ntip <- ape::Ntip(tr)
  parts <- ape::prop.part(tr)
  memb <- matrix(0L, nrow = ntip, ncol = ntip + tr$Nnode)
  memb[cbind(seq_len(ntip), seq_len(ntip))] <- 1L
  for (i in seq_along(parts)) memb[parts[[i]], ntip + i] <- 1L
  rownames(memb) <- tr$tip.label
  hit <- (comm[, rownames(memb), drop = FALSE] %*% memb) > 0   # parcelas x nodos
  L <- numeric(ncol(hit))
  L[tr$edge[, 2]] <- tr$edge.length
  has_sp <- comm[, tips, drop = FALSE] > 0

  a <- (1 - conf) / 2
  set.seed(seed)
  out <- lapply(sizes, function(t) {
    v <- vapply(seq_len(reps), function(r) {
      idx <- sample.int(Tn, t)
      c(sum(colSums(has_sp[idx, , drop = FALSE]) > 0),
        sum(L[colSums(hit[idx, , drop = FALSE]) > 0]))
    }, numeric(2))
    data.frame(n = t,
               sr_lo = quantile(v[1, ], a), sr_hi = quantile(v[1, ], 1 - a),
               pd_lo = quantile(v[2, ], a), pd_hi = quantile(v[2, ], 1 - a))
  })
  d <- do.call(rbind, out)
  d$meanpd_lo <- d$pd_lo / Tdepth
  d$meanpd_hi <- d$pd_hi / Tdepth
  rownames(d) <- NULL
  d
}


#' Curva de PD con la banda de `subsample_band()` adosada (NA en el tramo extrapolado).
pd_curve_ci <- function(comm, tree, sizes = NULL, knots = 40, reps = 100, conf = 0.95,
                        seed = 42) {
  base <- pd_curve(comm, tree, sizes, knots)
  if (reps < 2) {
    base[c("pd_lo", "pd_hi", "meanpd_lo", "meanpd_hi")] <- NA_real_
    return(base)
  }
  band <- subsample_band(comm, tree, base$n, reps = reps, conf = conf, seed = seed)
  i <- match(base$n, band$n)
  for (col in c("pd_lo", "pd_hi", "meanpd_lo", "meanpd_hi")) base[[col]] <- band[[col]][i]
  base
}


#' Comprueba esta implementacion contra iNEXT.3D sobre un submuestreo pequeno.
#'
#' iNEXT.3D es la referencia; aqui solo se reimplementa porque no escala. Si esta funcion
#' falla, la reimplementacion esta mal y no hay que creerle ninguna curva.
validate_pd_inext <- function(comm, tree, n_units = 150, tol = 1e-6, seed = 1) {
  set.seed(seed)
  idx <- sample.int(nrow(comm), n_units)
  cs <- comm[idx, , drop = FALSE]
  cs <- cs[, colSums(cs) > 0, drop = FALSE]
  tr <- ape::keep.tip(tree, colnames(cs))

  sizes <- unique(round(seq(1, 2 * n_units, length.out = 6)))
  ref <- iNEXT.3D::iNEXT3D(list(A = t(cs)), diversity = "PD", q = 0,
                           datatype = "incidence_raw", size = sizes, nboot = 0,
                           PDtree = tr, PDtype = "meanPD")$PDiNextEst$size_based
  eff <- intersect(c("nt", "mT", "m"), names(ref))[1]
  mine <- pd_curve(cs, tr, sizes = ref[[eff]])

  d <- data.frame(n = ref[[eff]], inext = ref$qPD, mine = mine$meanpd)
  d$rel <- abs(d$mine - d$inext) / pmax(d$inext, 1e-12)
  list(table = d, max_rel_error = max(d$rel), ok = max(d$rel) < tol)
}
