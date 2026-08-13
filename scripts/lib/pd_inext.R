# Rarefaccion y extrapolacion de diversidad filogenetica basada en muestras, ahora para
# q = 0, 1 y 2 (perfil completo de Hill: riqueza, tipo Shannon, tipo Simpson).
#
# POR QUE NO SE USA iNEXT.3D AQUI. Su ruta de PD es la funcion correcta pero no escala:
# medido en esta maquina con 6 nudos y sin bootstrap, 89 s con 100 unidades de muestreo y
# 315 s con 200 -- crecimiento cuadratico. Extrapolando, miles de parcelas costarian horas
# por conjunto y por orden de Hill. La parte taxonomica de iNEXT.3D, en cambio, tarda 0,7 s
# sobre la misma matriz (no depende de q) y se usa tal cual en scripts/56.
#
# DE DONDE SALEN LAS FORMULAS DE q=1,2. No se inventaron de memoria -- el reimplementado de
# q=0 (unico que habia hasta ahora) ya sigue a Chao, Chiu, Hsieh, Davis, Nipperess y Faith
# (2015, "Rarefaction and extrapolation of phylogenetic diversity", Methods Ecol Evol
# 6:380-388), pero q=1 (tipo Shannon) y q=2 (tipo Simpson) NO son una generalizacion simple
# de esa formula -- son ramas de codigo distintas incluso dentro del propio iNEXT.3D. Las
# formulas de abajo se extrajeron leyendo el codigo fuente real del paquete instalado
# (`asNamespace("iNEXT.3D")$RPD`, `$PhD.q.est`, `$PhD.m.est` para la parte R; el codigo C++
# publico de `AnneChao/iNEXT.3D` -- `RPD`, `PDq0`, `PDq1_2`, `PhD.q.est`'s `Sub()` -- para la
# parte compilada, ya que la instalada aqui no trae el `.cpp`, solo el `.so`).
#
# `validate_pd_inext()`/`validate_pd_inext_multiq()` (tests/test_phylo.R) comprueban que
# esta reimplementacion reproduce la salida de iNEXT.3D en un submuestreo donde esa
# referencia si se puede correr -- esa comprobacion es la que autoriza a confiar en esto,
# no una lectura del codigo por si sola.
#
# La idea del estimador (comun a los 3 ordenes): una rama del arbol es a la diversidad
# filogenetica lo que una especie es a la riqueza. Se cuenta en cuantas unidades de
# muestreo aparece cada rama (Y_b), se rarefaccionan las ramas igual que se
# rarefaccionarian especies, y cada una se pondera por su largo L_b. Con Y_b = 1 para
# todas las ramas terminales y todas de largo 1 se recupera exactamente la rarefaccion de
# riqueza taxonomica -- por eso `hill_from_incidence()` de abajo sirve para las dos cosas,
# solo cambia que se le pase como `L`.


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


#' Diversidad de Hill observada (sin rarefaccionar) de orden `q`, desde incidencias.
#'
#' Formula general de Hill (Chao & Jost) aplicada a una distribucion de incidencia
#' cualquiera: con `L = 1` para todas las unidades da los numeros de Hill taxonomicos
#' clasicos (riqueza en q=0, Shannon exponenciado en q=1, inverso de Simpson en q=2); con
#' `L` = largo de rama da su analogo filogenetico ("mean PD" de orden q).
#'
#' @param Y incidencia (en cuantas unidades de muestreo aparece) de cada especie/rama
#' @param L peso de cada especie/rama (1 para taxonomico, largo de rama para filogenetico)
#' @param Tn numero de unidades de muestreo del conjunto sobre el que se calcula
#' @param q vector de ordenes de Hill (0, 1 y/o 2)
#' @return vector nombrado (por `q`) con el valor para cada orden
hill_from_incidence <- function(Y, L, Tn, q) {
  keep <- Y > 0
  Y <- Y[keep]; L <- L[keep]
  tbar <- sum(Y * L) / Tn
  p <- Y / (Tn * tbar)
  out <- setNames(numeric(length(q)), as.character(q))
  for (j in seq_along(q)) {
    qj <- q[j]
    out[j] <- if (qj == 0) sum(L)
             else if (qj == 1) exp(sum(-p * log(p) * L))
             else if (qj == 2) 1 / sum(p^2 * L)
             else (sum(p^qj * L))^(1 / (1 - qj))
  }
  out
}


#' Rarefaccion (interpolacion) de Hill filogenetico en un tamano de muestra `t_size < Tn`.
#'
#' q=0 usa el atajo O(n_ramas) ya validado (`1 - C(Tn-Y,t)/C(Tn,t)`, equivalente algebraico
#' exacto de sumar la distribucion completa). q=1,2 SI necesitan la distribucion completa
#' sobre `k` (cuantas veces se observa cada rama en las `t_size` unidades sorteadas) porque
#' su formula pondera cada `k` distinto, no solo si la rama aparecio o no -- es el precio
#' real de q=1,2, no una limitacion de esta implementacion (`RPD` de iNEXT.3D hace lo mismo).
.rarefy_at_t <- function(Y, L, Tn, t_size, q, tbar) {
  need_dist <- any(q %in% c(1, 2)) || any(!(q %in% c(0, 1, 2)))
  out <- setNames(numeric(length(q)), as.character(q))

  if (!need_dist) {
    ok <- (Tn - Y) >= t_size
    r <- rep(0, length(Y))
    r[ok] <- exp(lchoose(Tn - Y[ok], t_size) - lchoose(Tn, t_size))
    out[as.character(0)] <- sum(L * (1 - r))
    return(out)
  }

  kk <- 0:(t_size - 1)
  K1 <- outer(Y, kk + 1, function(y, k1) ifelse(y >= k1, lchoose(y, k1), -Inf))
  K2 <- outer(Tn - Y, t_size - kk - 1, function(rr, k2) ifelse(rr >= k2, lchoose(rr, k2), -Inf))
  logC <- lchoose(Tn, t_size)
  ghat_pt2 <- exp(K1 + K2 - logC)          # n_ramas x t_size
  ghat <- as.numeric(L %*% ghat_pt2)       # largo esperado visto exactamente k+1 veces
  pk <- (kk + 1) / (t_size * tbar)

  for (j in seq_along(q)) {
    qj <- q[j]
    out[j] <- if (qj == 0) sum(ghat)
             else if (qj == 1) exp(sum(-pk * log(pk) * ghat))
             else if (qj == 2) 1 / sum(pk^2 * ghat)
             else (sum(pk^qj * ghat))^(1 / (1 - qj))
  }
  out
}


#' Estadisticos de ramas singleton/doubleton (f1,f2 = conteo; g1,g2 = largo total) y el
#' estimador de cobertura Good-Turing `A` -- insumos compartidos por la extrapolacion de
#' q=0 (ya existente) y la asintota de q=1,2 (nueva).
.branch_singleton_stats <- function(Y, L, Tn) {
  I1 <- Y == 1; I2 <- Y == 2
  f1 <- sum(I1); f2 <- sum(I2)
  g1 <- sum(L[I1]); g2 <- sum(L[I2])
  A <- if (f2 > 0) 2 * f2 / ((Tn - 1) * f1 + 2 * f2)
       else if (f1 > 0) 2 / ((Tn - 1) * (f1 - 1) + 2)
       else 1
  list(f1 = f1, f2 = f2, g1 = g1, g2 = g2, A = A)
}


#' Diversidad filogenetica asintotica (t -> infinito) de orden `q`.
#'
#' q=0 reusa la formula tipo Chao2 ya validada (`G0`), sin tocarla. q=1,2 son ramas nuevas,
#' extraidas de `PhD.q.est()`/`Sub()` de iNEXT.3D -- ver cabecera del archivo.
.pd_asymptotic <- function(Y, L, Tn, tbar, q, st = NULL) {
  st <- st %||% .branch_singleton_stats(Y, L, Tn)
  out <- setNames(numeric(length(q)), as.character(q))
  for (j in seq_along(q)) {
    qj <- q[j]
    if (qj == 0) {
      G0 <- if (st$g2 > 0) ((Tn - 1) / Tn) * st$g1^2 / (2 * st$g2)
            else ((Tn - 1) / Tn) * st$g1 * (st$g1 - 1) / 2
      out[j] <- sum(L) + max(G0, 0)
    } else if (qj == 1) {
      keep1 <- Y <= (Tn - 1)
      h1 <- sum(L[keep1] * (Y[keep1] / Tn) * (digamma(Tn) - digamma(Y[keep1])))
      h2 <- if (st$A == 1 || st$g1 == 0) 0 else {
        rs <- seq_len(Tn - 1)
        (st$g1 / Tn) * (1 - st$A)^(1 - Tn) * (-log(st$A) - sum((1 - st$A)^rs / rs))
      }
      out[j] <- tbar * exp((h1 + h2) / tbar)
    } else if (qj == 2) {
      out[j] <- tbar^2 / sum(L * Y * (Y - 1) / (Tn * (Tn - 1)))
    } else {
      stop("solo q en {0,1,2} soportado")
    }
  }
  out
}

`%||%` <- function(a, b) if (is.null(a)) b else a


#' Curva de rarefaccion/extrapolacion de Hill filogenetico (y su valor observado).
#'
#' @param comm parcelas x especies, 0/1
#' @param tree arbol filogenetico
#' @param sizes numero de unidades de muestreo en que evaluar la curva
#' @param q vector de ordenes de Hill (default 0, compatible con las llamadas existentes
#'   de scripts/26 y tests/test_phylo.R -- agrega una columna `q` a la salida pero con un
#'   solo valor, no cambia ninguna columna que ya se leia por nombre)
#' @return data.frame largo con n, q, pd, meanpd, method
pd_curve <- function(comm, tree, sizes = NULL, q = 0, knots = 40) {
  Tn <- nrow(comm)
  br <- branch_incidence(comm, tree)
  Y <- br$Y; L <- br$L
  Tdepth <- reference_time(tree, colnames(comm)[colSums(comm) > 0])
  if (is.null(sizes)) {
    sizes <- unique(round(seq(1, 2 * Tn, length.out = knots)))
    sizes <- sort(unique(c(sizes, Tn)))
  }
  tbar <- sum(Y * L) / Tn
  pd_obs_total <- sum(L)

  need_ext <- any(sizes > Tn)
  st <- if (need_ext) .branch_singleton_stats(Y, L, Tn) else NULL
  asy <- if (need_ext) .pd_asymptotic(Y, L, Tn, tbar, q, st) else NULL
  obs_vals <- hill_from_incidence(Y, L, Tn, q)
  rpd_tn1 <- if (need_ext && 1 %in% q) .rarefy_at_t(Y, L, Tn, Tn - 1, 1, tbar) else NULL

  rows <- lapply(sizes, function(t_size) {
    if (t_size < Tn) {
      vals <- .rarefy_at_t(Y, L, Tn, t_size, q, tbar)
      method <- "Rarefaction"
    } else if (t_size == Tn) {
      vals <- obs_vals
      method <- "Observed"
    } else {
      vals <- setNames(numeric(length(q)), as.character(q))
      for (j in seq_along(q)) {
        qj <- q[j]
        if (qj == 0) {
          # formula existente, sin cambios: extrapolacion Chao2-tipo directa sobre G0,
          # no pasa por asy/beta como q=1
          G0 <- if (st$g2 > 0) ((Tn - 1) / Tn) * st$g1^2 / (2 * st$g2)
                else ((Tn - 1) / Tn) * st$g1 * (st$g1 - 1) / 2
          G0 <- max(G0, 0)
          tstar <- t_size - Tn
          vals[j] <- if (G0 <= 0 || st$g1 <= 0) pd_obs_total
                    else pd_obs_total + G0 * (1 - (1 - st$g1 / (Tn * G0 + st$g1))^tstar)
        } else if (qj == 1) {
          rpd1 <- rpd_tn1[["1"]]
          asy1 <- asy[["1"]]; obs1 <- obs_vals[["1"]]
          denom <- asy1 - rpd1
          beta <- if (denom == 0) 1 else (obs1 - rpd1) / denom
          beta <- min(max(beta, 0), 1)
          vals[j] <- obs1 + (asy1 - obs1) * (1 - (1 - beta)^(t_size - Tn))
        } else if (qj == 2) {
          vals[j] <- 1 / sum((L / tbar^2) * ((1 / t_size) * (Y / Tn) +
                             ((t_size - 1) / t_size) * (Y * (Y - 1) / (Tn * (Tn - 1)))))
        } else {
          stop("solo q en {0,1,2} soportado para extrapolacion")
        }
      }
      method <- "Extrapolation"
    }
    data.frame(n = t_size, q = q, pd = as.numeric(vals), meanpd = as.numeric(vals) / Tdepth,
               method = method, stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}


#' Banda de dispersion por submuestreo SIN reemplazo de las unidades de muestreo.
#'
#' Devuelve, para cada tamano t <= T y cada orden `q`, los cuantiles de la diversidad
#' taxonomica (`sr_*`) y filogenetica (`pd_*`/`meanpd_*`) de Hill obtenidas al tomar t
#' parcelas al azar. Es la dispersion exacta de la cantidad que la curva de rarefaccion
#' estima como media, asi que la banda esta centrada por construccion -- misma convencion
#' que `vegan::specaccum`, extendida de riqueza pura a diversidad de Hill de cualquier
#' orden via `hill_from_incidence()`.
#'
#' Con `q` de un solo valor, las columnas quedan sin sufijo (`sr_lo`, `pd_hi`, ...) --
#' compatibilidad con scripts/26 y tests/test_phylo.R, que las leen por ese nombre. Con
#' varios `q` se agregan ademas columnas `*_q{orden}` por cada uno.
#'
#' POR QUE NO EL BOOTSTRAP DE iNEXT. Se probaron los dos alternativos y los dos fallan de
#' forma medible (remuestreo CON reemplazo: banda sistematicamente por debajo por las
#' copias repetidas; bootstrap parametrico Y_i/T: pierde singletons a una tasa distinta de
#' la real). Solo cubre el tramo interpolado -- para el extrapolado no hay analogo por
#' submuestreo, y dejarlo sin banda es la lectura honesta.
subsample_band <- function(comm, tree, sizes, q = 0, reps = 100, conf = 0.95, seed = 42) {
  Tn <- nrow(comm)
  sizes <- sort(unique(sizes[sizes >= 1 & sizes <= Tn]))
  Tdepth <- reference_time(tree, colnames(comm)[colSums(comm) > 0])

  tips <- intersect(colnames(comm), tree$tip.label)
  tr <- if (length(tips) < ape::Ntip(tree)) ape::keep.tip(tree, tips) else tree
  ntip <- ape::Ntip(tr)
  parts <- ape::prop.part(tr)
  memb <- matrix(0L, nrow = ntip, ncol = ntip + tr$Nnode)
  memb[cbind(seq_len(ntip), seq_len(ntip))] <- 1L
  for (i in seq_along(parts)) memb[parts[[i]], ntip + i] <- 1L
  rownames(memb) <- tr$tip.label
  hit <- (comm[, rownames(memb), drop = FALSE] %*% memb) > 0   # parcelas x nodos (ramas)
  Lb <- numeric(ncol(hit))
  Lb[tr$edge[, 2]] <- tr$edge.length
  has_sp <- comm[, tips, drop = FALSE] > 0                      # parcelas x especies

  a <- (1 - conf) / 2
  set.seed(seed)
  out <- lapply(sizes, function(t_size) {
    m <- vapply(seq_len(reps), function(r) {
      idx <- sample.int(Tn, t_size)
      Ysp <- colSums(has_sp[idx, , drop = FALSE])
      Ybr <- colSums(hit[idx, , drop = FALSE])
      c(hill_from_incidence(Ysp, rep(1, length(Ysp)), t_size, q),
        hill_from_incidence(Ybr, Lb, t_size, q))
    }, numeric(2 * length(q)))
    row <- data.frame(n = t_size)
    for (j in seq_along(q)) {
      row[[paste0("sr_lo_q", q[j])]] <- unname(quantile(m[j, ], a))
      row[[paste0("sr_hi_q", q[j])]] <- unname(quantile(m[j, ], 1 - a))
      row[[paste0("pd_lo_q", q[j])]] <- unname(quantile(m[length(q) + j, ], a))
      row[[paste0("pd_hi_q", q[j])]] <- unname(quantile(m[length(q) + j, ], 1 - a))
    }
    row
  })
  d <- do.call(rbind, out)
  for (j in seq_along(q)) {
    d[[paste0("meanpd_lo_q", q[j])]] <- d[[paste0("pd_lo_q", q[j])]] / Tdepth
    d[[paste0("meanpd_hi_q", q[j])]] <- d[[paste0("pd_hi_q", q[j])]] / Tdepth
  }
  if (0 %in% q) {
    for (base in c("sr_lo", "sr_hi", "pd_lo", "pd_hi", "meanpd_lo", "meanpd_hi")) {
      d[[base]] <- d[[paste0(base, "_q0")]]
    }
  }
  rownames(d) <- NULL
  d
}


#' Curva de PD con la banda de `subsample_band()` adosada (NA en el tramo extrapolado).
pd_curve_ci <- function(comm, tree, sizes = NULL, q = 0, knots = 40, reps = 100,
                        conf = 0.95, seed = 42) {
  base <- pd_curve(comm, tree, sizes, q, knots)
  if (reps < 2) {
    base[c("pd_lo", "pd_hi", "meanpd_lo", "meanpd_hi")] <- NA_real_
    return(base)
  }
  band <- subsample_band(comm, tree, unique(base$n), q, reps = reps, conf = conf, seed = seed)
  i <- match(base$n, band$n)
  for (col in c("pd_lo", "pd_hi", "meanpd_lo", "meanpd_hi")) base[[col]] <- band[[col]][i]
  base
}


#' Comprueba esta implementacion contra iNEXT.3D sobre un submuestreo pequeno, en q=0.
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


#' Igual que `validate_pd_inext()`, pero para q=1 y q=2 -- la compuerta que autoriza a usar
#' `pd_curve(..., q=c(0,1,2))` sobre datos reales en vez de la ruta O(n^2) de iNEXT.3D.
validate_pd_inext_multiq <- function(comm, tree, n_units = 150, q = c(1, 2), tol = 1e-4,
                                     seed = 1) {
  set.seed(seed)
  idx <- sample.int(nrow(comm), n_units)
  cs <- comm[idx, , drop = FALSE]
  cs <- cs[, colSums(cs) > 0, drop = FALSE]
  tr <- ape::keep.tip(tree, colnames(cs))

  sizes <- unique(round(seq(1, 2 * n_units, length.out = 6)))
  ref <- iNEXT.3D::iNEXT3D(list(A = t(cs)), diversity = "PD", q = q,
                           datatype = "incidence_raw", size = sizes, nboot = 0,
                           PDtree = tr, PDtype = "meanPD")$PDiNextEst$size_based
  eff <- intersect(c("nt", "mT", "m"), names(ref))[1]
  qcol <- intersect(c("Order.q", "order", "q"), names(ref))[1]

  mine <- pd_curve(cs, tr, sizes = unique(ref[[eff]]), q = q)

  d <- merge(
    data.frame(n = ref[[eff]], q = ref[[qcol]], inext = ref$qPD),
    mine[, c("n", "q", "meanpd")], by = c("n", "q")
  )
  names(d)[names(d) == "meanpd"] <- "mine"
  d$rel <- abs(d$mine - d$inext) / pmax(d$inext, 1e-12)
  res <- by(d, d$q, function(z) c(q = z$q[1], max_rel_error = max(z$rel)))
  list(table = d, by_q = do.call(rbind, res), max_rel_error = max(d$rel),
       ok = max(d$rel) < tol)
}
