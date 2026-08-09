# Qué limita el ajuste, y dónde está el margen que queda

Resultado de la fase de diagnóstico que precede al cribado de predictores. El punto de
partida era el de [`09_predictors.md`](09_predictors.md): *"el margen que queda está en el
predictor, no en el modelo"*. Eso resultó ser cierto solo para la mitad del problema, y por
una razón distinta a la esperada.

Tres cosas cambian lo que hay que hacer:

1. **La diversidad alfa no es predecible entre contribuyentes, y el R² negativo que se
   reporta es error de nivel, no falta de señal.**
2. **Los dos ejes PCoA que se modelan ponen un techo bajo a lo que "predecir composición"
   puede significar** — y modelar un solo eje representa la composición mejor que modelar
   dos.
3. **La disimilitud composicional está saturada:** el 58 % de los pares de parcelas no
   comparte ninguna especie. Eso favorece una familia de modelo que el proyecto no ha
   probado (GDM), y en la primera corrida esa familia ya supera al diseño actual.

---

## 1. El R² negativo de alfa es un desplazamiento de nivel entre contribuyentes

### La observación

Los mismos modelos, bajo esquemas de validación distintos:

| esquema | agrupa por | R²_alfa (mejor modelo) | R²_beta |
|---|---|---|---|
| `kfold5_block20` | geometría (20 km) | **+0,22 a +0,31** | +0,40 |
| `kfold5_window` | ventana compartida | **+0,39** | +0,49 |
| `kfold5_owner` | contribuyente | **−0,43 a −0,66** | +0,27 |
| `kfold5_dataset` | proyecto | −0,61 | +0,31 |
| `lodo_owner` | contribuyente, uno fuera | −0,61 | +0,27 |

Alfa se predice bien bajo los esquemas geométricos y colapsa exactamente bajo los que
agrupan por quién tomó el dato.

### La causa

La riqueza tiene **η² = 0,72 entre contribuyentes**: el 72 % de su varianza es *entre*
personas, no entre sitios. Las medias por contribuyente, con tamaños de parcela comparables:

| contribuyente | n | área mediana (m²) | hill_q0 medio |
|---|---|---|---|
| Dobbs, C. & Miranda, M.D. | 89 | 400 | **1,6** |
| Miranda, A. | 113 | 100 | 2,5 |
| Lopatin, J. | 181 | 113 | 2,9 |
| … | | | |
| sPlot | 21 | 100 | 24,2 |
| Becerra, P.I. | 60 | 500 | **28,2** |

Un factor 17 en riqueza media entre contribuyentes con parcelas del mismo tamaño. El área
explica poco de ese salto (ρ = +0,23 entre la media por contribuyente y su área mediana);
lo que cambia es qué se registra, no cuánta superficie se recorrió.

Cuando se excluye a Becerra, el modelo nunca vio una media de 28 y predice ~5. El R² respecto
de la varianza propia de Becerra se desploma. Cuando se agrupa por bloques geométricos, en
cambio, cada contribuyente aparece a ambos lados de la partición y el modelo **memoriza su
nivel vía las coordenadas** — que es exactamente por qué `B03_coords`, un modelo de solo
longitud, latitud y elevación, es el mejor de los 79 bajo `kfold5_random` (R² = +0,562,
contra +0,518 del mejor RF).

### Cuánto de eso es recuperable

Techo analítico bajo CV agrupada: un modelo que acierte la desviación dentro del grupo pero
que no pueda conocer el nivel del grupo excluido alcanza **R² = 1 − η²**. Contra eso, tres
variantes del mismo RF (`curve+topo+area`, kNDVI, `kfold5_owner`, 3 semillas):

| | A: como está | B: nivel de owner removido del entrenamiento | C: media global | techo |
|---|---|---|---|---|
| hill_q0 | −0,245 | −0,129 | −0,031 | +0,279 |
| hill_q1 | −0,633 | **−0,039** | −0,021 | +0,391 |
| hill_q2 | −0,793 | **−0,045** | −0,066 | +0,312 |
| **R²_alfa** | **−0,557** | **−0,071** | −0,039 | +0,327 |
| lcbd_pa | +0,170 | +0,148 | −0,165 | +0,659 |
| pcoa1_pa | +0,393 | +0,373 | −0,118 | +0,708 |
| pcoa2_pa | +0,189 | +0,098 | −0,120 | +0,635 |
| **R²_beta** | **+0,250** | +0,206 | −0,134 | +0,667 |

Dos lecturas, y las dos importan:

- **El 87 % del déficit de alfa era error de nivel.** El modelo no solo era poco informativo
  para un contribuyente nuevo: era *peor que no predecir nada*, porque usaba rasgos
  correlacionados con la identidad del contribuyente y los extrapolaba con el signo
  equivocado.
- **Lo que queda tras quitar el nivel es cero.** B llega a −0,071 y predecir la media global
  llega a −0,039. No hay señal de riqueza *dentro* de un contribuyente en estos predictores.

Beta se comporta al revés: quitar el nivel no la mejora (+0,250 → +0,206), así que su señal
no es memorización de nivel; y su techo es +0,667 contra +0,250 alcanzado.

### Qué hacer con esto

Alfa se reporta, nunca se optimiza, y el número que se reporta va acompañado del η². Todo el
esfuerzo de predictores va a beta. `kfold5_owner` sigue siendo el esquema primario **para
beta**; para alfa ningún esquema de este conjunto de datos produce un número interpretable, y
decirlo es el resultado.

Nótese que `kfold5_window` **no** es un esquema más estricto que `kfold5_owner`: cierra la
fuga por píxeles compartidos (642 componentes, cero solapamiento por construcción) pero
reparte a cada contribuyente entre folds, así que se comporta como los esquemas geométricos.
Las dos cosas son fugas distintas y ningún esquema único cierra ambas salvo
`kfold5_owner_window`, que solo deja 9 grupos y es demasiado grueso para 5 folds.

---

## 1b. El esquema queda fijado: `kfold5_window`

**Decisión de proyecto (J. Lopatin): `kfold5_window` es el esquema primario único de aquí en
adelante, y alfa se reporta sola, sin descomposición adjunta.** Esta sección deja registrada
la evidencia que respalda esa elección y la que la limita, porque un revisor va a pedir las
dos.

### Lo que la respalda

Bajo `kfold5_window` la validación cruzada es independiente en el sentido que importa para el
uso previsto: cada parcela se puntúa desde un modelo que nunca vio ni esa parcela ni ninguno
de los píxeles de su ventana de extracción. Es el único esquema del proyecto con **cero fuga
de ventana por construcción**, y el escenario que reproduce —predecir en una parcela nueva
dentro de un territorio ya muestreado— es exactamente el de un mapa.

La prueba decisiva se corrió dentro del propio esquema, sin comparar contra ningún otro
(`scripts/24_alpha_decomposition.py`). Descompone las mismas predicciones OOF que producen el
número titular en dos partes: acertar el nivel medio de cada contribuyente, y ordenar las
parcelas *dentro* de un contribuyente, que es la parte que solo puede venir del ambiente.

| bajo `kfold5_window` | R²_total | within-owner | between-owner |
|---|---|---|---|
| **beta**, 87 predictores | +0,562 | **+0,380** | +0,765 |
| alfa, 87 predictores | +0,569 | +0,063 | +0,751 |
| beta, solo lon/lat/elev (3 col) | +0,508 | +0,305 | +0,735 |
| alfa, solo lon/lat/elev (3 col) | +0,490 | −0,126 | +0,704 |

**Beta se sostiene sin reservas.** Con el nivel del contribuyente removido, el modelo todavía
explica +0,380 de la composición, sobre el 66–71 % de varianza que es intra-contribuyente. Eso
es señal ambiental real, medida en el esquema que se está usando para reportar.

### Lo que hay que declarar igual

- **Alfa es casi toda nivel de contribuyente.** Su R² intra-contribuyente es +0,063, y
  `hill_q2` queda en −0,095. El 88 % del +0,569 es acertar quién censó, lo que era esperable
  con η² = 0,72 (§1). Se reporta el número sin descomposición, por decisión de proyecto; el
  η² y esta tabla quedan aquí para quien pregunte.
- **Tres columnas de posición hacen casi todo el trabajo.** `lon/lat/elev` alcanzan el 90 % del
  beta y el 86 % del alfa que alcanzan los 87 predictores de fenología y clima. El aporte
  propio de la teledetección bajo este esquema es +0,054 en beta y +0,079 en alfa. Es el
  número que un revisor va a buscar, y conviene que esté escrito antes de que lo pida.
- **Pérdida de comparabilidad hacia atrás.** Los 79 modelos de `08_modelling.md` se corrieron
  con `kfold5_owner` como primario y no tienen `kfold5_window`. Para que las tablas hablen del
  mismo número hay que re-correr esa matriz bajo el esquema nuevo (ver
  [`11_next_steps.md`](11_next_steps.md), experimento 1).
- **`kfold5_owner` pasa a diagnóstico**, junto con `kfold5_random`: el primero acota cuánto del
  R² depende de la identidad del contribuyente, el segundo la brecha de optimismo.

### La medición que ordena los esquemas

Cuánta fuga de ventana queda realmente viva en cada partición realizada, que no es lo mismo
que cuántos componentes existen:

| esquema | componentes partidos (de 135) | parcelas con píxeles a ambos lados |
|---|---|---|
| `kfold5_random` | 126 | 557 (51,5 %) |
| `kfold5_owner` | **7** | **47 (4,3 %)** |
| `kfold5_block20` | 0 | 0 |
| `kfold5_window` | 0 | 0 |

`kfold5_window` y `kfold5_block20` llegan a cero, pero por razones distintas y solo el primero
lo hace **por construcción**: una línea de grilla de 20 km no sabe dónde cae una ventana de
150 m, y otro desplazamiento de grilla podría cortar un componente. `kfold5_owner` deja 7 de
135 componentes partidos (47 parcelas) porque cierra los otros 128 de forma incidental —los
contribuyentes están agrupados espacialmente—, y `kfold5_random` deja partido el 51,5 % de las
parcelas, que es por qué su R² no es un resultado sino la medida de la brecha de optimismo.

**Los dos esquemas no son comparables como "más o menos estricto": cierran fugas distintas.**
`kfold5_window` cierra la de ventana y reparte a cada contribuyente entre folds;
`kfold5_owner` hace lo inverso. Ninguno cierra ambas salvo `kfold5_owner_window`, que deja 9
grupos y es demasiado grueso para 5 folds. La consecuencia práctica de elegir `window` está en
la tabla de arriba: alfa deja de ser interpretable como señal ambiental, y por eso el proyecto
la reporta como descriptor y concentra el esfuerzo de predictores en beta.

Que la elección importa se ve en que la concordancia de ranking entre los dos esquemas sobre
las 19 filas del cribado es solo ρ = +0,53: coinciden en el ganador (X17) pero `clima+topo` es
3.º bajo `window` y 14.º bajo `owner`. Fijar uno antes de leer la tabla no es formalismo.

---

## 2. Los dos ejes PCoA son el cuello de botella del target

Tras la corrección de Cailliez, dos ejes llevan el **12,5 %** de la masa de autovalores
(presencia/ausencia) y el **13,8 %** (cover). Un R² de 1,0 sobre `pcoa1_pa` y `pcoa2_pa`
seguiría sin decir nada de los otros siete octavos.

La pregunta honesta es cuánto de la **disimilitud observada** puede representar un conjunto de
k ejes. Reconstruyendo la distancia euclidiana desde los k ejes verdaderos y correlacionándola
por rangos con el Jaccard observado sobre los 584.821 pares:

| ejes | varianza acumulada | ρ(Jaccard observado, distancia reconstruida) |
|---|---|---|
| 1 | 7,1 % | **+0,359** |
| **2** (lo que se modela hoy) | 12,5 % | **+0,305** |
| 4 | 21,0 % | +0,321 |
| 8 | 31,3 % | +0,386 |
| 16 | 42,8 % | +0,473 |
| 32 | 55,0 % | +0,550 |

**Un solo eje representa la composición mejor que dos.** El eje 2 no está dominado por
valores extremos (curtosis +0,3, ningún |z| > 6, las 10 parcelas más extremas aportan el 7 %
de su varianza), así que no es un artefacto de outliers: es que añadir un eje de baja varianza
a una reconstrucción euclidiana mete más distorsión que estructura mientras la matriz siga
saturada. La escalera se vuelve monótona recién a partir de 8 ejes.

Consecuencia directa: el techo de todo el diseño actual, con un modelo *perfecto*, es
ρ = +0,305.

---

## 3. La disimilitud está saturada, y eso selecciona la familia de modelo

Sobre los 584.821 pares de parcelas:

- **57,6 % tiene Jaccard = 1,0 exacto** — no comparten ninguna especie.
- La mediana de la disimilitud es exactamente 1,0; la media, 0,926.
- El 70,6 % está por encima de 0,9.

No es un problema de resolución taxonómica: agregando a género (313 taxones en vez de 570) la
saturación baja apenas de 57,6 % a 53,0 % y la varianza de dos ejes sube de 12,5 % a 13,4 %.
Es una propiedad real de muestrear parcelas con mediana de 5 especies a lo largo de ~1.500 km
de gradiente latitudinal.

Una ordenación lineal tiene que gastar ejes representando una distancia que dejó de variar.
**Generalized Dissimilarity Modelling** (Ferrier et al. 2007,
[10.1111/j.1472-4642.2007.00341.x](https://doi.org/10.1111/j.1472-4642.2007.00341.x)) fue
diseñado precisamente para esto: su enlace `d = 1 − exp(−η)` es asintótico en 1 por
construcción, y sus I-splines monótonas por predictor permiten que la tasa de recambio varíe a
lo largo de un gradiente. La variante para predictores de teledetección de alta dimensión es
**SGDM** (Leitão et al. 2015, *Methods in Ecology and Evolution* 6:1204-1214,
[10.1111/2041-210X.12378](https://doi.org/10.1111/2041-210X.12378)), que reduce el espacio
ambiental con una correlación canónica dispersa antes de ajustar el GDM.

### Primera corrida, `kfold5_owner`, `gm+seas_all+topo` (62 predictores)

Todo puntuado en el mismo espacio y sobre los mismos pares entre parcelas excluidas
(ρ de Spearman contra el Jaccard observado):

| | ρ |
|---|---|
| GDM solo distancia geográfica | +0,435 |
| **GDM con los predictores** | **+0,492** |
| SGDM (sCCA fold-local + GDM) | +0,436 |
| RF sobre 8 ejes PCoA → distancia | +0,425 |
| techo: los **2 ejes verdaderos** | +0,465 |
| techo: los 8 ejes verdaderos | +0,530 |

**El GDM desde satélite reproduce la composición observada mejor que conocer los valores
verdaderos de los dos ejes que el proyecto modela hoy** (+0,492 contra +0,465), y queda cerca
del techo de 8 ejes (+0,530).

> **Los dos "techos de 2 ejes" de este documento no son el mismo número y no deben
> compararse.** El +0,305 de la sección 2 es global, sobre los 584.821 pares del conjunto
> completo. El +0,465 de esta tabla es por fold, sobre los pares *entre parcelas excluidas*,
> que al venir de un subconjunto de contribuyentes cubren un rango geográfico más estrecho y
> están por tanto menos saturados. Cada tabla es internamente consistente; cruzarlas no lo
> es.

Dos cautelas que van en el diseño, no descubiertas después:

- La distancia geográfica sola ya da +0,435. El aporte propio de los predictores es
  +0,057, y hay que medirlo con geografía dentro del modelo, no contra el modelo nulo —
  por eso se añadió la variante `gdm_geo`.
- El SGDM va *por debajo* del GDM simple. Leitão et al. eligen las penalizaciones L1 por
  búsqueda en grilla contra el desempeño validado; con las penalizaciones por defecto la
  reducción está perdiendo información. Sin esa búsqueda el número de SGDM no es
  interpretable.

---

## 4. Lo que se construyó para responder esto

**El pipeline completo ya corre sin R.** `scripts/07b_compute_responses_python.py` reproduce
`scripts/07_compute_taxonomic_beta_responses.R` a precisión de máquina — Hill, LCBD (jaccard y
hellinger) y PCoA con corrección de Cailliez coinciden con la salida de R en las 11 columnas;
solo difieren el signo de `pcoa1` (arbitrario en cualquier descomposición espectral, sin
efecto sobre ningún R²) y `p_lcbd` (otra semilla de permutación, r = 0,999). Eso desbloquea
esta máquina, que es la que tiene los cubos.

**Los cubos `.nc` tenían la serie completa.** Los 1.082 archivos (346 MB) guardan
`obs_{ndvi,evi,kndvi,nbr,savi}` y `obs_band_{blue,…,swir2}` en `(time, y, x)`, 80–265 fechas
por parcela, con la máscara de nubes ya aplicada y la ventana causal de 3 años. La fase 0 del
doc 09 se resuelve por la afirmativa: **nada de lo que sigue necesita datacube.**

Predictores nuevos, todos por píxel y luego agregados con regla explícita
(`scripts/18`, `18b`):

| familia | qué es | columnas |
|---|---|---|
| `gm` | geomediana de 6 bandas + 5 índices derivados + las 3 MAD + `count` | 15 |
| `obscomp` | estadísticos de distribución sobre observaciones reales, no sobre la curva | 65 |
| `seas` | mediana por estación austral + contraste verano−invierno | 30 |
| `composite` | los mismos estadísticos sobre la curva de 52 pasos | 60 |
| `contrast` | diferencias y razones entre índices (NDVI−SAVI = exposición de suelo) | 30 |
| `svh` | dispersión entre los 25 píxeles | 220 |

**Cero NaN en las 80 columnas derivadas de los cubos**, contra el 5,1 % de píxeles que fallan
en LSP. La agregación conserva `median`, `mean`, `trimmed`, `sd`, `cv`, `center` y
`n_px_valid` en paralelo, así que "qué regla de agregación" es un factor cribado y no un
`np.nanmean` que nadie eligió.

---

## 4b. El clima es débil por separado y fuerte en conjunto

La hipótesis que motivó traer CR2MET era que la distancia geográfica, que sola alcanza
ρ = +0,435 contra la disimilitud observada, estaba haciendo de sustituto del gradiente de
aridez. Marginalmente, ningún predictor climático se acerca a los espectrales.

Correlación de Spearman con `pcoa1_pa`, sobre los 149 predictores de
`composite_all+contrast+gm+seas_all+clim`:

| predictor | ρ | familia |
|---|---|---|
| `ctr_kndvi_savi_diff` | **−0,697** | contraste entre índices (nuevo) |
| `ctr_ndvi_savi_diff` | −0,693 | contraste entre índices (nuevo) |
| `gmA_gm_band_red` | +0,680 | reflectancia cruda de la geomediana (nueva) |
| `gmA_gm_band_swir2` | +0,670 | reflectancia cruda de la geomediana (nueva) |
| `kndvi_comp_mean` | −0,664 | el mejor que ya existía |
| `clim_p_wettest_quarter` | **−0,386** | el mejor climático |

Dos lecturas:

- **Los predictores univariados más fuertes del proyecto son nuevos, y ninguno es
  fenológico.** La diferencia kNDVI−SAVI es una señal de exposición de suelo que ningún
  índice lleva por separado; la reflectancia roja y SWIR2 de la geomediana no existían
  porque el proyecto solo guardaba índices derivados, y los índices son irreversibles.
- **Fuerza marginal no es aporte conjunto.** X12, que reúne todo lo que no tiene forma
  temporal (161 columnas), da R²_beta = +0,197 bajo `kfold5_owner` contra +0,250 de la curva
  sola. Los predictores nuevos son fuertes por separado y redundantes entre sí: todos miden
  la misma posición en el gradiente de aridez, que es también lo que mide la curva.

**La inversa también resultó cierta, y era la que no se esperaba.** El peor predictor
marginal del proyecto es el que más aporta cuando se lo pone junto a la curva: `curve+clim`
(X17) es la mejor fila de las 19 del cribado bajo *ambos* esquemas. El clima no es redundante
con la fenología porque mide otra cosa — la fenología describe qué hace la vegetación en el
sitio, el clima describe con qué cuenta —, y esa complementariedad no se ve en ninguna
correlación univariada. La sección siguiente lo cuantifica.

---

## 4c. El cribado completo: qué bloque, y cuál era el contraste que importaba

19 filas × 2 esquemas × 3 semillas con Random Forest
(`results/tables/block_screen.csv`, resumen en `scripts/23_screen_report.py`). R²_beta es el
promedio sobre `lcbd_pa`, `pcoa1_pa` y `pcoa2_pa`; el ruido entre semillas es 0,016–0,024 bajo
`kfold5_owner` y 0,004 bajo `kfold5_window`, así que solo las diferencias que lo superan se
leen como diferencias.

| fila | bloque | p | R²_beta owner | R²_beta window |
|---|---|---|---|---|
| **X17** | **curva kNDVI + clima + topo** | 87 | **+0,309** | **+0,562** |
| X18 | las 5 curvas + clima + topo | 295 | +0,309 | +0,562 |
| X16 | clima + topo, sin teledetección | 35 | +0,202 | +0,542 |
| X13 | las 5 curvas apiladas (= RF06, el mejor publicado) | 281 | +0,268 | +0,516 |
| X08 | geomediana + curva | 88 | +0,257 | +0,526 |
| **X03** | **la curva de 52 pasos — LA REFERENCIA** (= RF03) | 73 | **+0,250** | **+0,507** |
| X04 | compuesto anual + curva | 133 | +0,245 | +0,504 |
| X02 | LSP actual, la línea base del campo (= RF01) | 39 | +0,239 | +0,510 |
| X07 | geomediana + las 3 MAD | 36 | +0,222 | +0,517 |
| X10 | fenología reducida a 4 medianas estacionales | 51 | +0,219 | +0,507 |
| X14 | todo junto, el techo empírico | 511 | +0,205 | +0,511 |
| X01 | el control no fenológico completo | 81 | +0,204 | +0,490 |
| X12 | todo lo que NO tiene forma temporal | 161 | +0,197 | +0,521 |
| X15 | clima solo | 18 | +0,177 | +0,516 |
| X00 | compuesto anual solo, sin forma ni fecha | 60 | +0,157 | +0,405 |
| X05 | heterogeneidad espectral entre píxeles, sola | 241 | +0,146 | +0,452 |

Cuatro resultados:

**1. La forma fenológica sí supera al compuesto anual.** X03 contra X01: +0,046 bajo `owner`
(ruido 0,016) y +0,017 bajo `window` (ruido 0,004). Contra el compuesto *sin topografía* (X00)
la distancia es mucho mayor: +0,093 y +0,102. La reformulación que `09_predictors.md` §6
obligaba a estar dispuesto a reportar —"un compuesto anual iguala a la curva"— **no ocurre**.

**2. El compuesto no aporta nada sobre la curva.** X04 contra X03: −0,005 y −0,002, ambas por
debajo del ruido. Toda la información del compuesto anual ya está en la curva; el recíproco es
falso. Ese era el contraste central del doc 09 y queda resuelto en la dirección favorable al
diseño actual.

**3. El clima es la única palanca real que apareció.** X17 supera a X03 en +0,058 (`owner`) y
+0,055 (`window`), y supera a X16 —el clima sin nada de teledetección— en +0,107 y +0,020. Los
dos bloques se necesitan mutuamente. X17 con 87 columnas supera además a `RF06` (X13, 281
columnas), el mejor modelo publicado del proyecto, y X18 muestra que apilar los otros cuatro
índices sobre kNDVI no añade nada una vez que el clima está dentro.

> **Cautela sobre `kfold5_window`.** Ahí el clima solo (X15, 18 columnas) alcanza +0,516 y casi
> iguala a la curva (+0,507). No hay que leerlo como que el clima es suficiente: las normales
> están en una grilla de 0,05° y las 1.082 parcelas caen en 253 celdas, así que el bloque es
> **casi una coordenada**, y `kfold5_window` reparte a cada contribuyente entre folds — el
> mismo mecanismo de memorización de nivel de la §1, por el que `B03_coords` era el mejor de
> los 79 modelos bajo `kfold5_random`. Bajo `kfold5_owner`, que es el esquema honesto para
> beta, el clima solo se queda en +0,202, por debajo de la curva. El aporte del clima es real,
> pero es *conjunto*, no autónomo.

**4. Más columnas es peor.** X14 (511 columnas, todo) da +0,205 contra +0,309 de X17 (87). Con
1.082 parcelas y 5 folds agrupados por contribuyente, el Random Forest se ahoga mucho antes de
quedarse sin señal. Cualquier modelo profundo hereda ese límite, y no es de arquitectura.

**5. La regla de agregación de los 25 píxeles no importa.** Fila X10 del doc 09, cribada sobre
los dos bloques que sí están hechos de píxeles, bajo `kfold5_owner`:

| | `median` | `mean` | `trimmed` | `center` | rango | ruido semillas |
|---|---|---|---|---|---|---|
| X07 `gm+MADs` | +0,222 | +0,228 | +0,221 | +0,234 | 0,013 | 0,013 |
| X12 todo sin forma | +0,197 | +0,208 | +0,199 | +0,196 | 0,012 | 0,014 |

El rango entre las cuatro reglas es exactamente el ruido entre semillas. Lo llamativo es
`center`: **el píxel central solo iguala a cualquier promedio de los 25**, así que la ventana
de 5×5 no está comprando señal, solo suavizando. Eso vuelve discutible el diseño de ventana
—y hace que `kfold5_window`, que existe para proteger esa ventana, pague un costo por algo que
aporta poco— pero también significa que la elección de `median` en todas las demás filas no
sesga ninguna conclusión.

---

## 4d. Las familias de modelo, sobre cinco conjuntos de predictores

`results/tables/gdm_comparison.csv`, `kfold5_owner`, ρ de Spearman contra el Jaccard observado
sobre los pares entre parcelas excluidas de cada fold:

| predictores | GDM | GDM+geo | SGDM | RF sobre 8 ejes |
|---|---|---|---|---|
| `curve+topo` | +0,482 | +0,522 | +0,434 | +0,417 |
| `gm+topo` | +0,489 | +0,520 | +0,435 | +0,407 |
| `composite_all+topo` | +0,469 | +0,519 | +0,419 | +0,406 |
| `gm+obscomp+seas+contrast+topo` | +0,491 | **+0,527** | +0,431 | +0,417 |
| `curve+gm+obscomp+seas+contrast+topo` | +0,493 | +0,526 | +0,444 | +0,418 |
| *referencia:* solo distancia geográfica | +0,435 | | | |
| *techo:* los 2 ejes verdaderos | +0,465 | | | |
| *techo:* los 8 ejes verdaderos | +0,530 | | | |

- **La elección de predictores casi no mueve la aguja: 0,024 separa a las cinco filas en
  `gdm_geo`.** El GDM con predictores *y* geografía llega a +0,527, prácticamente el techo de
  los 8 ejes verdaderos (+0,530) y muy por encima del de los 2 ejes que el proyecto modela hoy
  (+0,465). El cuello de botella dejó de ser el predictor.
- **En espacio GDM la forma fenológica no gana.** `gm+topo` (geomediana, sin ninguna noción de
  tiempo, 36 columnas) da +0,489 contra +0,482 de la curva de 52 pasos. Es lo opuesto al
  cribado RF bajo `owner`, y no es contradicción: son objetivos distintos —recambio entre
  pares contra posición en un eje— y el resultado es que **qué predictor gana depende de qué
  se está prediciendo**, que es en sí un hallazgo reportable.
- **El SGDM sigue por debajo del GDM simple en las cinco filas.** Con las penalizaciones por
  defecto la reducción sCCA pierde información; la búsqueda en grilla que hace el paper está
  corriendo y hasta que termine ese número no es interpretable.

---

## 5. Qué queda

El cribado respondió sus dos contrastes centrales (§4c) y la comparación de familias corrió
sobre cinco conjuntos de predictores (§4d). Lo que sigue abierto:

1. **En vuelo** (`scripts/run_climate_queue.sh`): X19 —la fila que junta todo con clima—, el
   factor de agregación de los 25 píxeles (`median` / `mean` / `trimmed` / `center` sobre X07
   y X12), y las tres corridas de GDM con clima en el diseño, una de ellas con la búsqueda en
   grilla de las penalizaciones del SGDM. Hasta que esa grilla termine, la comparación
   SGDM–GDM no es interpretable.
2. **Decidir el objetivo, que es ahora la decisión de diseño principal.** El GDM con
   predictores y geografía llega a +0,527 contra un techo de +0,530 para 8 ejes verdaderos:
   por el lado del recambio entre pares queda poco margen. Los dos caminos son abandonar los
   ejes PCoA y reportar GDM, o subir a 8–16 ejes y aceptar que el R² por eje deja de ser
   comparable con las 79 corridas anteriores. No son compatibles y hay que elegir uno.
3. **Rehacer la compuerta hacia los modelos profundos con X17, no con X03.** La compuerta de
   `09_predictors.md` §6 comparaba contra la curva sola; el bloque que hay que pasar a MLP,
   CNN 1D y CNN 2D es `curve+clim+topo`. El resultado 4 de §4c advierte que el margen no está
   en la capacidad del modelo, así que la expectativa razonable es un empate.
4. Solo si los bloques locales se estancan: traer fuentes nuevas de Data Cube Chile (SAR de
   Sentinel-1, red-edge de Sentinel-2). Ninguna está en los cubos guardados, y son lo único
   para lo que el datacube sigue haciendo falta ahora que el clima ya se extrajo.
