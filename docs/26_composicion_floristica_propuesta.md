# La historia del paper, y qué hacer con la composición florística

Fecha: 2026-09-29. Reescrito el mismo día tras dos correcciones del autor.
Reemplaza la primera versión, que tenía mal el papel de la riqueza y sacaba a Isomap sin
razón suficiente.

Base: lo medido en este repo el 2026-09-29 (`scripts/107`, `scripts/109`,
`results/tables/sintesis_lenoso_resumen.csv`, `contraste_estres_lenoso.csv`), el corpus
`Floristic-composition` del vault (9 papers, 2004–2021), `docs/10_findings.md`, y una revisión
de literatura 2021–2026.

---

## 1. La historia

> **El satélite aporta donde la señal es de la parcela; no aporta donde la señal es del pool
> regional. Y aporta cuando hay agua.**

Tiene sentido mecánico antes que estadístico: un sensor ve el rodal, no la historia
biogeográfica. Lo que está definido por el pool regional de especies —la estructura
filogenética, la diversidad oscura, el eje térmico de la composición— ya lo codifican tres
coordenadas. Lo que está definido por el estado del rodal —cuánta vegetación hay, cuán inusual
es su composición, cuán húmedo está— es lo que el sensor puede ver.

### Lo que la sostiene

> **Cambio de encuadre (2026-09-30, decisión del autor).** El piso de coordenadas (`B03c`) sale
> de la estructura del paper. Lo que se reporta es el **R² centrado dentro de bandas de 2° de
> latitud**, que sigue siendo el control del gradiente y ya está en todas las tablas; y la
> Discusión declara la correlación alta entre las facetas y el gradiente latitudinal (rho hasta
> 0,83 con la latitud en el primer eje de ordenación), esperable por el efecto climático.
>
> Las columnas `piso coords` que siguen abajo quedan como **referencia interna**, no como
> resultado del paper. Lo que se pierde al sacarlas está anotado en §4b.

**Donde el satélite aporta:**

| faceta | mejor bloque | piso coords | **sobre coords** |
|---|---|---|---|
| **riqueza cruda** (Hill q₀, n = 3.102) | 0,388 | 0,231 | **+0,157** |
| **LCBD Sørensen** | 0,210 | 0,109 | **+0,101** |
| LCBD p/a | 0,313 | 0,215 | +0,098 |
| LCBD frecuencia | 0,242 | 0,156 | +0,086 |
| PCoA 2 p/a (eje hídrico) | 0,541 | 0,465 | +0,076 |

**Donde no aporta, y el nulo es limpio porque hay piso:**

| faceta | mejor bloque | piso coords | sobre coords |
|---|---|---|---|
| SES PD | 0,187 | 0,172 | +0,015 |
| diversidad oscura | 0,343 | 0,329 | +0,014 |
| SES MPD | 0,214 | 0,202 | +0,011 |
| MPD | 0,212 | 0,205 | +0,007 |
| PCoA 1 p/a (eje térmico) | 0,563 | 0,569 | **−0,006** |
| PCoA 1 frecuencia | 0,429 | 0,455 | **−0,026** |

**Y aporta cuando hay agua.** Contraste seco contra húmedo por tercil de SPI-12 residualizado
contra latitud dentro del año de censo, descontando al bloque remoto el cambio que el piso tiene
entre los mismos terciles: **17 de 20 facetas mejoran en año húmedo**. Los tres negativos son
chicos (−0,024 a −0,074).

El control del piso descarta la explicación aburrida: si fuera varianza del target, el piso se
movería igual entre terciles, y no se mueve. **El patrón es sólido.**

> **PERO NO HAY MECANISMO, y el más obvio está descartado.** La explicación que se propuso —que
> en año seco la vegetación se expresa menos, la separación espectral entre comunidades se
> colapsa, y el predictor pierde información sin que la comunidad cambie— se puede medir sin
> modelar, y falla. `scripts/110_mecanismo_sequia.py`, sobre Living Trees, entre pares dentro de
> la misma banda de 2°:
>
> | tercil | rho(espectro, Jaccard) | dispersión espectral | NDVI mediano |
> |---|---|---|---|
> | seco | 0,124 | 0,0757 | 0,821 |
> | medio | 0,124 | 0,0739 | 0,834 |
> | húmedo | 0,130 | 0,0761 | 0,781 |
>
> El acoplamiento espectro–composición es el mismo, la dispersión espectral también, y el NDVI va
> **al revés** de lo que la hipótesis pide. (Cautela: el tercil es SPI residualizado contra
> latitud dentro del año, o sea seco *respecto de sus vecinos ese año*, no seco en absoluto —
> eso explica que el NDVI no ordene, pero los otros dos contrastes son dentro de banda y tampoco
> ordenan.)
>
> El paper puede **reportar el patrón**, que está bien controlado, pero **no puede afirmar la
> causa**. Una correlación de rango entre pares es un instrumento marginal y débil, así que esto
> no prueba que no exista ningún mecanismo espectral: prueba que el más obvio no se sostiene.

Eso explica además por qué el geomediano le gana a la fenología: en un sistema limitado por agua
la fenología es errática, y Feilhauer et al. 2013 ya reportó que lo multiestacional ayuda de
forma inconsistente y a veces empeora el ajuste.

---

## 2. La corrección sobre la riqueza, que cambia el marco

La primera versión de este documento decía "composición, no riqueza". **Es falso.** Se
construyó sobre el TD₀ estandarizado por cobertura sin mirar la otra riqueza de la misma tabla.

**La riqueza cruda tiene el mayor margen sobre la geografía de las veinte facetas** (+0,157).
No va contra la literatura de la hipótesis de variación espectral: la confirma.

Lo que no se predice es la versión estandarizada por cobertura (TD₀, n = 881, margen +0,009), y
la causa está medida:

- El piso de ≥5 especies que exige la estandarización deja solo las parcelas ricas.
- La desviación del target cae de **sd 3,04 a 1,70**.
- El R² es relativo a la varianza, así que se destruye.
- **Prueba decisiva:** la riqueza cruda evaluada en esas mismas 881 parcelas da **−0,625**, peor
  todavía. Es el subconjunto, no la estandarización.

Eso convierte un resultado incómodo en un hallazgo metodológico sobre el campo: **la
estandarización por cobertura, que la disciplina recomienda cada vez más, destruye la señal de
riqueza en sistemas pobres en especies, al restringir la muestra a las parcelas ricas.** Va en
el paper con las dos versiones lado a lado.

---

## 3. Qué entra en el paper

1. **Las dos riquezas, lado a lado** — cruda (+0,157) y estandarizada por cobertura (+0,009),
   con el diagnóstico de varianza. Es resultado y es crítica metodológica.
2. **LCBD como la faceta de composición**, en sus tres variantes. Margen consistente +0,086 a
   +0,101.
3. **El nulo filogenético**, bien controlado. Se predice a 0,21 pero **es todo geografía**.
4. **El control de latitud**, como R² centrado dentro de bandas de 2°, aplicado a todas las
   facetas. En la Discusión: las facetas correlacionan fuerte con el gradiente latitudinal, lo
   que es esperable por el gradiente climático que corre con él, y por eso ningún R² agrupado se
   interpreta sin el centrado.
5. **El contraste de sequía** como el mecanismo del resultado principal, no como sección aparte.
6. **La descripción de composición en las parcelas** (ver §5), que es la capa ecológica que hoy
   falta.
7. **El control de Becerra** (`scripts/106`) al suplemento: justifica el pool leñoso.
8. **El espectro del PCoA** (`figS7`) al suplemento: justifica no usar una ordenación como
   respuesta principal.

**Reducir la tabla de 20 facetas.** La mayoría son geografía y diluyen el mensaje. Principal:
riqueza (las dos), LCBD, una filogenética. El resto al suplemento.

---

## 4. La decisión de ordenación: Isomap sí, GDM no (en este paper)

### Isomap entra, como sustitución

Los ejes de PCoA **ya están en el paper como variables respuesta**. Si la representación es
mala, estamos reportando modelos de una representación mala. Cambiar PCoA por Isomap es una
sustitución —mismo pipeline, misma tabla, distinta función de ordenación—, no una ampliación de
la pregunta.

Y la mejora es sustancial y está medida aquí, no sólo citada:

| | PCoA | Isomap (k=30) |
|---|---|---|
| varianza en 3 ejes | 17,5 % | **36,8 %** |
| techo de reconstrucción, 2 ejes | 0,419 | **0,577** |
| techo, 3 ejes | 0,444 | **0,606** |
| techo, 8 ejes | 0,502 | 0,636 |

**Isomap con 2 ejes le gana a PCoA con 8.** Y dos riesgos que yo había marcado no se
materializan en estos datos: el grafo kNN queda **conexo** con k = 10, 30 y 80, y **k importa
poco** (0,577–0,606 entre k = 10 y 80 con 3 ejes), así que la sensibilidad a k que Feilhauer
2011 declara como limitación es mucho menor aquí.

Respaldo: Feilhauer et al. 2011 (74 % contra 54 % del DCA), Harris et al. 2015 (82 % en beta
alta). Mecanismo que encaja: Isomap mide distancias **geodésicas** sobre un grafo de k vecinos,
así que los pares saturados —el 77,6 % de los nuestros, con Jaccard 1— nunca se comparan
directamente.

**Lo que hay que medir, no argumentar:** una representación con más información no es
automáticamente más predecible. El techo sube seguro; que el satélite prediga mejor ese eje son
3–6 corridas y se decide con ellas.

**Lo que hay que declarar en Métodos:** Isomap no tiene proyección natural fuera de muestra
(PCoA sí, por la fórmula de Gower), así que los ejes de las parcelas de test salen del embedding
completo. Es la misma propiedad que ya tiene LCBD aquí —`beta.div.comp` se calcula sobre la
matriz entera— pero hay que decirlo.

### GDM queda para el segundo paper

Ver `docs/27`. Está implementado (`scripts/21_run_gdm.py`, `src/biodiv/gdm.py`), ya ganó una vez
(ρ +0,492 contra el techo de +0,465 de conocer los dos ejes verdaderos), y contesta una pregunta
distinta: no "¿se puede predecir?" sino "¿qué impulsa el recambio, y a qué tasa a lo largo de
cada gradiente?".

---

## 4b. RF contra CNN

Comparación sobre el mismo pool y el mismo target (LCBD Sørensen, pg_all leñoso,
`kfold5_block20_unified`):

| modelo | R² agrupado | sd entre semillas | **dentro de banda** |
|---|---|---|---|
| CNN 1D sobre la curva | 0,427 | 0,004 | 0,112 |
| CNN 2D serpentine | 0,429 | 0,006 | 0,109 |
| **CNN 2D + MAE** (el desplegado) | 0,431 | 0,010 | **0,115** |
| **piso de coordenadas** (`B03c`) | 0,437 | 0,003 | **0,109** |
| RF sobre la curva | 0,447 | 0,002 | 0,132 |
| RF gm + clima | 0,502 | 0,000 | 0,210 |

Lo que va al paper: **comparación justa de arquitectura**, mismo insumo (la curva de kNDVI).
RF 0,447 / 0,132 contra CNN 0,431 / 0,115. RF gana poco pero de forma consistente en las tres
variantes de CNN, con desviación entre semillas de 0,002 a 0,010.

> **Lo que se pierde al sacar el piso de coordenadas.** Con él, el resultado era mucho más
> fuerte y mucho más difícil de descartar: las tres CNN quedan agrupadas **por debajo** del piso
> (0,427–0,431 contra 0,437) y empatadas dentro de banda (0,109–0,115 contra 0,109), o sea el
> aporte de la arquitectura profunda sobre lon/lat/elevación es **cero**. Eso además era una
> instancia independiente de lo que arXiv [2609.28194](https://arxiv.org/abs/2609.28194)
> reporta para los embeddings fundacionales — las representaciones complejas lucen bien bajo
> validación débil y se colapsan bajo validación espacial.
>
> Sin el piso queda "RF le gana a la CNN por 0,016", que es pequeño y fácil de atribuir a
> afinado. Queda anotado por si se quiere recuperar en revisión.

> ### DECISIÓN PENDIENTE: el modelo del mapa
>
> El modelo desplegado para los mapas multitemporales es **C2D02 serpentine + MAE**, refitado
> sobre todas las parcelas (decisión del 2026-09-02, ejecutada en rapidita). Es exactamente el
> que empata con el piso de coordenadas.
>
> **DECIDIDO (2026-09-30): el mapa sale del paper.** Razón del autor: con ajustes tan bajos no
> vale la pena el producto, y la historia es de análisis. El paper lo declara en vez de omitirlo
> en silencio. Las opciones que se consideraron:
>
> 1. **Cambiar el modelo del mapa a RF gm+clima** (0,502 / 0,210, +0,101 sobre el piso). Exige
>    refitar sobre todas las parcelas y rehacer la inferencia, pero es el único bloque con
>    margen real.
> 2. **Publicar el mapa con la comparación declarada** y el piso de coordenadas como capa de
>    referencia. Honesto, pero debilita el producto.
> 3. **No publicar mapa en este paper.** La historia de §1 no lo necesita: es sobre cuándo y
>    dónde aporta el satélite, no sobre producir una capa.
>
> Recomendación: opción 1 si hay máquina, opción 3 si no. La 2 deja al lector preguntándose por
> qué se publicó.


---

## 5. La capa ecológica que falta: describir, sin abrir pregunta nueva

Hoy el paper tiene R² y no tiene ecología. La versión acotada que **sirve a la historia** es
una pregunta de residuos, no una exploración:

> ¿Qué distingue a las parcelas que el modelo acierta de las que falla?

Es análisis del OOF que ya existe, está limitado por construcción, y alimenta la Discusión.
Partiendo las parcelas por residuo del mejor modelo de LCBD:

- qué géneros y especies dominan en el cuartil bien predicho contra el mal predicho
- cuántas especies, y cuán dominante es la principal
- qué proporción son **generalistas altitudinales contra especialistas estrechos**
  (rasgo derivado del catálogo de Rodríguez et al. 2018; ver `docs/27` §2)

Esa última columna usa el rasgo como **descripción**, no como respuesta nueva a modelar. Si
resulta que el modelo falla donde dominan los especialistas estrechos, es una frase de Discusión
con respaldo, no un paper nuevo.

---

## 6. Lo que queda fuera, y por qué

- **Mapas RGB de ejes de ordenación.** Serían en buena parte una imagen de latitud y elevación:
  la geografía sola predice los ejes 1–3 de Isomap a R² 0,778 / 0,726 / 0,578. Si se hicieran,
  tendrían que ir **junto al mismo RGB hecho solo con geografía**, y la diferencia entre los dos
  sería el resultado. Eso es un paper de mapas, no este.
- **Rotación de ejes** (Neumann et al. 2016). Busca la dirección de máxima coherencia
  **espectral**, o sea ajusta la respuesta al predictor: tendría que ir dentro del fold o es fuga
  de selección, y multiplica el cómputo. Y sólo paga si los ejes valen algo.
- **Facetas de PFT por amplitud de nicho.** Es una respuesta nueva. `docs/27` §2.
- **Valores indicadores tipo Ellenberg.** Sin equivalente en uso en Chile; el vault marca su
  transferencia fuera de Centroeuropa como no tratada. Pero ver `docs/27` §2: el catálogo chileno
  da un sustituto derivado de catálogo.

---

## 7. El diagnóstico de fondo, para el registro

`docs/10_findings.md` §3 llegó aquí antes, con el pool anterior: 57,6 % de pares con Jaccard =
1,0 exacto, mediana de la disimilitud exactamente 1,0. Y sacó la conclusión correcta, que es de
familia de modelo:

> "Una ordenación lineal tiene que gastar ejes representando una distancia que dejó de variar."

Medido hoy sobre el pool leñoso unificado: 3.094 parcelas, 307 especies, **4,24 especies por
parcela**, **77,6 % de los pares sin ninguna especie en común**. PCoA 1/2/3 = 7,03 / 5,95 /
4,54 %; **26 ejes para el 50 %** del recambio; 260 para el 80 %. No hay codo: es una rampa.

La agregación taxonómica no lo arregla por sí sola —`docs/10` midió que a género la saturación
baja de 57,6 % a 53,0 % y la varianza de dos ejes del PCoA sube de 12,5 % a 13,4 %— **pero
combinada con Isomap sí paga**: a familia, el techo de reconstrucción sube de 0,606 a 0,749.
`docs/10` nunca probó esa combinación. No entra en este paper, pero queda anotado.

---

## 8. Literatura 2021–2026 (el corpus del vault llega a 2021)

- **Mokany et al. 2022**, *Ecography* ([10.1111/ecog.06426](https://doi.org/10.1111/ecog.06426)).
  El análogo más cercano a este proyecto: parcelas armonizadas de múltiples fuentes a escala
  continental (HAVPlot, 115.083 usadas), riqueza y disimilitud composicional estandarizadas a
  400 m², mapeadas a ~90 m. **D² = 33,0 % y 32,7 %.** Predictores más fuertes: temperatura y
  precipitación, luego textura de suelo y heterogeneidad topográfica.
- **White et al. 2024**, *MEE*, spGDMM
  ([10.1111/2041-210X.14259](https://doi.org/10.1111/2041-210X.14259)): GDM mixto espacial.
- **stGDMM** (arXiv [2608.05352](https://arxiv.org/abs/2608.05352), 2026): GDM espacial **y
  temporal** con cuantificación de incertidumbre.
- **GDUM** (bioRxiv [2025.09.28.679068](https://doi.org/10.1101/2025.09.28.679068)) — **aviso
  sobre LCBD, nuestra faceta de composición.** Los gradientes ambientales que producen cambio
  composicional direccional inflan y distorsionan la relación LCBD–ambiente, porque moldean la
  propia matriz de disimilitud de la que sale el LCBD. En su caso de estudio, controlar el
  gradiente **invirtió** el patrón. Nuestro LCBD se modela sobre 26 grados de latitud; el
  centrado por banda ataca parte del problema pero no este mecanismo. **Hay que declararlo aunque
  no se implemente.**
- **Embeddings fundacionales** (AlphaEarth, 64 dim, 10 m): prometedores, pero arXiv
  [2609.28194](https://arxiv.org/abs/2609.28194) (2026) muestra que su ventaja **se estrecha bajo
  validación espacial con buffer** —que es nuestro diseño— y sólo cubren desde ~2017, contra
  censos 2003–2026.
- **Abutaha et al. 2021**, *AVSC*
  ([10.1111/avsc.12582](https://doi.org/10.1111/avsc.12582)), con Feilhauer: NMDS1 como respuesta
  de beta, GAM sobre DEM + SoilGrids + PlanetScope, **80,6 %** — pero 133 parcelas en una sola
  montaña árida.
- **Robertson et al. 2023**, *JGR-B*
  ([10.1029/2022JG007350](https://doi.org/10.1029/2022JG007350)): la capacidad de detectar beta
  **cae con resolución espacial más gruesa**. Relevante porque trabajamos a 30 m con parcelas de
  250–900 m².

---

## 9. Orden de trabajo

1. **Corridas de Isomap** (3–6) para decidir con datos si sustituye a PCoA en la tabla.
2. **Análisis de residuos por composición y dominancia** (§5).
3. **Reescritura del texto** con la historia de §1, las dos riquezas de §2 y el piso de
   coordenadas subido a resultado.
4. Leer **GDUM** antes de fijar la redacción sobre LCBD.
