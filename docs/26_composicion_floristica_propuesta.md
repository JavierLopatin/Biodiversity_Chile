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

Todo con el piso de coordenadas (`B03c`: lon, lat, elevación) y dentro de bandas de 2° de
latitud, que es la lectura defendible. `sobre coords` es la columna que importa.

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

El mecanismo: si la señal funciona porque la vegetación **se expresa** —verdor, estructura,
amplitud fenológica—, en año seco se expresa menos, la separación espectral entre comunidades se
colapsa, y el predictor pierde información aunque la comunidad no haya cambiado. El control del
piso descarta la explicación aburrida: si fuera varianza del target, el piso se movería igual, y
no se mueve.

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
4. **El piso de coordenadas.** Es la contribución metodológica más fuerte y hoy no está en el
   texto. Casi toda la literatura del campo reporta R² agrupados sin ese control; aquí tres
   números de geografía dan 0,702 contra 0,704 del modelo completo en PCoA 1.
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
