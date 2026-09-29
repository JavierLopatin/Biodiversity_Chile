# Cómo abordar la composición florística — propuesta metodológica

Fecha: 2026-09-29. Estado: **propuesta, ninguna decisión tomada**.

> **CORRECCIÓN (misma fecha, tras revisar `docs/10` y la literatura 2021–2026).** La primera
> versión de este documento se escribió sin leer `docs/10_findings.md`, y por eso proponía como
> nuevas dos cosas que este proyecto ya hizo. Ver la sección **"Lo que ya se probó aquí"** al
> final: el GDM ya se corrió y **ganó**, y la agregación taxonómica ya se probó y **casi no
> movió nada**. El orden recomendado cambia en consecuencia.

Base: el corpus `Floristic-composition` del vault (9 papers, 2004–2021, paradigma
ordenación + regresión), cruzado con lo medido en este repo el 2026-09-29
(`scripts/107`, `scripts/109`, `results/tables/sintesis_lenoso_resumen.csv`).

## El diagnóstico, en una línea

Nuestro paso que falla no es la regresión: es la **ordenación**. El PCoA de Jaccard da 13,0 %
en dos ejes, y el corpus tiene una respuesta técnica central a ese problema exacto.

Lo medido aquí:

| | valor |
|---|---|
| parcelas × especies leñosas | 3.094 × 307 |
| especies por parcela | 4,24 |
| **pares de parcelas sin ninguna especie en común** | **77,6 %** (Jaccard = 1) |
| PCoA 1 / 2 / 3 | 7,03 % / 5,95 % / 4,54 % |
| ejes para el 50 % del recambio | 26 |

Y lo que eso cuesta, con el control de geografía de `B03` (lon, lat, elevación), dentro de
bandas de 2° de latitud:

| faceta | mejor bloque | piso coords | margen |
|---|---|---|---|
| LCBD Sørensen | 0,210 | 0,109 | **+0,101** |
| LCBD p/a | 0,313 | 0,215 | +0,098 |
| PCoA 2 p/a | 0,541 | 0,465 | +0,076 |
| **PCoA 1 p/a** | 0,563 | 0,569 | **−0,006** |
| PCoA 1 frecuencia | 0,429 | 0,455 | −0,026 |

PCoA 1 no le gana a tres coordenadas. Es coherente con su interpretación: PCoA 1 correlaciona
−0,73 con la temperatura media anual, o sea es el gradiente térmico, y eso la geografía ya lo
tiene. PCoA 2 es el gradiente hídrico (precipitación del trimestre seco +0,71) y ahí sí hay
margen. LCBD es donde más margen hay.

## Propuesta 1 — Isomap en lugar de PCoA (la de mayor valor esperado)

**Qué.** Sustituir el PCoA clásico por Isomap (Tenenbaum et al. 2000) para extraer los ejes
florísticos.

**Por qué funciona aquí, mecánicamente.** El PCoA embebe las distancias directamente. Cuando
el 77,6 % de los pares está saturado en el máximo, la geometría global es casi un símplex y
ninguna proyección de baja dimensión la representa — de ahí la rampa sin codo. Isomap no usa
las distancias directas: construye un grafo de k vecinos más próximos y mide **distancias
geodésicas** a lo largo del grafo. Dos parcelas que no comparten ninguna especie nunca se
comparan directamente; su distancia se calcula como un camino a través de parcelas intermedias
que sí comparten. Es exactamente el fallo que tenemos y exactamente lo que Isomap arregla.

**Evidencia del corpus.** Feilhauer et al. 2011: 74 % de la variación florística preservada en
3 ejes contra 54 % del DCA, en un paisaje heterogéneo. Harris et al. 2015: 82 % en 3 ejes a
nivel de especie en una turbera de **beta alta** — y el corpus señala que la ventaja de Isomap
aparece justo en heterogeneidad alta y beta alta. Nuestro caso es el extremo de eso.

**Riesgos, los dos documentados.**
- *k es sensible y es una fuga si se elige mal.* Feilhauer 2011 optimizó k por fuerza bruta
  sobre los mismos datos que después mapeó, y el propio paper lo declara como limitación. Aquí
  k tiene que elegirse **dentro del fold de entrenamiento**, no antes, o repetimos la clase de
  error que ya costó cuatro artefactos en este proyecto.
- *Ninguno de los 9 papers prueba transferencia entre sitios.* Todos declaran que sus
  coeficientes no transfieren sin recalibrar. Nuestro CV de bloques de 20 km sobre 26 grados de
  latitud es una prueba mucho más dura que la de cualquiera del corpus — puede que Isomap no
  aguante ahí. Eso es un resultado publicable, no un fracaso.

**Coste.** Bajo. `vegan`/`isopam` en R (pop-os los tiene), o `sklearn.manifold.Isomap`.

## Propuesta 2 — Agregación funcional antes de ordenar

**Qué.** Ordenar grupos funcionales en vez de las 307 especies.

**Por qué.** Harris et al. 2015 obtuvo **>96 % en 2 ejes** a nivel de tipo funcional contra
82 % a nivel de especie. Agregando se reduce la saturación directamente: con menos categorías,
menos pares de parcelas quedan sin nada en común. Nuestro 4,24 especies por parcela es la causa
raíz del 77,6 %.

**La trampa, que el mismo paper documenta.** El grupo tiene que ser coherente ecológica y
espectralmente. Su categoría "briofitas" **falló** (R²val 0,19 en el eje 2) hasta que la
restringieron a especies dominadas por *Sphagnum*. Agregar mal es peor que no agregar.

**Para Chile.** La infraestructura ya existe (`growth_form_lookup.csv`, `scripts/80-82`).
Candidatos: género; o siempreverde / caducifolio × esclerófilo / laurifolio, que mapea sobre la
estructura conocida de la vegetación leñosa chilena. **Esto lo decides tú, no yo** — es una
decisión de botánica, y elegir el agrupamiento por lo que mejore el R² sería exactamente el
error que Harris describe.

## Propuesta 3 — La métrica de preservación de información como evaluación

**Qué.** Adoptar el criterio de Feilhauer et al. 2021: el R² entre las **distancias originales
entre parcelas** y las **distancias predichas por el mapa**.

**Por qué nos hace falta.** Hoy comparamos R² por faceta, y eso no permite comparar entre
facetas: el 0,563 de PCoA 1 y el 0,210 de LCBD no son la misma escala ni miden lo mismo. La
métrica de preservación pone a LCBD, a los ejes del PCoA, a un Isomap y a una clasificación
**en un solo marco**: cuánta de la estructura florística real sobrevive al mapa.

En su comparación a tres bandas sobre los mismos datos: gradiente 0,42 > difusa 0,40 >
clasificación dura 0,31. La nota de `gaps/` la señala explícitamente como directamente
reutilizable para el trabajo chileno de este vault.

**Coste.** Muy bajo, es post-proceso sobre los OOF que ya existen. **Es lo que haría primero**,
porque sin ella no se puede decidir entre las propuestas 1 y 2 salvo por R² por eje, que es
justo lo que no compara.

## Propuesta 4 — Donde podríamos aportar en vez de seguir

El corpus tiene una pregunta abierta propia: *"todos los papers modelan los ejes de a uno,
descartando la estructura multivariada conjunta"*. Leutner et al. 2012 señaló el Random Forest
multivariado como dirección prometedora pero inmadura — **en 2012**.

Dos cosas nos ponen en posición de contestarla:

- **n.** El corpus entero trabaja con 129, 86, 57 parcelas, en paisajes de 12–26 ha. Nosotros
  tenemos 3.102 parcelas sobre 26 grados de latitud. Ninguno de los 9 papers tuvo con qué
  ajustar un modelo conjunto.
- **La pregunta de transferencia.** Los 9 declaran que sus coeficientes no transfieren y
  ninguno lo prueba. Nuestro CV de bloques lo está probando a escala continental.

## Propuesta 5 — El test de SWIR contra tiempo (pendiente, y mi hipótesis NO se confirmó)

Feilhauer et al. 2013 es el único paper del corpus sobre sensores **multiespectrales**, o sea
el único directamente comparable con Landsat. Su hallazgo central: lo decisivo es la
**cobertura espectral VIS–SWIR**, no el número de bandas ni la resolución temporal. Los
sensores sin SWIR fallaron específicamente en el gradiente **de humedad**; los de la familia
Landsat lo resolvieron.

Eso encaja sospechosamente bien con lo nuestro: el geomediano lleva `swir1` y `swir2` y no
lleva tiempo; la curva lleva tiempo y solo kNDVI (rojo/NIR); y el geomediano gana. Y nuestro
PCoA 2, donde hay margen sobre la geografía, es el gradiente hídrico.

**Lo comprobé y el atajo no lo sostiene.** Las correlaciones marginales de Spearman dentro de
banda entre las bandas del geomediano y PCoA 2 son todas débiles (swir1 −0,021, swir2 +0,014,
nir +0,063): el SWIR no destaca. Pero el RF usa interacciones y una correlación marginal de
rango es un test flojo. **La prueba que vale es una ablación** — geomediano con y sin las dos
bandas SWIR — y son dos corridas.

Nota adicional del mismo paper, relevante para nuestro resultado de curva y LSP: el
multiestacional ayudó de forma **inconsistente** y en un sitio **empeoró** el ajuste.

## Lo que NO propongo

- **Más ejes del PCoA.** PCoA 3 suma 4,54 puntos y deja el total en 17,5 %. El problema no es
  cuántos ejes se guardan, es que no hay estructura de baja dimensión que guardar
  (`scripts/109`, `figS7`).
- **Valores indicadores tipo Ellenberg.** Son la herramienta interpretativa de medio corpus,
  pero la nota de `gaps/` marca que su transferencia fuera de Centroeuropa y el Reino Unido no
  está tratada en ninguna parte, y Chile no tiene un sistema equivalente en uso.

## Orden que recomiendo

1. **Métrica de preservación de información** (propuesta 3). Barata, y sin ella las demás no se
   pueden comparar.
2. **Isomap** (propuesta 1), con k elegido dentro del fold.
3. **Agregación funcional** (propuesta 2), con el agrupamiento definido por criterio botánico y
   fijado antes de mirar ningún R².
4. **Ablación de SWIR** (propuesta 5), dos corridas, resuelve una explicación que hoy es
   especulación.

La 4 (multivariado) es la de mayor techo y la de mayor riesgo; la dejaría para cuando las
primeras tres digan si la ordenación se puede arreglar.

---

# Lo que ya se probó aquí (y yo no había mirado)

`docs/10_findings.md` §3 llegó a este mismo diagnóstico antes que yo, con los números del pool
anterior: **57,6 % de los pares con Jaccard = 1,0 exacto**, mediana de la disimilitud
exactamente 1,0, 70,6 % por encima de 0,9. Y sacó la conclusión correcta, que es de familia de
modelo y no de número de ejes:

> "Una ordenación lineal tiene que gastar ejes representando una distancia que dejó de variar."

## GDM ya se corrió, y ganó

`scripts/21_run_gdm.py`. Resultados sobre pares entre parcelas excluidas, ρ de Spearman contra
el Jaccard observado, esquema `kfold5_owner`, 62 predictores:

| | ρ |
|---|---|
| GDM solo distancia geográfica | +0,435 |
| **GDM con los predictores** | **+0,492** |
| SGDM (sCCA fold-local + GDM) | +0,436 |
| RF sobre 8 ejes PCoA → distancia | +0,425 |
| techo: los **2 ejes verdaderos** | +0,465 |
| techo: los 8 ejes verdaderos | +0,530 |

**El GDM desde satélite reproduce la composición observada mejor que conocer los valores
verdaderos de los dos ejes que el proyecto modela hoy** (+0,492 contra +0,465).

La razón por la que funciona es la que hace falta aquí: el enlace de GDM, `d = 1 − exp(−η)`, es
**asintótico en 1 por construcción** (Ferrier et al. 2007), o sea que la saturación deja de ser
un problema y pasa a ser la forma esperada del modelo. Las I-splines monótonas por predictor
dejan además que la tasa de recambio varíe a lo largo del gradiente.

**Por qué no está en el paper.** Por decisión de alcance, no por resultado: el banco de pruebas
de agosto-2026 sobre Parcelas-CL sola (`docs/14, 16, 17`) quedó fuera del manuscrito. El GDM
salió con él. **Esa decisión se tomó antes de que existieran el pool unificado, el control de
coordenadas y el espectro del PCoA.**

Cautela que ya venía anotada y que hoy se confirma por otra vía: la distancia geográfica sola
da +0,435 y el aporte propio de los predictores es **+0,057**. Encaja exacto con lo medido hoy
con `B03` — la geografía hace casi todo el trabajo, y el margen remoto real es de esa magnitud.

## La agregación taxonómica ya se probó, y casi no movió nada

`docs/10` §3: agregando a género (313 taxones en vez de 570), la saturación baja de 57,6 % a
**53,0 %** y la varianza de dos ejes sube de 12,5 % a **13,4 %**.

Eso rebaja mucho mi Propuesta 2. El >96 % de Harris et al. 2015 salió de una turbera con pocos
tipos funcionales y un gradiente corto; aquí el problema no es resolución taxonómica sino
muestrear parcelas de ~4 especies a lo largo de 26 grados de latitud. Sigue siendo posible que
un agrupamiento **funcional** (no taxonómico) se comporte distinto, pero la evidencia local
dice que el efecto será chico.

---

# Literatura 2021–2026 que el corpus del vault no cubre

El corpus llega a 2021. Revisión rápida en línea, 2026-09-29.

## 1. GDM tiene dos generaciones nuevas, y es el estándar continental

- **Mokany et al. 2022**, *Ecography*, "Patterns and drivers of plant diversity across
  Australia" ([10.1111/ecog.06426](https://doi.org/10.1111/ecog.06426)). **El análogo más
  cercano a este proyecto que he encontrado**: parcelas armonizadas de múltiples fuentes a
  escala continental (HAVPlot, 219.552 parcelas, 115.083 usadas), modelando riqueza y
  disimilitud composicional estandarizadas a 400 m², mapeadas a ~90 m. **D² = 33,0 % y 32,7 %.**
  Predictores más fuertes: combinación de temperatura y precipitación, luego textura de suelo y
  heterogeneidad topográfica. Usa GDM sobre disimilitud por pares, no LCBD por parcela.
- **White et al. 2024**, *Methods in Ecology and Evolution*, spGDMM
  ([10.1111/2041-210X.14259](https://doi.org/10.1111/2041-210X.14259)): GDM mixto espacial, con
  función de media que varía en el espacio y efectos aleatorios espaciales que capturan
  dependencia que los predictores no explican.
- **stGDMM** (arXiv [2608.05352](https://arxiv.org/abs/2608.05352), 2026): GDM conjunto
  **espacial y temporal**, con cuantificación de incertidumbre. Directamente relevante al eje
  de sequía y al mapa multitemporal 2000–2026.

## 2. Un aviso sobre LCBD, que es nuestra faceta titular

- **GDUM — Generalised Dissimilarity Uniqueness Models** (bioRxiv
  [2025.09.28.679068](https://doi.org/10.1101/2025.09.28.679068)). Tesis: los gradientes
  ambientales que producen cambio composicional direccional **inflan o distorsionan** la
  relación LCBD–ambiente, porque moldean la propia matriz de disimilitud de la que sale el
  LCBD. Las relaciones LCBD–ambiente en U que se reportan a lo largo de gradientes de elevación
  pueden venir del muestreo y no de un mecanismo ecológico. GDUM modela a la vez el gradiente
  de disimilitud por pares y el efecto de sitio sobre la unicidad, para separar lo direccional
  de lo no direccional. En su caso de estudio microbiano, **tener en cuenta el gradiente de pH
  invirtió el patrón de U a jorobado**.
  **Por qué importa aquí:** nuestro LCBD se modela a lo largo de 26 grados de latitud, que es
  el gradiente direccional más fuerte imaginable. El centrado dentro de banda ataca parte del
  problema, pero no el mecanismo que describe este paper, que actúa dentro de la matriz.

## 3. Embeddings de modelos fundacionales: promesa, y un aviso que nos apunta directo

- **AlphaEarth Foundations** (Google DeepMind): embeddings anuales de 64 dimensiones a 10 m,
  integrando Sentinel-1/2, Landsat, GEDI, ERA5-Land y más. Ya hay mapeo de comunidades
  forestales con composición por especie sobre 65 millones de hectáreas.
- **Pero**: *"Geospatial embeddings detect old-growth forests but buffered spatial validation
  narrows their advantage over Sentinel features"* (arXiv
  [2609.28194](https://arxiv.org/abs/2609.28194), 2026). Bajo validación cruzada **aleatoria**
  los embeddings ganan por bastante; bajo **validación espacial con buffer** la ventaja se
  estrecha mucho. Los autores concluyen que su superioridad aparente refleja en parte métricas
  infladas por dependencia espacial. **Nuestro diseño es exactamente ese: bloques de 20 km.**
- Y un segundo negativo, además en los Andes: *"Spectral indices outperform AlphaEarth
  foundation embeddings for aboveground biomass estimation in a regenerating tropical Andean
  forest"*.
- Limitación dura para nosotros: AlphaEarth es anual desde ~2017. Nuestros censos van de 2003 a
  2026 y el mapa pedido es 2000–2026. No cubre la mitad temprana.

## 4. Confirmaciones del paradigma de ordenación, a escala pequeña

- **Abutaha et al. 2021**, *Applied Vegetation Science*, Gebel Elba, Egipto
  ([10.1111/avsc.12582](https://doi.org/10.1111/avsc.12582)), con Feilhauer de coautor: usan
  **NMDS1 como la respuesta de beta** y un GAM sobre variables de DEM, SoilGrids y PlanetScope.
  **80,6 % en NMDS1** — pero con 133 parcelas en una sola montaña árida. Confirma el paradigma
  donde el gradiente es corto y el paisaje uno solo.
- **Robertson et al. 2023**, *JGR Biogeosciences*
  ([10.1029/2022JG007350](https://doi.org/10.1029/2022JG007350)): biodivMapR y especies
  espectrales; la capacidad de detectar beta **cae con resolución espacial más gruesa**, y la
  ventana de mapeo limita a su vez la resolución del mapa de beta. Relevante porque nosotros
  trabajamos a 30 m con parcelas de 250–900 m².

---

# Orden recomendado, corregido

1. **Rehacer el GDM sobre el pool unificado leñoso**, con el esquema `kfold5_block20_unified` y
   con la geografía dentro del modelo como control (la variante `gdm_geo` ya existe). Es la
   familia de modelo que el diagnóstico selecciona, ya está implementada aquí
   (`scripts/21_run_gdm.py`), ya ganó una vez, y la literatura continental de 2022–2026 la
   confirma como el estándar para este problema. **Coste bajo, valor alto, riesgo bajo.**
2. **La métrica de preservación de información** (Feilhauer et al. 2021) como criterio común,
   para poder comparar GDM, LCBD y ejes de ordenación en un solo marco.
3. **Leer GDUM** antes de fijar la redacción sobre LCBD. Si su crítica aplica —y a primera
   vista aplica—, hay que declararla aunque no se implemente.
4. **Isomap**, ya no como primera opción sino como comparador de la ordenación, con `k` elegido
   dentro del fold.
5. **Ablación de SWIR**, dos corridas, para cerrar la especulación de Feilhauer et al. 2013.

Lo que baja de prioridad: la agregación taxonómica (ya probada aquí, efecto chico) y los
embeddings fundacionales (no cubren 2000–2017, y su ventaja se estrecha justo bajo el tipo de
validación que usamos).
