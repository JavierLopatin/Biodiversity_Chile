# Cómo abordar la composición florística — propuesta metodológica

Fecha: 2026-09-29. Estado: **propuesta, ninguna decisión tomada**.

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
