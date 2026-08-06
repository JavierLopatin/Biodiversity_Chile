# Estado del arte

**Proyecto:** estimación multitemporal de facetas múltiples de biodiversidad en Chile central a partir de fenología satelital
**Fecha:** agosto 2026
**Referencias:** `refs.bib` (42 entradas, todos los DOI resueltos contra la API de Crossref)

---

## Resumen ejecutivo

El campo de la estimación de biodiversidad por sensores remotos se construyó sobre una hipótesis —la Spectral Variation Hypothesis (SVH)— que después de veinte años sigue sin comportarse de forma consistente entre ecosistemas. Las revisiones más recientes atribuyen buena parte de esa inconsistencia a factores que la SVH trata como ruido: resolución espacial, tamaño de la ventana móvil, y sobre todo **el momento fenológico de la adquisición**.

Paralelamente se consolidó una línea distinta que usa la **trayectoria temporal** en lugar de la variabilidad espacial de una fecha. Esa línea ya demostró que la fenología discrimina composición de comunidades vegetales, pero se quedó casi siempre en clasificación o en una sola faceta de diversidad a la vez.

El hueco que queda es específico: **nadie ha usado la curva fenológica completa para predecir simultáneamente varias facetas cuantitativas de diversidad** (taxonómica, filogenética, funcional, composicional), y nadie lo ha hecho de forma retrospectiva emparejando cada parcela con el año en que fue censada.

---

## Eje A — Spectral Variation Hypothesis: por qué no basta

La SVH postula que la heterogeneidad espectral de un píxel o vecindario es proxy de la heterogeneidad de hábitat y, por tanto, de la diversidad de especies. Es el paradigma dominante desde principios de los 2000.

**El balance a veinte años es ambiguo.** Torresani et al. (2024) revisan dos décadas de aplicaciones y encuentran relaciones que van de fuertemente positivas a negativas según ecosistema, sensor y diseño de muestreo. Schmidtlein & Fassnacht (2017), usando ocurrencias de plantas vasculares espacialmente contiguas en el sur de Alemania y MODIS en 14 fechas, concluyen directamente que la hipótesis **no se sostiene entre paisajes**: la relación cambia de signo entre hábitats. Fassnacht et al. (2022) reformulan el problema en términos de qué componente de la biodiversidad puede razonablemente esperar capturar una señal espectral.

**Dónde sí funciona.** Hayden et al. (2024) obtienen predicciones útiles de diversidad taxonómica con datos espectrales de alta resolución, pero en pastizales de **baja** diversidad. Donnini et al. (2024) revisan el caso forestal y documentan que la relación diversidad espectral–diversidad arbórea se debilita o se invierte a alta riqueza: cuando muchas especies comparten espacio espectral, añadir especies deja de añadir varianza espectral.

**Qué queda sin resolver.** Las revisiones coinciden en que uno de los factores de confusión principales es *cuándo* se adquirió la imagen. Eso convierte a la fenología en variable de estorbo dentro del marco SVH. La alternativa lógica —tratar la trayectoria fenológica como la señal misma, no como ruido a controlar— está poco explorada como sustituto de la heterogeneidad espacial de una fecha.

---

## Eje B — Fenología de superficie terrestre como predictor de biodiversidad

Dronova & Taddeo (2022, *Journal of Ecology*) es la síntesis de referencia. Argumentan que la fenología de superficie terrestre (LSP) funciona como indicador integrador de la dinámica de comunidades desde escala de especie hasta regional, y —punto central para este proyecto— que un enfoque fenológico holístico puede revelar sinergias entre los componentes **taxonómico, funcional y filogenético** de la diversidad, porque la estacionalidad espectral está ligada a la historia evolutiva de la estructura vegetal.

La dirección inversa también tiene evidencia. Dronova et al. (2022, *Science Advances*) muestran en humedales de EE.UU. a escala nacional que la diversidad de plantas **reduce** la variabilidad fenológica interanual, tras controlar covariables. Es decir: la variabilidad fenológica interanual no es solo ruido de medición, contiene señal de diversidad. Girardello et al. (2024) construyen un índice de diversidad fenológica para biomas forestales globales y lo interpretan como reflejo de partición de nicho en el tiempo, con valores máximos en regiones áridas y templadas.

El vínculo mecanístico con rasgos también se ha establecido en ambas direcciones: Liu et al. (2024) integran espectro y fenología con series temporales de Sentinel-2 para mapear rasgos foliares, y Zhao et al. (2025) muestran que los rasgos funcionales regulan la variabilidad fenológica interanual en bosques templados.

**Anclaje en trabajo propio.** Lopatin (2023, *IEEE GRSL*) demuestra que la **variabilidad interanual** de la fenología detectada remotamente se relaciona con las comunidades vegetales, y propone el RMSE segmentado (SOS/POS/EOS) como métrica de estabilidad fenológica —implementado en `PhenoSensing`. Lopatin et al. (2026, *Ecological Informatics*) extienden el enfoque a humedales costeros, mostrando controles ambientales y de manejo sobre comunidades vía fenología.

**Qué queda sin resolver.** La fenología ya está validada como discriminador de composición y como correlato de diversidad agregada. Lo que no existe es el paso a **predicción cuantitativa simultánea de varias facetas** con la curva completa como entrada, ni una evaluación de si la curva aporta más que sus métricas escalares resumidas.

---

## Eje C — Diversidad funcional y filogenética desde sensores remotos

Schweiger et al. (2018, *Nature Ecology & Evolution*) es la pieza fundacional: la diversidad espectral integra componentes funcionales y filogenéticos de la biodiversidad y predice función ecosistémica. La justificación es que el espacio espectral opera como análogo del espacio de rasgos, un hipervolumen n-dimensional poblado por espectros de especies. Cavender-Bares et al. (2022) sitúan esto dentro de un programa más amplio de integración entre teledetección, ecología y evolución.

**Los límites están bien documentados.** Pacheco-Labrador et al. (2022) usan modelado de transferencia radiativa para cuestionar el vínculo funcional-espectral: la relación no es garantizada por la física. Pacheco-Labrador et al. (2023, *MEE*) proponen una normalización generalizable porque las métricas de diversidad funcional derivadas de teledetección no son comparables entre escalas sin corrección. Helfenstein et al. (2022) cuantifican cuánto degradan la resolución espacial y espectral las estimaciones de diversidad funcional basada en rasgos. Pacheco-Labrador et al. (2026, *Ecological Informatics*) hacen el benchmark sistemático con escenas sintéticas (BOSSE), comparando reflectancia hiperespectral, índices, SIF y temperatura superficial.

Mederer et al. (2025, *Communications Earth & Environment*) añaden la dimensión temporal explícitamente: con >4.000 escenas hiperespectrales de EnMAP (2022–2024) muestran que la diversidad funcional **varía estacionalmente** y que esa variación difiere entre biomas. Es el argumento más directo disponible de que una sola fecha es insuficiente.

**Revisión propia como fuente de gaps.** Cerda-Paredes et al. (2026, preprint bioRxiv) hacen la revisión sistemática de la brecha entre ecología de campo y teledetección para estimar diversidad funcional vegetal. Debe usarse como fuente primaria de gaps declarados por el propio equipo.

**Qué queda sin resolver.** Toda esta línea depende de rasgos ópticamente activos (pigmentos, agua, estructura foliar) recuperables desde el espectro. Cuando los rasgos disponibles en campo son mayoritariamente **categóricos y no ópticos** —el caso de Rasgos-CL: 21 categóricos, 2 continuos, ninguno del espectro de economía foliar— la cadena espectro→rasgo→diversidad funcional se rompe. La ruta alternativa, predecir la métrica de diversidad funcional directamente sin pasar por rasgos individuales, está poco formalizada.

---

## Eje D — Composición florística y beta: regresión sobre ejes de ordenación

Existe precedente sólido y cuantificado. Adams et al. (2019, *Forest Ecology and Management*) proyectaron 699 parcelas forestales con 99 especies/géneros sobre una solución NMDS y modelaron los ejes con Random Forest a partir de reflectancia Landsat 8 OLI y variables de terreno. Resultado: **61%, 49% y 25%** de la variación de los ejes NMDS 1, 2 y 3 explicada. Este es el número de referencia contra el cual comparar.

Pinto-Ledezma & Cavender-Bares (2021, *Scientific Reports*) predicen distribuciones de especies y composición de comunidad usando predictores satelitales, mostrando que la señal satelital aporta a escala de comunidad y no solo de especie.

Para la faceta beta, Legendre & De Cáceres (2013, *Ecology Letters*) proveen el marco operativo: la beta total como varianza de la matriz de comunidad, particionable en contribuciones locales (**LCBD**, singularidad ecológica de cada sitio) y de especies (SCBD). LCBD es un valor por parcela, lo que la hace directamente utilizable como variable respuesta —a diferencia de la beta pareada.

**Qué queda sin resolver.** El enfoque ordination-regression se ha aplicado siempre con **reflectancia de una o pocas fechas**. No hay precedente de usar la curva fenológica completa como predictor de ejes de ordenación, ni de predecir LCBD desde fenología.

---

## Eje E — Deep learning multi-tarea en biodiversidad

Pettorelli et al. (2024/2025, *Remote Sensing in Ecology and Conservation*) revisan el estado del deep learning con teledetección para monitoreo de biodiversidad. El cuello de botella recurrente que identifican es la escasez de datos de entrenamiento etiquetados, no la arquitectura.

Gillespie et al. (2024, *PNAS*) es el ejemplo mejor logrado de multi-tarea aplicado a plantas: una TResNet modificada con **múltiples capas de salida correspondientes a tres rangos taxonómicos** (familia, género, especie), entrenada con imágenes satelitales de California y ~500.000 observaciones de ciencia ciudadana, mapeando >2.000 especies. Demuestra que las cabezas múltiples sobre un tronco compartido funcionan en este dominio.

En rasgos, Cherif et al. (2023, *RSE*) entrenan un 1D-CNN que predice **20 rasgos simultáneamente** desde espectros heterogéneos y dispersos, con pérdida enmascarada para manejar etiquetas faltantes. El repositorio propio `Trait_2DCNN` construye sobre ese baseline: convierte el espectro 1D en imagen 2D mediante nueve transformaciones (reshape, serpentine, Hilbert, GAF, MTF, espectrograma, CWT, COS2D, NDI) y alimenta backbones de visión, mejorando R² de 0,587 a 0,684. Punto relevante: **GAF, MTF, espectrograma y CWT fueron diseñadas originalmente para señales temporales** y allí se aplican a un eje espectral.

**Qué queda sin resolver.** Multi-tarea existe para rasgos (Cherif, Trait_2DCNN) y para composición taxonómica (Gillespie). **No existe para facetas de diversidad conjuntas** —α, β/LCBD, filogenética, funcional y composición predichas desde una representación compartida. Tampoco existe evidencia sobre si las transformaciones señal→imagen mantienen su ventaja cuando la señal de entrada es genuinamente temporal.

---

## Eje F — Dark diversity

Pärtel et al. (2011, *TREE*) introducen el concepto: el conjunto de especies del pool regional que podrían habitar un sitio dadas sus condiciones ecológicas pero están ausentes. Lewis et al. (2015/2016, *MEE*) evalúan empíricamente los dos métodos principales de estimación. Carmona & Pärtel (2020/2021, *GEB*) desarrollan el enfoque probabilístico basado en co-ocurrencias implementado en el paquete R `DarkDiv` (índice de Beals e hipergeométrico). Pärtel et al. (2025, *Nature*) llevan el marco a escala global, revelando empobrecimiento generalizado de la vegetación natural mediante dark diversity.

**Qué queda sin resolver.** Las búsquedas realizadas no localizaron **ningún trabajo que estime dark diversity a partir de sensores remotos**. Tampoco hay ningún documento sobre dark diversity en el research-vault del usuario. La estimación actual depende enteramente de matrices de co-ocurrencia de datos de campo. Un modelo que prediga dark diversity —o el ratio de completitud de comunidad— desde variables ambientales continuas derivadas de satélite no tiene precedente localizado.

Advertencia metodológica seria: `DarkDiv` requiere una matriz de co-ocurrencia bien muestreada para estimar el pool. Las curvas de acumulación de Parcelas-CL **no alcanzan asíntota** (declarado por sus autores), lo que significa que el pool regional está incompletamente capturado. Estimar dark diversity sobre un pool incompleto produce sesgo de dirección desconocida.

---

## Eje G — Contexto chileno

Chile mediterráneo es hotspot de biodiversidad y está casi ausente de la literatura de teledetección de biodiversidad. Los antecedentes locales directos son propios: Ceballos et al. (2015, *Remote Sensing*) comparan LiDAR aerotransportado e hiperespectral satelital para estimar riqueza de plantas vasculares en bosques mediterráneos deciduos de Chile central; Lopatin et al. (2016, *RSE*) comparan GLM y Random Forest para riqueza de especies vasculares con LiDAR en bosque natural de Chile central. Ambos son de una sola fecha y una sola faceta (riqueza).

Las bases de datos que habilitan el salto son recientes:
- **Parcelas-CL** (Cerda-Paredes et al. 2026; Zenodo 10.5281/zenodo.20602096): 1.485 parcelas, 675 especies leñosas, 1976–2026, 30,25°S–54,82°S. Pasa la disponibilidad chilena de 117 registros en sPlot a 1.485. Unidades ecológicas según Luebert & Pliscoff (2017).
- **Rasgos-CL** (Alfaro et al. 2023, *GEB*): 662 especies leñosas, 25.174 registros, 23 rasgos.
- Contexto global: sPlot (Bruelheide et al. 2019) y sPlotOpen (Sabatini et al. 2021).

Las conclusiones del propio paper de Parcelas-CL nombran la **interoperabilidad con Rasgos-CL** como la vía para análisis multi-faceta de biodiversidad entre escalas. Es un llamado explícito, aún no ejecutado.

Marco de política: Skidmore et al. (2021, *Nature Ecology & Evolution*) publican la lista prioritaria de métricas de biodiversidad observables desde el espacio, que sirve de referencia para justificar qué facetas vale la pena mapear.

---

## Tabla de gaps

| # | Gap | Evidencia de que está abierto | Qué lo cerraría aquí | Riesgo |
|---|---|---|---|---|
| **G1** | La trayectoria fenológica completa como predictor, en vez de heterogeneidad espectral de una fecha | Torresani 2024 y Schmidtlein & Fassnacht 2017 atribuyen la inconsistencia de la SVH en parte al momento de adquisición; Mederer 2025 muestra que la diversidad funcional varía estacionalmente | Usar `PhenoShape()` (curva de 52 pasos) además de las 18 métricas LSP, y comparar curva vs métricas escalares | Bajo. Es una comparación bien definida y publicable aunque salga negativa |
| **G2** | Predicción **simultánea** de varias facetas de diversidad desde una representación compartida | Multi-tarea existe para rasgos (Cherif 2023) y para taxones (Gillespie 2024), no para facetas de diversidad | CNN multivariante con cabezas para α, LCBD, PD, FD y ejes NMDS; pérdida enmascarada para targets faltantes | Medio. Depende de que n sea suficiente |
| **G3** | Emparejamiento retrospectivo parcela–año sobre décadas | La literatura RS-biodiversidad usa típicamente una imagen o un año; Parcelas-CL cubre 1976–2026 | Extraer la fenología de cada parcela en su propio año de censo (Landsat 1999–2026, HLS/S2 2017+) | Medio. Calidad de serie desigual entre épocas |
| **G4** | Diversidad funcional desde rasgos mayoritariamente **categóricos y no ópticos** | Toda la línea Schweiger–Pacheco-Labrador asume rasgos ópticamente activos; Rasgos-CL tiene 2 rasgos continuos y ninguno de economía foliar | Predecir FDis/RaoQ sobre distancia de Gower mixta directamente, sin pasar por recuperación de rasgos individuales | Alto. Un referee puede objetar que "diversidad funcional" con estos rasgos mide otra cosa |
| **G5** | Dark diversity desde teledetección | Sin precedente localizado; la estimación depende de co-ocurrencia de campo (Carmona & Pärtel 2020) | Estimar dark diversity con `DarkDiv` sobre Parcelas-CL y predecirla desde fenología + topografía | Muy alto. Las curvas de acumulación de Parcelas-CL no saturan → el pool está mal estimado |

---

## Fuentes consultadas en línea

- [Reviewing the Spectral Variation Hypothesis (Torresani et al. 2024)](https://www.sciencedirect.com/science/article/pii/S1574954124002449)
- [The spectral variability hypothesis does not hold across landscapes (Schmidtlein & Fassnacht 2017)](https://www.sciencedirect.com/science/article/abs/pii/S0034425717300482)
- [About the link between biodiversity and spectral variation (Fassnacht et al. 2022)](https://onlinelibrary.wiley.com/doi/10.1111/avsc.12643)
- [High-resolution spectral data predict taxonomic diversity in low diversity grasslands (Hayden et al. 2024)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1002/2688-8319.12365)
- [Spectral Diversity as a Predictor of Tree Diversity (Donnini et al. 2024)](https://www.tandfonline.com/doi/full/10.1080/07038992.2024.2403495)
- [Remote sensing of phenology (Dronova & Taddeo 2022)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2745.13897)
- [Plant diversity reduces satellite-observed phenological variability in wetlands (Dronova et al. 2022)](https://www.science.org/doi/10.1126/sciadv.abl8214)
- [Spatial heterogeneity of land surface phenology of global forests (Girardello et al. 2024)](https://iopscience.iop.org/article/10.1088/2515-7620/ad3c16)
- [Plant spectral diversity integrates functional and phylogenetic components (Schweiger et al. 2018)](https://www.nature.com/articles/s41559-018-0551-1)
- [Unraveling the seasonality of functional diversity through remote sensing (Mederer et al. 2025)](https://www.nature.com/articles/s43247-025-02646-x)
- [Benchmarking remote sensing methods to capture plant functional diversity from space (Pacheco-Labrador et al. 2026)](https://www.sciencedirect.com/science/article/pii/S1574954126000427)
- [Deep learning and satellite remote sensing for biodiversity monitoring (Pettorelli et al. 2024)](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.415)
- [Deep learning models map rapid plant species changes (Gillespie et al. 2024)](https://www.pnas.org/doi/10.1073/pnas.2318296121)
- [Mapping floristic gradients using an ordination-regression approach (Adams et al. 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0378112718317286)
- [Predicting species distributions and community composition (Pinto-Ledezma & Cavender-Bares 2021)](https://www.nature.com/articles/s41598-021-96047-7)
- [Beta diversity as the variance of community data (Legendre & De Cáceres 2013)](https://onlinelibrary.wiley.com/doi/abs/10.1111/ele.12141)
- [Dark diversity: shedding light on absent species (Pärtel et al. 2011)](https://www.sciencedirect.com/science/article/abs/pii/S0169534710002922)
- [Estimating probabilistic site-specific species pools and dark diversity (Carmona & Pärtel 2020)](https://onlinelibrary.wiley.com/doi/10.1111/geb.13203)
- [Global impoverishment of natural vegetation revealed by dark diversity (Pärtel et al. 2025)](https://www.nature.com/articles/s41586-025-08814-5)
- [From spectra to plant functional traits (Cherif et al. 2023)](https://sciencedirect.com/science/article/abs/pii/S0034425723001311)
- [sPlot – A new tool for global vegetation analyses (Bruelheide et al. 2019)](https://onlinelibrary.wiley.com/doi/10.1111/jvs.12710)
- [sPlotOpen (Sabatini et al. 2021)](https://onlinelibrary.wiley.com/doi/full/10.1111/geb.13346)
