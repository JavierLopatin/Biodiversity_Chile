# Introducción: revisión y propuesta de historia

Insumo para escribir la Introducción de `paper/BiodiversityChile_Lopatin.tex`. Extiende
`docs/01_state_of_the_art.md` (agosto 2026) con la literatura localizada el 2026-09-02
mediante ocho búsquedas académicas y verificación de DOI contra Crossref. Todas las
referencias nuevas están en `paper/refs.bib`; los números citados abajo son textuales de
las fuentes, no estimaciones.

---

## 1. La historia en una línea

La teledetección de biodiversidad vegetal produce mapas de una faceta, de una fecha y
validados de forma optimista; nosotros predecimos tres facetas cuantitativas de diversidad
a partir de la **trayectoria fenológica de tres años** del píxel de cada parcela, sobre
3.102 parcelas repartidas en 25 grados de latitud, y las evaluamos con dos esquemas que
miden lo que un mapa realmente tendría que hacer: extrapolar en el espacio y extrapolar a
años no observados.

## 2. Estructura por párrafos

### P1 — Por qué hacen falta facetas, no una sola métrica
La biodiversidad no es un número. Riqueza, singularidad composicional (beta) y diversidad
filogenética responden a procesos distintos y no son intercambiables como objetivos de
conservación. La agenda de variables esenciales de biodiversidad observables desde el
espacio pide explícitamente varias de ellas (`skidmore2021priority`), y las síntesis del
campo llevan una década señalando que la teledetección puede aportar a más de una
(`wang2019remote`, `cavenderbares2022integrating`). **Gancho:** un mapa de riqueza no dice
si un sitio es singular, y ninguna de las dos cosas dice si el sitio conserva linajes
profundos.

### P2 — Lo que el paradigma dominante logra y dónde se detiene
La hipótesis de variación espectral (SVH) domina desde hace veinte años. Su balance es
ambiguo entre ecosistemas (`torresani2024reviewing`, `schmidtlein2017the`,
`fassnacht2022about`), y las revisiones coinciden en cuatro factores de contexto: tipo de
vegetación, escala y resolución, métrica elegida y **cambios de reflectancia en el tiempo
por fenología** (`lenormand2025spectral`, citando a Fassnacht). En la práctica los modelos
de una fecha rinden poco: en Sudáfrica, índices de Landsat-8 y Sentinel-2 con Random Forest
dan "relaciones débiles" con riqueza y Shannon (`mashiane2024`, R² ≤ 0,04); en Suiza la
correlación entre riqueza y complejidad espectral resulta **negativa** (`rossi2021`); una
revisión multi-ecosistema en China concluye que "los modelos existentes de teledetección
para estimar diversidad alfa vegetal exhiben típicamente una exactitud relativamente baja"
(`wang2026plant`). Donde sí funciona es en sistemas de baja diversidad
(`hayden2024highresolution`) y se debilita a alta riqueza (`donnini2024spectral`).

### P3 — La fenología como señal, no como estorbo
La alternativa lógica es tratar la trayectoria temporal como la señal. La evidencia se
acumuló en los últimos cinco años: la fenología de superficie integra dinámica de
comunidades (`dronova2022remote`); la diversidad vegetal **reduce** la variabilidad
fenológica interanual a escala nacional (`dronova2022plant`); la diversidad funcional
**varía estacionalmente** entre biomas, de modo que una sola fecha es por construcción
insuficiente (`mederer2025unraveling`); la asincronía espectral entre píxeles se propone
como medida de diversidad de respuesta en bosques secos brasileños
(`mazzochini2024spectral`); y agrupar series MODIS en "clusters espectrales" sobre más de
23.000 celdas en Francia captura gradientes alfa y beta (`lenormand2025spectral`).
Gholizadeh et al. lo dicen sin rodeos: "un aspecto crítico de la teledetección de la
diversidad vegetal que ha sido pasado por alto en muchos estudios es la dinámica temporal
de las comunidades vegetales (o fenología)" (`gholizadeh2020multitemporal`). Trabajo
propio previo ancla el mecanismo: la variabilidad interanual de la fenología se relaciona
con las comunidades vegetales (`lopatin2023interannual`, `lopatin2026remotely`).
**Lo que falta:** usar la curva completa como entrada de un modelo, no unas pocas métricas
resumidas, y hacerlo para varias facetas a la vez.

### P4 — Qué se ha logrado a gran escala, y con qué exactitud
Aquí va la comparación que le da escala al paper (Tabla 1 de este documento). Dos rutas
distintas:
- **Desde bases de parcelas con predictores climáticos**, sin teledetección: los mapas
  globales de diversidad alfa con 170.272 parcelas alcanzan "r de Pearson = 0,49" bajo
  validación cruzada por bloques espaciales (`sabatini2022global`), es decir del orden de
  0,24 de varianza explicada; los modelos globales de riqueza y riqueza filogenética por
  aprendizaje automático llegan a "70,3 % de variación explicada bajo validación cruzada
  espacial y 80,9 % bajo validación cruzada aleatoria" para riqueza y "73,7 % y 83,3 %"
  para riqueza filogenética (`cai2023global`), pero a grano de 7.774 km² y con
  predictores ambientales, no satelitales; en bosques europeos, Random Forest sobre 73.134
  parcelas explica "de 51,0 % a 70,9 %" de la diversidad alfa (`vecera2019alpha`).
- **Desde satélite**, la escala cae: los mejores resultados publicados son de paisaje
  (4.000 km², 135 parcelas de 1 ha, R² medio 0,477 con Sentinel-2;
  `liu2024tree`) o de textura a escala continental para otro taxón (41 % de la varianza de
  riqueza de aves con textura de EVI de Landsat; `farwell2020habitat`).
**El vacío:** no localizamos ningún mapa nacional o continental de facetas de diversidad
**vegetal** producido con series temporales satelitales y validado contra parcelas. La
revisión sistemática de América Latina y el Caribe lo dice desde el otro lado: "el uso de
teledetección en ALC es desproporcionadamente bajo en relación con la biodiversidad que
alberga" (`garzonlopez2024remote`).

### P5 — Por qué las exactitudes publicadas no se pueden creer sin más
Este párrafo justifica el diseño de validación y es donde el paper se distingue. La
validación cruzada aleatoria sobre datos agrupados infla el resultado: en series
temporales de Planet con 43 arquitecturas profundas, la validación espacialmente
independiente muestra que el muestreo aleatorio "infla el desempeño en 16 a 25 puntos
porcentuales" (`arrudabruno2026deep`); en mapeo de sitios ecológicos en Estados Unidos, la
exactitud pasa de 70–79 % en validación cruzada a 56 % y 44 % contra un conjunto
independiente (`maynard2019`). A eso se suma que los modelos globales extrapolan fuera de
su espacio de predictores y "las predicciones fuera del área de aplicabilidad deberían
evitarse" (`ludwig2023assessing`, `meyer2022machine`). Y la dimensión temporal, casi nunca
evaluada: la relación entre diversidad de campo y espectral "se debilitó respecto de datos
de la misma estación del año anterior" (`gholizadeh2020multitemporal`), y extrapolar
modelos de rasgos a nuevas ubicaciones, estaciones y tipos funcionales reduce el R² a
rangos de 0,12–0,49, 0,15–0,42 y 0,25–0,56 frente a la validación aleatoria no espacial
(`ji2024leaf`).

### P6 — El vacío específico y la oportunidad chilena
Tres huecos convergen: (i) la curva fenológica completa como predictor en vez de
heterogeneidad espectral de una fecha; (ii) predicción **simultánea** de facetas —el
multi-tarea existe para rasgos (`cherif2023from`) y para rangos taxonómicos
(`gillespie2024deep`), no para facetas de diversidad—; (iii) evaluación explícita de la
transferencia espacial **y** temporal. Chile mediterráneo y templado es un hotspot casi
ausente de esta literatura, y los antecedentes locales son de una fecha y una faceta
(`ceballos2015comparison`, `lopatin2016comparing`). Dos bases recientes lo desbloquean:
Parcelas-CL (`cerdaparedes2026parcelascl`) y el inventario Living Trees, que juntas dan
3.102 parcelas entre 30° y 55°S.

### P7 — Qué hacemos y qué preguntamos
Cerrar con el diseño y tres preguntas explícitas:
1. ¿Predice la trayectoria fenológica de tres años facetas de diversidad estandarizadas
   por cobertura, y cuáles?
2. ¿Aporta la representación (imagen 2D de la curva, convolución) sobre un Random Forest
   sobre la misma información?
3. ¿Sobrevive esa habilidad cuando el modelo tiene que predecir en bloques espaciales no
   vistos y en períodos censales no vistos?
Y anunciar el resultado sin adornos: la riqueza estandarizada por cobertura se predice bien
bajo bloqueo espacial (R² 0,58–0,78), la singularidad composicional moderadamente
(0,42–0,45), las facetas ponderadas por abundancia no se predicen, y la extrapolación
temporal —no la capacidad del modelo— es el límite.

---

## 3. Tabla 1 — Puntos de comparación de gran escala (para el párrafo 4)

| Estudio | Escala / grano | Datos de entrenamiento | Predictores | Faceta | Exactitud reportada | Validación |
|---|---|---|---|---|---|---|
| `sabatini2022global` | global, 3 granos | 170.272 parcelas | clima, suelo, topografía | riqueza alfa | **r = 0,49** (≈ R² 0,24) | bloques espaciales |
| `cai2023global` | global, 7.774 km² | 830 inventarios regionales | ambientales (presente y pasado) | riqueza; riqueza filogenética | **70,3 % / 73,7 %** espacial; 80,9 % / 83,3 % aleatoria | espacial y aleatoria |
| `vecera2019alpha` | Europa, 400 m² | 73.134 parcelas | ambientales, uso de suelo, historia | riqueza alfa forestal | **51,0–70,9 %** | CV estándar |
| `liu2024tree` | paisaje, 4.000 km² | 135 parcelas de 1 ha | Sentinel-2, RapidEye, Landsat-8, PlanetScope | diversidad arbórea (Shannon) | **R² medio 0,477** (Sentinel-2) | CV |
| `farwell2020habitat` | EE.UU. continental | red de aves | textura de EVI Landsat | riqueza de aves | **41 %** (hasta 51 % con topografía) | CV |
| **Este estudio** | Chile 30–55°S, 30 m | 3.102 parcelas | serie kNDVI de 3 años + topografía | TD₀, PD₀, LCBD | **R² 0,78 / 0,59 / 0,43** | bloques de 20 km; además LLTO |

Lectura para el texto: nuestro TD₀ bajo bloqueo espacial (0,78) es comparable con los
mejores modelos globales de riqueza que usan clima a grano de miles de km²
(`cai2023global`, 0,703), y muy superior al único mapa global validado por bloques con
parcelas (`sabatini2022global`, r = 0,49), **pero** a 30 m, con predictores satelitales y
sobre un dominio de un país. No es un récord: es una escala distinta con una fuente de
información distinta. Y el mismo modelo cae por debajo de cero al extrapolar a períodos no
vistos, que es la parte que la literatura de gran escala casi nunca reporta.

## 4. Claims que el paper puede hacer, y el que no

**Puede afirmar:**
- Primera predicción simultánea de facetas taxonómica, filogenética y composicional de
  diversidad vegetal desde series fenológicas satelitales, con validación espacial.
- Primer conjunto de modelos de diversidad vegetal para Chile que cubre 25° de latitud y
  combina dos redes de parcelas.
- Evidencia cuantitativa de que la curva cruda de tres años supera al compuesto anual, y de
  que la arquitectura (2D vs 1D vs Random Forest) importa menos que el esquema de
  validación.
- Cuantificación explícita de la pérdida por extrapolación temporal, que la literatura de
  mapas de gran escala rara vez mide.

**No puede afirmar:** causalidad fenología→diversidad; que las facetas ponderadas por
abundancia sean mapeables; que el mapa sea válido fuera del área de aplicabilidad ni fuera
del período muestreado.

## 5. Referencias nuevas incorporadas a `refs.bib`

`sabatini2022global`, `cai2023global`, `vecera2019alpha`, `meyer2022machine`,
`ludwig2023assessing`, `wang2019remote`, `kattenborn2021review`, `lenormand2025spectral`,
`gholizadeh2020multitemporal`, `frye2021plant`, `ji2024leaf`, `garzonlopez2024remote`,
`liu2024tree`, `mazzochini2024spectral`, `hauser2021towards`, `rocchini2021zero`,
`kacic2022forest`. DOI verificados contra Crossref el 2026-09-02.

Faltan por incorporar (citadas arriba por apellido, sin entrada aún; requieren
verificación): Mashiane et al. 2024 (10.1111/avsc.12778), Rossi et al. 2021
(10.1002/rse2.244), Wang et al. 2026 (10.1002/ece3.72899), Arruda Bruno et al. 2026
(10.1111/tgis.70250), Maynard et al. 2019 (10.2136/sssaj2018.09.0346).
