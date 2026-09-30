# Material para un segundo paper, y qué haría falta añadir

Fecha: 2026-09-29. Complemento de `docs/26`, que define qué entra en el primero.

Regla de corte que se usó para repartir: **el primero contesta "¿qué aporta el satélite sobre
la geografía, y cuándo?"**. Todo lo que conteste otra pregunta va aquí, aunque esté medido y
aunque sea bueno. Lo que sigue está ordenado por madurez: lo primero ya está corrido.

---

## 1. GDM: qué impulsa el recambio, y a qué tasa — **ya corrido, es lo más maduro**

Implementado en `src/biodiv/gdm.py` y `scripts/21_run_gdm.py`, con las I-splines monótonas
escritas a mano porque GDM necesita la forma integrada.

Resultados sobre el pool anterior (PCL, `kfold5_owner`, 62 predictores), ρ de Spearman contra el
Jaccard observado, sobre pares **entre parcelas excluidas**:

| | ρ |
|---|---|
| GDM solo distancia geográfica | +0,435 |
| **GDM con los predictores** | **+0,492** |
| SGDM (sCCA fold-local) | +0,436 |
| RF sobre 8 ejes PCoA → distancia | +0,425 |
| techo: los 2 ejes verdaderos | +0,465 |
| techo: los 8 ejes verdaderos | +0,530 |

**El GDM desde satélite reproduce la composición observada mejor que conocer los valores
verdaderos de los dos ejes que el proyecto modela.** Funciona porque su enlace
`d = 1 − exp(−η)` es asintótico en 1 por construcción: la saturación deja de ser el error del
modelo y pasa a ser su forma.

**Qué falta:** rehacerlo sobre el pool unificado leñoso con `kfold5_block20_unified`, y con la
variante `gdm_geo` como control obligatorio (la geografía sola ya da +0,435 de los +0,492, o sea
el aporte propio es +0,057 — coherente con lo medido hoy con `B03`).

**Qué lo hace un paper y no una sección:** el entregable de GDM no es un R², son **las alturas
de las I-splines**, que dicen cuánto recambio compra cada variable y **a qué tasa a lo largo del
gradiente**. Eso es una pregunta ecológica, no de predicción.

**Literatura que lo respalda ahora mismo:** Mokany et al. 2022 (continental, D² 32,7 %),
spGDMM (White et al. 2024), y **stGDMM** (arXiv 2608.05352, 2026) que añade el eje temporal —
con censos 2003–2026 y Landsat desde 2000, tenemos con qué.

---

## 2. Composición funcional por amplitud de nicho — **el hallazgo más original**

Medido hoy. Agrupando las leñosas por **amplitud altitudinal de nicho** (terciles del rango del
catálogo de Rodríguez et al. 2018) en vez de por taxonomía:

| agrupamiento | clases | **\|rho con latitud\|** | techo |
|---|---|---|---|
| especie | 307 | 0,825 | 0,606 |
| familia | 70 | 0,800 | 0,517 |
| hábito × amplitud de rango | 9 | 0,408 | 0,364 |
| **amplitud altitudinal** | 3 | **0,221** | 0,370 |
| amplitud de rango | 3 | 0,229 | 0,296 |
| hábito (árbol/arbusto) | 3 | 0,350 | 0,219 |
| origen (endémico/nativo) | 3 | 0,375 | 0,196 |

**Existe un eje composicional en estos datos que la geografía no da ya.** La correlación con
latitud cae de 0,825 a 0,221 —casi un factor 4— y conserva más información de especie que
cualquier otro agrupamiento funcional probado. Domina a `hábito × rango` en las dos columnas.

Tiene sentido por construcción: "qué proporción de las leñosas de esta parcela son generalistas
altitudinales contra especialistas estrechos" no es una pregunta latitudinal — hay generalistas
y especialistas a todas las latitudes.

**Por qué es un paper aparte:** es una **respuesta nueva**, no una representación distinta de la
misma. El techo cae de 0,606 a 0,370: se pierde el 40 % de la capacidad de reconstruir la
disimilitud entre especies. Si el objetivo es qué especies hay, es un retroceso; si es ensamblaje
funcional independiente del gradiente, es lo correcto. Son dos preguntas.

**Y cierra un hueco que el propio vault marca:** *"la transferencia de sistemas tipo Ellenberg
fuera de Centroeuropa y el Reino Unido no está tratada en ninguna parte — directamente relevante
para el foco chileno de este vault"*. El catálogo de Rodríguez da rango altitudinal y
distribución por región para la flora chilena: **es un sustituto chileno de un valor indicador,
derivado de catálogo en vez de consenso de expertos.** Eso es publicable por sí solo.

**Qué falta:**
- **Cobertura.** Del parseo del catálogo salen 201/347 leñosas con hábito, 252/347 con
  distribución, 229/347 con rango altitudinal. Un tercio se cae; hay que cerrarlo.
- **Los cortes son terciles arbitrarios.** Habría que fijarlos por criterio ecológico, o al menos
  demostrar que el resultado no depende del corte.
- **Declarar que es un rasgo de especie a escala país**, no una medición local — como los
  valores de Ellenberg.

---

## 3. Estabilidad temporal de los predictores — **la pregunta que originó la línea de sequía**

La pregunta del autor que abrió todo esto: *"si las LSP o la curva predicen diversidad, ¿esas
curvas varían mucho en el tiempo?"*. Sigue sin contestarse, porque hace falta la serie completa.

**Qué falta:** la extracción 2000–2026 por parcela, ya especificada en
`docs/25_unified_series_extraction_spec.md` y encargada al datacube, sin entregar.

**Qué contestaría:** (1) si dos clases —seca y húmeda— bastan o hacen falta más; (2) la varianza
interanual real de cada métrica de LSP; (3) si la ventana causal de 3 años es la correcta.

Es el complemento natural del resultado de sequía del primer paper: ese dice *que* el predictor
funciona peor en año seco, este diría *cuánto se mueve el predictor* y por tanto cuánto de la
degradación es inestabilidad del predictor contra cambio real.

---

## 4. Transferencia temporal (LLTO) — **arreglado y verificado, sin explotar**

El buffer causal del LLTO no excluía nada para Living Trees (`win_start`/`win_end` venían NaN).
Arreglado; verificado el 2026-09-29: **0 filas de train dentro del rango de años del test y 0
solapes de ventana causal, en los 29 folds y en las dos fuentes**. Antes eran 1.104 sólo en
`s0t1`.

Con el buffer correcto, la curva leñosa da LCBD **0,335 bajo LLTO contra 0,447 bajo block20**,
misma n, mismas parcelas.

**No se puede interpretar todavía**: `results/models_llto_fixed` tiene 8 corridas y **las 8 son
de curva**. Sin un piso bajo LLTO, el 0,335 podría ser todo topografía — el piso de block20 es
0,355, más alto. Falta `B01c` bajo `kfold_loc_time_unified`, y con eso ya se lee.

Cautela que va en el diseño: los bloques temporales del LLTO están casi perfectamente
confundidos con la fuente (t1+t2 son 92 % Living Trees, t4+t5 100 % Parcelas-CL), así que aun
arreglado el LLTO mide en parte el contraste entre inventarios.

---

## 5. Corrección topográfica SCS+C — **nota metodológica negativa, y útil**

Ya corrido. Resultado: la corrección es **estructural en la reflectancia y nula en la
predicción**. Mueve EVI +17,8 % y kNDVI +1,7 %; el ΔR² para kNDVI queda dentro del nulo; el
efecto sobre el modelo desplegado es **+0,002 agrupado** y +0,023 en pendientes > 25°.

Un negativo bien medido y con control nulo bien construido es publicable como nota corta, y
ahorra trabajo a quien venga detrás. También sostiene la decisión de publicar el mapa sin
corregir.

---

## 6. La crítica a la estandarización por cobertura

Va en el primer paper como resultado (`docs/26` §2), pero **da para una nota metodológica
propia** si crece: la estandarización por cobertura destruye la señal de riqueza en sistemas
pobres en especies, al restringir la muestra a las parcelas ricas. Medido: sd del target
3,04 → 1,70, y la riqueza cruda en el mismo subconjunto da **−0,625**, peor que la
estandarizada.

Generalizarlo exigiría replicarlo en otros conjuntos, no sólo Chile. Es la diferencia entre una
observación y una nota metodológica.

---

## 7. Datos que habría que añadir, por orden de valor

| dato | para qué | estado |
|---|---|---|
| **Hábito foliar y forma de crecimiento** de las 347 leñosas (Rodríguez et al. 2018) | el PFT funcional de §2, y descripción en el primer paper | parseado a medias (201/347 hábito); falta cerrar |
| **Serie Landsat 2000–2026 por parcela** | §3, estabilidad de predictores | especificada (`docs/25`), encargada, sin entregar |
| **Suelo (SoilGrids)** | Mokany 2022 encuentra textura de suelo entre los predictores más fuertes; el corpus marca la falta de suelo como limitación recurrente | no pedido |
| **Perfil de horizonte desde DEM** | sombra proyectada contra auto-sombreado, para cerrar §5 | encargado al datacube, sin entregar |
| **Embeddings AlphaEarth 2017+** | comparador contra el geomediano, bajo el mismo CV de bloques | no pedido; ojo: sólo cubre desde ~2017 y su ventaja se estrecha bajo validación espacial |

---

## 8. Bloqueante que no es de análisis

**Living Trees Chile no tiene cita en el repo.** `[CITATION REQUIRED]`, clave
`livingtrees_chile`. Es un bloqueante de envío del primer paper y lo resuelve el autor, no el
análisis.
