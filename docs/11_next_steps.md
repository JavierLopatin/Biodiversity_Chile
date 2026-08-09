# Qué falta, en qué orden, y con qué comando

Documento de traspaso. La fase de predictores terminó: los cubos `.nc` están destilados, el
cribado corrió sus 20 bloques y la comparación de familias de modelo corrió sobre 5 conjuntos
de predictores. Los resultados están en [`10_findings.md`](10_findings.md); esto es solo lo que
queda por hacer.

**Dos decisiones de proyecto ya tomadas, que no se re-discuten aquí:**

1. **`kfold5_window` es el esquema de validación primario y único.** Todo lo que sigue se corre
   bajo él. `kfold5_owner` y `kfold5_random` se calculan como diagnóstico y se reportan, nunca
   se optimiza contra ellos.
2. **Alfa se reporta sola**, sin descomposición within/between adjunta. La descomposición
   existe (`scripts/24_alpha_decomposition.py`, `results/tables/alpha_decomposition.csv`) y
   está resumida en `10_findings.md` §1b por si un revisor la pide, pero no va en las tablas de
   resultados.

---

## 0. Montar la máquina local (10 min, sin datacube)

**Data Cube Chile ya no hace falta.** Se verificó rastreando las importaciones: `datacube` solo
lo usan `src/biodiv/cube.py` y los scripts **02** (fenología), **03** (topografía) y **22**
(clima), los tres ya corridos y con su salida versionada. Nada del modelamiento vuelve a
tocarlo.

```bash
git clone git@github.com:JavierLopatin/Biodiversity_Chile.git
cd Biodiversity_Chile
pip install -r requirements.txt
```

Los 12 archivos que lee el pipeline (50 MB) están todos en git — `phenoshape_pixels.parquet`,
`cube_predictors.parquet`, `phenoshape_by_index.parquet`, `lsp_all_auto.parquet`,
`topography/`, `climate.parquet`, `composition_axes.parquet`, `plots_subset.parquet`,
`biodiversity_responses.parquet`, `cv_folds_modelling.parquet`. **Los 346 MB de cubos `.nc` no
se transfieren:** ya están destilados.

En la máquina con GPU hay que reemplazar la línea de `torch` de `requirements.txt` por la build
de CUDA del driver correspondiente. Y **`PyWavelets` es obligatoria**: sin ella,
`biodiv.transforms1d.CWTTransform` aborta y con él todas las corridas `substrate=cwt` de la CNN
1D, además de `tests/test_modelling.py::test_every_substrate_runs_through_one_model`.

Comprobación de que quedó bien montado:

```bash
python -m pytest tests/ -q          # 44 pasan; si falla solo el de sustratos, falta pywt
```

---

## 1. La cola que quedó a medias (≈1 h de CPU, ninguna GPU)

`scripts/run_climate_queue.sh` alcanzó a correr las filas de clima (X15–X19) y el factor de
agregación; **las tres corridas de GDM con clima nunca arrancaron.** El proceso se detuvo a
mano, no falló: quedó bloqueado 40 min en su propia guarda, porque `pgrep -f run_gdm_queue`
sin anclar también matcheaba un proceso de monitoreo que mencionaba ese nombre. La guarda ya
está anclada en el script, así que el bug no se repite.

`21_run_gdm.py` hace *append* sobre `results/tables/gdm_comparison.csv` y `19_screen_blocks.py`
reanuda por `(fila, esquema, agregación)`, así que **relanzar la cola completa es seguro**: lo
ya hecho se salta solo y un kill cuesta como mucho la corrida en vuelo.

```bash
nohup bash scripts/run_climate_queue.sh >> logs/climate_queue.log 2>&1 &
```

`nohup … &` importa: así el proceso sobrevive si se cae la sesión — que es exactamente lo que
pasó la vez anterior y por lo que el cómputo no se perdió. Para seguirlo:

```bash
tail -f logs/climate_queue.log
pgrep -af '^bash scripts/run_climate_queue\.sh'     # vacío = terminó
```

Lo que falta ejecutar dentro de esa cola, y por qué importa cada una:

| corrida | pregunta |
|---|---|
| `--spec clim+topo` | ¿cuánto del recambio composicional explica el clima solo, en espacio GDM? |
| `--spec curve+clim+topo` | el bloque ganador del cribado (X17), ahora como GDM |
| `--sgdm-grid` sobre el spec completo | **la que más importa**: busca en grilla las penalizaciones L1 del sCCA, como hace Leitão et al. Con las penalizaciones por defecto el SGDM queda por debajo del GDM simple en las 5 filas corridas, y hasta que esta grilla termine ese resultado no es interpretable — no se puede afirmar que SGDM sea peor si nunca se lo sintonizó. |

**Falta además la mitad del cribado bajo el esquema nuevo.** El factor de agregación (X07 y
X12 × `mean`/`trimmed`/`center`) corrió solo bajo `kfold5_owner`, porque cuando se lanzó ese
era el primario:

```bash
for agg in mean trimmed center; do
  python scripts/19_screen_blocks.py --rows X07 X12 --schemes kfold5_window --agg "$agg"
done
```

---

## 2. Re-correr los 79 modelos bajo `kfold5_window` (≈1 día, GPU)

**Este es el bloqueante de todo lo demás.** `08_modelling.md` reporta 79 modelos con
`kfold5_owner` como primario, y ninguno tiene `kfold5_window`. Mientras no se re-corran, las
tablas viejas y las nuevas hablan de números distintos y no se pueden poner en la misma figura.

```bash
python scripts/14_run_matrix.py --schemes kfold5_window
python scripts/12_model_report.py
```

Al re-generar el reporte hay que cambiar dos referencias que quedaron obsoletas:

- La comparación base pasa de `X03` (la curva sola) a **`X17` = `curve+clim+topo`**, que es el
  bloque ganador del cribado.
- `kfold5_owner` pasa de primario a diagnóstico.

---

## 3. Los modelos profundos sobre el bloque ganador (≈2–3 días, GPU)

La compuerta de `09_predictors.md` §6 decía "los bloques que superen a X03 pasan a MLP, CNN 1D
y CNN 2D". Con el cribado terminado el bloque a llevar es **`curve+clim+topo`, 87 columnas**,
con kNDVI como índice.

```bash
python scripts/14_run_matrix.py --schemes kfold5_window --spec "curve+clim+topo+area" --index kndvi
```

Toda la infraestructura se reusa sin cambios: `dl_runner`, los sustratos, la pérdida enmascarada.
Solo cambian los bloques de features.

**Expectativa honesta: un empate con el Random Forest.** El resultado 4 del cribado dice que el
margen no está en la capacidad del modelo — X14 con 511 columnas cae a +0,205 contra +0,309 de
X17 con 87. Con 1.082 parcelas, cualquier arquitectura hereda ese límite. Vale la pena correrlo
porque es la comparación que el paper necesita, no porque se espere que gane.

---

## 4. La decisión de diseño que sigue abierta: qué se predice

Es la más importante y no la resuelve ningún cómputo pendiente; hay que elegir.

El GDM con predictores y geografía llega a ρ = +0,527 contra un techo de +0,530 para los 8 ejes
PCoA verdaderos. **Por el lado del recambio entre pares casi no queda margen.** Al mismo tiempo,
los 2 ejes que el proyecto modela hoy tienen un techo de +0,465, o sea que el diseño actual está
persiguiendo un objetivo peor que el que el GDM ya alcanza.

Los dos caminos son incompatibles y hay que tomar uno:

- **Abandonar los ejes y reportar GDM.** No tiene el cuello de botella de la ordenación, y su
  número ya supera al techo de los 2 ejes. Cuesta reescribir el marco de resultados.
- **Subir a 8–16 ejes** (ya están calculados en `composition_axes.parquet`). Recupera masa de
  varianza —31 % con 8 ejes contra 12,5 % con 2— pero el R² por eje deja de ser comparable con
  las 79 corridas anteriores.

---

## 5. Solo si lo anterior se estanca

Traer fuentes nuevas de Data Cube Chile: **SAR de Sentinel-1** y **red-edge de Sentinel-2**.
Ninguna de las dos está en los cubos `.nc` guardados, y son lo único para lo que el datacube
vuelve a hacer falta ahora que el clima ya se extrajo. Es el único punto de esta lista que
requiere volver a una máquina con credenciales.

---

## Apéndice — deudas técnicas conocidas

Ninguna bloquea lo de arriba; van anotadas para que no se descubran dos veces.

- **47 parcelas (4,3 %) con fuga de ventana bajo `kfold5_owner`.** No afecta a `kfold5_window`,
  que es cero por construcción. Solo importa si se vuelve a reportar algo bajo `owner`; el
  parche (soltar 12 parcelas) está descrito en `10_findings.md` §1b.
- **La ventana de 5×5 puede no estar comprando nada.** El píxel central solo iguala a cualquier
  promedio de los 25 (§4c, resultado 5). Vale la pena confirmarlo sobre más bloques antes de
  defender el diseño de ventana en el paper.
- **`order_coherent` sigue en `BLOCK_QC`** y es algebraicamente idéntico a `rog_nan | ros_nan`;
  `degenerate_px_frac` solo detecta `sos == pos` y no ve `eos == pos` ni `eos < sos`
  (`09_predictors.md` §2.4).
- **La fase 1 del doc 09 (LSP sin NaN) nunca se hizo** y quedó sin hacer falta: los predictores
  derivados de los cubos tienen cero NaN, contra el 5,1 % de píxeles que falla en LSP, así que
  el bloque `lsp` dejó de ser la ruta principal. Si se retoma, la compuerta de reproducción de
  §2.2.1 sigue siendo obligatoria antes de creerle a cualquier mejora.
