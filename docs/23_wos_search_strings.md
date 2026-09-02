# Cadenas de búsqueda para Web of Science

Para reproducir y ampliar la revisión de `docs/22_introduction_storyline.md` en Web of
Science Core Collection. Sintaxis de la búsqueda avanzada de WoS: `TS=` busca en título,
resumen y palabras clave; `TI=` solo título; `PY=` año; `WC=` categoría; los operadores son
`AND`, `OR`, `NOT`, `NEAR/n` (proximidad en la misma frase) y `SAME` (misma oración). Los
comodines son `*` (cero o más caracteres), `$` (cero o uno) y `?` (exactamente uno).

Recomendación de uso: correr cada bloque por separado, guardarlo como conjunto (`#1`, `#2`,
…) y combinarlos al final. Fija el índice a *Web of Science Core Collection*, tipo de
documento `Article OR Review`, y el rango de años que se indica en cada bloque.

---

## B1 — Núcleo: diversidad vegetal desde teledetección

```
TS=(("remote sensing" OR satellite* OR spaceborne OR "earth observation" OR Landsat OR
     Sentinel-2 OR MODIS OR hyperspectral OR imaging spectroscopy)
    AND ("plant diversity" OR "species richness" OR "vegetation diversity" OR
     "floristic composition" OR "community composition" OR "beta diversity" OR
     "phylogenetic diversity" OR "functional diversity" OR biodiversity)
    AND (map* OR predict* OR estimat* OR model*))
```
`PY=2015-2026`. Es el bloque ancho; devuelve miles de registros y sirve como universo del
que se recortan los demás. Refinar con `WC=(Remote Sensing OR Ecology OR Biodiversity
Conservation OR "Environmental Sciences" OR "Plant Sciences")`.

## B2 — Fenología como predictor (el eje del paper)

```
TS=(("land surface phenology" OR "phenolog*" OR "time series" OR "temporal trajector*" OR
     "seasonal dynamic*" OR "interannual variab*" OR "growing season" OR "phenological
     metric*")
    NEAR/5 (satellit* OR "remote sensing" OR Landsat OR Sentinel OR MODIS))
AND TS=("plant diversity" OR "species richness" OR "species composition" OR
        "community composition" OR "beta diversity" OR "phylogenetic diversity" OR
        "functional diversity")
```
`PY=2015-2026`. La proximidad `NEAR/5` evita los papers de fenología agronómica sin
componente de diversidad. Es la búsqueda que aporta la mayor parte de la sección 3 de la
historia.

## B3 — Hipótesis de variación espectral, y sus críticas

```
TS=("spectral variation hypothesis" OR "spectral diversity" OR "spectral heterogeneity" OR
    "spectral variability" OR "Rao's Q" OR "spectral species")
AND TS=(biodiversity OR "species richness" OR "plant diversity" OR "tree diversity")
```
`PY=2000-2026`. Añadir `AND TS=(inconsisten* OR contradict* OR "does not hold" OR fail* OR
negative OR limitation* OR caveat*)` para aislar la literatura crítica, que es la que
justifica el párrafo 2.

## B4 — Predicciones de gran escala (el punto de comparación)

```
TS=(global OR continental OR national OR "country-wide" OR "wall-to-wall" OR macroecolog*)
AND TS=("species richness" OR "plant diversity" OR "phylogenetic diversity" OR
        "alpha diversity" OR "vascular plant*")
AND TS=(map* OR "spatial prediction*" OR "machine learning" OR "random forest" OR
        "boosted regression" OR "deep learning" OR "neural network*")
AND TS=("vegetation plot*" OR sPlot OR "plot database" OR "forest inventory" OR
        "field plot*" OR "ground data")
```
`PY=2018-2026`. De aquí salen Sabatini 2022, Cai 2023 y Večeřa 2019, que son los tres
números con los que el paper se compara.

## B5 — Validación honesta: bloqueo espacial, área de aplicabilidad, transferencia

```
TS=("spatial cross-validation" OR "block cross-validation" OR "spatial block*" OR
    "leave-location-out" OR "leave-one-region-out" OR "area of applicability" OR
    "spatial autocorrelation" NEAR/5 (validat* OR overfit* OR overestimat*))
AND TS=(ecolog* OR biodiversity OR vegetation OR "species distribution" OR
        "remote sensing" OR mapping)
```
`PY=2017-2026`. Para el párrafo 5. Complementar con
`TS=("temporal transferability" OR "temporal extrapolation" OR "leave-time-out" OR
"across years" OR "interannual transfer*") AND TS=(model* AND (vegetation OR biodiversity
OR "remote sensing"))`, que es la parte peor cubierta de la literatura y donde el paper
aporta.

## B6 — Aprendizaje profundo sobre series temporales satelitales

```
TS=("deep learning" OR "convolutional neural network*" OR CNN OR transformer* OR
    "recurrent neural network*" OR LSTM OR "self-supervised" OR "masked autoencoder*")
AND TS=("satellite image time series" OR "image time series" OR "time series" OR
        phenolog* OR "temporal profile*")
AND TS=(vegetation OR biodiversity OR "species richness" OR "plant communit*" OR
        "land cover" OR "tree species")
```
`PY=2019-2026`. Sirve para posicionar la arquitectura y para el argumento de que el cuello
de botella son las etiquetas, no el modelo.

## B7 — Facetas específicas poco cubiertas

```
TS=("phylogenetic diversity" OR "Faith's PD" OR "phylogenetic endemism" OR
    "evolutionary distinctiveness")
AND TS=("remote sensing" OR satellite* OR spectral* OR Sentinel OR Landsat)
```
```
TS=("local contribution to beta diversity" OR LCBD OR "beta diversity partition*" OR
    "generalized dissimilarity model*" OR GDM OR "compositional turnover")
AND TS=("remote sensing" OR satellite* OR Landsat OR Sentinel OR "spectral")
```
```
TS=("dark diversity" OR "species pool" NEAR/3 (absent OR "site-specific"))
AND TS=("remote sensing" OR satellite* OR predict* OR map*)
```
`PY=2010-2026`. Los tres son búsquedas de vacío: se espera que devuelvan poco, y ese poco
es el argumento de novedad. Guardar el número de resultados de cada una, porque una frase
del tipo "no localizamos ningún estudio que…" se defiende mejor con la cadena y la fecha de
la búsqueda declaradas.

## B8 — Contexto regional

```
TS=(Chile OR "South America" OR Patagonia* OR "Mediterranean climate" OR
    "temperate rainforest" OR Valdivian OR sclerophyll* OR matorral)
AND TS=(biodiversity OR "plant diversity" OR "species richness" OR vegetation)
AND TS=("remote sensing" OR satellite* OR Landsat OR Sentinel OR MODIS)
```
`PY=2010-2026`. Confirma la escasez regional; complementar con la revisión sistemática de
América Latina ya citada.

---

## Combinaciones útiles

| Objetivo | Combinación |
|---|---|
| Universo del tema | `#1` |
| Fenología como señal | `#2` |
| Vacío central del paper | `#2 AND #7` (fenología × facetas poco cubiertas) |
| Comparación de escala | `#4` |
| Justificación del diseño de validación | `#5` |
| Novedad metodológica | `#6 AND #1` |
| Vacío regional | `#8` |

## Notas de reproducibilidad

- Anotar fecha de la búsqueda, edición del índice (SCI-EXPANDED, SSCI, ESCI) y número de
  resultados por bloque; una afirmación de ausencia solo es verificable con esos tres datos.
- WoS no indexa preprints de bioRxiv ni EGUsphere, y varias referencias de este paper lo
  son. Complementar con Scopus, Google Scholar y una búsqueda directa en bioRxiv/EcoEvoRxiv.
- El operador `NEAR/n` no funciona dentro de comillas; las frases exactas deben ir como
  términos separados unidos por `NEAR/n`.
- Exportar en formato BibTeX desde WoS produce claves distintas a las de `paper/refs.bib`;
  conviene exportar a RIS o texto plano y resolver los DOI contra Crossref, que es el
  procedimiento usado en este repositorio.
