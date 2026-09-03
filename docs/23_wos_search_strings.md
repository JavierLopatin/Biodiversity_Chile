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

---

## Acceso programático

Tres vías, en orden de fidelidad a Web of Science:

| vía | cobertura | clave | paquete |
|---|---|---|---|
| **WoS Starter API** | Core Collection, metadatos y citas, sin resúmenes | sí, gratuita con registro en developer.clarivate.com | Python: `clarivate/wosstarter_python_client` (o REST directo). R: `rwosstarter` (github.com/FRBCesab/rwosstarter) |
| **WoS Expanded API** | registros completos, afiliaciones, financiamiento, referencias citadas | sí, requiere suscripción institucional | R: `wosr` (CRAN 0.3.0); Python: cliente REST propio |
| **OpenAlex** | ~250 M de trabajos, sin etiquetas de campo ni operadores de proximidad | no | Python: `pyalex` o REST. R: `openalexR` |

La Starter API acepta las mismas etiquetas de campo que la interfaz web (`TS=`, `TI=`,
`PY=`, `AND`/`OR`/`NOT`), así que las cadenas de arriba corren sin cambios; la Expanded
añade el registro completo. OpenAlex no tiene `TS=` ni `NEAR/n`, de modo que cada búsqueda
se traduce a términos en título y resumen más filtros de año y tipo, y sus conteos no son
comparables uno a uno con los de WoS.

`scripts/75_literature_search.py` implementa las once búsquedas de este documento contra
los tres backends y escribe un CSV por bloque más un `summary.csv` con el conteo, la
consulta, el índice y la fecha, que es lo que hace verificable una afirmación de ausencia:

```bash
python scripts/75_literature_search.py --list
python scripts/75_literature_search.py --backend openalex --out results/literature
WOS_API_KEY=xxxx python scripts/75_literature_search.py --backend wos --queries B7a,B7b,B7c
```

Prueba con OpenAlex (2026-09-02): B7a diversidad filogenética desde teledetección 3.928
resultados, B7b LCBD y recambio composicional 295, B7c dark diversity desde teledetección
129, B2 fenología como predictor 14.978. Los conteos de OpenAlex son mucho más altos que
los de WoS porque la traducción sin etiquetas de campo busca también en el texto completo;
sirven para ordenar magnitudes y para no perder trabajos que WoS no indexa, no para
sustituir la búsqueda estructurada.

## Sesgo de cada índice, medido

Los dos índices tienen sesgos opuestos, y conviene saber cuál se está pagando.

**Web of Science selecciona.** Un comité decide qué revistas entran, y esa curaduría deja
fuera la mayor parte de la literatura de acceso abierto: de unas 62.700 revistas OA activas,
WoS indexa 6.157 y Scopus 7.351, contra 34.217 en OpenAlex. El efecto conocido es un sesgo
hacia revistas en inglés y del norte global, que es justamente donde la teledetección de
biodiversidad en América Latina queda subrepresentada.

**OpenAlex no selecciona, pero su metadato es heterogéneo.** Ingiere Crossref completo, así
que no hay sesgo editorial en qué entra; el sesgo aparece en qué se puede *recuperar*.
Elsevier no deposita resúmenes en Crossref, de modo que OpenAlex no los tiene, y una
búsqueda por resumen los pierde. Medido sobre 600 trabajos de este dominio
(`"remote sensing" OR satellite` × `"plant diversity" OR "species richness"`, 2015-2026,
2026-09-02):

| editorial | trabajos | con resumen |
|---|---:|---:|
| Wiley | 158 | 100 % |
| **Elsevier** | 134 | **54 %** |
| MDPI | 106 | 100 % |
| Nature Portfolio | 49 | 80 % |
| Springer | 32 | 59 % |
| resto (PLOS, OUP, IOP, Frontiers, PNAS, Royal Society, AAAS) | 121 | 100 % |
| **total** | **600** | **86 %** |

Esto importa aquí más que en otros dominios: *Remote Sensing of Environment*, *ISPRS
Journal of Photogrammetry*, *Ecological Informatics* y *Science of Remote Sensing* son todas
de Elsevier. Verificado por DOI: los dos artículos de Elsevier de la lista de referencias no
tienen `abstract_inverted_index` en OpenAlex, mientras que los de Wiley, Springer y Nature
sí.

**Cómo trabajar con eso.** La búsqueda por resumen en OpenAlex se complementa, no se
sustituye:
1. Buscar también por título (`title.search:`), que sí está completo para todas las
   editoriales.
2. Encadenar citas: el grafo de referencias de OpenAlex sí cubre Elsevier (239 referencias
   enlazadas para el artículo de RSE probado), así que partir de dos o tres trabajos ancla y
   recorrer `referenced_works` y `cited_by_api_url` recupera lo que la búsqueda por resumen
   pierde.
3. Declarar en el paper qué índice se usó para cada afirmación de ausencia. Una frase de
   vacío basada solo en OpenAlex con búsqueda por resumen sobreestima el vacío en revistas
   de Elsevier, que es donde más literatura de teledetección hay.

Dos advertencias adicionales sobre OpenAlex: no filtra por calidad, de modo que incluye
revistas depredadoras que WoS excluye por diseño; y su clasificación temática es automática,
no las categorías curadas de WoS (`WC=`), así que un filtro por disciplina no es equivalente
entre ambos. En sentido contrario, el número de resultados en inglés frente a otros idiomas
no es evidencia de sesgo del índice cuando la consulta está en inglés: la consulta misma
selecciona el idioma.

## Cómo obtener la clave y correr la búsqueda en WoS

1. Entrar a `https://developer.clarivate.com`, crear cuenta (el correo institucional de la
   UAI da acceso a lo que la suscripción de la universidad habilite) y confirmar el mail.
2. En *My Portal → Register an app*, crear una aplicación y suscribirla al producto
   **Web of Science Starter API**. La clave aparece en la ficha de la aplicación como
   *Primary key*. Si la biblioteca tiene contratada la **Expanded API**, suscribir también
   esa: devuelve resúmenes, afiliaciones, financiamiento y referencias citadas, que la
   Starter no trae.
3. Exportarla en el entorno, sin escribirla en ningún archivo versionado:

```bash
export WOS_API_KEY='...'            # o añadir a ~/.zshrc, nunca al repositorio
python scripts/75_literature_search.py --backend wos --out results/literature
python scripts/75_literature_search.py --backend wos --queries B7a,B7b,B7c --bibtex
```

El endpoint verificado el 2026-09-02 es
`https://api.clarivate.com/apis/wos-starter/v1/documents`, con la consulta en `q`, la base
en `db=WOS` y la clave en la cabecera `X-ApiKey`; responde 401 sin clave, que es la prueba
de que la URL y los parámetros son correctos.

Límites a tener en cuenta: la Starter tiene un tope mensual de registros y admite pocas
peticiones por segundo, de ahí `--sleep`; `--limit` acota cuántos registros se descargan por
bloque, mientras que el conteo total de la consulta se guarda igual en `summary.csv`. Para
las búsquedas de vacío (B7a, B7b, B7c) basta `--limit 50`, porque lo que interesa es el
conteo y los primeros títulos.

`--bibtex` resuelve cada DOI encontrado contra Crossref y escribe un `.bib` por bloque, con
el mismo procedimiento con que se verificó `paper/refs.bib`. Deliberadamente no se usa el
BibTeX que exporta Web of Science, porque trae claves propias y campos sin verificar.

Advertencia sobre el backend `crossref`: su parámetro de búsqueda es difuso y devuelve
cientos de miles de coincidencias con relevancia decreciente, así que sirve para resolver un
título conocido, no para contar. Los conteos citables salen de `wos`, y los de `openalex`
solo como referencia de magnitud.

## Barrido provisional en OpenAlex (2026-09-03)

La suscripción al *Free Institutional Member Plan* de la Web of Science Starter API quedó
en `Subscription approval is pending`, así que estos conteos vienen de OpenAlex y son
**órdenes de magnitud, no cifras citables**: la búsqueda de OpenAlex es difusa sobre texto
completo, por lo que B1 devuelve 47.265 registros para una consulta que en WoS, restringida
a título, resumen y palabras clave, devolverá bastante menos. Cuando llegue la clave hay que
repetir el barrido con `--backend wos` y reemplazar esta tabla.

| Bloque | Tema | Años | Registros |
|--------|------|------|-----------|
| B1  | núcleo: diversidad de plantas por teledetección | 2015-2026 | 47.265 |
| B2  | fenología / series de tiempo como predictor | 2015-2026 | 14.978 |
| B3  | hipótesis de variación espectral y sus críticas | 2000-2026 | 1.076 |
| B4  | mapas de diversidad a gran escala desde bases de parcelas | 2018-2026 | 1.117 |
| B5  | validación cruzada espacial, área de aplicabilidad | 2017-2026 | 2.117 |
| B5b | transferibilidad temporal entre años | 2015-2026 | 1.284 |
| B6  | aprendizaje profundo sobre series de imágenes | 2019-2026 | 7.637 |
| B7a | VACÍO: diversidad filogenética por teledetección | 2010-2026 | 3.928 |
| B7b | VACÍO: LCBD / recambio composicional | 2010-2026 | 295 |
| B7c | VACÍO: dark diversity por teledetección | 2010-2026 | 129 |
| B8  | contexto regional: Chile y Sudamérica | 2010-2026 | 3.625 |

Lo que el cruce contra el manuscrito arrojó:

- **B7b confirma el encuadre.** De los 295 registros, los primeros están dominados por beta
  diversidad acuática y por *generalized dissimilarity modelling*, no por LCBD desde
  sensores ópticos. Pero el barrido destapó dos ausencias reales en `refs.bib`, ya añadidas
  y verificadas por DOI: `rocchini2018measuring` (Methods Ecol Evol) y `rocchini2021from`
  (Ecological Informatics).
- **B7c no está vacío.** Existe un grupo consolidado de trabajos que estiman dark diversity
  con LiDAR aéreo en Europa. El manuscrito no reclama esa faceta, así que no toca ninguna
  afirmación, pero conviene no escribir nunca que la dark diversity no se ha estimado por
  teledetección.
- Candidatas periféricas no citadas, por si hacen falta en la Discusión: BioSCape
  (`10.1038/s44185-024-00071-5`) y la estimación de alfa y beta diversidad de líquenes
  (`10.1016/j.ecolind.2023.110173`).
