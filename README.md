# Biodiversity_Chile

Estimación multitemporal de facetas múltiples de biodiversidad vegetal en Chile central a partir de
fenología satelital.

Se predicen simultáneamente diversidad alfa taxonómica, contribución local a la diversidad beta
(LCBD), diversidad filogenética, diversidad funcional y composición florística (ejes de ordenación),
usando la curva fenológica reconstruida desde series temporales Landsat y Sentinel-2 más variables
topográficas derivadas de un DEM.

**Estado:** fase de diseño. Solo documentación; sin pipeline todavía.

---

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/01_state_of_the_art.md`](docs/01_state_of_the_art.md) | Revisión crítica en 7 ejes: Spectral Variation Hypothesis, fenología como predictor de biodiversidad, diversidad funcional y filogenética desde teledetección, regresión sobre ejes de ordenación, deep learning multi-tarea, dark diversity, contexto chileno. Cierra con tabla de 5 gaps. |
| [`docs/02_innovation_and_impact.md`](docs/02_innovation_and_impact.md) | Evaluación graduada de innovación, tabla de 9 riesgos cuantificados, impacto esperado, y sección explícita de lo que el proyecto **no** va a demostrar. |
| [`docs/03_cnn_architecture.md`](docs/03_cnn_architecture.md) | Diseño de `PhenoNet-S`, CNN de ~15 k parámetros para n ≈ 1.000 parcelas. Sustrato *phenocube* (año × DOY), catálogo de transformaciones señal→imagen, régimen de entrenamiento y matriz experimental. |
| [`docs/refs.bib`](docs/refs.bib) | 42 referencias; todos los DOI resueltos contra la API de Crossref. |

---

## Datos

Ninguna base de datos se versiona en este repositorio. Todas son públicas:

| Base | Contenido | Acceso |
|---|---|---|
| **Parcelas-CL** | 1.485 parcelas de vegetación georreferenciadas, 675 especies leñosas, 1976–2026, 30,25°S–54,82°S | [10.5281/zenodo.20602096](https://doi.org/10.5281/zenodo.20602096) — preprint: [10.21203/rs.3.rs-9986019/v1](https://doi.org/10.21203/rs.3.rs-9986019/v1) |
| **Rasgos-CL** | 662 especies leñosas chilenas, 25.174 registros, 23 rasgos funcionales | [github.com/dylancraven/Rasgos-CL](https://github.com/dylancraven/Rasgos-CL) — paper: [10.1111/geb.13755](https://doi.org/10.1111/geb.13755) |

Descarga esperada en `data/` (ignorada por git).

---

## Métodos y dependencias propias

| Repositorio | Rol |
|---|---|
| [PhenoSensing](https://github.com/JavierLopatin/PhenoSensing) | Reconstrucción de curvas fenológicas (`PhenoShape`) y 18 métricas LSP (`PhenoLSP`) sobre cubos xarray. Aporta el predictor. |
| [Trait_2DCNN](https://github.com/JavierLopatin/Trait_2DCNN) | Transformaciones señal→imagen, pérdida enmascarada multi-target y pretraining MAE. Aporta el marco de modelado. |

---

## Diseño acordado

- **Alcance:** Chile central. Landsat 1999–2026 como serie base; Sentinel-2 / HLS desde ~2017 para mayor
  resolución temporal en el período reciente. El sensor se trata como estrato, no se mezclan las series.
- **Abundancia (dos niveles):** métricas de presencia/ausencia sobre todas las parcelas; métricas
  ponderadas solo sobre el subset con abundancia comparable. Los targets faltantes se manejan con
  pérdida enmascarada, sin descartar parcelas ni imputar.
- **Beta y composición:** LCBD (Legendre & De Cáceres 2013) más 2–3 ejes NMDS como respuestas continuas
  por parcela.
- **Validación:** validación cruzada por bloques espaciales desde el primer experimento. El CV aleatorio
  se reporta solo como referencia optimista.

---

## Financiamiento

ANID FONDECYT Iniciación 11241088 · FSEQ210022 · Fundación Data Observatory.

## Licencia

Por definir.
