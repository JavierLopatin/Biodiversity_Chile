#!/usr/bin/env python3
# ---
# Source for `notebooks/03_explore_modelling_results.ipynb`.
#
# Cells are separated by `# %%` (jupytext "percent" format). This .py file is the version
# that gets reviewed and diffed — a .ipynb does not review. Regenerate the notebook with:
#
#     python notebooks/build_notebook.py notebooks/03_explore_modelling_results.py
#
# Edit the .py, never the .ipynb.
# ---

# %% [markdown]
# # Resultados del modelamiento
#
# Lectura de `results/models/summary.csv` y de las tablas de `results/tables/`. Nada se
# recalcula aquí: este cuaderno **lee** lo que produjeron los scripts 09–13 y lo ordena para
# mirarlo. Si un número no cuadra, el lugar donde arreglarlo es el script, no el cuaderno.
#
# Orden de lectura, que es el orden en que las conclusiones dependen unas de otras:
#
# 1. **Los controles.** Sin `B02` (sólo área) y `B03` (sólo coordenadas) al lado, ningún R²
#    de riqueza significa nada.
# 2. **La brecha de optimismo.** `R²(CV aleatoria) − R²(CV por dueño)`. Cuánto del puntaje
#    era saber dónde está la parcela en vez de qué crece en ella.
# 3. **Los tres contrastes:** curva contra LSP, convolución contra MLP, 2D contra 1D.
# 4. **Por índice de vegetación** y **por sustrato**.
# 5. **Atribución sobre el año fenológico** — la única salida que se lee como ecología.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from biodiv import targets as tg          # noqa: E402

MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
INTERP = ROOT / "results" / "interpretation"

PRIMARY, SPATIAL, OPTIMISTIC = "kfold5_owner", "kfold5_block20", "kfold5_random"
pd.set_option("display.width", 200, "display.max_columns", 60)

summary = pd.read_csv(MODELS / "summary.csv")
print(f"{summary['run_id'].nunique()} corridas, {summary['scheme'].nunique()} esquemas, "
      f"{len(summary):,} filas (run x scheme x target x semilla)")
summary.head()

# %% [markdown]
# ## 1. Los controles primero
#
# `B00` es el predictor de la media entrenada. Bajo CV aleatoria **debe** dar R² ≈ 0; si no,
# la partición o el escalado de targets están rotos y ningún otro número es confiable.
#
# `B02` usa sólo el área de parcela y el estrato de abundancia. Su R² es la parte de la
# riqueza atribuible al esfuerzo de muestreo, con cero teledetección.

# %%
def pooled(scheme=PRIMARY):
    """Media entre semillas del R² agrupado fuera de fold, una fila por corrida y target."""
    s = summary[summary["scheme"] == scheme]
    return (s.groupby(["run_id", "family", "target"])["R2"].mean()
            .unstack("target")
            .reindex(columns=tg.TARGETS_ALL))


ctrl = pooled(PRIMARY).loc[lambda d: d.index.get_level_values("family") == "BASE"]
print("Controles bajo kfold5_owner (CV agrupada por contribuyente):")
display(ctrl.round(3))

print("\nB00 bajo kfold5_random — debe ser ~0 en las nueve columnas:")
b00 = pooled(OPTIMISTIC)
display(b00[b00.index.get_level_values("run_id").str.startswith("B00")].round(3))

# %% [markdown]
# ## 2. La brecha de optimismo
#
# Es un resultado, no un diagnóstico: el riesgo R8 de `docs/02_innovation_and_impact.md`
# pide explícitamente reportar los dos números para poder citar la diferencia. Una brecha
# grande significa que el modelo aprendió *dónde* están las parcelas.

# %%
gap_path = TABLES / "optimism_gap.csv"
if gap_path.exists():
    gap = pd.read_csv(gap_path)
    g = gap.dropna(subset=["gap_vs_owner"]).sort_values("gap_vs_owner", ascending=False)
    display(g.head(15).round(3))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for t, sub in g.groupby("target"):
        ax.scatter(sub[PRIMARY], sub[OPTIMISTIC], s=22, alpha=0.75, label=t)
    lim = [min(-1.0, g[[PRIMARY, OPTIMISTIC]].min().min()), 1.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$R^2$ — CV agrupada por contribuyente")
    ax.set_ylabel("$R^2$ — CV aleatoria (optimista)")
    ax.set_title("Cuánto del puntaje es autocorrelación espacial y de protocolo")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    plt.show()
else:
    print("todavía no existe optimism_gap.csv — corre scripts/12_model_report.py")

# %% [markdown]
# ## 3. Los tres contrastes
#
# | Contraste | Corridas | Pregunta |
# |---|---|---|
# | curva vs LSP | `RF03` vs `RF01` | ¿aporta la curva sobre sus 18 resúmenes escalares? (gap G1) |
# | convolución vs MLP | `C1D01` vs `MLP02` | ¿aporta leer la curva como secuencia? |
# | 2D vs 1D | mejor `C2D*` vs `C1D01` | ¿aporta una segunda dimensión? |

# %%
# Alfa y beta se resumen por separado, y no es cosmética: bajo CV por contribuyente la
# fenología predice **composición** y no predice **diversidad alfa**. Un solo promedio sobre
# los seis targets completos mezcla un resultado real con uno nulo y no reporta ninguno.
ALPHA = ["hill_q0", "hill_q1", "hill_q2"]
BETA = ["lcbd_pa", "pcoa1_pa", "pcoa2_pa"]


def block_mean(df, cols):
    return df[[c for c in cols if c in df.columns]].mean(axis=1)


P = pooled(PRIMARY)
P = P.assign(R2_beta=block_mean(P, BETA), R2_alpha=block_mean(P, ALPHA),
             R2_main=block_mean(P, tg.TARGETS_MAIN)).sort_values("R2_beta", ascending=False)
print("Top 20 corridas por R² medio sobre los targets de composición (kfold5_owner):")
display(P.head(20).round(3))

# %%
d = P.head(28).iloc[::-1]
colors = {"BASE": "#999999", "RF": "#4c72b0", "MLP": "#dd8452",
          "C1D": "#55a868", "C2D": "#c44e52"}
cols = [colors.get(f, "#8172b3") for f in d.index.get_level_values("family")]
fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
for ax, key, title in zip(axes, ("R2_beta", "R2_alpha"),
                          ("composición y unicidad\n(lcbd_pa, pcoa1_pa, pcoa2_pa)",
                           "diversidad alfa\n(hill_q0, hill_q1, hill_q2)")):
    ax.barh(np.arange(len(d)), d[key], color=cols)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$R^2$ agrupado fuera de fold")
    ax.set_title(title, fontsize=9)
axes[0].set_yticks(np.arange(len(d)))
axes[0].set_yticklabels(d.index.get_level_values("run_id"), fontsize=7)
axes[1].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
               colors.keys(), fontsize=7, loc="lower right", frameon=False)
fig.suptitle("Comparación de modelos — kfold5_owner", fontsize=11)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Índice de vegetación y sustrato
#
# Dos preguntas distintas: cuál índice lleva más información, y si los índices se combinan o
# son redundantes. `stack5` y `curve5` usan los cinco a la vez — el primero como filas de una
# imagen, el segundo como canales de una señal 1D — así que el par aísla lo que gana el eje
# índice al ser una dimensión espacial.

# %%
s = summary[(summary["scheme"] == PRIMARY) & summary["target"].isin(tg.TARGETS_MAIN)]

by_index = (s[s["index"].notna() & (s["index"] != "")]
            .groupby(["family", "index"], as_index=False)["R2"].mean()
            .pivot(index="index", columns="family", values="R2"))
print("R² medio por índice de vegetación y familia:")
display(by_index.round(3))

by_sub = (s[s["substrate"].notna() & (s["substrate"] != "")]
          .groupby("substrate")["R2"].agg(R2="mean", sd="std", n="size")
          .reset_index()
          .sort_values("R2", ascending=False))
print("\nR² medio por sustrato:")
display(by_sub.round(3))

# %%
if len(by_sub) > 1:
    fig, ax = plt.subplots(figsize=(7, 4))
    manufactured = {"reshape", "serpentine", "hilbert", "gaf", "mtf", "ndi", "cwt",
                    "cos2d", "spectrogram"}
    d = by_sub.iloc[::-1]
    cols = ["#c44e52" if x in manufactured else "#4c72b0" for x in d["substrate"]]
    ax.barh(np.arange(len(d)), d["R2"], xerr=d["sd"].fillna(0), capsize=2, color=cols)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["substrate"], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$R^2$ medio sobre los seis targets completos")
    ax.set_title("Sustratos: rojo = segunda dimensión manufacturada, azul = real", fontsize=9)
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 5. Por target
#
# Los nueve targets no son igual de predecibles y no tienen por qué serlo. `hill_q0` está
# confundido con el área de parcela (ρ = 0,33); `lcbd_cover` y los ejes `*_cover` sólo
# existen para 546 parcelas; los ejes PCoA explican ~13 % de la varianza composicional, así
# que hay un techo bajo por construcción.

# %%
best = P.head(8).index.get_level_values("run_id")   # ordenadas por R2_beta
d = summary[(summary["scheme"] == PRIMARY) & summary["run_id"].isin(best)]
piv = d.groupby(["run_id", "target"])["R2"].mean().unstack().reindex(columns=tg.TARGETS_ALL)

fig, ax = plt.subplots(figsize=(9, 4))
w = 0.8 / len(piv)
for i, (r, row) in enumerate(piv.iterrows()):
    ax.bar(np.arange(len(row)) + i * w, row.to_numpy(), width=w, label=r)
ax.set_xticks(np.arange(len(tg.TARGETS_ALL)) + 0.4)
ax.set_xticklabels(tg.TARGETS_ALL, rotation=30, ha="right", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("$R^2$ agrupado fuera de fold")
ax.legend(fontsize=6, ncol=2, frameon=False)
ax.set_title("Rendimiento por target de las ocho mejores corridas", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Observado contra predicho
#
# El R² resume; el diagrama muestra *cómo* falla. Buscar: compresión hacia la media (el
# modelo predice la media y poco más), sesgo por fold (desplazamiento entre contribuyentes) y
# parcelas de riqueza alta sistemáticamente subestimadas.

# %%
top_run = P.index.get_level_values("run_id")[0]
oof = pd.read_csv(MODELS / top_run / PRIMARY / "oof_predictions.csv")
oof = oof[oof["seed"] == oof["seed"].min()]

show = ["hill_q0", "hill_q1", "lcbd_pa", "pcoa1_pa", "pcoa2_pa", "lcbd_cover"]
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, t in zip(axes.ravel(), show):
    if f"{t}_pred" not in oof:
        ax.axis("off"); continue
    x, y = oof[f"{t}_obs"], oof[f"{t}_pred"]
    ok = x.notna() & y.notna()
    ax.scatter(x[ok], y[ok], s=8, alpha=0.35, c=oof.loc[ok, "fold"], cmap="viridis")
    lim = [min(x[ok].min(), y[ok].min()), max(x[ok].max(), y[ok].max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    r2 = 1 - ((y[ok] - x[ok]) ** 2).sum() / ((x[ok] - x[ok].mean()) ** 2).sum()
    ax.set_title(f"{t}   $R^2$={r2:.3f}", fontsize=9)
    ax.set_xlabel("observado", fontsize=7); ax.set_ylabel("predicho", fontsize=7)
    ax.tick_params(labelsize=6)
fig.suptitle(f"{top_run} — {PRIMARY}, color = fold", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Atribución sobre el año fenológico
#
# La salida que se lee como ecología en vez de como métrica: qué semanas del año mira el
# modelo para cada faceta de diversidad. Producida por `scripts/13_interpretability.py` con
# Integrated Gradients, replegada al eje de 52 semanas por el mapa inverso propio de cada
# sustrato, y etiquetada con el DOY real desde `phenoshape_doy_grid.parquet`.

# %%
attr_files = sorted(INTERP.glob("doy_attribution_*.csv")) if INTERP.exists() else []
if attr_files:
    attr = pd.read_csv(attr_files[0])
    ts = [t for t in tg.TARGETS_ALL if t in set(attr["target"])]
    fig, axes = plt.subplots(3, 3, figsize=(11, 7), sharex=True)
    for ax, t in zip(axes.ravel(), ts):
        g = attr[attr["target"] == t].sort_values("step")
        ax.fill_between(g["step"], 0, g["attribution"], alpha=0.3, color="#c44e52")
        ax.plot(g["step"], g["attribution"], lw=1.2, color="#c44e52")
        ax.set_title(t, fontsize=8); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(ts):]:
        ax.axis("off")
    doy = attr.groupby("step")["doy"].first()
    for ax in axes[-1]:
        ax.set_xticks(np.arange(0, 52, 8))
        ax.set_xticklabels([int(doy.get(s, 0)) for s in np.arange(0, 52, 8)], fontsize=6)
        ax.set_xlabel("día del año", fontsize=7)
    fig.suptitle(f"Integrated gradients — {attr_files[0].stem.replace('doy_attribution_', '')}",
                 fontsize=10)
    plt.tight_layout(); plt.show()

    print("\nSemana de máxima atribución por target:")
    display(attr.loc[attr.groupby("target")["attribution"].idxmax()]
            [["target", "step", "doy", "attribution"]].round(4))
else:
    print("todavía no hay atribuciones — corre scripts/13_interpretability.py --auto")

# %%
imp = sorted(INTERP.glob("rf_block_importance_*.csv")) if INTERP.exists() else []
if imp:
    b = pd.read_csv(imp[0]).set_index("target")
    print("Importancia por bloque de features (Random Forest, permutación sobre OOF):")
    display(b.round(3))
    b.plot(kind="barh", stacked=True, figsize=(7, 4),
           title="De dónde viene la señal, por bloque de predictores")
    plt.xlabel("participación de la importancia"); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Significancia
#
# Con 9 targets y 3–5 semillas, diferencias de 0,02 en R² son ruido. Cualquier afirmación de
# que un modelo gana necesita esta tabla: t pareado y Wilcoxon sobre los puntajes
# (target × semilla), con corrección de Bonferroni dentro de la familia de comparación.

# %%
tests_path = TABLES / "paired_tests.csv"
if tests_path.exists():
    tests = pd.read_csv(tests_path)
    sig = tests[tests["significant"]].sort_values("delta", ascending=False)
    print(f"{len(sig)} de {len(tests)} pares significativos tras Bonferroni")
    display(sig.head(20).round(4))
else:
    print("todavía no existe paired_tests.csv — corre scripts/12_model_report.py")

# %% [markdown]
# ## 9. Píxel central contra media 5×5
#
# `results/px_ablation/` es una corrida aparte que repite RF01/RF03/RF04 y la CNN 1D con todo
# el diseño movido al píxel central, contra el mismo diseño sobre la ventana 5×5. El
# emparejamiento es exacto: mismo modelo, índice, folds y semilla, difieren solo en el
# footprint. Tres cosas tiran en direcciones opuestas — calce con el tamaño de parcela
# (78,5–10.000 m² contra 900 m²/píxel), ruido de reconstrucción por píxel, y datos faltantes
# (200 celdas NaN contra 2) — por eso se mide en vez de decidirse.

# %%
px_path = TABLES / "pixel_ablation.csv"
if px_path.exists():
    px = pd.read_csv(px_path)
    for block in ("beta", "alpha"):
        d = px[px.block == block].sort_values("delta_center_minus_5x5")
        print(f"\n=== {block} ===  (delta > 0 favorece al píxel central)")
        display(d[["stem", "mean5x5", "center", "delta_center_minus_5x5"]].round(3))
        print(f"  delta medio = {d.delta_center_minus_5x5.mean():+.4f}   "
              f"el central gana en {int((d.delta_center_minus_5x5 > 0).sum())}/{len(d)}")

    d = px[px.block == "beta"].sort_values("mean5x5")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(d))))
    ax.barh(y - 0.19, d["mean5x5"], height=0.36, label="media 5×5", color="#4c72b0")
    ax.barh(y + 0.19, d["center"], height=0.36, label="píxel central", color="#dd8452")
    ax.set_yticks(y); ax.set_yticklabels(d["stem"], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("$R^2$ agrupado fuera de fold — targets de composición")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout(); plt.show()
else:
    print("aún no existe pixel_ablation.csv — corre scripts/15_pixel_ablation.py --run")

# %% [markdown]
# ## 10. Diversidad filogenética y curvas de acumulación
#
# La faceta que `scripts/07_compute_taxonomic_beta_responses.R` dejó pendiente por no haber
# "ninguna filogenia referenciada en este repo". `scripts/25_compute_phylo_responses.R` la
# cierra con el megaárbol `GBOTB.extended.TPL` de V.PhyloMaker2 (74.529 tips, backbone GBOTB
# de Smith & Brown 2018 extendido con Zanne et al. 2014).
#
# Las curvas replican la **Fig. 4b,c del paper de Parcelas-CL** con el método que esa figura
# usa —rarefacción/extrapolación de iNEXT (Chao et al.)— y añaden la comparación que ahí no
# existe: el subset de 1.082 parcelas de este proyecto contra las 1.485 del dataset completo.
# La pregunta es si filtrar por Chile central, año ≥ 1999 y coordenada única costó
# representatividad, y si la costó en la dimensión taxonómica, en la filogenética o en las dos.
#
# **El n del paper.** Parcelas-CL declara 675 especies, pero eso cuenta 54 registros
# determinados sólo a género y 7 sólo a familia. A rango de especie hay 597 taxones, y
# colapsando subespecies y variedades al binomio quedan **601**, que es lo que un árbol puede
# representar: darle un tip a un registro de género inventaría un linaje que nadie observó.
#
# **La escala del panel filogenético.** El eje del paper llega a ~60, no a las decenas de miles
# de una PD sumada en millones de años: es `meanPD`, la PD dividida por la profundidad del
# árbol (390,7 Ma aquí), que se lee como número efectivo de linajes.
#
# **La banda.** No es la de iNEXT. Es la dispersión al submuestrear parcelas sin reemplazo, que
# está centrada en la curva por construcción, y cubre sólo el tramo interpolado. Los dos
# alternativos fallan de forma medible y están documentados en `scripts/lib/pd_inext.R`. Para
# el tramo extrapolado no hay análogo por submuestreo, así que va sin banda: su incertidumbre
# vive justo en las especies que no se han visto.
#
# de aquí es propio y queda documentado; si Cerda-Paredes comparte el suyo, es preferible.

# %%
CURVES = ROOT / "data" / "derived" / "rarefaction_inext.csv"
ASYMPT = ROOT / "data" / "derived" / "rarefaction_asymptote.csv"
PHYLO = ROOT / "data" / "derived" / "phylo_responses.parquet"

if CURVES.exists():
    cur = pd.read_csv(CURVES)
    colors = {"Parcelas-CL completo": "#2f6f7f", "subset del proyecto": "#c1553b"}
    panels = [("taxonomica", "Riqueza taxonómica"),
              ("filogenetica_meanPD", "Riqueza filogenética (meanPD)")]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)
    for ax, (metric, ylab) in zip(axes, panels):
        for d, g in cur[cur["metric"] == metric].groupby("dataset"):
            g = g.sort_values("n")
            obs = g[g["method"] == "Observed"]
            interp = g[g["method"] != "Extrapolation"]
            # el punto observado pertenece a los dos tramos, si no la línea se corta
            extrap = pd.concat([obs, g[g["method"] == "Extrapolation"]])
            ax.plot(interp["n"], interp["value"], lw=1.8, color=colors[d], label=d)
            ax.plot(extrap["n"], extrap["value"], lw=1.8, ls="--", color=colors[d])
            ax.plot(obs["n"], obs["value"], "o", ms=6, color=colors[d])
            b = g[g["lo"].notna()]
            ax.fill_between(b["n"], b["lo"], b["hi"], alpha=0.18, color=colors[d], lw=0)
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[1].set_xlabel("Unidades de muestreo (parcelas)")
    fig.suptitle("Rarefacción y extrapolación — réplica de la Fig. 4b,c de Parcelas-CL\n"
                 "sólido = observado, punteado = extrapolado a 2n", fontsize=10)
    plt.tight_layout(); plt.show()

    print("Cuánto queda por descubrir, según el propio muestreo:\n")
    for metric, lab in panels:
        for d, g in cur[cur["metric"] == metric].groupby("dataset"):
            g = g.sort_values("n")
            o = g[g["method"] == "Observed"].iloc[0]; e = g.iloc[-1]
            print(f"  {lab:32s} {d:22s} n={o['n']:>4.0f}: {o['value']:8.1f}"
                  f"  ->  2n={e['n']:>4.0f}: {e['value']:8.1f}  (+{100*(e['value']/o['value']-1):.1f} %)")

    if ASYMPT.exists():
        asy = pd.read_csv(ASYMPT)
        print("\nAsíntota de riqueza taxonómica (Chao2) — el total que el muestreo implica:\n")
        for _, a in asy.iterrows():
            print(f"  {a['Assemblage']:22s} observado {a['TD_obs']:5.1f}  ->  "
                  f"asíntota {a['TD_asy']:6.1f} ± {a['s.e.']:.1f}   "
                  f"(sin ver el {100*(1-a['TD_obs']/a['TD_asy']):.0f} % de la flora)")

    # Lo que decide si el filtrado costó representatividad: a igual número de parcelas,
    # ¿las dos curvas coinciden? Si el subset queda por debajo, el filtro no fue neutral.
    # Los dos conjuntos tienen rejillas de n distintas (cada uno se evalúa en sus propios
    # nudos), así que cruzarlos por valor exacto no encuentra nada: hay que interpolar.
    print("\nA igual esfuerzo (tramo interpolado, sobre rejilla común):\n")
    for metric, lab in panels:
        g = cur[(cur["metric"] == metric) & (cur["method"] != "Extrapolation")]
        a = g[g["dataset"] == "Parcelas-CL completo"].sort_values("n")
        b = g[g["dataset"] == "subset del proyecto"].sort_values("n")
        grid = np.linspace(20, min(a["n"].max(), b["n"].max()), 200)
        ratio = (np.interp(grid, b["n"], b["value"])
                 / np.interp(grid, a["n"], a["value"]))
        print(f"  {lab:32s} el subset retiene el {100*ratio.mean():.1f} % "
              f"(de {100*ratio[0]:.1f} % con 20 parcelas a {100*ratio[-1]:.1f} % "
              f"con {grid[-1]:.0f})")
else:
    print("aún no existen las curvas — corre scripts/26_rarefaction_inext.R")

# %% [markdown]
# ### 10.1 Qué facetas filogenéticas sirven como target
#
# `PD` de Faith correlaciona +0,95 con la riqueza: es riqueza reetiquetada, y la riqueza es
# justo lo que no se puede predecir bajo CV por contribuyente (§7.1 de `docs/08_modelling.md`).
# Las que aportan información nueva son **MPD** (casi ortogonal a la riqueza) y **MNTD**, más
# los **SES**, que son los que separan "hay muchas especies" de "hay muchos linajes distintos".

# %%
if PHYLO.exists():
    ph = pd.read_parquet(PHYLO)
    print(f"{len(ph)} parcelas, {ph.pd_faith.isna().sum()} sin cobertura filogenética\n")
    metrics = ["pd_faith", "mpd", "mntd", "ses_pd", "ses_mpd", "ses_mntd"]
    display(ph[metrics + ["n_sp_tree"]].describe().round(3))

    # el criterio de seleccion de targets, en una tabla
    rows = []
    resp_tax = pd.read_parquet(ROOT / "data/derived/biodiversity_responses.parquet")
    m = ph.merge(resp_tax, on="PlotObservationID", how="left")
    from scipy import stats as st
    for v in metrics:
        ok = m[v].notna()
        rows.append(dict(
            metrica=v,
            r_riqueza=st.spearmanr(m.loc[ok, v], m.loc[ok, "n_sp_tree"]).statistic,
            r_hill_q0=st.spearmanr(m.loc[ok, v], m.loc[ok, "hill_q0"]).statistic,
            r_pcoa1=st.spearmanr(m.loc[ok, v], m.loc[ok, "pcoa1_pa"]).statistic,
            r_lcbd=st.spearmanr(m.loc[ok, v], m.loc[ok, "lcbd_pa"]).statistic))
    tab = pd.DataFrame(rows).round(3)
    print("Spearman contra riqueza y contra los targets taxonómicos ya existentes.")
    print("Una métrica con |r| alto contra hill_q0 no aporta un target nuevo:\n")
    display(tab)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, v in zip(axes.ravel(), metrics):
        ax.scatter(m["n_sp_tree"], m[v], s=7, alpha=0.3, c="#4c72b0")
        ok = m[v].notna() & m["n_sp_tree"].notna()
        r = st.spearmanr(m.loc[ok, "n_sp_tree"], m.loc[ok, v]).statistic
        ax.set_title(f"{v}   ρ={r:+.3f}", fontsize=9)
        ax.set_xlabel("especies en el árbol", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Cada métrica filogenética contra la riqueza — cuanto más plana, "
                 "más información nueva aporta", fontsize=10)
    plt.tight_layout(); plt.show()
else:
    print("aún no existen las respuestas — corre scripts/25_compute_phylo_responses.R")

# %% [markdown]
# ## 11. Dark diversity
#
# Las especies que **podrían** estar en una parcela y no están (Pärtel, Szava-Kovats & Zobel
# 2011), estimadas por co-ocurrencia con `DarkDiv` (Carmona & Pärtel 2021), método
# hipergeométrico. `scripts/27_compute_dark_diversity.R`.
#
# Tres decisiones que cambian el resultado y están medidas en `docs/12_phylo_and_rarefaction.md`:
#
# - **El pool se estima con las 1.485 parcelas**, no con las 1.082 del subset. Con el pool
#   reducido la dependencia del área de parcela sube de ρ = +0,085 a **+0,315**.
# - **Se cuenta con umbral (p > 0,9), no sumando probabilidades.** Sumarlas da 253 especies
#   oscuras de mediana para parcelas con 5 observadas — es sumar 596 números pequeños, no un
#   resultado ecológico.
# - **La completitud `log(obs/oscuras)` no sirve como target**: ρ = +0,988 con la riqueza.
#   Es riqueza reetiquetada, igual que la PD de Faith, y por la misma razón estructural.
#
# Lo que sale de aquí es **`dark_n`**, y es la faceta con menos confundido de contribuyente
# de todo el proyecto: R² por `Owner` de 0,22, contra 0,70 de la log-riqueza. Eso importa
# porque es justo el confundido que hunde la α bajo `kfold5_window`.

# %%
DARK = ROOT / "data" / "derived" / "dark_diversity.parquet"
DARKSP = ROOT / "data" / "derived" / "dark_diversity_spatial.csv"

if DARK.exists():
    dk = pd.read_parquet(DARK)
    tax = pd.read_parquet(ROOT / "data/derived/biodiversity_responses.parquet")
    md = dk.merge(tax, on="PlotObservationID", how="left")
    print(f"{len(dk)} parcelas, {dk.dark_n.isna().sum()} sin cobertura")
    print(f"especies oscuras: mediana {dk.dark_n.median():.0f}, "
          f"rango {dk.dark_n.min():.0f}-{dk.dark_n.max():.0f}   "
          f"(observadas: mediana {dk.n_obs.median():.0f})\n")

    from scipy import stats as st
    rows = []
    for v in ["dark_n", "pool_n", "dark_pd", "dark_mpd", "dark_prob", "completeness"]:
        ok = md[v].notna()
        rows.append(dict(
            metrica=v,
            r_riqueza=st.spearmanr(md.loc[ok, v], md.loc[ok, "n_obs"]).statistic,
            r_hill_q0=st.spearmanr(md.loc[ok, v], md.loc[ok, "hill_q0"]).statistic,
            r_pcoa1=st.spearmanr(md.loc[ok, v], md.loc[ok, "pcoa1_pa"]).statistic,
            r_lcbd=st.spearmanr(md.loc[ok, v], md.loc[ok, "lcbd_pa"]).statistic))
    print("Criterio de selección: |r| bajo contra la riqueza y contra los targets que ya "
          "existen.\nUna métrica con r alto contra hill_q0 no aporta un target nuevo:\n")
    display(pd.DataFrame(rows).round(3))

    if DARKSP.exists():
        sp = pd.read_csv(DARKSP)
        print("\nValidación espacial — ¿el pool es local, o una lista nacional?")
        print("El número absoluto no dice nada sin la línea base (una ausente cualquiera):\n")
        for _, r in sp.iterrows():
            print(f"  {r['km']:>3.0f} km: de las oscuras se ha visto cerca el "
                  f"{100*r['oscuras']:4.1f} %, de una ausente cualquiera el "
                  f"{100*r['base']:4.1f} %  ->  {r['razon']:.1f}×")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (v, lab) in zip(axes, [
            ("dark_n", "especies oscuras (p > 0,9)"),
            ("completeness", "completitud  log(obs/oscuras)"),
            ("near_frac", "fracción de oscuras vista a < 50 km")]):
        ok = md[v].notna()
        ax.scatter(md.loc[ok, "n_obs"], md.loc[ok, v], s=7, alpha=0.25, c="#2f6f7f")
        r = st.spearmanr(md.loc[ok, "n_obs"], md.loc[ok, v]).statistic
        ax.set_title(f"{lab}\nρ = {r:+.3f}", fontsize=9)
        ax.set_xlabel("especies observadas", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("El panel del medio es el resultado negativo: la completitud es la riqueza "
                 "otra vez", fontsize=10)
    plt.tight_layout(); plt.show()
else:
    print("aún no existe — corre scripts/27_compute_dark_diversity.R")
