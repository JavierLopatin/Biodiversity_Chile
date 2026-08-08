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
