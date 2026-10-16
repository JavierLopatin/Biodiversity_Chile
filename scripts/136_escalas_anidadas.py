#!/usr/bin/env python3
"""El margen sobre el nulo geografico a tres extensiones anidadas.

Si lo que el sensor no ve es la posicion de la parcela en un pool REGIONAL, entonces achicar la
region deberia cambiar el balance: el modelo nulo geografico pierde el gradiente que explota, y
el margen del sensor deberia crecer. Si no crece, el limite no es el rango del gradiente sino
el sensor, y la afirmacion central del paper se vuelve mas fuerte.

TRES VENTANAS ANIDADAS, TODAS DE LIVING TREES. Parcelas-CL no entra: es un compendio de
muestreos distintos -- estratos de cobertura, de conteo y de area basal, mas 60 parcelas de
flora completa-- y mezclarla haria variar el protocolo junto con la extension, que es
exactamente el confundido que este diseno existe para evitar. Living Trees tiene un solo
diseno de principio a fin y cubre 30,6 a 55 S por si sola.

    nacional   30,6-55 S   2.020 parcelas   564 bloques de 20 km   24,3 grados
    regional   36-44 S     1.246            281                     8,0
    local      39-41 S       488             88                     2,0

DOS CONTROLES, Y UNA CONSECUENCIA QUE NO SE PUEDE EVITAR

1. Mismo n. Las tres se submuestrean al mismo numero de parcelas, con varias extracciones
   independientes, para que "menos gradiente" no se confunda con "menos datos".
2. Mismo target. Las facetas conservan su valor NACIONAL; no se recalculan por ventana. Esto
   aisla el lado de los predictores. Recalcularlas -- que cambiaria el significado de LCBD, los
   ejes de ordenacion y la diversidad oscura, todos definidos respecto a un pool-- es el otro
   experimento, y su diferencia con este mide el efecto de la definicion del pool.

Lo que no se puede fijar: con n constante, achicar la extension AUMENTA la densidad de
parcelas. Fijar n y densidad a la vez equivale a fijar el area. Asi que la lectura es
"extension, con la densidad que le corresponde", que es lo que se entiende por escala.

METRICA. R2 agrupado, no el centrado dentro de bandas de 2 grados que usan las tablas del
paper: la ventana local mide 2 grados, o sea UNA banda, y el centrado seria degenerado. Eso no
es un parche sino el experimento mismo -- el centrado existe para quitar el gradiente, y aqui
la variable manipulada ES cuanto gradiente hay.

Salida: `results/tables/escalas_anidadas.csv`.

Uso:
    python scripts/136_escalas_anidadas.py [--n 400] [--draws 10] [--jobs 4]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = ROOT / "results" / "tables" / "escalas_anidadas.csv"
sys.path.insert(0, str(ROOT / "src"))
from biodiv import facetas_paper as FP          # noqa: E402

VENTANAS = {"nacional": (-56.0, -29.0), "regional": (-44.0, -36.0), "local": (-41.0, -39.0)}
NULO = ["lon", "lat", "elevation"]
FUENTE = {
    **{f: "unified_diversity_responses_woody.parquet" for f in
       ["hill_q0_unified", "lcbd_pa_unified", "lcbd_freq_unified",
        *[f"isomap{i}_{v}_unified" for i in (1, 2, 3) for v in ("pa", "freq")]]},
    "lcbd_count_sorensen": "lcbd_count_sorensen_unified_padded_woody.parquet",
    **{f"td_inext_q{q}": "td_inext_coverage_unified_padded_woody.parquet" for q in (0, 1, 2)},
    **{f"pd_inext_q{q}": "pd_inext_coverage_unified_padded_woody.parquet" for q in (0, 1, 2)},
    **{f: "unified_phylo_responses_woody.parquet" for f in
       ["mpd_unified", "mntd_unified", "ses_pd_unified"]},
    "dark_n_unified": "unified_dark_diversity_woody.parquet",
}


def datos(inventario: str = "living_trees") -> tuple[pd.DataFrame, list[str]]:
    gm = pd.read_parquet(DERIVED / "gm_center_nc.parquet")
    cl = pd.read_parquet(DERIVED / "climate_unified.parquet")
    if cl.index.name:
        cl = cl.reset_index()
    tp = (pd.read_parquet(DERIVED / "topography_unified.parquet")
          .rename(columns={"plot_id": "PlotObservationID"}))
    pl = pd.read_parquet(DERIVED / "plots_unified.parquet")[
        ["PlotObservationID", "lon", "lat", "X_m", "Y_m", "source"]]
    d = (gm.merge(cl, on="PlotObservationID").merge(tp, on="PlotObservationID")
         .merge(pl, on="PlotObservationID"))
    for f, src in FUENTE.items():
        t = pd.read_parquet(DERIVED / src)
        if f in t.columns:
            d = d.merge(t[["PlotObservationID", f]], on="PlotObservationID", how="left")
    # lon/lat FUERA del diseno: son el nulo. Si entraran, el diseno contendria al nulo y el
    # margen seria no-negativo por construccion. El area de parcela tampoco entra -- Living
    # Trees usa un solo tamano, asi que el bloque `ctr-area` del paper aqui es constante.
    reservadas = {"PlotObservationID", "X_m", "Y_m", "source", "lon", "lat", *FUENTE}
    pred = [c for c in d.columns if c not in reservadas
            and pd.api.types.is_numeric_dtype(d[c])]
    # Las columnas de clima se identifican por su archivo y no por un prefijo: el diseno de
    # reflectancia sola del paper (`RFG4c_gm-topo`) lleva topografia, asi que quitar "todo lo
    # que no empieza con gm_" se llevaria tambien la topografia y seria otro modelo.
    clima = [c for c in cl.columns if c in pred]
    # El inventario es un PARAMETRO y no una constante porque la diferencia entre los dos
    # es el experimento: Parcelas-CL tiene su vecino mas cercano a 0,15 km y Living Trees a
    # 3,07 km, y esa distancia --- no la extension --- es lo que fija cuanto infla el
    # protocolo aleatorio. Correr el MISMO codigo, las mismas variables y la misma ventana
    # cambiando solo la fuente aisla esa variable de todo lo demas.
    sel = d.source.unique() if inventario == "ambos" else [inventario]
    out = d[d.source.isin(sel)]
    assert len(out), f"el inventario {inventario!r} no dejo parcelas; fuentes: {sel}"
    return out.reset_index(drop=True), pred, clima


def r2(o, p) -> float:
    o, p = np.asarray(o, float), np.asarray(p, float)
    return float(1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def oof(X: np.ndarray, y: np.ndarray, bloque: np.ndarray, rng, jobs: int,
        folds: str = "bloques") -> np.ndarray:
    """Prediccion fuera de muestra, con los folds armados de una de dos maneras.

    `bloques` asigna BLOQUES de 20 km enteros, que es el protocolo del paper. `aleatorio`
    asigna PARCELAS sueltas, que es el protocolo que el campo usa por defecto y el que la
    Fig. 2 mide. Todo lo demas --las parcelas sorteadas, la semilla, el modelo-- es identico
    entre los dos, asi que la diferencia de $R^2$ es atribuible al protocolo y a nada mas.
    """
    if folds == "aleatorio":
        bloque_fold = rng.integers(0, 5, len(y))
    else:
        bl = pd.unique(bloque)
        fold = pd.Series(rng.integers(0, 5, len(bl)), index=bl)
        bloque_fold = fold.reindex(bloque).to_numpy()
    p = np.full(len(y), np.nan)
    for k in range(5):
        te = bloque_fold == k
        if te.sum() < 5 or (~te).sum() < 50:
            continue
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.33,
                                  n_jobs=jobs, random_state=0).fit(X[~te], y[~te])
        p[te] = m.predict(X[te])
    return p


def generador(faceta: str, ventana: str, draw: int) -> np.random.Generator:
    """Un generador por celda, sembrado por su propia clave.

    Antes habia UN generador para toda la corrida, consumido en el orden del bucle de facetas.
    Eso hace que los sorteos dependan de CUANTAS facetas tiene la lista: al pasar de veinte a
    dieciseis variables, el flujo se corre y la misma celda recibe otra submuestra y otros
    pliegues. Medido: el R2 de la linea base --el mismo modelo sobre las mismas parcelas--
    difirio hasta 0,17 entre dos corridas que debian compartirlo, en 90 de 480 celdas.

    Con la semilla derivada de (faceta, ventana, sorteo) dos corridas cualesquiera comparten
    submuestra y pliegues celda por celda, aunque cambie la lista de variables o el orden. El
    digest es blake2b y no `hash()` porque `hash()` lleva sal por proceso y no es reproducible
    entre corridas.
    """
    h = hashlib.blake2b(f"{faceta}|{ventana}|{draw}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "big"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400,
                    help="parcelas por extraccion, igual en las tres; 0 = TODAS las de cada "
                         "ventana, sin submuestrear y sin n comun")
    ap.add_argument("--draws", type=int, default=10)
    #: 4 era un default de maquina chica: en la Mac de 14 nucleos dejaba 10 ociosos y la
    #: corrida sobre el pool completo tardaba el doble de lo necesario. `0` toma los nucleos
    #: de rendimiento menos uno, para no dejar la maquina sin aire.
    ap.add_argument("--jobs", type=int, default=0,
                    help="hilos del random forest; 0 = nucleos disponibles menos uno")
    #: `geo` es la linea base de ubicacion; `clima` es el diseno sin el bloque de reflectancia.
    ap.add_argument("--referencia", choices=("geo", "clima"), default="geo")
    #: `completo` es reflectancia + clima + topografia (RFG5c); `reflectancia` es el bloque de
    #: teledeteccion con topografia y sin clima (RFG4c), que es el otro diseno de la figura 3.
    ap.add_argument("--diseno", choices=("completo", "reflectancia"), default="completo")
    #: El protocolo de validacion. `bloques` es el del paper; `aleatorio` existe para medir
    #: cuanto infla el protocolo DENTRO DE CADA VENTANA, que es lo que falta para llevar la
    #: Fig. 2 a las cuatro extensiones: hoy el salto solo esta medido a extension nacional y
    #: global, y las dos ventanas chilenas cortas solo tienen la corrida por bloques.
    ap.add_argument("--folds", choices=("bloques", "aleatorio"), default="bloques")
    #: Que red de parcelas. El default es el del paper; `parcelas_cl` existe para el control
    #: que separa la geometria del muestreo de la extension (ver `datos`). Las ventanas que
    #: el inventario elegido no cubre se descartan declarandolo, no en silencio.
    ap.add_argument("--inventario", choices=("living_trees", "parcelas_cl", "ambos"),
                    default="living_trees")
    ap.add_argument("--min-parcelas", type=int, default=120,
                    help="una ventana con menos parcelas que esto se descarta: con pocas "
                         "decenas el R2 fuera de fold es ruido, no una extension mas chica")
    ap.add_argument("--salida", type=Path, default=None)
    a = ap.parse_args()
    if a.jobs == 0:
        a.jobs = max(1, (os.cpu_count() or 2) - 1)

    d, pred, clima = datos(a.inventario)
    #: El conjunto COMPLETO se guarda aparte: la mascara de parcelas utilizables se calcula
    #: siempre sobre el, nunca sobre el diseno en curso. Si se calculara sobre el diseno, el de
    #: reflectancia sola --que tiene menos columnas y por lo tanto menos NaN-- sobreviviria en
    #: mas parcelas, los sorteos dejarian de coincidir y las dos corridas ya no compartirian ni
    #: las parcelas ni la linea base. Medido al pisarlo: el R2 del nulo llegaba a diferir 0,17
    #: entre corridas, que es mas que cualquier margen que la figura quiera mostrar.
    pred_todo = list(pred)
    if a.diseno == "reflectancia":
        pred = [c for c in pred if c not in set(clima)]
        assert any(c.startswith("gm_") for c in pred), "el diseno se quedo sin reflectancia"
        assert not set(pred) & set(clima), "quedo clima dentro del diseno de reflectancia sola"
    d["bloque"] = (np.floor(d.X_m / 20000).astype(int).astype(str) + "_" +
                   np.floor(d.Y_m / 20000).astype(int).astype(str))
    if a.referencia == "clima" and a.diseno == "reflectancia":
        raise SystemExit("reflectancia sola contra clima no tiene sentido: el diseno no "
                         "contiene al nulo y la resta mezcla dos preguntas")
    ref = NULO if a.referencia == "geo" else [c for c in pred if not c.startswith("gm_")]
    Xf, Xn = d[pred].to_numpy(float), d[ref].to_numpy(float)
    finito = (np.isfinite(d[pred_todo].to_numpy(float)).all(1)
              & np.isfinite(Xn).all(1))
    hay = [f for f in FP.ORDEN if f in d.columns]
    print(f"{a.inventario}: {len(d)} parcelas, {len(pred)} predictores, {len(hay)} facetas; "
          f"{finito.sum()} con predictores completos")
    filas = []

    disp = {}          # (faceta, ventana) -> indices utilizables
    for vent, (lo, hi) in VENTANAS.items():
        vmask = (d.lat >= lo).to_numpy() & (d.lat < hi).to_numpy()
        print(f"  {vent}: {vmask.sum()} parcelas, {d.bloque[vmask].nunique()} bloques, "
              f"{d.lat[vmask].max() - d.lat[vmask].min():.1f} grados")
        for fac in hay:
            disp[(fac, vent)] = np.flatnonzero(vmask & finito & np.isfinite(d[fac]).to_numpy())

    # Una ventana que el inventario elegido no cubre no es una extension mas chica: es una
    # ventana vacia. Parcelas-CL llega de -37,9 a -30,3, asi que la regional le queda a
    # medias y la local fuera. Se descartan aqui y se declara, antes de que el n comun las
    # arrastre a todas a cero.
    ventanas = {v: VENTANAS[v] for v in VENTANAS
                if max(len(disp[(f, v)]) for f in hay) >= a.min_parcelas}
    fuera = [v for v in VENTANAS if v not in ventanas]
    if fuera:
        print(f"  ventanas sin cobertura en {a.inventario}, fuera: {', '.join(fuera)}")
    assert ventanas, f"{a.inventario} no cubre ninguna ventana con >= {a.min_parcelas} parcelas"

    for fac in hay:
        # El n comun lo fija la ventana MAS POBRE, por faceta: las facetas estandarizadas por
        # cobertura solo existen donde inext converge, y esa cobertura no es uniforme en
        # latitud, asi que un n global dejaria variar el tamano de muestra junto con la
        # extension -- el confundido que este diseno existe para cerrar.
        # `--n 0` renuncia al n comun a proposito. El submuestreo a 400 parcelas existe para
        # que solo cambie la extension y no el tamano de muestra, y para medir MARGENES eso es
        # lo correcto. Pero para medir el OPTIMISMO DEL PROTOCOLO desactiva el protocolo que
        # se quiere medir: con 400 parcelas en ~250 bloques de 20 km quedan 1,6 parcelas por
        # bloque, asi que asignar bloques enteros a folds es casi asignar parcelas sueltas y
        # el salto se desploma a +0,01 cuando sobre el pool completo es +0,16. Con `--n 0`
        # cada ventana usa todas sus parcelas, el n VARIA entre ventanas y hay que declararlo.
        if a.n == 0:
            n_fac = None
        else:
            n_fac = min(a.n, min(len(disp[(fac, v)]) for v in ventanas))
            if n_fac < a.min_parcelas:
                print(f"{fac}: n comun {n_fac} < {a.min_parcelas}, fuera")
                continue
        cuota = "/".join(f"{v}={len(disp[(fac, v)])}" for v in ventanas)
        print(f"\n{fac}: n {'por ventana (sin submuestreo)' if n_fac is None else n_fac} "
              f"(disponibles {cuota})")
        for vent in ventanas:
            idx = disp[(fac, vent)]
            n_v = len(idx) if n_fac is None else n_fac
            if n_v < a.min_parcelas:
                print(f"  {vent}: {n_v} parcelas < {a.min_parcelas}, fuera")
                continue
            for draw in range(a.draws):
                g = generador(fac, vent, draw)
                sel = g.choice(idx, n_v, replace=False)
                y, bl = d[fac].to_numpy(float)[sel], d.bloque.to_numpy()[sel]
                # El MISMO generador para los dos modelos no sirve: el segundo heredaria el
                # estado que dejo el primero y los pliegues serian distintos entre diseno y
                # linea base. Cada uno arranca de su propia semilla derivada.
                pf = oof(Xf[sel], y, bl, generador(fac, vent, draw), a.jobs, a.folds)
                pn = oof(Xn[sel], y, bl, generador(fac, vent, draw), a.jobs, a.folds)
                m = np.isfinite(pf) & np.isfinite(pn)
                if m.sum() < 100:
                    continue
                rf, rn = r2(y[m], pf[m]), r2(y[m], pn[m])
                filas.append(dict(faceta=fac, ventana=vent, diseno=a.diseno,
                                  inventario=a.inventario,
                                  folds=a.folds, draw=draw, n=n_v,
                                  n_evaluado=int(m.sum()), n_disponible=len(idx),
                                  frac_ventana=n_v / len(idx),
                                  n_bloques=len(pd.unique(bl)), R2_diseno=rf, R2_nulo=rn,
                                  margen=rf - rn))
            pd.DataFrame(filas).to_csv(a.salida or OUT, index=False)
            u = pd.DataFrame([x for x in filas if x["faceta"] == fac and x["ventana"] == vent])
            if len(u):
                print(f"  {vent:9s} margen {u.margen.mean():+.3f} +- {u.margen.std():.3f} "
                      f"(diseno {u.R2_diseno.mean():+.3f}, nulo {u.R2_nulo.mean():+.3f}, "
                      f"{u.frac_ventana.iloc[0]:.0%} de la ventana)")

    f = pd.DataFrame(filas)
    f.to_csv(a.salida or OUT, index=False)
    print(f"\n-> {a.salida or OUT}")
    g = f.groupby("ventana")[["R2_nulo", "R2_diseno", "margen"]].mean()
    pos = (f.groupby(["ventana", "faceta"]).margen.mean() > 0).groupby("ventana").sum()
    g["facetas_con_margen_pos"] = pos
    orden = [v for v in VENTANAS if v in g.index]
    print(g.reindex(orden).round(3).to_string())


if __name__ == "__main__":
    main()
