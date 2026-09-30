"""El mecanismo propuesto para el efecto de sequia, y por que hay que descartarlo.

`scripts/107` mide que 17 de 20 facetas se predicen mejor en ano humedo, descontando al bloque
remoto el cambio que el piso tiene entre los mismos terciles. El patron es solido. La
explicacion que se propuso para el no lo es, y este script es lo que la descarta.

LA HIPOTESIS. Si la senal remota funciona porque la vegetacion *se expresa* -- verdor,
estructura, amplitud fenologica-- entonces en ano seco se expresa menos, dos comunidades
distintas deberian verse mas parecidas, y el predictor perderia informacion sin que la comunidad
cambie.

EL TEST. Se mide sin modelar nada, en dos formas complementarias, sobre Living Trees (un solo
protocolo, area fija) y entre pares dentro de la misma banda de 2 grados de latitud para que el
gradiente no lo infle:

  1. correlacion entre distancia ESPECTRAL (bandas del geomediano) y distancia COMPOSICIONAL
     (Jaccard lenoso), por tercil. Si la hipotesis vale, baja en seco.
  2. dispersion espectral entre parcelas, por tercil. Si la hipotesis vale, baja en seco.

EL RESULTADO. Las dos salen planas:

    tercil      rho(espectro, Jaccard)    dispersion espectral    NDVI mediano
    seco                         0,124                  0,0757           0,821
    medio                        0,124                  0,0739           0,834
    humedo                       0,130                  0,0761           0,781

El acoplamiento entre espectro y composicion es el mismo en seco y en humedo, la dispersion
espectral tambien, y el NDVI va AL REVES de lo que la hipotesis pide: es mas alto en el tercil
seco. La hipotesis queda descartada.

Cautela al leer el NDVI: el tercil es de SPI residualizado contra latitud DENTRO del ano, o sea
es seco-respecto-de-sus-vecinos-ese-ano, no seco en absoluto. Una parcela del tercil seco a 45 S
sigue siendo mas humeda que una del tercil humedo a 32 S. Eso explica que el NDVI no ordene, pero
no rescata la hipotesis: los otros dos contrastes son dentro de banda y tampoco ordenan.

QUE QUEDA. El patron de `scripts/107` sigue en pie -- esta controlado contra el piso, que es lo
que descarta la explicacion de varianza del target. Lo que no hay es mecanismo. Una correlacion
de rango entre pares es un instrumento marginal y debil, asi que esto no prueba que NINGUN
mecanismo espectral exista; prueba que el mas obvio no se sostiene, y que el paper no puede
afirmarlo.

Uso:
    python scripts/110_mecanismo_sequia.py
"""
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

g = pd.read_parquet('data/derived/gm_center_nc.parquet')
bands = [c for c in g.columns if c.startswith('gm_band')]
occ = pd.read_parquet('data/derived/occurrences_unified.parquet')
lk = pd.read_csv('data/derived/growth_form_lookup.csv')
occ = occ[occ.species.isin(set(lk.loc[lk.forma == 'lenosa', 'species']))]
P = pd.read_parquet('data/derived/plots_unified.parquet')[
    ['PlotObservationID', 'lat', 'Year', 'source']]
spi = pd.read_parquet('data/derived/spi_unified.parquet')
if spi.index.name: spi = spi.reset_index()
col = [c for c in spi.columns if 'spi12' in c and 'mean' in c][0]
P = P.merge(spi[['PlotObservationID', col]].rename(columns={col: 'spi'}), on='PlotObservationID')

comm = occ.assign(v=1).pivot_table(index='PlotObservationID', columns='species',
                                   values='v', aggfunc='max', fill_value=0)
d = P[(P.source == 'living_trees') & P.spi.notna()].copy()
d = d[d.PlotObservationID.isin(comm.index) & d.PlotObservationID.isin(set(g.PlotObservationID))]

# tercil de SPI residualizado contra latitud, dentro del ano de censo -- igual que scripts/107
d['t'] = np.nan
for _, gy in d.groupby('Year'):
    if len(gy) < 120 or gy.lat.nunique() < 3: continue
    res = gy.spi - np.polyval(np.polyfit(gy.lat, gy.spi, 1), gy.lat)
    d.loc[gy.index, 't'] = pd.qcut(res, 3, labels=[0, 1, 2]).astype(float)
d = d.dropna(subset=['t'])
gm = g.set_index('PlotObservationID')

print('Correlacion entre distancia ESPECTRAL y distancia COMPOSICIONAL, por tercil.')
print('Pares dentro de la misma banda de 2 grados de latitud, para que la geografia no la infle.\n')
print(f'{"tercil":10s} {"n parc":>7s} {"n pares":>9s} {"rho(espectro, Jaccard)":>24s}')
d['b'] = np.floor(d.lat / 2) * 2
for k, nm in enumerate(['seco', 'medio', 'humedo']):
    s = d[d.t == k]
    rr, nn = [], 0
    for b, gb in s.groupby('b'):
        ids = gb.PlotObservationID.to_numpy()
        if len(ids) < 40: continue
        S = gm.loc[ids, bands].to_numpy(float)
        C = (comm.loc[ids].to_numpy() > 0).astype(float)
        ok = np.isfinite(S).all(1)
        if ok.sum() < 40: continue
        ds = pdist(S[ok]); dc = pdist(C[ok], metric='jaccard')
        v = np.isfinite(ds) & np.isfinite(dc)
        rr.append(spearmanr(ds[v], dc[v]).statistic); nn += v.sum()
    print(f'{nm:10s} {len(s):7d} {nn:9,d} {np.mean(rr):24.3f}')
