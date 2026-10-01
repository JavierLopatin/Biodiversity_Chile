"""Las facetas que el paper reporta, su orden y sus rotulos, en un solo lugar.

Esta lista estaba duplicada en `scripts/115`, `117`, `120` y `124`, que es exactamente por que
sacar dos facetas tocaba cuatro archivos y se podian desincronizar. Las figuras y las tablas la
importan de aqui.

QUE ESTA FUERA Y POR QUE
------------------------
`ses_mpd_unified` y `ses_mntd_unified` (decision del 2026-10-01). Un SES estandariza la metrica
contra un nulo de barajado de etiquetas para quitarle la dependencia de la riqueza, y eso sirve
para preguntar por procesos de ensamblaje. Aqui no hay nada que quitar y la pregunta es otra.
Medido sobre el pool lenoso:

    rho(MPD, SES MPD)   = +0,971     rho(MPD, riqueza)  = +0,160
    rho(MNTD, SES MNTD) = +0,874     rho(MNTD, riqueza) = -0,221, y SES MNTD queda en +0,129

O sea MPD casi no depende de la riqueza, asi que su SES es la misma variable; y en MNTD el SES
no quita la dependencia sino que le cambia el signo. Los margenes de cada par son ademas casi
iguales columna por columna: |MPD - SES MPD| <= 0,011 y |MNTD - SES MNTD| <= 0,038. Eran filas
duplicadas.

`ses_pd_unified` SE QUEDA, y no por simetria. La PD de Faith correlaciona +0,843 con la riqueza
-- es casi una medida de riqueza-- y su SES baja a +0,095, con rho 0,560 entre las dos. Ahi el
SES si hace su trabajo, y responde una pregunta que el paper si hace: si la estructura
filogenetica es predecible MAS ALLA de la riqueza.

Los cuatro ejes de PCoA no estan en `FAMILIAS` porque el texto principal usa Isomap; viven en
`PCOA` y entran solo en el suplemento.
"""

from __future__ import annotations

#: (nombre de la familia, facetas) en el orden en que el paper las presenta.
FAMILIAS: list[tuple[str, list[str]]] = [
    ("Taxonomic richness", ["hill_q0_unified", "td_inext_q0", "td_inext_q1", "td_inext_q2"]),
    ("Phylogenetic", ["pd_inext_q0", "pd_inext_q1", "pd_inext_q2",
                      "mpd_unified", "mntd_unified", "ses_pd_unified"]),
    ("Compositional uniqueness", ["lcbd_count_sorensen", "lcbd_pa_unified",
                                  "lcbd_freq_unified"]),
    ("Dark diversity", ["dark_n_unified"]),
    ("Floristic composition (ordination)", ["isomap1_pa_unified", "isomap2_pa_unified",
                                            "isomap3_pa_unified", "isomap1_freq_unified",
                                            "isomap2_freq_unified", "isomap3_freq_unified"]),
]
ORDEN: list[str] = [f for _, fs in FAMILIAS for f in fs]

#: Fuera del texto principal, con el motivo. No se borran del pipeline: siguen calculandose y
#: siguen en el suplemento, pero no entran en figuras, tablas ni familias de correccion.
EXCLUIDAS = {
    "ses_mpd_unified": "redundante con MPD (rho 0,971); MPD apenas depende de la riqueza",
    "ses_mntd_unified": "redundante con MNTD (rho 0,874); el SES no quita la dependencia",
}

#: Ordenacion del suplemento.
PCOA = ["pcoa1_pa_unified", "pcoa2_pa_unified", "pcoa1_freq_unified", "pcoa2_freq_unified"]

LAB: dict[str, str] = {
    "hill_q0_unified": r"Richness $q_0$ (raw)",
    "td_inext_q0": r"TD $q_0$ (cov.-std.)", "td_inext_q1": r"TD $q_1$ (cov.-std.)",
    "td_inext_q2": r"TD $q_2$ (cov.-std.)",
    "pd_inext_q0": r"PD $q_0$ (cov.-std.)", "pd_inext_q1": r"PD $q_1$ (cov.-std.)",
    "pd_inext_q2": r"PD $q_2$ (cov.-std.)",
    "mpd_unified": "MPD", "mntd_unified": "MNTD", "ses_pd_unified": "SES PD",
    "ses_mpd_unified": "SES MPD", "ses_mntd_unified": "SES MNTD",
    "lcbd_count_sorensen": "LCBD Sørensen", "lcbd_pa_unified": "LCBD (p/a)",
    "lcbd_freq_unified": "LCBD (freq.)",
    "dark_n_unified": "Dark diversity",
    **{f"{m}{i}_{v}_unified": f"{'Isomap' if m == 'isomap' else 'PCoA'} {i} "
       f"({'p/a' if v == 'pa' else 'freq.'})"
       for m in ("isomap", "pcoa") for i in (1, 2, 3) for v in ("pa", "freq")},
}
