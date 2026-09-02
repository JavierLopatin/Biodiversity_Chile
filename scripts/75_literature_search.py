#!/usr/bin/env python3
"""Run the state-of-the-art searches of `docs/23_wos_search_strings.md` programmatically.

Three backends, chosen with ``--backend``:

``wos``
    Web of Science Starter API (`api.clarivate.com/apis/wos-starter/v1`). Needs a key from
    developer.clarivate.com, passed in ``WOS_API_KEY`` or ``--api-key``. Takes the same
    field tags as the web interface (``TS=``, ``TI=``, ``PY=``, ``AND``/``OR``/``NOT``), so
    the strings in docs/23 run unchanged. The Starter tier returns bibliographic metadata
    and times-cited, not abstracts; the Expanded API returns full records but is a separate
    subscription. Rate limit for Starter is low (of the order of a few requests per second
    and a monthly record cap), hence ``--sleep``.

``openalex``
    OpenAlex (`api.openalex.org`), free and without a key. Not a WoS replacement — its
    query language has no field tags — so each search is translated into a title/abstract
    search plus filters. Use it to get counts and DOIs when there is no WoS key, and to
    cross-check that a "we found no study that..." claim is not an artefact of one index.

``crossref``
    Crossref (`api.crossref.org`), free, best for resolving a known title to a canonical
    record. Included because it is what the rest of this repository already uses to verify
    every DOI before it enters `paper/refs.bib`.

Outputs, per query: a CSV of hits (doi, year, title, journal, citations) and a line in
`summary.csv` with the result count, so that a statement of absence can be reported with
its query, backend, index and date, which is what makes it checkable.

Usage:
    python scripts/75_literature_search.py --backend openalex --out results/literature
    WOS_API_KEY=xxxx python scripts/75_literature_search.py --backend wos --queries B2,B7
    python scripts/75_literature_search.py --list

R equivalent, if you would rather stay in R: `wosr` (CRAN 0.3.0) wraps the Expanded API and
InCites, `rwosstarter` (github.com/FRBCesab/rwosstarter) wraps the Starter API, and
`openalexR` covers the OpenAlex path. The Python client Clarivate publishes is
`github.com/clarivate/wosstarter_python_client`; this script calls the REST endpoint
directly to avoid the dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

MAILTO = "javierlopatin@gmail.com"
WOS_URL = "https://api.clarivate.com/apis/wos-starter/v1/documents"
OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"

#: The eight blocks of `docs/23_wos_search_strings.md`. ``wos`` is the exact string for the
#: Web of Science query field; ``openalex`` is the closest translation, since OpenAlex has
#: no field tags and no proximity operator (NEAR/n is approximated by co-occurrence).
QUERIES: dict[str, dict] = {
    "B1": dict(
        label="core: plant diversity from remote sensing",
        years=(2015, 2026),
        wos='TS=(("remote sensing" OR satellite* OR spaceborne OR "earth observation" OR '
            'Landsat OR "Sentinel-2" OR MODIS OR hyperspectral) AND ("plant diversity" OR '
            '"species richness" OR "vegetation diversity" OR "community composition" OR '
            '"beta diversity" OR "phylogenetic diversity" OR "functional diversity") AND '
            '(map* OR predict* OR estimat* OR model*))',
        openalex='("remote sensing" OR satellite OR Landsat OR Sentinel OR hyperspectral) '
                 'AND ("plant diversity" OR "species richness" OR "community composition")',
    ),
    "B2": dict(
        label="phenology / time series as predictor of diversity",
        years=(2015, 2026),
        wos='TS=(("land surface phenology" OR phenolog* OR "time series" OR "seasonal '
            'dynamic*" OR "interannual variab*") NEAR/5 (satellit* OR "remote sensing" OR '
            'Landsat OR Sentinel OR MODIS)) AND TS=("plant diversity" OR "species richness" '
            'OR "species composition" OR "community composition" OR "beta diversity" OR '
            '"phylogenetic diversity" OR "functional diversity")',
        openalex='(phenology OR "time series") AND ("remote sensing" OR satellite) AND '
                 '("plant diversity" OR "species richness" OR "community composition")',
    ),
    "B3": dict(
        label="spectral variation hypothesis and its critiques",
        years=(2000, 2026),
        wos='TS=("spectral variation hypothesis" OR "spectral diversity" OR "spectral '
            'heterogeneity" OR "spectral variability" OR "Rao\'s Q" OR "spectral species") '
            'AND TS=(biodiversity OR "species richness" OR "plant diversity" OR '
            '"tree diversity")',
        openalex='("spectral variation hypothesis" OR "spectral diversity" OR "spectral '
                 'heterogeneity") AND (biodiversity OR "species richness")',
    ),
    "B4": dict(
        label="large-scale diversity maps from plot databases",
        years=(2018, 2026),
        wos='TS=(global OR continental OR national OR "wall-to-wall" OR macroecolog*) AND '
            'TS=("species richness" OR "plant diversity" OR "phylogenetic diversity" OR '
            '"alpha diversity" OR "vascular plant*") AND TS=(map* OR "spatial prediction*" '
            'OR "machine learning" OR "random forest" OR "deep learning") AND '
            'TS=("vegetation plot*" OR sPlot OR "plot database" OR "forest inventory" OR '
            '"field plot*")',
        openalex='(global OR continental OR national) AND ("species richness" OR "plant '
                 'diversity") AND ("machine learning" OR "random forest") AND '
                 '("vegetation plots" OR "forest inventory")',
    ),
    "B5": dict(
        label="spatial cross-validation, area of applicability, transferability",
        years=(2017, 2026),
        wos='TS=("spatial cross-validation" OR "block cross-validation" OR "spatial block*" '
            'OR "leave-location-out" OR "area of applicability" OR "spatial autocorrelation") '
            'AND TS=(ecolog* OR biodiversity OR vegetation OR "species distribution" OR '
            '"remote sensing" OR mapping)',
        openalex='("spatial cross-validation" OR "block cross-validation" OR "area of '
                 'applicability") AND (ecology OR biodiversity OR "remote sensing")',
    ),
    "B5b": dict(
        label="temporal transferability across years",
        years=(2015, 2026),
        wos='TS=("temporal transferability" OR "temporal extrapolation" OR '
            '"leave-time-out" OR "across years" OR "interannual transfer*") AND '
            'TS=(model* AND (vegetation OR biodiversity OR "remote sensing"))',
        openalex='("temporal transferability" OR "temporal extrapolation") AND '
                 '(vegetation OR biodiversity OR "remote sensing")',
    ),
    "B6": dict(
        label="deep learning on satellite image time series for vegetation",
        years=(2019, 2026),
        wos='TS=("deep learning" OR "convolutional neural network*" OR transformer* OR '
            '"self-supervised" OR "masked autoencoder*") AND TS=("satellite image time '
            'series" OR "image time series" OR phenolog* OR "temporal profile*") AND '
            'TS=(vegetation OR biodiversity OR "species richness" OR "plant communit*" OR '
            '"tree species")',
        openalex='("deep learning" OR "convolutional neural network") AND ("satellite image '
                 'time series" OR "image time series" OR phenology) AND (vegetation OR '
                 'biodiversity)',
    ),
    "B7a": dict(
        label="GAP: phylogenetic diversity from remote sensing",
        years=(2010, 2026),
        wos='TS=("phylogenetic diversity" OR "Faith\'s PD" OR "phylogenetic endemism") AND '
            'TS=("remote sensing" OR satellite* OR spectral* OR Sentinel OR Landsat)',
        openalex='"phylogenetic diversity" AND ("remote sensing" OR satellite OR spectral)',
    ),
    "B7b": dict(
        label="GAP: LCBD / compositional turnover from remote sensing",
        years=(2010, 2026),
        wos='TS=("local contribution to beta diversity" OR LCBD OR "generalized '
            'dissimilarity model*" OR "compositional turnover") AND TS=("remote sensing" OR '
            'satellite* OR Landsat OR Sentinel OR spectral*)',
        openalex='("local contribution to beta diversity" OR LCBD OR "generalized '
                 'dissimilarity modelling") AND ("remote sensing" OR satellite)',
    ),
    "B7c": dict(
        label="GAP: dark diversity from remote sensing",
        years=(2010, 2026),
        wos='TS=("dark diversity") AND TS=("remote sensing" OR satellite* OR predict* OR map*)',
        openalex='"dark diversity" AND ("remote sensing" OR satellite)',
    ),
    "B8": dict(
        label="regional context: Chile and South America",
        years=(2010, 2026),
        wos='TS=(Chile OR "South America" OR Patagonia* OR sclerophyll* OR matorral OR '
            'Valdivian) AND TS=(biodiversity OR "plant diversity" OR "species richness" OR '
            'vegetation) AND TS=("remote sensing" OR satellite* OR Landsat OR Sentinel OR '
            'MODIS)',
        openalex='(Chile OR Patagonia OR sclerophyll OR matorral) AND ("plant diversity" OR '
                 '"species richness") AND ("remote sensing" OR satellite)',
    ),
}


def _get(url: str, headers: dict | None = None, tries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": f"biodiv-chile/1.0 ({MAILTO})",
                                               **(headers or {})})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------

def search_wos(query: str, years: tuple[int, int], api_key: str, limit: int,
               sleep: float) -> tuple[int, list[dict]]:
    """Web of Science Starter API. Returns (total hits, records up to `limit`)."""
    q = f"{query} AND PY={years[0]}-{years[1]}"
    rows: list[dict] = []
    total = 0
    page = 1
    while len(rows) < limit:
        url = f"{WOS_URL}?" + urllib.parse.urlencode(
            {"q": q, "db": "WOS", "limit": min(50, limit - len(rows)), "page": page})
        data = _get(url, {"X-ApiKey": api_key})
        total = int(data.get("metadata", {}).get("total", 0))
        hits = data.get("hits", [])
        if not hits:
            break
        for h in hits:
            src = h.get("source", {})
            ids = h.get("identifiers", {})
            rows.append(dict(
                doi=ids.get("doi", ""),
                year=src.get("publishYear", ""),
                title=h.get("title", ""),
                journal=src.get("sourceTitle", ""),
                citations=(h.get("citations") or [{}])[0].get("count", ""),
                uid=h.get("uid", "")))
        page += 1
        time.sleep(sleep)
    return total, rows


def search_openalex(query: str, years: tuple[int, int], limit: int,
                    sleep: float) -> tuple[int, list[dict]]:
    """OpenAlex. `search` covers title, abstract and fulltext; filters do the year range."""
    rows: list[dict] = []
    total = 0
    cursor = "*"
    while len(rows) < limit:
        url = f"{OPENALEX_URL}?" + urllib.parse.urlencode({
            "search": query,
            "filter": f"from_publication_date:{years[0]}-01-01,"
                      f"to_publication_date:{years[1]}-12-31,type:article",
            "per-page": min(50, limit - len(rows)),
            "cursor": cursor,
            "mailto": MAILTO})
        data = _get(url)
        total = data.get("meta", {}).get("count", 0)
        cursor = data.get("meta", {}).get("next_cursor")
        results = data.get("results", [])
        if not results:
            break
        for w in results:
            loc = (w.get("primary_location") or {}).get("source") or {}
            rows.append(dict(
                doi=(w.get("doi") or "").replace("https://doi.org/", ""),
                year=w.get("publication_year", ""),
                title=(w.get("title") or "")[:250],
                journal=loc.get("display_name", ""),
                citations=w.get("cited_by_count", ""),
                uid=w.get("id", "")))
        if not cursor:
            break
        time.sleep(sleep)
    return total, rows


def search_crossref(query: str, years: tuple[int, int], limit: int,
                    sleep: float) -> tuple[int, list[dict]]:
    url = f"{CROSSREF_URL}?" + urllib.parse.urlencode({
        "query.bibliographic": query,
        "filter": f"from-pub-date:{years[0]}-01-01,until-pub-date:{years[1]}-12-31,"
                  "type:journal-article",
        "rows": min(100, limit), "mailto": MAILTO})
    data = _get(url)["message"]
    rows = [dict(doi=i.get("DOI", ""), year=(i.get("issued", {}).get("date-parts",
                 [[None]])[0][0] or ""), title=(i.get("title") or [""])[0][:250],
                 journal=(i.get("container-title") or [""])[0],
                 citations=i.get("is-referenced-by-count", ""), uid=i.get("DOI", ""))
            for i in data.get("items", [])]
    time.sleep(sleep)
    return data.get("total-results", 0), rows


# --------------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="openalex", choices=["wos", "openalex", "crossref"])
    p.add_argument("--api-key", default=os.environ.get("WOS_API_KEY"), dest="api_key")
    p.add_argument("--queries", default="all",
                   help="comma-separated block ids (see --list), or 'all'")
    p.add_argument("--limit", type=int, default=200, help="records to retrieve per query")
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--out", default="results/literature")
    p.add_argument("--list", action="store_true", help="print the queries and exit")
    args = p.parse_args()

    if args.list:
        for k, v in QUERIES.items():
            print(f"{k:5s} {v['years'][0]}-{v['years'][1]}  {v['label']}")
        return
    if args.backend == "wos" and not args.api_key:
        raise SystemExit("the wos backend needs a key: WOS_API_KEY=... or --api-key. "
                         "Request one at developer.clarivate.com (Starter API); with an "
                         "institutional subscription your library can enable Expanded.")

    keys = list(QUERIES) if args.queries == "all" else [k.strip() for k in args.queries.split(",")]
    unknown = [k for k in keys if k not in QUERIES]
    if unknown:
        raise SystemExit(f"unknown query ids: {unknown}. Use --list.")

    out = Path(args.out) / args.backend
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for k in keys:
        q = QUERIES[k]
        query = q["wos"] if args.backend == "wos" else q["openalex"]
        print(f"[{k}] {q['label']} ({q['years'][0]}-{q['years'][1]})", flush=True)
        try:
            if args.backend == "wos":
                total, rows = search_wos(query, q["years"], args.api_key, args.limit, args.sleep)
            elif args.backend == "openalex":
                total, rows = search_openalex(query, q["years"], args.limit, args.sleep)
            else:
                total, rows = search_crossref(query, q["years"], args.limit, args.sleep)
        except Exception as e:                                   # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}")
            summary.append(dict(query_id=k, label=q["label"], backend=args.backend,
                                years=f"{q['years'][0]}-{q['years'][1]}", total="error",
                                retrieved=0, date=str(date.today()), query=query))
            continue
        f = out / f"{k}.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["doi", "year", "title", "journal",
                                               "citations", "uid"])
            w.writeheader()
            w.writerows(rows)
        print(f"    {total} hits, {len(rows)} retrieved -> {f}")
        summary.append(dict(query_id=k, label=q["label"], backend=args.backend,
                            years=f"{q['years'][0]}-{q['years'][1]}", total=total,
                            retrieved=len(rows), date=str(date.today()), query=query))

    s = out / "summary.csv"
    with s.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_id", "label", "backend", "years", "total",
                                           "retrieved", "date", "query"])
        w.writeheader()
        w.writerows(summary)
    print(f"\n-> {s}")
    print("Report absence claims as: query id, backend, index, date and hit count.")


if __name__ == "__main__":
    main()
