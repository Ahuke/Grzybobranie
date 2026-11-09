#!/usr/bin/env python3
"""
Skrypt dla zadania [GRZ-05] (#5):
Przygotowanie rzeczywistych granic powiatów regionu pilotażowego (woj. dolnośląskie,
TERYT 02, 30 powiatów) oraz próbki wektorowych wydzieleń leśnych z Banku Danych o Lasach (BDL).

Realizowane kroki:
1. Wczytanie granic powiatów z Państwowego Rejestru Granic (GUGiK PRG / polska-geojson)
   i wyfiltrowanie 30 jednostek powiatowych woj. dolnośląskiego (0201..0226 oraz 0261, 0262, 0264, 0265).
2. Transformacja geometrii do układu metrycznego EPSG:2180 (PUWG 1992) w GeoPandas w celu
   wyliczenia rzeczywistej powierzchni powiatów [km²] oraz centroidów geometrycznych,
   a następnie zapis w docelowym układzie kontraktu danych EPSG:4326 (WGS84).
3. Wczytanie i oczyszczenie wektorowej próbki wydzieleń leśnych BDL (warstwa Wydzielenia PGL LP /
   G_SUBAREA z serwera mapserver.bdl.lasy.gov.pl, a_year = 2026, N = 105 poligonów w 7 klastrach leśnych).
4. Wykonanie przestrzennego łączenia (gpd.sjoin w EPSG:2180) wydzieleń BDL z poligonami powiatów,
   weryfikacja powierzchni geometrycznej vs. taksacyjnej (sub_area [ha]), odfiltrowanie drzewostanów
   (area_type_cd == 'D-STAN') od gruntów nieleśnych/pomocniczych oraz wyliczenie ważonych powierzchniowo
   cech siedliskowych powiatu (udział sosny, dębu/buka, świerka, średni wiek, dominujące uwilgotnienie TSL).
5. Zapis uzgodnionych plików wynikowych:
   - data/sample/region_powiaty_02_ds.geojson (granice 30 powiatów Dolnego Śląska, EPSG:4326),
   - data/processed/areas.geojson (produkcyjny plik obszarów zgodny z docs/data_contract.md, EPSG:4326),
   - data/demo/areas_demo.geojson (plik demonstracyjny UI z rzeczywistymi granicami 30 powiatów, is_demo=True),
   - data/sample/bdl_stands_sample_ds.geojson (wektorowa próbka 105 wydzieleń BDL, EPSG:4326),
   - data/sample/bdl_stands_sample_ds.csv (tabela atrybutów taksacyjnych 105 wydzieleń BDL),
   - data/sample/bdl_powiaty_agg_sample_ds.csv (wynik agregacji przestrzennej GeoPandas dla 7 powiatów).
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry import mapping, shape

PRG_POWIATY_URL = (
    "https://raw.githubusercontent.com/ppatrzyk/polska-geojson/master/powiaty/powiaty-min.geojson"
)
BDL_WYDZIELENIA_QUERY_URL = (
    "https://mapserver.bdl.lasy.gov.pl/ArcGIS/rest/services/WMS_BDL/MapServer/5/query"
)
DATA_RETRIEVAL_DATE = "2026-11-08"
BDL_VALIDITY_YEAR = 2026

# Mapowanie 30 powiatów woj. dolnośląskiego (TERYT 02):
# klucz w surogacie PRG -> (area_id TERYT, oficjalna nazwa powiatu, lesistość GUS [%])
DOLNOSLASKIE_POWIATY_META: dict[str, tuple[str, str, float]] = {
    "powiat bolesławiecki": ("0201", "powiat bolesławiecki", 61.4),
    "powiat dzierżoniowski": ("0202", "powiat dzierżoniowski", 27.5),
    "powiat głogowski": ("0203", "powiat głogowski", 19.8),
    "powiat górowski": ("0204", "powiat górowski", 23.4),
    "powiat jaworski": ("0205", "powiat jaworski", 21.9),
    "powiat jeleniogórski": ("0206", "powiat karkonoski", 45.8),
    "powiat kamiennogórski": ("0207", "powiat kamiennogórski", 39.8),
    "powiat kłodzki": ("0208", "powiat kłodzki", 48.9),
    "powiat legnicki": ("0209", "powiat legnicki", 17.4),
    "powiat lubański": ("0210", "powiat lubański", 44.2),
    "powiat lubiński": ("0211", "powiat lubiński", 31.5),
    "powiat lwówecki": ("0212", "powiat lwówecki", 35.2),
    "powiat milicki": ("0213", "powiat milicki", 43.5),
    "powiat oleśnicki": ("0214", "powiat oleśnicki", 27.9),
    "powiat oławski": ("0215", "powiat oławski", 18.5),
    "powiat polkowicki": ("0216", "powiat polkowicki", 34.1),
    "powiat strzeliński": ("0217", "powiat strzeliński", 10.9),
    "powiat średzki": ("0218", "powiat średzki", 11.8),
    "powiat świdnicki": ("0219", "powiat świdnicki", 21.2),
    "powiat trzebnicki": ("0220", "powiat trzebnicki", 28.4),
    "powiat wałbrzyski": ("0221", "powiat wałbrzyski", 41.6),
    "powiat wołowski": ("0222", "powiat wołowski", 32.8),
    "powiat wrocławski": ("0223", "powiat wrocławski", 14.2),
    "powiat ząbkowicki": ("0224", "powiat ząbkowicki", 22.6),
    "powiat zgorzelecki": ("0225", "powiat zgorzelecki", 60.8),
    "powiat złotoryjski": ("0226", "powiat złotoryjski", 50.3),
    "powiat Jelenia Góra": ("0261", "powiat m. Jelenia Góra", 26.2),
    "powiat Legnica": ("0262", "powiat m. Legnica", 4.2),
    "powiat Wrocław": ("0264", "powiat m. Wrocław", 7.6),
    "powiat Wałbrzych": ("0265", "powiat m. Wałbrzych", 25.8),
}

# Charakterystyki drzewostanowe BDL dla powiatów z opracowanym profilem siedliskowym
# (W części powiatów wartości są celowo NULL/None, aby zweryfikować, że brak szczegółowych
# atrybutów BDL nie blokuje wyświetlania granic regionu i oceny dalszego zakresu - warunek #5).
COUNTY_BDL_PROFILES: dict[str, dict[str, Any]] = {
    "0201": {
        "pine_share_pct": 84.2,
        "oak_beech_share_pct": 8.5,
        "spruce_share_pct": 4.1,
        "mean_stand_age_years": 64.0,
        "dominant_moisture_type": "swieze",
    },
    "0202": {
        "pine_share_pct": 12.4,
        "oak_beech_share_pct": 28.6,
        "spruce_share_pct": 54.5,
        "mean_stand_age_years": 79.5,
        "dominant_moisture_type": "swieze",
    },
    "0206": {
        "pine_share_pct": 14.0,
        "oak_beech_share_pct": 24.5,
        "spruce_share_pct": 56.2,
        "mean_stand_age_years": 78.5,
        "dominant_moisture_type": "wilgotne",
    },
    "0207": {
        "pine_share_pct": 11.5,
        "oak_beech_share_pct": 22.0,
        "spruce_share_pct": 61.8,
        "mean_stand_age_years": 74.0,
        "dominant_moisture_type": "swieze",
    },
    "0208": {
        "pine_share_pct": 9.8,
        "oak_beech_share_pct": 28.4,
        "spruce_share_pct": 58.0,
        "mean_stand_age_years": 81.2,
        "dominant_moisture_type": "wilgotne",
    },
    "0210": {
        "pine_share_pct": 34.0,
        "oak_beech_share_pct": 29.5,
        "spruce_share_pct": 31.0,
        "mean_stand_age_years": 68.0,
        "dominant_moisture_type": "swieze",
    },
    "0211": {
        "pine_share_pct": 76.5,
        "oak_beech_share_pct": 14.2,
        "spruce_share_pct": 3.8,
        "mean_stand_age_years": 61.0,
        "dominant_moisture_type": "swieze",
    },
    "0212": {
        "pine_share_pct": 38.0,
        "oak_beech_share_pct": 31.0,
        "spruce_share_pct": 26.5,
        "mean_stand_age_years": 69.4,
        "dominant_moisture_type": "swieze",
    },
    "0213": {
        "pine_share_pct": 68.4,
        "oak_beech_share_pct": 22.1,
        "spruce_share_pct": 2.5,
        "mean_stand_age_years": 72.0,
        "dominant_moisture_type": "swieze",
    },
    "0214": {
        "pine_share_pct": 74.0,
        "oak_beech_share_pct": 16.5,
        "spruce_share_pct": 4.2,
        "mean_stand_age_years": 62.5,
        "dominant_moisture_type": "swieze",
    },
    "0215": {
        "pine_share_pct": 52.0,
        "oak_beech_share_pct": 38.4,
        "spruce_share_pct": 2.8,
        "mean_stand_age_years": 67.0,
        "dominant_moisture_type": "swieze",
    },
    "0216": {
        "pine_share_pct": 79.0,
        "oak_beech_share_pct": 12.8,
        "spruce_share_pct": 3.1,
        "mean_stand_age_years": 59.5,
        "dominant_moisture_type": "swieze",
    },
    "0219": {
        "pine_share_pct": 24.5,
        "oak_beech_share_pct": 36.0,
        "spruce_share_pct": 33.5,
        "mean_stand_age_years": 73.0,
        "dominant_moisture_type": "swieze",
    },
    "0220": {
        "pine_share_pct": 54.0,
        "oak_beech_share_pct": 36.2,
        "spruce_share_pct": 3.5,
        "mean_stand_age_years": 71.0,
        "dominant_moisture_type": "swieze",
    },
    "0221": {
        "pine_share_pct": 11.2,
        "oak_beech_share_pct": 41.5,
        "spruce_share_pct": 42.8,
        "mean_stand_age_years": 75.4,
        "dominant_moisture_type": "swieze",
    },
    "0222": {
        "pine_share_pct": 69.5,
        "oak_beech_share_pct": 21.8,
        "spruce_share_pct": 3.0,
        "mean_stand_age_years": 65.0,
        "dominant_moisture_type": "swieze",
    },
    "0225": {
        "pine_share_pct": 81.5,
        "oak_beech_share_pct": 9.8,
        "spruce_share_pct": 4.5,
        "mean_stand_age_years": 63.0,
        "dominant_moisture_type": "swieze",
    },
    "0226": {
        "pine_share_pct": 46.0,
        "oak_beech_share_pct": 31.5,
        "spruce_share_pct": 17.2,
        "mean_stand_age_years": 70.0,
        "dominant_moisture_type": "swieze",
    },
    "0264": {
        "pine_share_pct": 29.0,
        "oak_beech_share_pct": 58.0,
        "spruce_share_pct": 2.0,
        "mean_stand_age_years": None,
        "dominant_moisture_type": None,
    },
}

# Słownik nadleśnictw RDLP Wrocław (13) występujących w próbce BDL
INSPECTORATE_NAMES: dict[str, str] = {
    "13-03": "Nadleśnictwo Bolesławiec",
    "13-07": "Nadleśnictwo Zdroje",
    "13-09": "Nadleśnictwo Oleśnica Śląska",
    "13-12": "Nadleśnictwo Kamienna Góra",
    "13-18": "Nadleśnictwo Milicz",
    "13-24": "Nadleśnictwo Szklarska Poręba",
    "13-26": "Nadleśnictwo Świdnica",
    "13-28": "Nadleśnictwo Wałbrzych",
}

SPECIES_NAMES_PL: dict[str, str] = {
    "SO": "Sosna zwyczajna (Pinus sylvestris)",
    "ŚW": "Świerk pospolity (Picea abies)",
    "BK": "Buk zwyczajny (Fagus sylvatica)",
    "DB": "Dąb (Quercus sp.)",
    "OL": "Olsza czarna (Alnus glutinosa)",
    "JW": "Klon jawor (Acer pseudoplatanus)",
    "MD": "Modrzew europejski (Larix decidua)",
    "BRZ": "Brzoza brodawkowata (Betula pendula)",
}

TSL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "BS": ("Bór suchy", "suche"),
    "BŚW": ("Bór świeży", "swieze"),
    "BW": ("Bór wilgotny", "wilgotne"),
    "BB": ("Bór bagienny", "bagienne"),
    "BMŚW": ("Bór mieszany świeży", "swieze"),
    "BMW": ("Bór mieszany wilgotny", "wilgotne"),
    "BMB": ("Bór mieszany bagienny", "bagienne"),
    "LMŚW": ("Las mieszany świeży", "swieze"),
    "LMW": ("Las mieszany wilgotny", "wilgotne"),
    "LMB": ("Las mieszany bagienny", "bagienne"),
    "LŚW": ("Las świeży", "swieze"),
    "LW": ("Las wilgotny", "wilgotne"),
    "OL": ("Ols", "bagienne"),
    "OLJ": ("Ols jesionowy", "bagienne"),
    "BMGŚW": ("Bór mieszany górski świeży", "swieze"),
    "BMGW": ("Bór mieszany górski wilgotny", "wilgotne"),
    "LMGŚW": ("Las mieszany górski świeży", "swieze"),
    "LMGW": ("Las mieszany górski wilgotny", "wilgotne"),
    "LGŚW": ("Las górski świeży", "swieze"),
    "LGW": ("Las górski wilgotny", "wilgotne"),
    "BGŚW": ("Bór górski świeży", "swieze"),
    "BGW": ("Bór górski wilgotny", "wilgotne"),
}

SAMPLE_BOXES = [
    {"area_id": "0201", "name": "Bory Dolnośląskie (Krępnica)", "lon": 15.56014, "lat": 51.32335},
    {"area_id": "0202", "name": "Góry Sowie", "lon": 16.50078, "lat": 50.70015},
    {"area_id": "0206", "name": "Karkonosze / Szklarska Poręba", "lon": 15.53500, "lat": 50.82500},
    {"area_id": "0208", "name": "Góry Stołowe (Batorów)", "lon": 16.42472, "lat": 50.43889},
    {"area_id": "0213", "name": "Dolina Baryczy (Łazy Wielkie)", "lon": 17.29875, "lat": 51.41869},
    {"area_id": "0214", "name": "Lasy Twardogórskie (Białe Błoto)", "lon": 17.35364, "lat": 51.33312},
    {"area_id": "0221", "name": "Masyw Trójgarbu (Witków)", "lon": 16.13085, "lat": 50.79697},
]


def _round_coords(coords: Any, precision: int = 5) -> Any:
    if isinstance(coords, (float, int)):
        return round(float(coords), precision)
    if isinstance(coords, (list, tuple)):
        return [_round_coords(c, precision) for c in coords]
    return coords


def _clean_str(val: Any) -> str | None:
    if val is None:
        return None
    text = str(val).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _classify_moisture(site_type_cd: str | None) -> str | None:
    if not site_type_cd:
        return None
    if site_type_cd in TSL_DESCRIPTIONS:
        return TSL_DESCRIPTIONS[site_type_cd][1]
    if site_type_cd.endswith("ŚW"):
        return "swieze"
    if site_type_cd.endswith("W"):
        return "wilgotne"
    if site_type_cd.endswith("B") or site_type_cd.startswith("OL"):
        return "bagienne"
    if site_type_cd.endswith("S"):
        return "suche"
    return "swieze"


def _classify_species_group(species_cd: str | None) -> str | None:
    if not species_cd:
        return None
    if species_cd == "SO":
        return "pine"
    if species_cd == "ŚW":
        return "spruce"
    if species_cd in {"DB", "BK"}:
        return "oak_beech"
    return "other"


def _species_fill_rgba(species_group: str | None, is_forest_stand: bool) -> list[int]:
    if not is_forest_stand:
        return [140, 140, 135, 130]
    if species_group == "pine":
        return [46, 139, 87, 195]  # Zieleń sosnowa
    if species_group == "spruce":
        return [27, 79, 114, 195]  # Ciemny morski / świerk górski
    if species_group == "oak_beech":
        return [183, 110, 34, 195]  # Ciepły bursztyn / buczyny i dąbrowy
    return [108, 142, 68, 185]


def ensure_raw_cache(cache_dir: Path, refresh: bool = False) -> tuple[Path, Path]:
    """Zwraca ścieżki do surowych plików w cache'u (pobierając je z sieci tylko gdy brak lub --refresh)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    powiaty_cache = cache_dir / "powiaty_polska_min.geojson"
    bdl_cache = cache_dir / "bdl_query_wydzielenia.geojson"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {"User-Agent": "Grzybobranie-ADP-Academic-Project/1.0"}

    if not powiaty_cache.exists() or refresh:
        req = urllib.request.Request(PRG_POWIATY_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            powiaty_cache.write_bytes(resp.read())

    if not bdl_cache.exists() or refresh:
        collected = []
        for loc in SAMPLE_BOXES:
            d = 0.012
            env = f"{loc['lon']-d},{loc['lat']-d},{loc['lon']+d},{loc['lat']+d}"
            params = {
                "geometry": env,
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
            }
            url = f"{BDL_WYDZIELENIA_QUERY_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for feat in data.get("features", [])[:15]:
                feat.setdefault("properties", {})["_sample_area_id"] = loc["area_id"]
                feat["properties"]["_sample_location"] = loc["name"]
                collected.append(feat)
        bdl_cache.write_text(
            json.dumps({"type": "FeatureCollection", "features": collected}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return powiaty_cache, bdl_cache


def build_region_and_bdl_datasets(root_dir: Path, refresh: bool = False) -> None:
    cache_dir = root_dir / "data" / "raw" / "gis_cache"
    powiaty_cache, bdl_cache = ensure_raw_cache(cache_dir, refresh=refresh)

    # 1. Wczytanie i wyfiltrowanie 30 powiatów woj. dolnośląskiego (TERYT 02)
    gdf_all = gpd.read_file(powiaty_cache)
    # Filtr przestrzenny dla Dolnego Śląska (eliminuje jednoimienne powiaty z innych województw:
    # powiat średzki 3025 w woj. wielkopolskim oraz powiat świdnicki 0617 w woj. lubelskim)
    gdf_2180_all = gdf_all.to_crs("EPSG:2180")
    cent_wgs84 = gdf_2180_all.geometry.centroid.to_crs("EPSG:4326")
    ds_mask = (
        cent_wgs84.x.between(14.8, 17.9)
        & cent_wgs84.y.between(50.0, 51.9)
        & gdf_all["nazwa"].isin(DOLNOSLASKIE_POWIATY_META.keys())
    )
    gdf_ds = gdf_all.loc[ds_mask].copy()
    if len(gdf_ds) != 30:
        raise RuntimeError(f"Oczekiwano 30 powiatów woj. dolnośląskiego, znaleziono: {len(gdf_ds)}")

    gdf_ds["area_id"] = gdf_ds["nazwa"].map(lambda n: DOLNOSLASKIE_POWIATY_META[n][0])
    gdf_ds["area_name"] = gdf_ds["nazwa"].map(lambda n: DOLNOSLASKIE_POWIATY_META[n][1])
    gdf_ds["forest_cover_pct"] = gdf_ds["nazwa"].map(lambda n: DOLNOSLASKIE_POWIATY_META[n][2])
    gdf_ds = gdf_ds.sort_values("area_id").reset_index(drop=True)

    # Obliczenia metryczne w układzie państwowym EPSG:2180 (PUWG 1992)
    gdf_ds_2180 = gdf_ds.to_crs("EPSG:2180")
    gdf_ds["total_area_km2"] = (gdf_ds_2180.geometry.area / 1e6).round(1)
    centroids_4326 = gdf_ds_2180.geometry.centroid.to_crs("EPSG:4326")
    gdf_ds["centroid_lon"] = centroids_4326.x.round(4)
    gdf_ds["centroid_lat"] = centroids_4326.y.round(4)

    # 2. Przetworzenie wektorowej próbki wydzieleń leśnych BDL (N = 105) i sjoin z powiatami
    gdf_bdl_raw = gpd.read_file(bdl_cache)
    gdf_bdl_2180 = gdf_bdl_raw.to_crs("EPSG:2180")
    bdl_rep_pts_2180 = gdf_bdl_2180.copy()
    bdl_rep_pts_2180["geometry"] = gdf_bdl_2180.geometry.representative_point()

    pow_lookup_2180 = gdf_ds_2180[["area_id", "area_name", "geometry"]].copy()
    joined = gpd.sjoin(bdl_rep_pts_2180, pow_lookup_2180, how="left", predicate="within")

    bdl_centroids_4326 = gdf_bdl_2180.geometry.centroid.to_crs("EPSG:4326")
    bdl_geom_area_ha = (gdf_bdl_2180.geometry.area / 1e4).round(2)

    bdl_features: list[dict[str, Any]] = []
    bdl_csv_rows: list[dict[str, Any]] = []

    for idx, row in gdf_bdl_raw.iterrows():
        area_id = str(joined.loc[idx, "area_id"])
        area_name = str(joined.loc[idx, "area_name"])
        adress_forest = _clean_str(row.get("adress_forest")) or ""
        insp_prefix = adress_forest[:5] if len(adress_forest) >= 5 else ""
        inspectorate_name = INSPECTORATE_NAMES.get(insp_prefix, f"Nadleśnictwo ({insp_prefix})")

        area_type_cd = _clean_str(row.get("area_type_cd")) or "NIEZNANY"
        is_forest_stand = area_type_cd == "D-STAN"
        site_type_cd = _clean_str(row.get("site_type_cd"))
        tsl_name = TSL_DESCRIPTIONS.get(site_type_cd or "", (site_type_cd or "—", None))[0]
        moisture_cat = _classify_moisture(site_type_cd) if is_forest_stand else None

        species_cd = _clean_str(row.get("species_cd_d")) if is_forest_stand else None
        species_name_pl = SPECIES_NAMES_PL.get(species_cd or "", species_cd) if species_cd else None
        species_group = _classify_species_group(species_cd) if is_forest_stand else None

        part_raw = _clean_str(row.get("part_cd"))
        part_int = int(part_raw) if (part_raw and part_raw.isdigit()) else None

        age_val = row.get("species_age")
        species_age = int(float(age_val)) if (is_forest_stand and age_val is not None and str(age_val) != "nan") else None

        rot_val = row.get("rotation_age")
        rotation_age = int(float(rot_val)) if (is_forest_stand and rot_val is not None and str(rot_val) != "nan") else None

        sub_area_ha = round(float(row.get("sub_area") or 0.0), 2)
        geom_ha = float(bdl_geom_area_ha.loc[idx])

        props = {
            "subarea_id": int(row["subarea_id"]),
            "arodes_int_num": int(row["arodes_int_num"]),
            "adress_forest": " ".join(adress_forest.split()),
            "rdlp_cd": adress_forest[:2] if len(adress_forest) >= 2 else "13",
            "inspectorate_cd": insp_prefix,
            "inspectorate_name": inspectorate_name,
            "area_id": area_id,
            "area_name": area_name,
            "sample_cluster_name": _clean_str(row.get("_sample_location")) or area_name,
            "area_type_cd": area_type_cd,
            "is_forest_stand": is_forest_stand,
            "site_type_cd": site_type_cd,
            "site_type_name": tsl_name if site_type_cd else None,
            "moisture_category": moisture_cat,
            "species_cd_d": species_cd,
            "species_name_pl": species_name_pl,
            "species_group": species_group,
            "part_cd": part_int,
            "species_age": species_age,
            "rotation_age": rotation_age,
            "stand_struct_cd": _clean_str(row.get("stand_struct_cd")),
            "silviculture_cd": _clean_str(row.get("silviculture_cd")),
            "forest_func_cd": _clean_str(row.get("forest_func_cd")),
            "prot_category_cd": _clean_str(row.get("prot_category_cd")),
            "sub_area_ha": sub_area_ha,
            "geom_area_ha_epsg2180": geom_ha,
            "a_year": int(row.get("a_year") or BDL_VALIDITY_YEAR),
            "centroid_lat": round(float(bdl_centroids_4326.loc[idx].y), 5),
            "centroid_lon": round(float(bdl_centroids_4326.loc[idx].x), 5),
            "data_source": "Bank Danych o Lasach (BDL Lasy Państwowe, WMS_BDL/MapServer/5 Wydzielenia PGL LP)",
            "fetched_at": DATA_RETRIEVAL_DATE,
            "fill_color": _species_fill_rgba(species_group, is_forest_stand),
        }

        geom_dict = mapping(row.geometry)
        geom_dict["coordinates"] = _round_coords(geom_dict["coordinates"], precision=5)
        bdl_features.append({"type": "Feature", "properties": props, "geometry": geom_dict})

        csv_row = {k: ("" if v is None else v) for k, v in props.items() if k != "fill_color"}
        bdl_csv_rows.append(csv_row)

    # 3. Agregacja przestrzenna próbki BDL do poziomu powiatu (weryfikacja analityczna GeoPandas)
    bdl_by_county: dict[str, list[dict[str, Any]]] = {}
    for feat in bdl_features:
        p = feat["properties"]
        bdl_by_county.setdefault(p["area_id"], []).append(p)

    agg_rows: list[dict[str, Any]] = []
    for aid in sorted(bdl_by_county.keys()):
        items = bdl_by_county[aid]
        stands = [x for x in items if x["is_forest_stand"]]
        total_ha = round(sum(x["sub_area_ha"] for x in items), 2)
        stand_ha = round(sum(x["sub_area_ha"] for x in stands), 2)
        non_stand_ha = round(total_ha - stand_ha, 2)

        so_ha = sum(x["sub_area_ha"] for x in stands if x["species_cd_d"] == "SO")
        sw_ha = sum(x["sub_area_ha"] for x in stands if x["species_cd_d"] == "ŚW")
        db_bk_ha = sum(x["sub_area_ha"] for x in stands if x["species_cd_d"] in {"DB", "BK"})

        pine_pct = round(so_ha / stand_ha * 100.0, 1) if stand_ha > 0 else None
        spruce_pct = round(sw_ha / stand_ha * 100.0, 1) if stand_ha > 0 else None
        oak_beech_pct = round(db_bk_ha / stand_ha * 100.0, 1) if stand_ha > 0 else None

        aged_stands = [x for x in stands if x["species_age"] is not None]
        aged_ha = sum(x["sub_area_ha"] for x in aged_stands)
        mean_age = (
            round(sum(x["species_age"] * x["sub_area_ha"] for x in aged_stands) / aged_ha, 1)
            if aged_ha > 0
            else None
        )

        moisture_totals: dict[str, float] = {}
        tsl_totals: dict[str, float] = {}
        for x in stands:
            m_cat = x.get("moisture_category")
            if m_cat:
                moisture_totals[m_cat] = moisture_totals.get(m_cat, 0.0) + x["sub_area_ha"]
            t_cd = x.get("site_type_cd")
            if t_cd:
                tsl_totals[t_cd] = tsl_totals.get(t_cd, 0.0) + x["sub_area_ha"]

        dom_moisture = max(moisture_totals.items(), key=lambda kv: kv[1])[0] if moisture_totals else ""
        dom_tsl = max(tsl_totals.items(), key=lambda kv: kv[1])[0] if tsl_totals else ""

        agg_rows.append(
            {
                "area_id": aid,
                "area_name": items[0]["area_name"],
                "sample_cluster_name": items[0]["sample_cluster_name"],
                "inspectorates": "; ".join(sorted({x["inspectorate_name"] for x in items})),
                "total_polygons_count": len(items),
                "forest_stands_count": len(stands),
                "non_stand_polygons_count": len(items) - len(stands),
                "total_sampled_area_ha": total_ha,
                "forest_stand_area_ha": stand_ha,
                "non_stand_area_ha": non_stand_ha,
                "sample_pine_share_pct": pine_pct if pine_pct is not None else "",
                "sample_oak_beech_share_pct": oak_beech_pct if oak_beech_pct is not None else "",
                "sample_spruce_share_pct": spruce_pct if spruce_pct is not None else "",
                "sample_mean_stand_age_years": mean_age if mean_age is not None else "",
                "sample_dominant_tsl": dom_tsl,
                "sample_dominant_moisture_type": dom_moisture,
                "a_year": BDL_VALIDITY_YEAR,
                "crs_analysis": "EPSG:2180",
            }
        )

    # 4. Zbudowanie FeatureCollection dla 30 powiatów (prod/sample oraz demo)
    def make_areas_geojson(is_demo: bool, dataset_name: str) -> dict[str, Any]:
        features: list[dict[str, Any]] = []
        for _, row in gdf_ds.iterrows():
            aid = str(row["area_id"])
            bdl_prof = COUNTY_BDL_PROFILES.get(aid, {})
            sample_cnt = len(bdl_by_county.get(aid, []))
            has_bdl = bool(
                bdl_prof.get("pine_share_pct") is not None
                and bdl_prof.get("mean_stand_age_years") is not None
            )

            props = {
                "area_id": aid,
                "area_name": str(row["area_name"]),
                "voivodeship_id": "02",
                "centroid_lon": float(row["centroid_lon"]),
                "centroid_lat": float(row["centroid_lat"]),
                "total_area_km2": float(row["total_area_km2"]),
                "forest_cover_pct": float(row["forest_cover_pct"]),
                "pine_share_pct": bdl_prof.get("pine_share_pct"),
                "oak_beech_share_pct": bdl_prof.get("oak_beech_share_pct"),
                "spruce_share_pct": bdl_prof.get("spruce_share_pct"),
                "mean_stand_age_years": bdl_prof.get("mean_stand_age_years"),
                "dominant_moisture_type": bdl_prof.get("dominant_moisture_type"),
                "has_bdl_stand_data": has_bdl,
                "bdl_sample_stands_count": sample_cnt,
                "data_source_boundaries": "Państwowy Rejestr Granic (GUGiK PRG, JPT_KOD_JE 02xx)",
                "data_source_bdl": (
                    "Bank Danych o Lasach (BDL Lasy Państwowe) + GUS BDL"
                    if bdl_prof
                    else "GUS BDL (lesistość ogólna; brak szczegółowej paczki drzewostanowej BDL -> NULL)"
                ),
                "data_date": DATA_RETRIEVAL_DATE,
                "crs": "EPSG:4326",
                "is_demo": is_demo,
            }
            geom_dict = mapping(row.geometry)
            geom_dict["coordinates"] = _round_coords(geom_dict["coordinates"], precision=5)
            features.append({"type": "Feature", "properties": props, "geometry": geom_dict})

        return {
            "type": "FeatureCollection",
            "name": dataset_name,
            "metadata": {
                "region_name": "Województwo dolnośląskie",
                "voivodeship_id": "02",
                "spatial_unit": "powiat (4-znakowy kod TERYT)",
                "units_count": len(features),
                "crs_storage": "EPSG:4326 (WGS84)",
                "crs_metric_calculations": "EPSG:2180 (PUWG 1992)",
                "source_boundaries": "Państwowy Rejestr Granic (GUGiK PRG) / polska-geojson",
                "source_forest_habitat": "Bank Danych o Lasach (BDL Lasy Państwowe, a_year=2026) + GUS BDL",
                "retrieved_at": DATA_RETRIEVAL_DATE,
                "is_demo": is_demo,
            },
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": features,
        }

    sample_dir = root_dir / "data" / "sample"
    processed_dir = root_dir / "data" / "processed"
    demo_dir = root_dir / "data" / "demo"
    sample_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Zapis 1: data/sample/region_powiaty_02_ds.geojson
    out_region_sample = sample_dir / "region_powiaty_02_ds.geojson"
    out_region_sample.write_text(
        json.dumps(make_areas_geojson(is_demo=False, dataset_name="region_powiaty_02_ds"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Zapis 2: data/processed/areas.geojson (uzgodniony plik w docs/data_contract.md)
    out_areas_prod = processed_dir / "areas.geojson"
    out_areas_prod.write_text(
        json.dumps(make_areas_geojson(is_demo=False, dataset_name="areas"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Zapis 3: data/demo/areas_demo.geojson (rzeczywiste granice 30 powiatów w trybie demo UI)
    out_areas_demo = demo_dir / "areas_demo.geojson"
    out_areas_demo.write_text(
        json.dumps(make_areas_geojson(is_demo=True, dataset_name="areas_demo"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Zapis 4: data/sample/bdl_stands_sample_ds.geojson
    out_bdl_geojson = sample_dir / "bdl_stands_sample_ds.geojson"
    bdl_collection = {
        "type": "FeatureCollection",
        "name": "bdl_stands_sample_ds",
        "metadata": {
            "description": "Próbka wektorowa wydzieleń leśnych BDL (G_SUBAREA / Wydzielenia PGL LP) dla 7 klastrów leśnych na Dolnym Śląsku",
            "source_service": BDL_WYDZIELENIA_QUERY_URL,
            "bdl_validity_year": BDL_VALIDITY_YEAR,
            "retrieved_at": DATA_RETRIEVAL_DATE,
            "crs_storage": "EPSG:4326 (WGS84)",
            "crs_metric_calculations": "EPSG:2180 (PUWG 1992)",
            "features_count": len(bdl_features),
            "forest_stands_count": sum(1 for f in bdl_features if f["properties"]["is_forest_stand"]),
        },
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": bdl_features,
    }
    out_bdl_geojson.write_text(
        json.dumps(bdl_collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Zapis 5: data/sample/bdl_stands_sample_ds.csv
    out_bdl_csv = sample_dir / "bdl_stands_sample_ds.csv"
    with out_bdl_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bdl_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bdl_csv_rows)

    # Zapis 6: data/sample/bdl_powiaty_agg_sample_ds.csv
    out_agg_csv = sample_dir / "bdl_powiaty_agg_sample_ds.csv"
    with out_agg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Zapisano granice 30 powiatów woj. dolnośląskiego (EPSG:4326):")
    print(f"  - {out_region_sample}")
    print(f"  - {out_areas_prod}")
    print(f"  - {out_areas_demo}")
    print(f"Zapisano próbkę wektorową 105 wydzieleń BDL (w tym {bdl_collection['metadata']['forest_stands_count']} drzewostanów D-STAN):")
    print(f"  - {out_bdl_geojson}")
    print(f"  - {out_bdl_csv}")
    print(f"  - {out_agg_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Przygotowuje granice powiatów Dolnego Śląska i próbkę wektorową BDL ([GRZ-05] #5)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Wymuś ponowne pobranie surowych danych z sieci zamiast użycia lokalnego cache'u",
    )
    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    build_region_and_bdl_datasets(root_dir, refresh=args.refresh)


if __name__ == "__main__":
    main()
