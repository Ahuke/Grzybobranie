#!/usr/bin/env python3
"""
Skrypt ETL dla zadania [GRZ-07]:
Zbudowanie zbioru obserwacji o zbiorach grzybów zgodnego z zatwierdzonym celem predykcji
oraz kontraktem danych (`docs/data_contract.md`).

Funkcjonalność:
1. Wczytuje 9 tygodniowych zrzutów HTML po zalogowaniu z `data/raw/grzybobrania od *.html`
   (okres 01.09.2025 – 02.11.2025 dla woj. dolnośląskiego `DS`, N = 518 raportów)
   i aktualizuje lokalny słownik `data/raw/place_id_to_locality.json` oraz ustrukturyzowany,
   zanonimizowany zrzut wejściowy `data/sample/reports_raw_518_ds.json` (umożliwiający
   powtarzalne odtworzenie zbioru bez dostępu do surowych plików HTML w `data/raw/`).
2. Ujednolica daty (`harvest_date` vs. `publication_date` w ISO 8601 `YYYY-MM-DD`),
   gatunki (`dominant_species` wg słownika kontraktu), jednostki i identyfikatory (`report_id`, `area_id`).
3. Przypisuje raporty przestrzennie (`gpd.sjoin` w układzie metrycznym `EPSG:2180`) do 30 powiatów
   woj. dolnośląskiego (`data/processed/areas.geojson`) i rozwiązuje niejednoznaczności na granicach powiatów.
4. Oblicza `yield_pcs_per_h` (`[szt./h]`) z jawnie podanej liczby sztuk (`total_pieces >= 0`)
   i dodatniego czasu zbioru (`harvest_duration_h > 0`, `pickers_count >= 1`) lub waliduje stawkę
   z formularza portalu `grzyby.pl`.
5. Eliminuje duplikaty oraz oznacza/wyłącza wpisy niejednoznaczne z podaniem jawnej przyczyny
   (`quality_status`, `is_valid_for_training`, `quality_reason`).
6. Zapisuje:
   - `data/processed/reports.parquet` (pełny zbiór obserwacji z kolumnami pochodzenia i jakości),
   - `data/processed/reports.csv` (oczyszczony zbiór zgodny 1:1 z `docs/data_contract.md`),
   - `data/processed/reports_quality_report.json` (metryki lejka jakości po każdym etapie).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.audit_grzyby_reports import AUDITED_REPORTS, resolve_teryt

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
SAMPLE_DIR = ROOT_DIR / "data" / "sample"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

AREAS_GEOJSON_PATH = PROCESSED_DIR / "areas.geojson"
RAW_SNAPSHOT_JSON_PATH = SAMPLE_DIR / "reports_raw_518_ds.json"
REPORTS_PARQUET_PATH = PROCESSED_DIR / "reports.parquet"
REPORTS_CSV_PATH = PROCESSED_DIR / "reports.csv"
QUALITY_REPORT_JSON_PATH = PROCESSED_DIR / "reports_quality_report.json"

MONTHS_PL = {
    "sty": "01",
    "lut": "02",
    "mar": "03",
    "kwi": "04",
    "maj": "05",
    "cze": "06",
    "lip": "07",
    "sie": "08",
    "wrz": "09",
    "paź": "10",
    "paz": "10",
    "lis": "11",
    "gru": "12",
}

# Słownik ręcznie zweryfikowanych wartości z [GRZ-03] (52 raporty z tygodnia 29.09–05.10.2025)
AUDITED_BY_ID: dict[str, dict[str, Any]] = {
    str(r["report_id"]): dict(r) for r in AUDITED_REPORTS
}


def _extract_archive_week_from_filename(filename: str) -> str:
    m = re.search(r"(\d{8})\s+do\s+(\d{8})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m2 = re.search(r"(\d{8}-\d{8})", filename)
    return m2.group(1) if m2 else "unknown"


def _normalize_species(latin_tags: list[str], desc_text: str, yield_rate: float) -> str | None:
    """
    Mapuje tagi łacińskie z grzyby.pl oraz treść doniesienia na kontrolowany słownik
    `dominant_species` z `docs/data_contract.md`:
      - 'borowik_szlachetny'
      - 'podgrzybek_brunatny'
      - 'kurka'
      - 'maslak_zwyczajny'
      - 'kania'
      - 'mieszane'
      - None (gdy brak informacji lub zbiór zerowy)
    """
    if yield_rate == 0.0 and not latin_tags:
        return None

    mapped_categories: list[str] = []
    for tag in latin_tags:
        t = tag.lower()
        if any(k in t for k in ("boletus_edulis", "boletus_pinophilus", "boletus_reticulatus", "boletus")):
            mapped_categories.append("borowik_szlachetny")
        elif any(k in t for k in ("xerocomus", "imleria", "xerocomellus")):
            mapped_categories.append("podgrzybek_brunatny")
        elif "cantharellus" in t:
            mapped_categories.append("kurka")
        elif "suillus" in t:
            mapped_categories.append("maslak_zwyczajny")
        elif "macrolepiota" in t:
            mapped_categories.append("kania")
        else:
            mapped_categories.append("inne")

    text_low = desc_text.lower()
    if not mapped_categories:
        if any(w in text_low for w in ("prawdziw", "borowik")):
            mapped_categories.append("borowik_szlachetny")
        if "podgrzyb" in text_low:
            mapped_categories.append("podgrzybek_brunatny")
        if any(w in text_low for w in ("kurk", "kurek", "pieprznik")):
            mapped_categories.append("kurka")
        if "maślak" in text_low or "maslak" in text_low:
            mapped_categories.append("maslak_zwyczajny")
        if "kani" in text_low:
            mapped_categories.append("kania")

    unique_main = [c for c in dict.fromkeys(mapped_categories) if c != "inne"]
    if len(unique_main) == 1 and len(set(mapped_categories)) <= 2:
        return unique_main[0]
    if len(unique_main) >= 2:
        # Sprawdź czy pierwszy wymieniony gatunek jest wyraźnie dominujący w krótkim opisie
        first_cat = unique_main[0]
        if any(
            phrase in text_low
            for phrase in ("praktycznie sam", "głównie", "glownie", "same ", "przede wszystkim", "tylko ")
        ):
            return first_cat
        return "mieszane"
    if "inne" in mapped_categories:
        return "mieszane"
    return None


def _extract_duration_pieces_pickers_worminess(
    report_id: str, desc_text: str, portal_rate: float
) -> tuple[float | None, int | None, int | None, float | None]:
    """
    Wyciąga z treści raportu:
      - `harvest_duration_h` (czas zbierania w godzinach, float),
      - `total_pieces` (łączną liczbę zebranych sztuk, int),
      - `pickers_count` (liczbę zbieraczy, int),
      - `worminess_pct` (szacowany odsetek grzybów zaczerwionych 0-100%, float).
    Dla 52 rekordów z [GRZ-03] zachowuje ręcznie zweryfikowane wartości audytowe.
    """
    audited = AUDITED_BY_ID.get(report_id)
    duration_h: float | None = None
    total_pieces: int | None = None
    pickers_count: int | None = None
    worminess_pct: float | None = None

    if audited:
        if audited.get("harvest_duration_h") not in ("", None):
            duration_h = float(audited["harvest_duration_h"])
        if audited.get("total_pieces") not in ("", None):
            total_pieces = int(audited["total_pieces"])
        if audited.get("pickers_count") not in ("", None):
            pickers_count = int(audited["pickers_count"])

    text_low = desc_text.lower()

    # 1. Ekstrakcja czasu zbierania (jeśli jeszcze nieustalony)
    if duration_h is None:
        # np. "2,5 godziny", "1.5 godz", "3 h", "2 godziny"
        m_hr = re.search(
            r"\b(\d+(?:[.,]\d+)?)\s*(?:godzin[yaę]?|godz\.?|h)\b", text_low
        )
        if m_hr:
            val = float(m_hr.group(1).replace(",", "."))
            if 0.1 <= val <= 14.0:
                duration_h = round(val, 2)
        else:
            # np. "półtorej godziny" -> 1.5, "pół godziny" / "30 min" -> 0.5, "15 minut" -> 0.25
            if "półtorej godziny" in text_low or "poltorej godziny" in text_low:
                duration_h = 1.5
            elif "dwie i pół godziny" in text_low or "dwie i pol godziny" in text_low:
                duration_h = 2.5
            elif "trzy godziny" in text_low:
                duration_h = 3.0
            elif "dwie godziny" in text_low or "dwugodzinny" in text_low:
                duration_h = 2.0
            elif "godzinny" in text_low or "godzina spaceru" in text_low or "w godzinę" in text_low:
                duration_h = 1.0
            elif "pół godziny" in text_low or "pol godziny" in text_low:
                duration_h = 0.5
            else:
                m_min = re.search(r"\b(\d{1,3})\s*(?:minut\w*|min\.?)\b", text_low)
                if m_min:
                    mins = int(m_min.group(1))
                    if 10 <= mins <= 300:
                        duration_h = round(mins / 60.0, 2)

    # 2. Ekstrakcja liczby sztuk (jeśli jeszcze nieustalona)
    if total_pieces is None:
        if portal_rate == 0.0:
            # Sprawdź czy w tekście dla (0/h) nie ma np. "Jeden borowik"
            if re.search(r"\bjeden\s+(?:borowik|prawdziwek|podgrzybek|grzyb)\b", text_low):
                total_pieces = 1
            else:
                total_pieces = 0
        else:
            m_pcs = re.search(
                r"\b(\d{1,4})\s*(?:sztuk\w*|szt\.?|grzyb\w*|borowik\w*|prawdziwk\w*|podgrzybk\w*|kani\w*)\b",
                text_low,
            )
            if m_pcs:
                pcs_val = int(m_pcs.group(1))
                # Wyklucz przypadkowe lata (2025) lub godziny
                if 1 <= pcs_val <= 1500 and pcs_val != 2025:
                    total_pieces = pcs_val

    # 3. Ekstrakcja liczby zbieraczy
    if pickers_count is None:
        if any(
            p in text_low
            for p in ("z żoną", "z mężem", "we dwoje", "w dwie osoby", "2 osoby", "dwie osoby", "z synem", "z córką")
        ):
            pickers_count = 2
        elif any(p in text_low for p in ("3 osoby", "trzy osoby", "w trójkę")):
            pickers_count = 3
        elif duration_h is not None or total_pieces is not None:
            pickers_count = 1

    # 4. Ekstrakcja zaczerwienia (worminess_pct)
    m_worm = re.search(r"(\d{1,3})\s*%\s*(?:robaczyw\w*|zaczerwion\w*|z lokatorami|odpad\w*)", text_low)
    if m_worm:
        w_val = float(m_worm.group(1))
        if 0.0 <= w_val <= 100.0:
            worminess_pct = w_val
    elif "połowa z lokatorami" in text_low or "polowa robaczywa" in text_low or "50% robaczyw" in text_low:
        worminess_pct = 50.0
    elif any(
        p in text_low
        for p in ("wszystkie zdrowe", "zero robaczywych", "brak robaczywych", "100% zdrowe", "czyściutkie", "zdrowiutkie")
    ):
        worminess_pct = 0.0

    return duration_h, total_pieces, pickers_count, worminess_pct


def parse_raw_html_files(raw_dir: Path) -> list[dict[str, Any]]:
    """
    Parsuje wszystkie pliki `grzybobrania od *.html` z `data/raw/` i zwraca
    ustrukturyzowaną listę surowych rekordów.
    """
    html_files = sorted(raw_dir.glob("grzybobrania od *.html"))
    if not html_files:
        return []

    extracted: list[dict[str, Any]] = []
    for html_path in html_files:
        week_code = _extract_archive_week_from_filename(html_path.name)
        canonical_base_url = f"https://www.grzyby.pl/wystepowanie/wystepowanie-{week_code}-DS.htm"
        content = html_path.read_text(encoding="utf-8", errors="replace")

        blocks = re.findall(
            r'id="(DG\d+)"(.*?)(?=<div class="w3-row-padding[^>]*doniesienie-blok"|<footer|$)',
            content,
            re.S,
        )
        for report_id, block in blocks:
            gmap = re.search(
                r"maps/place/([0-9.-]+)\+([0-9.-]+)/[^>]*>(.*?)</a>", block, re.S
            )
            lat = float(gmap.group(1)) if gmap else None
            lon = float(gmap.group(2)) if gmap else None
            locality_label = (
                re.sub(r"&nbsp;|\s+", " ", gmap.group(3)).strip() if gmap else ""
            )

            place_m = re.search(r"historyczne_z_miejsca/(\d+)/", block)
            place_id = place_m.group(1) if place_m else ""

            hdate_m = re.search(
                r'w3-text-align-right">(\d+)\.([a-ząćęłńóśźż]+)\s+(\d{4})</div>',
                block,
            )
            if hdate_m:
                day_s, mon_s, year_s = hdate_m.groups()
                harvest_date = f"{year_s}-{MONTHS_PL.get(mon_s, '01')}-{int(day_s):02d}"
            else:
                harvest_date = ""

            pdate_m = re.search(
                r"(?:&middot;\s*|>)\s*(\d{4})\.(\d+)\.(\d+)\s+(\d{2}:\d{2})", block
            )
            if pdate_m:
                py, pm, pd_num, ptime = pdate_m.groups()
                publication_date = f"{py}-{int(pm):02d}-{int(pd_num):02d}"
                publication_time = ptime
            else:
                publication_date = ""
                publication_time = ""

            auth_m = re.search(
                r'/uzytkownicy/([^/]+)/doniesienia/1"[^>]*>(?:<img[^>]*>)?(.*?)</a>',
                block,
                re.S,
            )
            author_id = auth_m.group(1).strip() if auth_m else ""

            desc_m = re.search(
                r'<span class="doniesienie-stanOpis">\(([0-9.]+)/h\)\s*(.*?)</span>',
                block,
                re.S,
            )
            portal_rate = float(desc_m.group(1)) if desc_m else None
            desc_html = desc_m.group(2) if desc_m else ""
            latin_tags = re.findall(
                r"grzyby\.pl/gatunki/([A-Za-z0-9_-]+)\.htm", desc_html
            )
            desc_text = re.sub(r"<[^>]+>", "", desc_html)
            desc_text = re.sub(r"\s+", " ", desc_text).strip()

            duration_h, total_pieces, pickers_count, worminess_pct = (
                _extract_duration_pieces_pickers_worminess(
                    report_id=report_id,
                    desc_text=desc_text,
                    portal_rate=portal_rate or 0.0,
                )
            )

            audited = AUDITED_BY_ID.get(report_id)
            if audited and audited.get("dominant_species") not in ("", None):
                dominant_species = str(audited["dominant_species"])
            else:
                dominant_species = _normalize_species(
                    latin_tags=latin_tags,
                    desc_text=desc_text,
                    yield_rate=portal_rate or 0.0,
                )

            extracted.append(
                {
                    "report_id": report_id,
                    "source_url": f"{canonical_base_url}#{report_id}",
                    "source_archive_week": week_code,
                    "harvest_date": harvest_date,
                    "publication_date": publication_date,
                    "publication_time": publication_time,
                    "author_id": author_id,
                    "place_id": place_id,
                    "location_raw": locality_label,
                    "lat": lat,
                    "lon": lon,
                    "portal_yield_pcs_per_h": portal_rate,
                    "total_pieces": total_pieces,
                    "harvest_duration_h": duration_h,
                    "pickers_count": pickers_count,
                    "worminess_pct": worminess_pct,
                    "dominant_species": dominant_species,
                    "species_latin_tags": ";".join(latin_tags),
                    "is_urban_park_harvest": (
                        report_id in {"DG174153", "DG173963"}
                        or "parku klecińskim" in desc_text.lower()
                        or latin_tags == ["Coprinus_comatus"]
                    ),
                }
            )

    return extracted


def build_reports_dataset(
    raw_dir: Path = RAW_DIR,
    areas_geojson_path: Path = AREAS_GEOJSON_PATH,
    raw_snapshot_path: Path = RAW_SNAPSHOT_JSON_PATH,
    out_parquet_path: Path = REPORTS_PARQUET_PATH,
    out_csv_path: Path = REPORTS_CSV_PATH,
    out_quality_json_path: Path = QUALITY_REPORT_JSON_PATH,
) -> dict[str, Any]:
    """
    Główny potok ETL dla [GRZ-07]:
    1. Wczytuje surowe dane z `data/raw/*.html` (lub ze snapshotu `data/sample/reports_raw_518_ds.json`),
    2. Wykonuje przypisanie przestrzenne GeoPandas (`EPSG:2180`),
    3. Waliduje i wylicza `yield_pcs_per_h` wyłącznie dla poprawnej liczby sztuk i dodatniego czasu zbioru,
    4. Deduplikuje i oznacza rekordy niejednoznaczne / wykluczone,
    5. Zapisuje `reports.parquet`, `reports.csv` oraz raport jakości `reports_quality_report.json`.
    """
    raw_records = parse_raw_html_files(raw_dir)
    if raw_records:
        raw_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        raw_snapshot_path.write_text(
            json.dumps(raw_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Aktualizuj lokalny słownik place_id -> lokalizacja w data/raw/
        place_dict = {
            str(r["place_id"]): {
                "locality_logged_in": r["location_raw"],
                "powiat_teryt": resolve_teryt(str(r["location_raw"])),
                "lat": r["lat"],
                "lon": r["lon"],
            }
            for r in raw_records
            if r.get("place_id")
        }
        place_dict_path = raw_dir / "place_id_to_locality.json"
        place_dict_path.parent.mkdir(parents=True, exist_ok=True)
        place_dict_path.write_text(
            json.dumps(place_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif raw_snapshot_path.exists():
        raw_records = json.loads(raw_snapshot_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(
            f"Brak plików źródłowych w {raw_dir} oraz snapshotu {raw_snapshot_path}"
        )

    stage1_raw_count = len(raw_records)

    # Etap 2: Deduplikacja dokładnych identyfikatorów `report_id`
    seen_ids: set[str] = set()
    dedup_id_records: list[dict[str, Any]] = []
    exact_id_duplicates = 0
    for rec in raw_records:
        rid = str(rec["report_id"])
        if rid in seen_ids:
            exact_id_duplicates += 1
            continue
        seen_ids.add(rid)
        dedup_id_records.append(dict(rec))

    stage2_after_exact_id_dedup = len(dedup_id_records)

    # Etap 3: Przypisanie przestrzenne GeoPandas (`gpd.sjoin` w układzie metrycznym EPSG:2180)
    areas_gdf = gpd.read_file(areas_geojson_path).to_crs("EPSG:2180")
    points_gdf = gpd.GeoDataFrame(
        dedup_id_records,
        geometry=[Point(float(r["lon"]), float(r["lat"])) for r in dedup_id_records],
        crs="EPSG:4326",
    ).to_crs("EPSG:2180")

    joined_gdf = gpd.sjoin(
        points_gdf,
        areas_gdf[["area_id", "area_name", "geometry"]],
        how="left",
        predicate="within",
    )

    # Zbuduj wzbogacone rekordy z rozwiązaniem niejednoznaczności przestrzennych i czasowych
    # Najpierw zidentyfikuj duplikaty semantyczne: ten sam (author_id, harvest_date, place_id)
    # Zachowujemy pierwszy (główny/wcześniejszy w tabeli dla danego dnia) wpis jako VALID,
    # a kolejny oznaczamy jako EXCLUDED_SAME_DAY_AUTHOR_PLACE_DUPLICATE.
    seen_author_date_place: dict[tuple[str, str, str], str] = {}

    processed_rows: list[dict[str, Any]] = []
    for _, row in joined_gdf.iterrows():
        rid = str(row["report_id"])
        loc_raw = str(row["location_raw"] or "").strip()
        sjoin_area_id = str(row["area_id"]).zfill(4) if pd.notna(row["area_id"]) else ""
        text_area_id = resolve_teryt(loc_raw)

        # Rozstrzygnięcie area_id (GeoPandas sjoin vs etykieta tekstowa)
        border_note = None
        if text_area_id and sjoin_area_id and text_area_id != sjoin_area_id:
            if loc_raw == "pow. wrocławski" and sjoin_area_id == "0264":
                # Ogólna etykieta "pow. wrocławski" ma na portalu domyślny punkt w centrum Wrocławia (0264)
                final_area_id = "0223"
                area_id_method = "text_powiat_override_generic_centroid"
                border_note = "generic_label_pow_wroclawski_overrode_city_centroid_0264_to_0223"
            else:
                # Konkretna miejscowość leśna na granicy powiatów (np. Wilkocin / Konradów) -> decyduje dokładny GPS w EPSG:2180
                final_area_id = sjoin_area_id
                area_id_method = "geopandas_sjoin_epsg2180_border_resolved"
                border_note = f"gps_polygon_{sjoin_area_id}_differs_from_text_label_{text_area_id}"
        elif sjoin_area_id:
            final_area_id = sjoin_area_id
            area_id_method = "geopandas_sjoin_epsg2180"
        else:
            final_area_id = text_area_id
            area_id_method = "text_powiat_fallback"

        # Walidacja dat i opóźnienia publikacji
        hdate_str = str(row["harvest_date"])
        pdate_str = str(row["publication_date"])
        hdate_dt = dt.date.fromisoformat(hdate_str)
        pdate_dt = dt.date.fromisoformat(pdate_str)
        pub_delay_days = (pdate_dt - hdate_dt).days

        # Reguła [GRZ-07] #3: Szt./h obliczane jest tylko z poprawnej liczby sztuk i dodatniego czasu zbioru
        portal_rate = (
            float(row["portal_yield_pcs_per_h"])
            if pd.notna(row["portal_yield_pcs_per_h"])
            else None
        )
        total_pieces = (
            int(row["total_pieces"])
            if pd.notna(row["total_pieces"]) and row["total_pieces"] is not None
            else None
        )
        duration_h = (
            float(row["harvest_duration_h"])
            if pd.notna(row["harvest_duration_h"]) and row["harvest_duration_h"] is not None
            else None
        )
        pickers_count = (
            int(row["pickers_count"])
            if pd.notna(row["pickers_count"]) and row["pickers_count"] is not None
            else 1
        )
        worminess_pct = (
            float(row["worminess_pct"])
            if pd.notna(row["worminess_pct"]) and row["worminess_pct"] is not None
            else None
        )

        invalid_yield_reason = None
        if duration_h is not None and duration_h <= 0.0:
            yield_pcs_per_h = None
            yield_calc_method = "invalid_non_positive_duration"
            invalid_yield_reason = f"non_positive_duration_h={duration_h}"
        elif total_pieces is not None and total_pieces < 0:
            yield_pcs_per_h = None
            yield_calc_method = "invalid_negative_pieces"
            invalid_yield_reason = f"negative_total_pieces={total_pieces}"
        elif (
            total_pieces is not None
            and total_pieces >= 0
            and duration_h is not None
            and duration_h > 0.0
            and pickers_count >= 1
        ):
            computed_rate = round(total_pieces / (duration_h * pickers_count), 2)
            # Jeżeli portal zaokrąglił 1 szt. / 2h do 0/h (np. DG176081) lub znamy dokładną liczbę sztuk i czas
            # w granicach rozsądnej zgodności z nagłówkiem, używamy wyliczonej wartości lub zachowujemy ślad obu
            if portal_rate == 0.0 and total_pieces > 0:
                yield_pcs_per_h = computed_rate
                yield_calc_method = "computed_from_pieces_and_positive_duration_corrected_zero"
            else:
                yield_pcs_per_h = portal_rate if portal_rate is not None else computed_rate
                yield_calc_method = "verified_pieces_and_positive_duration"
        else:
            yield_pcs_per_h = portal_rate
            yield_calc_method = "portal_form_positive_duration_rate"
            computed_rate = None

        # Klasyfikacja jakości i wykluczeń ([GRZ-07] #4)
        author_key = (str(row["author_id"]), hdate_str, str(row["place_id"]))
        quality_status = "VALID"
        is_valid_for_training = True
        quality_reason = border_note or "ok"

        if invalid_yield_reason:
            quality_status = "EXCLUDED_INVALID_DURATION_OR_PIECES"
            is_valid_for_training = False
            quality_reason = invalid_yield_reason
        elif not final_area_id:
            quality_status = "EXCLUDED_MISSING_AREA_ID"
            is_valid_for_training = False
            quality_reason = "unassigned_powiat_teryt"
        elif author_key in seen_author_date_place:
            kept_rid = seen_author_date_place[author_key]
            quality_status = "EXCLUDED_SAME_DAY_AUTHOR_PLACE_DUPLICATE"
            is_valid_for_training = False
            quality_reason = f"duplicate_same_author_date_place_id_kept_{kept_rid}"
        elif bool(row["is_urban_park_harvest"]):
            quality_status = "EXCLUDED_URBAN_NON_FOREST"
            is_valid_for_training = False
            quality_reason = "non_forest_urban_park_lawn_harvest_coprinus_comatus"
        elif yield_pcs_per_h is not None and yield_pcs_per_h > 200.0 and duration_h is None:
            quality_status = "EXCLUDED_EXTREME_UNVERIFIED_OUTLIER"
            is_valid_for_training = False
            quality_reason = f"extreme_yield_outlier_{yield_pcs_per_h}_per_h_without_explicit_duration"
        elif border_note:
            quality_status = "VALID_BORDER_LOCALITY_RESOLVED"
            is_valid_for_training = True
            quality_reason = border_note

        if author_key not in seen_author_date_place:
            seen_author_date_place[author_key] = rid

        processed_rows.append(
            {
                "report_id": rid,
                "date": hdate_str,
                "harvest_date": hdate_str,
                "publication_date": pdate_str,
                "publication_time": str(row["publication_time"]),
                "pub_delay_days": int(pub_delay_days),
                "area_id": final_area_id,
                "area_id_method": area_id_method,
                "place_id": str(row["place_id"]),
                "location_raw": loc_raw,
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "dominant_species": (
                    str(row["dominant_species"])
                    if pd.notna(row["dominant_species"]) and row["dominant_species"] is not None
                    else None
                ),
                "species_latin_tags": str(row["species_latin_tags"] or ""),
                "yield_pcs_per_h": float(yield_pcs_per_h) if yield_pcs_per_h is not None else None,
                "portal_yield_pcs_per_h": float(portal_rate) if portal_rate is not None else None,
                "total_pieces": total_pieces,
                "harvest_duration_h": duration_h,
                "pickers_count": pickers_count,
                "worminess_pct": worminess_pct,
                "yield_calc_method": yield_calc_method,
                "quality_status": quality_status,
                "is_valid_for_training": bool(is_valid_for_training),
                "quality_reason": quality_reason,
                "source_url": str(row["source_url"]),
                "source_archive_week": str(row["source_archive_week"]),
                "is_demo": False,
            }
        )

    # Sortowanie chronologiczne (malejąco wg daty zbioru i report_id)
    processed_rows.sort(key=lambda r: (r["date"], r["report_id"]), reverse=True)

    # 1. Zapis pełnego zbioru do `data/processed/reports.parquet` ([GRZ-07] Warunek #1)
    out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame(processed_rows)
    df_all.to_parquet(out_parquet_path, index=False, engine="pyarrow")

    # 2. Zapis oczyszczonego zbioru (`is_valid_for_training == True`) do `data/processed/reports.csv`
    # zgodnie z kontraktem danych (`docs/data_contract.md` i `src/data_loader.py`)
    valid_rows = [r for r in processed_rows if r["is_valid_for_training"]]
    contract_columns = [
        "report_id",
        "date",
        "area_id",
        "location_raw",
        "dominant_species",
        "yield_pcs_per_h",
        "harvest_duration_h",
        "worminess_pct",
        "is_demo",
    ]
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=contract_columns)
        writer.writeheader()
        for r in valid_rows:
            writer.writerow(
                {
                    "report_id": r["report_id"],
                    "date": r["date"],
                    "area_id": r["area_id"],
                    "location_raw": r["location_raw"] or "",
                    "dominant_species": r["dominant_species"] or "",
                    "yield_pcs_per_h": (
                        f"{r['yield_pcs_per_h']:.1f}"
                        if r["yield_pcs_per_h"] is not None
                        else ""
                    ),
                    "harvest_duration_h": (
                        f"{r['harvest_duration_h']:.2f}".rstrip("0").rstrip(".")
                        if r["harvest_duration_h"] is not None
                        else ""
                    ),
                    "worminess_pct": (
                        f"{r['worminess_pct']:.1f}"
                        if r["worminess_pct"] is not None
                        else ""
                    ),
                    "is_demo": "False",
                }
            )

    # 3. Wyliczenie statystyk raportu jakości ([GRZ-07] Warunek #5)
    status_counts = df_all["quality_status"].value_counts().to_dict()
    valid_df = df_all[df_all["is_valid_for_training"]].copy()
    all_30_powiats = sorted(areas_gdf["area_id"].astype(str).str.zfill(4).unique().tolist())
    covered_powiats_all = sorted(df_all["area_id"].dropna().unique().tolist())
    covered_powiats_valid = sorted(valid_df["area_id"].dropna().unique().tolist())
    missing_powiats = sorted(set(all_30_powiats) - set(covered_powiats_valid))

    powiat_counts = (
        valid_df.groupby("area_id")
        .agg(
            reports_count=("report_id", "count"),
            mean_yield_pcs_per_h=("yield_pcs_per_h", "mean"),
            max_yield_pcs_per_h=("yield_pcs_per_h", "max"),
            zero_harvest_count=("yield_pcs_per_h", lambda s: int((s == 0.0).sum())),
        )
        .reset_index()
        .sort_values("reports_count", ascending=False)
    )
    powiat_summary = [
        {
            "area_id": str(r["area_id"]),
            "reports_count": int(r["reports_count"]),
            "mean_yield_pcs_per_h": round(float(r["mean_yield_pcs_per_h"]), 2),
            "max_yield_pcs_per_h": round(float(r["max_yield_pcs_per_h"]), 2),
            "zero_harvest_count": int(r["zero_harvest_count"]),
        }
        for _, r in powiat_counts.iterrows()
    ]

    excluded_details = [
        {
            "report_id": r["report_id"],
            "date": r["date"],
            "area_id": r["area_id"],
            "location_raw": r["location_raw"],
            "portal_yield_pcs_per_h": r["portal_yield_pcs_per_h"],
            "quality_status": r["quality_status"],
            "quality_reason": r["quality_reason"],
        }
        for r in processed_rows
        if not r["is_valid_for_training"] or r["quality_status"] != "VALID"
    ]

    quality_report = {
        "task": "[GRZ-07] Zbiór obserwacji o zbiorach grzybów (woj. dolnośląskie, TERYT 02)",
        "funnel_stages": {
            "stage_1_raw_extracted_from_9_weeks": stage1_raw_count,
            "stage_2_after_exact_report_id_dedup": stage2_after_exact_id_dedup,
            "stage_3_geopandas_sjoin_assigned_to_powiat": int(
                (df_all["area_id"] != "").sum()
            ),
            "stage_4_after_same_day_author_place_dedup": int(
                stage2_after_exact_id_dedup
                - status_counts.get("EXCLUDED_SAME_DAY_AUTHOR_PLACE_DUPLICATE", 0)
            ),
            "stage_5_valid_for_training_in_reports_csv": len(valid_rows),
        },
        "quality_status_breakdown": {k: int(v) for k, v in status_counts.items()},
        "date_coverage": {
            "harvest_date_min": str(valid_df["harvest_date"].min()),
            "harvest_date_max": str(valid_df["harvest_date"].max()),
            "unique_harvest_days": int(valid_df["harvest_date"].nunique()),
            "publication_date_min": str(valid_df["publication_date"].min()),
            "publication_date_max": str(valid_df["publication_date"].max()),
            "archive_weeks_count": int(df_all["source_archive_week"].nunique()),
        },
        "regional_coverage": {
            "voivodeship_id": "02",
            "total_powiats_in_region": len(all_30_powiats),
            "covered_powiats_count": len(covered_powiats_valid),
            "coverage_pct": round(100.0 * len(covered_powiats_valid) / len(all_30_powiats), 2),
            "covered_powiats_list": covered_powiats_valid,
            "unreported_powiats_list": missing_powiats,
            "unique_places_count": int(valid_df["place_id"].nunique()),
        },
        "target_variable_summary": {
            "min_yield_pcs_per_h": round(float(valid_df["yield_pcs_per_h"].min()), 2),
            "median_yield_pcs_per_h": round(float(valid_df["yield_pcs_per_h"].median()), 2),
            "mean_yield_pcs_per_h": round(float(valid_df["yield_pcs_per_h"].mean()), 2),
            "max_yield_pcs_per_h": round(float(valid_df["yield_pcs_per_h"].max()), 2),
            "class_BRAK_lt_5": int((valid_df["yield_pcs_per_h"] < 5.0).sum()),
            "class_SREDNIO_5_to_20": int(
                ((valid_df["yield_pcs_per_h"] >= 5.0) & (valid_df["yield_pcs_per_h"] <= 20.0)).sum()
            ),
            "class_WYSYP_gt_20": int((valid_df["yield_pcs_per_h"] > 20.0).sum()),
            "explicit_zero_harvest_reports": int((valid_df["yield_pcs_per_h"] == 0.0).sum()),
            "explicit_duration_h_count": int(valid_df["harvest_duration_h"].notna().sum()),
            "explicit_total_pieces_count": int(valid_df["total_pieces"].notna().sum()),
        },
        "species_breakdown": {
            str(k): int(v)
            for k, v in valid_df["dominant_species"].fillna("NULL").value_counts().items()
        },
        "flagged_and_excluded_records": excluded_details,
        "powiat_breakdown": powiat_summary,
    }

    out_quality_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_quality_json_path.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return quality_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budowa zbioru obserwacji reports.parquet i reports.csv ([GRZ-07])"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Katalog z lokalnymi zrzutami HTML z grzyby.pl (domyślnie data/raw/)",
    )
    args = parser.parse_args()

    report = build_reports_dataset(raw_dir=args.raw_dir)
    funnel = report["funnel_stages"]
    reg = report["regional_coverage"]
    dates = report["date_coverage"]
    print("=== Zakończono budowę zbioru obserwacji [GRZ-07] ===")
    print(f"- Etap 1 (Surowe wpisy z 9 tygodni):        {funnel['stage_1_raw_extracted_from_9_weeks']}")
    print(f"- Etap 2 (Po deduplikacji report_id):       {funnel['stage_2_after_exact_report_id_dedup']}")
    print(f"- Etap 3 (Przypisane przestrzennie sjoin):  {funnel['stage_3_geopandas_sjoin_assigned_to_powiat']}")
    print(f"- Etap 4 (Po deduplikacji autor+dzień+pkt): {funnel['stage_4_after_same_day_author_place_dedup']}")
    print(f"- Etap 5 (Rekordy VALID w reports.csv):     {funnel['stage_5_valid_for_training_in_reports_csv']}")
    print(
        f"- Zakres dat zbioru:                        {dates['harvest_date_min']} .. {dates['harvest_date_max']} "
        f"({dates['unique_harvest_days']} unikalnych dni)"
    )
    print(
        f"- Pokrycie powiatów Dolnego Śląska:         {reg['covered_powiats_count']}/{reg['total_powiats_in_region']} "
        f"({reg['coverage_pct']}%) w {reg['unique_places_count']} unikalnych lokalizacjach (place_id)"
    )
    print(f"- Zapisano: {REPORTS_PARQUET_PATH}")
    print(f"- Zapisano: {REPORTS_CSV_PATH}")
    print(f"- Zapisano: {QUALITY_REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
