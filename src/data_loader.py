"""
Moduł wczytywania i walidacji danych zgodny z kontraktem danych (docs/data_contract.md).

Zapewnia:
- ścisłą separację środowiska demonstracyjnego (`data/demo/`, `is_demo = True`)
  od środowiska produkcyjnego (`data/processed/`, `is_demo = False`),
- wczytywanie pełnego zbioru rzeczywistych raportów i pokrycia danych dla widoku
  eksploracyjnego w Streamlit ([GRZ-12]: `areas.geojson`, `reports.parquet`,
  `reports_spatial.parquet`, `forest_features.parquet`, `weather.csv`, `features.parquet`),
- zachowanie 4-znakowego kodu TERYT (`area_id` / `region_id`, np. `"0201"`),
- poprawne rozróżnienie pomiaru zerowego (`0.0`) od braku informacji (`NULL` / `None`),
  w tym pokazywanie powiatów z `0` raportów jako `BRAK_DANYCH (NULL)`, a nie `0.0 szt./h`,
- weryfikację zgodności pojedynczych rekordów z tabelami źródłowymi ([GRZ-12]),
- dostęp do próbek audytowych ze Sprintu 1 (`data/sample/`, w tym granic regionu i BDL [GRZ-05]),
- odporność na brak danych BDL (brak plików lub atrybutów BDL nie blokuje wczytania granic regionu).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEMO_DIR, PROCESSED_DIR, SAMPLE_DIR
from src.geo_utils import haversine_km


@dataclass(frozen=True)
class DatasetBundle:
    """Komplet danych wczytanych dla wybranego środowiska kontraktu danych."""

    env_mode: str  # "demo" lub "prod"
    is_fallback_to_demo: bool
    status_message: str
    areas_geojson: dict[str, Any]
    areas_by_id: dict[str, dict[str, Any]]
    predictions: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    weather: list[dict[str, Any]]


@dataclass(frozen=True)
class RealObservationsBundle:
    """
    Komplet rzeczywistych danych obserwacyjnych, przestrzennych, pogodowych i cech
    wczytanych z `data/processed/` na potrzeby eksploracji w Streamlit ([GRZ-12]).
    """

    status_message: str
    areas_geojson: dict[str, Any]
    areas_by_id: dict[str, dict[str, Any]]
    reports_df: pd.DataFrame
    reports_spatial_df: pd.DataFrame
    forest_features_df: pd.DataFrame
    features_df: pd.DataFrame
    weather_rows: list[dict[str, Any]]
    min_date: str
    max_date: str


def _parse_optional_float(value: str | float | int | None) -> float | None:
    """Konwertuje wartość na float, zachowując NULL (pusty ciąg -> None) i 0.0 jako 0.0."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"null", "none", "nan"}:
        return None
    return float(text)


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "tak"}


def _normalize_area_id(raw_id: Any) -> str:
    """Wymusza 4-znakowy kod TERYT z wiodącym zerem (np. '0201')."""
    text = str(raw_id or "").strip()
    return text.zfill(4) if text else ""


def resolve_environment_paths(requested_mode: str = "demo") -> tuple[str, bool, dict[str, Path]]:
    """
    Zwraca faktyczny tryb ('demo' lub 'prod'), flagę fallbacku oraz słownik ścieżek
    do 4 plików kontraktu danych (`areas`, `reports`, `weather`, `predictions`).
    """
    demo_paths = {
        "areas": DEMO_DIR / "areas_demo.geojson",
        "reports": DEMO_DIR / "reports_demo.csv",
        "weather": DEMO_DIR / "weather_demo.csv",
        "predictions": DEMO_DIR / "predictions_demo.csv",
    }
    prod_paths = {
        "areas": PROCESSED_DIR / "areas.geojson",
        "reports": PROCESSED_DIR / "reports.csv",
        "weather": PROCESSED_DIR / "weather.csv",
        "predictions": PROCESSED_DIR / "predictions.csv",
    }

    if requested_mode == "prod":
        if all(p.exists() for p in prod_paths.values()):
            return "prod", False, prod_paths
        return "demo", True, demo_paths

    return "demo", False, demo_paths


def load_areas_geojson(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """
    Wczytuje plik GeoJSON z granicami powiatów i cechami BDL.
    Gwarantuje poprawne działanie nawet wtedy, gdy atrybuty BDL są równe NULL (None).
    """
    if not path.exists():
        raise FileNotFoundError(f"Brak wymaganego pliku obszarów GeoJSON: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    areas_by_id: dict[str, dict[str, Any]] = {}
    for feature in data.get("features", []):
        props = feature.setdefault("properties", {})
        area_id = _normalize_area_id(
            props.get("area_id") or props.get("region_id") or props.get("JPT_KOD_JE")
        )
        props["area_id"] = area_id
        props["region_id"] = area_id
        # Upewnij się, że opcjonalne pola BDL zawsze istnieją w słowniku (jako wartość lub None)
        for optional_bdl_field in (
            "forest_cover_pct",
            "pine_share_pct",
            "oak_beech_share_pct",
            "spruce_share_pct",
            "mean_stand_age_years",
            "dominant_moisture_type",
        ):
            props.setdefault(optional_bdl_field, None)
        areas_by_id[area_id] = props

    return data, areas_by_id


def load_predictions_csv(path: Path) -> list[dict[str, Any]]:
    """Wczytuje plik predykcji zgodnie z kontraktem danych."""
    if not path.exists():
        raise FileNotFoundError(f"Brak wymaganego pliku predykcji: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "target_date": str(row.get("target_date", "")).strip(),
                    "area_id": _normalize_area_id(row.get("area_id")),
                    "species_filter": str(row.get("species_filter", "all")).strip(),
                    "predicted_yield_pcs_per_h": _parse_optional_float(
                        row.get("predicted_yield_pcs_per_h")
                    ),
                    "abundance_class": str(row.get("abundance_class", "")).strip(),
                    "confidence_score": _parse_optional_float(row.get("confidence_score")),
                    "habitat_match_note": (
                        str(row["habitat_match_note"]).strip()
                        if row.get("habitat_match_note") not in (None, "")
                        else None
                    ),
                    "generated_at": str(row.get("generated_at", "")).strip(),
                    "model_version": str(row.get("model_version", "")).strip(),
                    "is_demo": _parse_bool(row.get("is_demo")),
                }
            )
    return rows


def load_reports_csv(path: Path) -> list[dict[str, Any]]:
    """Wczytuje plik raportów o zbiorach zgodnie z kontraktem danych."""
    if not path.exists():
        raise FileNotFoundError(f"Brak wymaganego pliku raportów: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "report_id": str(row.get("report_id", "")).strip(),
                    "date": str(row.get("date", "")).strip(),
                    "area_id": _normalize_area_id(row.get("area_id")),
                    "location_raw": (
                        str(row["location_raw"]).strip()
                        if row.get("location_raw") not in (None, "")
                        else None
                    ),
                    "dominant_species": (
                        str(row["dominant_species"]).strip()
                        if row.get("dominant_species") not in (None, "")
                        else None
                    ),
                    "yield_pcs_per_h": _parse_optional_float(row.get("yield_pcs_per_h")),
                    "harvest_duration_h": _parse_optional_float(row.get("harvest_duration_h")),
                    "worminess_pct": _parse_optional_float(row.get("worminess_pct")),
                    "is_demo": _parse_bool(row.get("is_demo")),
                }
            )
    return rows


def load_weather_csv(path: Path) -> list[dict[str, Any]]:
    """Wczytuje plik danych pogodowych zgodnie z kontraktem danych."""
    if not path.exists():
        raise FileNotFoundError(f"Brak wymaganego pliku pogody: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "date": str(row.get("date", "")).strip(),
                    "area_id": _normalize_area_id(row.get("area_id")),
                    "precip_mm_1d": _parse_optional_float(row.get("precip_mm_1d")),
                    "precip_lag_7d_mm": _parse_optional_float(row.get("precip_lag_7d_mm")),
                    "precip_lag_14d_mm": _parse_optional_float(row.get("precip_lag_14d_mm")),
                    "precip_lag_21d_mm": _parse_optional_float(row.get("precip_lag_21d_mm")),
                    "temp_min_night_c": _parse_optional_float(row.get("temp_min_night_c")),
                    "temp_mean_7d_c": _parse_optional_float(row.get("temp_mean_7d_c")),
                    "soil_moisture_0_7cm_m3m3": _parse_optional_float(
                        row.get("soil_moisture_0_7cm_m3m3")
                    ),
                    "is_forecast": _parse_bool(row.get("is_forecast")),
                    "is_demo": _parse_bool(row.get("is_demo")),
                }
            )
    return rows


def load_dataset_bundle(requested_mode: str = "demo") -> DatasetBundle:
    """Wczytuje pełny pakiet danych dla wybranego trybu ('demo' lub 'prod')."""
    actual_mode, is_fallback, paths = resolve_environment_paths(requested_mode)
    areas_geojson, areas_by_id = load_areas_geojson(paths["areas"])
    predictions = load_predictions_csv(paths["predictions"])
    reports = load_reports_csv(paths["reports"])
    weather = load_weather_csv(paths["weather"])

    if is_fallback:
        status_msg = (
            "Pliki produkcyjne (`reports.csv`, `weather.csv`, `predictions.csv`) w `data/processed/` "
            "nie zostały jeszcze w pełni wygenerowane (dostępny plik granic `data/processed/areas.geojson`). "
            "Automatycznie przełączono na próbkę `data/demo/` (`is_demo = True`, 30 rzeczywistych powiatów Dolnego Śląska)."
        )
    elif actual_mode == "demo":
        status_msg = (
            "Aktywny tryb demonstracyjny UI (`data/demo/`, `is_demo = True`) z rzeczywistymi granicami "
            "30 powiatów woj. dolnośląskiego (`EPSG:4326`). Wykluczony z treningu ML."
        )
    else:
        status_msg = "Aktywny tryb produkcyjny (`data/processed/`, `is_demo = False`)."

    return DatasetBundle(
        env_mode=actual_mode,
        is_fallback_to_demo=is_fallback,
        status_message=status_msg,
        areas_geojson=areas_geojson,
        areas_by_id=areas_by_id,
        predictions=predictions,
        reports=reports,
        weather=weather,
    )


def load_real_observations_bundle() -> RealObservationsBundle:
    """
    Wczytuje rzeczywisty zbiór danych przetworzonych z `data/processed/` ([GRZ-07]..[GRZ-10], [GRZ-12]):
    - `areas.geojson` (30 powiatów Dolnego Śląska, `is_demo = False`),
    - `reports.parquet` (518 raportów ogółem z grzyby.pl),
    - `reports_spatial.parquet` (518 raportów po złączeniu przestrzennym z powiatami i BDL),
    - `forest_features.parquet` (30 powiatów z cechami leśnymi BDL),
    - `features.parquet` (513 ważnych obserwacji treningowych z oknami pogodowymi Point-in-Time),
    - `weather.csv` (2910 rekordów dobowych Open-Meteo ERA5-Land).
    """
    areas_path = PROCESSED_DIR / "areas.geojson"
    reports_path = PROCESSED_DIR / "reports.parquet"
    reports_spatial_path = PROCESSED_DIR / "reports_spatial.parquet"
    forest_path = PROCESSED_DIR / "forest_features.parquet"
    features_path = PROCESSED_DIR / "features.parquet"
    weather_path = PROCESSED_DIR / "weather.csv"

    areas_geojson, areas_by_id = load_areas_geojson(areas_path)
    reports_df = pd.read_parquet(reports_path)
    reports_spatial_df = pd.read_parquet(reports_spatial_path)
    forest_features_df = pd.read_parquet(forest_path)
    features_df = pd.read_parquet(features_path)
    weather_rows = load_weather_csv(weather_path)

    # Normalizacja kodów TERYT z zachowaniem wiodącego zera
    reports_df["area_id"] = reports_df["area_id"].astype(str).str.zfill(4)
    reports_spatial_df["region_id"] = reports_spatial_df["region_id"].astype(str).str.zfill(4)
    reports_spatial_df["area_id"] = reports_spatial_df["area_id"].astype(str).str.zfill(4)
    forest_features_df["region_id"] = forest_features_df["region_id"].astype(str).str.zfill(4)
    features_df["region_id"] = features_df["region_id"].astype(str).str.zfill(4)
    features_df["area_id"] = features_df["area_id"].astype(str).str.zfill(4)

    min_d = str(features_df["observation_date"].min())
    max_d = str(features_df["observation_date"].max())
    active_cnt = int(features_df["region_id"].nunique())

    status_msg = (
        f"Aktywny zbiór rzeczywistych obserwacji (`data/processed/`, `is_demo = False`): "
        f"<b>{len(features_df)}</b> zwalidowanych raportów z portalu grzyby.pl "
        f"(z {len(reports_df)} wyodrębnionych ogółem), okres <b>{min_d} – {max_d}</b>, "
        f"pokrycie <b>{active_cnt} z {len(areas_by_id)}</b> powiatów Dolnego Śląska."
    )

    return RealObservationsBundle(
        status_message=status_msg,
        areas_geojson=areas_geojson,
        areas_by_id=areas_by_id,
        reports_df=reports_df,
        reports_spatial_df=reports_spatial_df,
        forest_features_df=forest_features_df,
        features_df=features_df,
        weather_rows=weather_rows,
        min_date=min_d,
        max_date=max_d,
    )


def _classify_observed_yield(mean_yield: float | None) -> str:
    """Klasyfikuje obserwowaną średnią wydajność w powiecie lub zwraca BRAK_DANYCH dla None."""
    if mean_yield is None or pd.isna(mean_yield):
        return "BRAK_DANYCH"
    val = float(mean_yield)
    if val < 5.0:
        return "BRAK"
    if val <= 20.0:
        return "SREDNIO"
    return "WYSYP"


def filter_real_observations_and_aggregate_by_county(
    real_bundle: RealObservationsBundle,
    date_from: str,
    date_to: str,
    species_filter: str = "all",
    origin_lat: float = 51.1079,
    origin_lon: float = 17.0385,
    radius_km: float = 160.0,
    region_filter: str = "ALL",
) -> dict[str, Any]:
    """
    Filtruje rzeczywiste obserwacje z `features.parquet` (dołączając `location_raw` i `lat`/`lon`
    z `reports_spatial.parquet`) po zakresie dat `[date_from, date_to]`, gatunku `species_filter`,
    promieniu odległości `radius_km` oraz opcjonalnym powiecie `region_filter`.

    Wyznacza dla każdego z 30 powiatów woj. dolnośląskiego (`0201`–`0265`) zagregowane pokrycie
    i statystyki zbiorów, gwarantując spełnienie warunków [GRZ-12]:
    - Filtry zmieniają zarówno liczebność dostępnych raportów, jak i mapę powiatów.
    - Brak raportów w powiecie (`reports_count == 0`) jest jawnie oznaczony jako
      `status_code = "BRAK_DANYCH"` i `mean_yield_pcs_per_h = None` (`NULL`), a NIGDY jako `0.0 szt./h`.
    """
    df = real_bundle.features_df.copy()

    # Dołączenie surowej nazwy lokalizacji i współrzędnych z reports_spatial (relacja 1:1 po report_id)
    spatial_extra = real_bundle.reports_spatial_df[
        ["report_id", "location_raw", "lat", "lon", "is_county_aggregate", "spatial_precision_note"]
    ].drop_duplicates(subset=["report_id"])
    df = df.merge(spatial_extra, on="report_id", how="left")

    # 1. Filtr zakresu dat obserwacji [date_from, date_to]
    mask_date = (df["observation_date"] >= str(date_from)) & (
        df["observation_date"] <= str(date_to)
    )
    df = df[mask_date]

    # 2. Filtr gatunku (dotyczy obserwacji, a nie osobnego modelu ML dla gatunku!)
    if species_filter and species_filter != "all":
        if species_filter == "__NULL__":
            df = df[df["dominant_species"].isna()]
        else:
            df = df[df["dominant_species"] == species_filter]

    # 3. Odległość powiatów od punktu odniesienia
    dist_by_region: dict[str, float] = {}
    in_radius_by_region: dict[str, bool] = {}
    for area_id, props in real_bundle.areas_by_id.items():
        d_km = haversine_km(
            origin_lat,
            origin_lon,
            float(props["centroid_lat"]),
            float(props["centroid_lon"]),
        )
        dist_by_region[area_id] = d_km
        in_radius_by_region[area_id] = d_km <= float(radius_km)

    # Odfiltrowanie raportów spoza promienia i ewentualnego filtra pojedynczego powiatu
    df["distance_from_origin_km"] = df["region_id"].map(dist_by_region)
    df["in_radius"] = df["region_id"].map(in_radius_by_region)
    df = df[df["in_radius"]]

    if region_filter and region_filter != "ALL":
        reg_norm = _normalize_area_id(region_filter)
        df = df[df["region_id"] == reg_norm]

    df = df.sort_values(
        by=["observation_date", "region_id", "report_id"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # 4. Najnowszy rekord pogodowy z końca wybranego okresu (date <= date_to) dla każdego powiatu
    weather_by_area: dict[str, dict[str, Any]] = {}
    for w in sorted(real_bundle.weather_rows, key=lambda x: x["date"]):
        if w["date"] <= str(date_to):
            weather_by_area[w["area_id"]] = w

    # 5. Agregacja na poziomie wszystkich 30 powiatów
    county_records: list[dict[str, Any]] = []
    county_by_id: dict[str, dict[str, Any]] = {}

    grouped = {reg_id: grp for reg_id, grp in df.groupby("region_id")}

    for area_id, area_props in real_bundle.areas_by_id.items():
        d_km = dist_by_region[area_id]
        in_rad = in_radius_by_region[area_id]
        if region_filter and region_filter != "ALL" and area_id != _normalize_area_id(region_filter):
            in_rad = False

        grp = grouped.get(area_id)
        rep_cnt = int(len(grp)) if grp is not None else 0

        if rep_cnt > 0 and grp is not None and in_rad:
            mean_y = round(float(grp["yield_pcs_per_h"].mean()), 2)
            med_y = round(float(grp["yield_pcs_per_h"].median()), 2)
            max_y = round(float(grp["yield_pcs_per_h"].max()), 2)
            zero_cnt = int((grp["yield_pcs_per_h"] == 0.0).sum())
            precise_cnt = int((grp["point_represents_specific_forest"] == True).sum())
            county_agg_cnt = int((grp["point_represents_specific_forest"] == False).sum())
            last_rep_date = str(grp["observation_date"].max())
            status_code = _classify_observed_yield(mean_y)
            has_reports = True
        else:
            # Kluczowa reguła [GRZ-12]: brak raportów to BRAK_DANYCH (None / NULL), nigdy 0.0 szt./h!
            mean_y = None
            med_y = None
            max_y = None
            zero_cnt = 0
            precise_cnt = 0
            county_agg_cnt = 0
            last_rep_date = None
            status_code = "BRAK_DANYCH" if in_rad else "POZA_PROMIENIEM"
            has_reports = False

        w_rec = weather_by_area.get(area_id, {})

        rec = {
            "area_id": area_id,
            "region_id": area_id,
            "area_name": area_props.get("area_name", area_id),
            "total_area_km2": area_props.get("total_area_km2"),
            "distance_km": d_km,
            "in_radius": in_rad,
            "reports_count": rep_cnt if in_rad else 0,
            "has_reports": has_reports,
            "status_code": status_code,
            "abundance_class": status_code if status_code in {"WYSYP", "SREDNIO", "BRAK"} else "BRAK_DANYCH",
            "mean_yield_pcs_per_h": mean_y,
            "median_yield_pcs_per_h": med_y,
            "max_yield_pcs_per_h": max_y,
            "zero_harvest_reports_count": zero_cnt,
            "point_precise_count": precise_cnt,
            "county_aggregate_count": county_agg_cnt,
            "last_report_date": last_rep_date,
            "forest_cover_pct": area_props.get("forest_cover_pct"),
            "pine_share_pct": area_props.get("pine_share_pct"),
            "oak_beech_share_pct": area_props.get("oak_beech_share_pct"),
            "spruce_share_pct": area_props.get("spruce_share_pct"),
            "mean_stand_age_years": area_props.get("mean_stand_age_years"),
            "dominant_moisture_type": area_props.get("dominant_moisture_type"),
            "has_bdl_stand_data": bool(
                area_props.get("has_bdl_stand_data")
                or (
                    area_props.get("pine_share_pct") is not None
                    and area_props.get("mean_stand_age_years") is not None
                )
            ),
            "bdl_sample_stands_count": int(area_props.get("bdl_sample_stands_count") or 0),
            "precip_lag_7d_mm": w_rec.get("precip_lag_7d_mm"),
            "precip_lag_14d_mm": w_rec.get("precip_lag_14d_mm"),
            "precip_lag_21d_mm": w_rec.get("precip_lag_21d_mm"),
            "temp_min_night_c": w_rec.get("temp_min_night_c"),
            "soil_moisture_0_7cm_m3m3": w_rec.get("soil_moisture_0_7cm_m3m3"),
            "is_demo": False,
        }
        county_records.append(rec)
        county_by_id[area_id] = rec

    county_records.sort(
        key=lambda r: (
            r["in_radius"],
            r["has_reports"],
            r["reports_count"],
            r["mean_yield_pcs_per_h"] if r["mean_yield_pcs_per_h"] is not None else -1.0,
            -r["distance_km"],
        ),
        reverse=True,
    )

    in_radius_counties = [r for r in county_records if r["in_radius"]]
    covered_counties = [r for r in in_radius_counties if r["has_reports"]]
    missing_counties = [r for r in in_radius_counties if not r["has_reports"]]

    return {
        "filtered_reports_df": df,
        "filtered_reports_count": int(len(df)),
        "total_valid_reports_count": int(len(real_bundle.features_df)),
        "county_records": county_records,
        "county_by_id": county_by_id,
        "in_radius_counties_count": len(in_radius_counties),
        "covered_counties_count": len(covered_counties),
        "no_data_counties_count": len(missing_counties),
        "covered_counties_ids": [r["area_id"] for r in covered_counties],
        "no_data_counties_ids": [r["area_id"] for r in missing_counties],
        "date_from": str(date_from),
        "date_to": str(date_to),
        "species_filter": species_filter,
    }


def verify_record_against_source_tables(
    report_id: str,
    real_bundle: RealObservationsBundle,
) -> dict[str, Any]:
    """
    Porównuje pojedynczy rekord `report_id` pomiędzy:
    1) tabelą źródłową raportów `data/processed/reports.parquet` ([GRZ-07]),
    2) tabelą przestrzenną `data/processed/reports_spatial.parquet` ([GRZ-08]),
    3) tabelą cech leśnych `data/processed/forest_features.parquet` ([GRZ-08]),
    4) finalną tabelą analityczną `data/processed/features.parquet` ([GRZ-10]).

    Pozwala użytkownikowi w Streamlit sprawdzić ręcznie zgodność próbki rekordów
    z tabelami źródłowymi (warunek ukończenia #5 zadania [GRZ-12]).
    """
    rid = str(report_id).strip()
    rep_rows = real_bundle.reports_df[real_bundle.reports_df["report_id"] == rid]
    spat_rows = real_bundle.reports_spatial_df[real_bundle.reports_spatial_df["report_id"] == rid]
    feat_rows = real_bundle.features_df[real_bundle.features_df["report_id"] == rid]

    if rep_rows.empty or spat_rows.empty or feat_rows.empty:
        raise KeyError(f"Nie znaleziono raportu o identyfikatorze {rid} w tabelach przetworzonych.")

    r_src = rep_rows.iloc[0]
    r_spat = spat_rows.iloc[0]
    r_feat = feat_rows.iloc[0]

    reg_id = str(r_feat["region_id"]).zfill(4)
    f_rows = real_bundle.forest_features_df[real_bundle.forest_features_df["region_id"] == reg_id]
    r_forest = f_rows.iloc[0] if not f_rows.empty else {}

    def _fmt(v: Any) -> str:
        if v is None or pd.isna(v):
            return "NULL"
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) < 1.0 and v != 0.0 else f"{v:.2f}"
        return str(v)

    def _same(a: Any, b: Any) -> bool:
        if (a is None or pd.isna(a)) and (b is None or pd.isna(b)):
            return True
        if (a is None or pd.isna(a)) or (b is None or pd.isna(b)):
            return False
        if isinstance(a, (float, int)) and isinstance(b, (float, int)):
            return abs(float(a) - float(b)) < 1e-4
        return str(a).strip() == str(b).strip()

    checks = [
        (
            "report_id (ID raportu)",
            "reports.parquet [GRZ-07]",
            r_src.get("report_id"),
            r_feat.get("report_id"),
        ),
        (
            "harvest_date / observation_date",
            "reports.parquet [GRZ-07]",
            r_src.get("harvest_date"),
            r_feat.get("observation_date"),
        ),
        (
            "publication_date (data publikacji)",
            "reports.parquet [GRZ-07]",
            r_src.get("publication_date"),
            r_feat.get("publication_date"),
        ),
        (
            "region_id / area_id (TERYT powiatu)",
            "reports.parquet + reports_spatial.parquet",
            r_src.get("area_id"),
            r_feat.get("region_id"),
        ),
        (
            "place_id (ID lokalizacji grzyby.pl)",
            "reports.parquet [GRZ-07]",
            r_src.get("place_id"),
            r_feat.get("place_id"),
        ),
        (
            "yield_pcs_per_h [szt./h] (etykieta)",
            "reports.parquet [GRZ-07]",
            r_src.get("yield_pcs_per_h"),
            r_feat.get("yield_pcs_per_h"),
        ),
        (
            "dominant_species (gatunek dominujący)",
            "reports.parquet [GRZ-07]",
            r_src.get("dominant_species"),
            r_feat.get("dominant_species"),
        ),
        (
            "harvest_duration_h [h] (czas zbioru)",
            "reports.parquet [GRZ-07]",
            r_src.get("harvest_duration_h"),
            r_feat.get("harvest_duration_h"),
        ),
        (
            "total_pieces [szt.] (liczba sztuk)",
            "reports.parquet [GRZ-07]",
            r_src.get("total_pieces"),
            r_feat.get("total_pieces"),
        ),
        (
            "spatial_precision (dokładność lokalizacji)",
            "reports_spatial.parquet [GRZ-08]",
            r_spat.get("spatial_precision"),
            r_feat.get("spatial_precision"),
        ),
        (
            "forest_cover_pct [%] (lesistość powiatu)",
            "forest_features.parquet [GRZ-08]",
            r_forest.get("forest_cover_pct"),
            r_feat.get("forest_cover_pct"),
        ),
        (
            "pine_share_pct [%] (udział sosny BDL)",
            "forest_features.parquet [GRZ-08]",
            r_forest.get("pine_share_pct"),
            r_feat.get("pine_share_pct"),
        ),
        (
            "source_url (adres źródła grzyby.pl)",
            "reports.parquet [GRZ-07]",
            r_src.get("source_url"),
            r_feat.get("source_url"),
        ),
    ]

    comparison_rows: list[dict[str, Any]] = []
    all_match = True
    for field_label, source_table, src_val, feat_val in checks:
        matched = _same(src_val, feat_val)
        if not matched:
            all_match = False
        comparison_rows.append(
            {
                "Pole": field_label,
                "Tabela źródłowa": source_table,
                "Wartość w tabeli źródłowej": _fmt(src_val),
                "Wartość w features.parquet": _fmt(feat_val),
                "Zgodność": "TAK (1:1)" if matched else "NIEZGODNOŚĆ",
            }
        )

    return {
        "report_id": rid,
        "region_id": reg_id,
        "region_name": str(r_feat.get("region_name", reg_id)),
        "location_raw": str(r_src.get("location_raw") or "—"),
        "observation_date": str(r_feat.get("observation_date")),
        "feature_cutoff_date": str(r_feat.get("feature_cutoff_date")),
        "source_url": str(r_src.get("source_url") or ""),
        "all_fields_match": all_match,
        "comparison_rows": comparison_rows,
    }


def load_sprint1_audit_samples() -> dict[str, Any]:
    """
    Wczytuje rzeczywiste próbki audytowe przygotowane w Sprincie 1 ([GRZ-03], [GRZ-04], [GRZ-05])
    z katalogu `data/sample/`, jeżeli są obecne w repozytorium.
    Brak plików BDL nie powoduje błędu i nie blokuje działania aplikacji.
    """
    result: dict[str, Any] = {
        "reports_audit": [],
        "weather_sample": [],
        "region_areas_geojson": None,
        "bdl_stands_geojson": None,
        "bdl_stands_sample": [],
        "bdl_powiaty_agg": [],
    }
    reports_sample_path = SAMPLE_DIR / "reports_audit_52_ds.csv"
    weather_sample_path = SAMPLE_DIR / "weather_daily_sample_ds.csv"
    region_geojson_path = SAMPLE_DIR / "region_powiaty_02_ds.geojson"
    bdl_geojson_path = SAMPLE_DIR / "bdl_stands_sample_ds.geojson"
    bdl_csv_path = SAMPLE_DIR / "bdl_stands_sample_ds.csv"
    bdl_agg_path = SAMPLE_DIR / "bdl_powiaty_agg_sample_ds.csv"

    if reports_sample_path.exists():
        with reports_sample_path.open("r", encoding="utf-8", newline="") as f:
            result["reports_audit"] = list(csv.DictReader(f))

    if weather_sample_path.exists():
        with weather_sample_path.open("r", encoding="utf-8", newline="") as f:
            result["weather_sample"] = list(csv.DictReader(f))

    if region_geojson_path.exists():
        result["region_areas_geojson"] = json.loads(
            region_geojson_path.read_text(encoding="utf-8")
        )

    if bdl_geojson_path.exists():
        result["bdl_stands_geojson"] = json.loads(
            bdl_geojson_path.read_text(encoding="utf-8")
        )

    if bdl_csv_path.exists():
        with bdl_csv_path.open("r", encoding="utf-8", newline="") as f:
            result["bdl_stands_sample"] = list(csv.DictReader(f))

    if bdl_agg_path.exists():
        with bdl_agg_path.open("r", encoding="utf-8", newline="") as f:
            result["bdl_powiaty_agg"] = list(csv.DictReader(f))

    return result
