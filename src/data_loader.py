"""
Moduł wczytywania i walidacji danych zgodny z kontraktem danych (docs/data_contract.md).

Zapewnia:
- ścisłą separację środowiska demonstracyjnego (`data/demo/`, `is_demo = True`)
  od środowiska produkcyjnego (`data/processed/`, `is_demo = False`),
- zachowanie 4-znakowego kodu TERYT (`area_id`, np. `"0201"`),
- poprawne rozróżnienie pomiaru zerowego (`0.0`) od braku informacji (`NULL` / `None`),
- dostęp do próbek audytowych ze Sprintu 1 (`data/sample/`).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import DEMO_DIR, PROCESSED_DIR, SAMPLE_DIR


@dataclass(frozen=True)
class DatasetBundle:
    """Komplet danych wczytanych dla wybranego środowiska."""

    env_mode: str  # "demo" lub "prod"
    is_fallback_to_demo: bool
    status_message: str
    areas_geojson: dict[str, Any]
    areas_by_id: dict[str, dict[str, Any]]
    predictions: list[dict[str, Any]]
    reports: list[dict[str, Any]]
    weather: list[dict[str, Any]]


def _parse_optional_float(value: str | float | int | None) -> float | None:
    """Konwertuje wartość na float, zachowując NULL (pusty ciąg -> None) i 0.0 jako 0.0."""
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() == "null" or text.lower() == "none":
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
    """Wczytuje plik GeoJSON z granicami powiatów i cechami BDL."""
    if not path.exists():
        raise FileNotFoundError(f"Brak wymaganego pliku obszarów GeoJSON: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    areas_by_id: dict[str, dict[str, Any]] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        area_id = _normalize_area_id(props.get("area_id"))
        props["area_id"] = area_id
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
            "Pliki produkcyjne w `data/processed/` nie zostały jeszcze wygenerowane. "
            "Automatycznie przełączono na bezpieczną próbkę demonstracyjną `data/demo/` (`is_demo = True`)."
        )
    elif actual_mode == "demo":
        status_msg = (
            "Aktywny tryb demonstracyjny UI (`data/demo/`, `is_demo = True`). "
            "Dane służą do budowy i integracji interfejsu i są wykluczone z treningu ML."
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


def load_sprint1_audit_samples() -> dict[str, list[dict[str, str]]]:
    """
    Wczytuje rzeczywiste próbki audytowe przygotowane w Sprincie 1 ([GRZ-03] i [GRZ-04])
    z katalogu `data/sample/`, jeżeli są obecne w repozytorium.
    """
    result: dict[str, list[dict[str, str]]] = {
        "reports_audit": [],
        "weather_sample": [],
    }
    reports_sample_path = SAMPLE_DIR / "reports_audit_52_ds.csv"
    weather_sample_path = SAMPLE_DIR / "weather_daily_sample_ds.csv"

    if reports_sample_path.exists():
        with reports_sample_path.open("r", encoding="utf-8", newline="") as f:
            result["reports_audit"] = list(csv.DictReader(f))

    if weather_sample_path.exists():
        with weather_sample_path.open("r", encoding="utf-8", newline="") as f:
            result["weather_sample"] = list(csv.DictReader(f))

    return result
