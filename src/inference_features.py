"""
Moduł budowania cech wejściowych do prognoz w aplikacji i symulacji historycznych [GRZ-15].

Zapewnia 100% zgodność nazw, jednostek i typów cech z tabelą treningową (`src/features.py` / `features.parquet`),
przy ścisłym przestrzeganiu zasady Point-in-Time Correctness:
- W chwili wystawienia prognozy `issued_at` ($t_0$) dostępne są wyłącznie dane pogodowe i glebowe
  do końca doby odcięcia `feature_cutoff_date = issued_at` włącznie.
- Przyszła rzeczywista pogoda (`date > issued_at`, w tym z dnia docelowego `target_date = issued_at + horizon_days`)
  NIGDY nie jest używana.
- Braki historii pogodowej lub brak profilu BDL są zachowywane jako `None`/`NaN` (nigdy `0.0`) i jawnie oznaczane.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd

from src.features import (
    FEATURE_DICTIONARY,
    _clean_nullable_float,
    _clean_nullable_str,
    compute_point_in_time_weather_features,
    compute_seasonality_features,
    index_weather_by_region_and_date,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

DEFAULT_SUPPORTED_HORIZONS = (1, 2, 3)
MAX_SUPPORTED_HORIZON_DAYS = 7

# Kolumny pogodowe, leśne, przestrzenne i sezonowe współdzielone 1:1 z treningiem (src/features.py)
WEATHER_FEATURE_COLUMNS = [
    "precip_lag_7d_mm",
    "precip_lag_14d_mm",
    "precip_lag_21d_mm",
    "precip_cutoff_1d_mm",
    "precip_hours_lag_7d_h",
    "rainy_days_lag_7d_count",
    "rainy_days_lag_14d_count",
    "temp_min_night_c",
    "temp_min_lag_3d_c",
    "temp_min_lag_7d_c",
    "frost_nights_lag_7d_count",
    "temp_mean_cutoff_1d_c",
    "temp_mean_7d_c",
    "temp_mean_14d_c",
    "temp_max_cutoff_1d_c",
    "wind_speed_mean_7d_kmh",
    "soil_moisture_0_7cm_m3m3",
    "soil_moisture_mean_7d_m3m3",
    "soil_moisture_mean_14d_m3m3",
    "soil_temperature_0_7cm_c",
    "soil_temp_mean_7d_c",
    "grid_elevation_m",
]

FOREST_FEATURE_COLUMNS = [
    "total_area_km2",
    "forest_cover_pct",
    "forest_area_km2",
    "pine_share_pct",
    "oak_beech_share_pct",
    "spruce_share_pct",
    "mean_stand_age_years",
    "dominant_moisture_type",
    "has_bdl_stand_data",
]

SPATIAL_CONTEXT_COLUMNS = [
    "spatial_precision",
    "point_represents_specific_forest",
    "distance_to_county_centroid_km",
]

SEASONALITY_FEATURE_COLUMNS = [
    "month",
    "day_of_year",
    "week_of_year",
    "day_of_week",
    "is_weekend",
    "day_of_year_sin",
    "day_of_year_cos",
    "season_phase",
]

# Pełny zestaw cech wejściowych modelu (bez kolumn etykiet i identyfikatorów raportowych)
MODEL_INPUT_FEATURE_COLUMNS = (
    WEATHER_FEATURE_COLUMNS
    + FOREST_FEATURE_COLUMNS
    + SPATIAL_CONTEXT_COLUMNS
    + SEASONALITY_FEATURE_COLUMNS
)

INFERENCE_METADATA_COLUMNS = [
    "region_id",
    "area_id",
    "region_name",
    "issued_at",
    "feature_cutoff_date",
    "target_date",
    "horizon_days",
    "window_7d_start_date",
    "window_14d_start_date",
    "window_21d_start_date",
    "window_end_date",
    "is_region_covered",
    "history_completeness",
    "missing_weather_days_21d",
    "missing_feature_columns",
]

ALL_PREDICTION_INPUT_COLUMNS = INFERENCE_METADATA_COLUMNS + MODEL_INPUT_FEATURE_COLUMNS


class RegionOutOfCoverageError(ValueError):
    """Zgłaszany, gdy żądany obszar `region_id` leży poza obszarem pilotażowym (Dolny Śląsk)."""


def build_forest_lookup(forest_features_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Buduje słownik cech leśnych po 4-znakowym kluczu `region_id`."""
    region_col = "region_id" if "region_id" in forest_features_df.columns else "area_id"
    if forest_features_df[region_col].nunique() != len(forest_features_df):
        raise ValueError("Tabela forest_features zawiera zduplikowane wartości region_id!")

    lookup: dict[str, dict[str, Any]] = {}
    for _, f_row in forest_features_df.iterrows():
        reg_id = str(f_row[region_col]).strip().zfill(4)
        lookup[reg_id] = {
            "region_name": str(f_row["region_name"]),
            "total_area_km2": float(f_row["total_area_km2"]),
            "forest_cover_pct": _clean_nullable_float(f_row.get("forest_cover_pct")),
            "forest_area_km2": _clean_nullable_float(f_row.get("forest_area_km2")),
            "pine_share_pct": _clean_nullable_float(f_row.get("pine_share_pct")),
            "oak_beech_share_pct": _clean_nullable_float(f_row.get("oak_beech_share_pct")),
            "spruce_share_pct": _clean_nullable_float(f_row.get("spruce_share_pct")),
            "mean_stand_age_years": _clean_nullable_float(f_row.get("mean_stand_age_years")),
            "dominant_moisture_type": _clean_nullable_str(f_row.get("dominant_moisture_type")),
            "has_bdl_stand_data": bool(f_row.get("has_bdl_stand_data", False)),
        }
    return lookup


def load_inference_reference_data(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Wczytuje tabele referencyjne (`forest_features.parquet`, `weather_daily.parquet`) do pamięci."""
    forest_path = processed_dir / "forest_features.parquet"
    weather_path = processed_dir / "weather_daily.parquet"

    if not forest_path.exists():
        raise FileNotFoundError(f"Brak tabeli cech leśnych: {forest_path}")
    if not weather_path.exists():
        raise FileNotFoundError(f"Brak tabeli pogody dziennej: {weather_path}")

    forest_df = pd.read_parquet(forest_path)
    weather_df = pd.read_parquet(weather_path)

    return build_forest_lookup(forest_df), index_weather_by_region_and_date(weather_df)


def _count_missing_weather_days_in_21d(
    region_id: str,
    cutoff_date_str: str,
    weather_lookup: dict[tuple[str, str], dict[str, Any]],
) -> int:
    """Zlicza brakujące dni pogodowe w 21-dniowym oknie historii przed i włącznie z `cutoff_date_str`."""
    t_cutoff = date.fromisoformat(cutoff_date_str)
    missing = 0
    for offset in range(21):
        d_iso = (t_cutoff - timedelta(days=20 - offset)).isoformat()
        if (region_id, d_iso) not in weather_lookup:
            missing += 1
    return missing


def build_prediction_input_row(
    region_id: str,
    issued_at: str,
    horizon_days: int = 1,
    forest_lookup: dict[str, dict[str, Any]] | None = None,
    weather_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    allow_out_of_coverage: bool = False,
) -> dict[str, Any]:
    """
    Buduje pojedynczy rekord cech wejściowych dla modelu prognozy wysypu grzybów:
    - `region_id`: 4-znakowy kod TERYT powiatu (np. `"0201"`),
    - `issued_at`: data wystawienia prognozy `YYYY-MM-DD` (równa `feature_cutoff_date`),
    - `horizon_days`: horyzont wyprzedzenia w dniach (domyślnie 1..3, max 7 dni).

    Gwarancje:
    - Nazwy, jednostki i typy cech w 100% odpowiadają słownikowi `FEATURE_DICTIONARY` z `src/features.py`.
    - Do wyliczenia cech używane są wyłącznie dni `<= issued_at` (brak wycieku przyszłej pogody).
    - Niepełna historia zwraca `None` w odpowiednich oknach (nigdy `0.0`) i ustawia `history_completeness`.
    """
    if horizon_days < 1 or horizon_days > MAX_SUPPORTED_HORIZON_DAYS:
        raise ValueError(
            f"horizon_days={horizon_days} poza dozwolonym zakresem [1, {MAX_SUPPORTED_HORIZON_DAYS}]."
        )

    if forest_lookup is None or weather_lookup is None:
        loaded_forest, loaded_weather = load_inference_reference_data()
        forest_lookup = forest_lookup or loaded_forest
        weather_lookup = weather_lookup or loaded_weather

    reg_id = str(region_id).strip().zfill(4)
    t_issued = date.fromisoformat(str(issued_at).strip())
    issued_iso = t_issued.isoformat()
    target_iso = (t_issued + timedelta(days=horizon_days)).isoformat()

    # Obsługa obszaru spoza pokrycia pilotażowego (TERYT 02)
    if reg_id not in forest_lookup:
        if not allow_out_of_coverage:
            raise RegionOutOfCoverageError(
                f"Obszar region_id='{reg_id}' leży poza pokryciem pilotażowym (30 powiatów woj. dolnośląskiego)."
            )
        empty_row: dict[str, Any] = {
            "region_id": reg_id,
            "area_id": reg_id,
            "region_name": None,
            "issued_at": issued_iso,
            "feature_cutoff_date": issued_iso,
            "target_date": target_iso,
            "horizon_days": int(horizon_days),
            "window_7d_start_date": (t_issued - timedelta(days=6)).isoformat(),
            "window_14d_start_date": (t_issued - timedelta(days=13)).isoformat(),
            "window_21d_start_date": (t_issued - timedelta(days=20)).isoformat(),
            "window_end_date": issued_iso,
            "is_region_covered": False,
            "history_completeness": "OUT_OF_COVERAGE",
            "missing_weather_days_21d": 21,
            "missing_feature_columns": list(MODEL_INPUT_FEATURE_COLUMNS),
        }
        for col in MODEL_INPUT_FEATURE_COLUMNS:
            empty_row[col] = None
        return empty_row

    f_meta = forest_lookup[reg_id]

    # Wyliczenie cech pogodowych z twardym odcięciem feature_cutoff_date == issued_at
    # (compute_point_in_time_weather_features dla target_date i horizon_days wyznacza cutoff = target - horizon = issued_at)
    w_feats = compute_point_in_time_weather_features(
        region_id=reg_id,
        target_date_str=target_iso,
        weather_lookup=weather_lookup,
        horizon_days=horizon_days,
    )
    if w_feats["issued_at"] != issued_iso or w_feats["feature_cutoff_date"] != issued_iso:
        raise AssertionError(
            f"Niezgodność odcięcia czasowego: oczekiwano {issued_iso}, otrzymano {w_feats['feature_cutoff_date']}"
        )

    # Cechy kalendarzowo-sezonowe dotyczą dnia docelowego prognozy (target_date)
    s_feats = compute_seasonality_features(target_iso)

    missing_days_21d = _count_missing_weather_days_in_21d(reg_id, issued_iso, weather_lookup)

    row: dict[str, Any] = {
        "region_id": reg_id,
        "area_id": reg_id,
        "region_name": f_meta["region_name"],
        "issued_at": issued_iso,
        "feature_cutoff_date": issued_iso,
        "target_date": target_iso,
        "horizon_days": int(horizon_days),
        "window_7d_start_date": w_feats["window_7d_start_date"],
        "window_14d_start_date": w_feats["window_14d_start_date"],
        "window_21d_start_date": w_feats["window_21d_start_date"],
        "window_end_date": w_feats["window_end_date"],
        "is_region_covered": True,
        # Cechy pogodowe
        "precip_lag_7d_mm": w_feats["precip_lag_7d_mm"],
        "precip_lag_14d_mm": w_feats["precip_lag_14d_mm"],
        "precip_lag_21d_mm": w_feats["precip_lag_21d_mm"],
        "precip_cutoff_1d_mm": w_feats["precip_cutoff_1d_mm"],
        "precip_hours_lag_7d_h": w_feats["precip_hours_lag_7d_h"],
        "rainy_days_lag_7d_count": w_feats["rainy_days_lag_7d_count"],
        "rainy_days_lag_14d_count": w_feats["rainy_days_lag_14d_count"],
        "temp_min_night_c": w_feats["temp_min_night_c"],
        "temp_min_lag_3d_c": w_feats["temp_min_lag_3d_c"],
        "temp_min_lag_7d_c": w_feats["temp_min_lag_7d_c"],
        "frost_nights_lag_7d_count": w_feats["frost_nights_lag_7d_count"],
        "temp_mean_cutoff_1d_c": w_feats["temp_mean_cutoff_1d_c"],
        "temp_mean_7d_c": w_feats["temp_mean_7d_c"],
        "temp_mean_14d_c": w_feats["temp_mean_14d_c"],
        "temp_max_cutoff_1d_c": w_feats["temp_max_cutoff_1d_c"],
        "wind_speed_mean_7d_kmh": w_feats["wind_speed_mean_7d_kmh"],
        "soil_moisture_0_7cm_m3m3": w_feats["soil_moisture_0_7cm_m3m3"],
        "soil_moisture_mean_7d_m3m3": w_feats["soil_moisture_mean_7d_m3m3"],
        "soil_moisture_mean_14d_m3m3": w_feats["soil_moisture_mean_14d_m3m3"],
        "soil_temperature_0_7cm_c": w_feats["soil_temperature_0_7cm_c"],
        "soil_temp_mean_7d_c": w_feats["soil_temp_mean_7d_c"],
        "grid_elevation_m": w_feats["grid_elevation_m"],
        # Cechy leśne BDL
        "total_area_km2": f_meta["total_area_km2"],
        "forest_cover_pct": f_meta["forest_cover_pct"],
        "forest_area_km2": f_meta["forest_area_km2"],
        "pine_share_pct": f_meta["pine_share_pct"],
        "oak_beech_share_pct": f_meta["oak_beech_share_pct"],
        "spruce_share_pct": f_meta["spruce_share_pct"],
        "mean_stand_age_years": f_meta["mean_stand_age_years"],
        "dominant_moisture_type": f_meta["dominant_moisture_type"],
        "has_bdl_stand_data": f_meta["has_bdl_stand_data"],
        # Kontekst przestrzenny w trybie prognozy powiatowej (brak fałszywej precyzji punktu)
        "spatial_precision": "county_aggregate",
        "point_represents_specific_forest": False,
        "distance_to_county_centroid_km": 0.0,
        # Cechy sezonowości dla target_date
        "month": s_feats["month"],
        "day_of_year": s_feats["day_of_year"],
        "week_of_year": s_feats["week_of_year"],
        "day_of_week": s_feats["day_of_week"],
        "is_weekend": s_feats["is_weekend"],
        "day_of_year_sin": s_feats["day_of_year_sin"],
        "day_of_year_cos": s_feats["day_of_year_cos"],
        "season_phase": s_feats["season_phase"],
    }

    missing_cols = [col for col in MODEL_INPUT_FEATURE_COLUMNS if row.get(col) is None]
    if missing_days_21d == 21:
        history_status = "NO_WEATHER_HISTORY"
    elif missing_days_21d > 0 or any(row.get(c) is None for c in WEATHER_FEATURE_COLUMNS):
        history_status = "PARTIAL_HISTORY"
    elif not f_meta["has_bdl_stand_data"]:
        history_status = "COMPLETE_WEATHER_NO_BDL_STAND"
    else:
        history_status = "COMPLETE"

    row["history_completeness"] = history_status
    row["missing_weather_days_21d"] = int(missing_days_21d)
    row["missing_feature_columns"] = missing_cols

    return row


def _cast_prediction_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ujednolica typy kolumn wejściowych z `FEATURE_DICTIONARY` w `src/features.py`."""
    out = df.copy()
    nullable_int_cols = [
        "horizon_days",
        "rainy_days_lag_7d_count",
        "rainy_days_lag_14d_count",
        "frost_nights_lag_7d_count",
        "month",
        "day_of_year",
        "week_of_year",
        "day_of_week",
        "missing_weather_days_21d",
    ]
    for col in nullable_int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    bool_cols = ["has_bdl_stand_data", "point_represents_specific_forest", "is_weekend", "is_region_covered"]
    for col in bool_cols:
        if col in out.columns:
            out[col] = out[col].astype("bool")

    # Pozostałe cechy ciągłe zgodne z FEATURE_DICTIONARY -> float64
    for col, meta in FEATURE_DICTIONARY.items():
        if col in out.columns and meta["type"] in ("float64", "float64 (nullable)"):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    return out


def build_prediction_inputs_batch(
    issued_at: str,
    horizons: Sequence[int] = DEFAULT_SUPPORTED_HORIZONS,
    region_ids: Iterable[str] | None = None,
    forest_lookup: dict[str, dict[str, Any]] | None = None,
    weather_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    allow_out_of_coverage: bool = False,
) -> pd.DataFrame:
    """
    Generuje gotową do predykcji tabelę `DataFrame` dla wybranych obszarów (`region_ids`,
    domyślnie wszystkie 30 powiatów) oraz horyzontów (domyślnie 1, 2, 3 dni od `issued_at`).
    """
    if forest_lookup is None or weather_lookup is None:
        loaded_forest, loaded_weather = load_inference_reference_data()
        forest_lookup = forest_lookup or loaded_forest
        weather_lookup = weather_lookup or loaded_weather

    target_regions = (
        [str(r).strip().zfill(4) for r in region_ids]
        if region_ids is not None
        else sorted(forest_lookup.keys())
    )

    rows: list[dict[str, Any]] = []
    for h in horizons:
        for reg_id in target_regions:
            row = build_prediction_input_row(
                region_id=reg_id,
                issued_at=issued_at,
                horizon_days=int(h),
                forest_lookup=forest_lookup,
                weather_lookup=weather_lookup,
                allow_out_of_coverage=allow_out_of_coverage,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ALL_PREDICTION_INPUT_COLUMNS)

    # Zapewnienie ustalonej kolejności kolumn
    ordered_cols = [c for c in ALL_PREDICTION_INPUT_COLUMNS if c in df.columns]
    df = df[ordered_cols]
    return _cast_prediction_dataframe_types(df)
