"""
Moduł usług predykcyjnych łączący wytrenowany potok ML ([GRZ-14]), generator cech
Point-in-Time ([GRZ-15]) oraz aplikację mapową Streamlit ([GRZ-16]).

Zapewnia:
1. Wczytanie zapisanych potoków `.joblib` (RandomForestRegressor + RandomForestClassifier)
   oraz wyraźne rozdzielenie trzech źródeł wyników:
   - `ML_MODEL` (`rf_pipeline_v1.0`, `is_demo = False`),
   - `RULE_BASED_INDEX` (Jawny Wskaźnik Warunków Grzybobrania, `RULE_SUITABILITY_INDEX_v1.0`, `is_demo = False`),
   - `DEMO_MOCK` (`DEMO_MOCK_v0`, `is_demo = True` — osobny tryb z `data/demo/`).
2. Generowanie predykcji dla dowolnej daty docelowej `target_date`, horyzontu `1..3` dni
   (`issued_at = target_date - horizon_days`), filtra gatunku i obszaru — zarówno z pliku
   prekomputowanego (`data/processed/predictions.csv`), jak i na żywo bezpośrednio z potoku ML.
3. Rzetelną obsługę obszarów bez wystarczających danych (`PARTIAL_HISTORY`, `NO_WEATHER_HISTORY`,
   `OUT_OF_COVERAGE`) — jawne oznaczenie jako `NIEDOSTEPNY` (`predicted_yield_pcs_per_h = None`).
4. Funkcję audytu zgodności wartości wyświetlanych na mapie z bezpośrednim wywołaniem modelu
   (`verify_map_predictions_against_direct_model`, [GRZ-16] Warunek #5).
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, SPECIES_LABELS
from src.evaluation import classify_abundance
from src.geo_utils import haversine_km
from src.inference_features import (
    build_prediction_input_row,
    build_prediction_inputs_batch,
    load_inference_reference_data,
)
from src.model_pipeline import (
    MODEL_VERSION,
    TARGET_UNIT,
    compute_species_habitat_multiplier,
)

RULE_INDEX_VERSION: str = "RULE_SUITABILITY_INDEX_v1.0"
SUFFICIENT_HISTORY_STATUSES: set[str] = {"COMPLETE", "COMPLETE_WEATHER_NO_BDL_STAND"}

# Zakres dat prekomputowanego pliku data/processed/predictions.csv (pełny okres testowy OoT + końcówka archiwum)
PRECOMPUTED_START_DATE: str = "2025-10-13"
PRECOMPUTED_END_DATE: str = "2025-11-06"

# Pełny zakres kalendarza w aplikacji (obejmuje również daty brzegowe 2025-08-10..2025-08-21
# z niepełną historią 21d, aby można było zweryfikować oznaczanie obszarów jako niedostępnych)
APP_FORECAST_MIN_DATE: str = "2025-08-10"
APP_FORECAST_MAX_DATE: str = "2025-11-08"
APP_FORECAST_DEFAULT_DATE: str = "2025-10-19"


@lru_cache(maxsize=2)
def load_saved_model_pipelines(
    reg_path_str: str | None = None,
    clf_path_str: str | None = None,
) -> tuple[Any, Any]:
    """Wczytuje i cache'uje zapisane potoki scikit-learn (.joblib) z [GRZ-14]."""
    reg_path = Path(reg_path_str) if reg_path_str else (MODELS_DIR / "rf_yield_pipeline.joblib")
    clf_path = Path(clf_path_str) if clf_path_str else (MODELS_DIR / "rf_abundance_pipeline.joblib")

    if not reg_path.exists():
        raise FileNotFoundError(
            f"Brak potoku regresyjnego ML: {reg_path}. Uruchom: python scripts/train_model.py"
        )
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Brak potoku klasyfikacyjnego ML: {clf_path}. Uruchom: python scripts/train_model.py"
        )
    return joblib.load(reg_path), joblib.load(clf_path)


@lru_cache(maxsize=2)
def load_cached_inference_reference(
    processed_dir_str: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Wczytuje i cache'uje słowniki `forest_features` i `weather_daily` dla szybkich zapytań UI."""
    p_dir = Path(processed_dir_str) if processed_dir_str else PROCESSED_DIR
    return load_inference_reference_data(processed_dir=p_dir)


def load_precomputed_predictions_df(
    predictions_csv_path: Path | None = None,
) -> pd.DataFrame | None:
    """Wczytuje prekomputowaną tabelę `data/processed/predictions.csv` (jeśli istnieje)."""
    path = predictions_csv_path or (PROCESSED_DIR / "predictions.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"area_id": str})
    df["area_id"] = df["area_id"].astype(str).str.strip().str.zfill(4)
    return df


def compute_rule_based_suitability_yield(
    row: dict[str, Any] | pd.Series,
    species_filter: str = "all",
) -> dict[str, Any]:
    """
    Oblicza **Jawny Wskaźnik Warunków Grzybobrania** (model regułowy / Tor A z `docs/data_audit.md`),
    całkowicie niezależny od wytrenowanego modelu ML (`prediction_source = 'RULE_BASED_INDEX'`).

    Zasady domenowe:
    - Wymaga pełnej 21-dniowej historii pogodowej (`history_completeness in SUFFICIENT_HISTORY_STATUSES`).
      Przy brakach historii zwraca `is_available = False` i `predicted_yield_pcs_per_h = None`.
    - Komponent 1 (nawodnienie grzybni, max 16 szt./h): opad 21-dniowy i impuls 15–21d.
    - Komponent 2 (uwilgotnienie gleby ERA5-Land 0–7 cm, max 10 szt./h): optimum 0.25–0.38 m³/m³.
    - Komponent 3 (termika gleby i kara za przymrozki nocne, max 8 szt./h): optimum 9–16°C,
      kara za przymrozki (`temp_min_night_c < 2.0°C`, `frost_nights_lag_7d_count > 0`).
    - Komponent 4 (potencjał leśny powiatu z BDL/GUS, max 10 szt./h): `forest_cover_pct`.
    """
    hist_status = str(row.get("history_completeness") or "NO_WEATHER_HISTORY")
    if hist_status not in SUFFICIENT_HISTORY_STATUSES:
        missing_d = row.get("missing_weather_days_21d")
        reason = (
            f"Niedostępny: brak pełnej 21-dniowej historii pogodowej "
            f"(status={hist_status}, brakuje {missing_d}/21 dni)."
        )
        return {
            "is_available": False,
            "availability_status": "UNAVAILABLE_INSUFFICIENT_DATA",
            "unavailable_reason": reason,
            "predicted_yield_pcs_per_h": None,
            "base_predicted_yield_pcs_per_h": None,
            "species_habitat_multiplier": None,
            "abundance_class": "NIEDOSTEPNY",
            "confidence_score": None,
            "habitat_match_note": reason,
            "model_version": RULE_INDEX_VERSION,
            "prediction_source": "RULE_BASED_INDEX",
        }

    p14 = float(row["precip_lag_14d_mm"])
    p21 = float(row["precip_lag_21d_mm"])
    p_slice_15_21 = max(0.0, p21 - p14)
    soil_m = float(row["soil_moisture_0_7cm_m3m3"])
    soil_t = float(row["soil_temperature_0_7cm_c"])
    tmin = float(row["temp_min_night_c"])
    frost_cnt = int(row.get("frost_nights_lag_7d_count") or 0)
    fc_raw = row.get("forest_cover_pct")
    fc = float(fc_raw) if fc_raw is not None and not pd.isna(fc_raw) else 25.0

    # 1. Nawodnienie opadowe (0..16 szt./h)
    rain_score = min(10.0, (p21 / 55.0) * 10.0) + min(6.0, (p_slice_15_21 / 25.0) * 6.0)

    # 2. Wilgotność płytkiej gleby 0-7 cm (0..10 szt./h, optimum ~0.30 m3/m3)
    moist_dist = abs(soil_m - 0.30)
    moisture_score = max(0.0, 10.0 * (1.0 - moist_dist / 0.18))

    # 3. Termika gleby i kara za przymrozki (0..8 szt./h, kara do -6 szt./h)
    if 8.0 <= soil_t <= 16.0:
        temp_score = 8.0
    elif soil_t < 8.0:
        temp_score = max(0.0, 8.0 - (8.0 - soil_t) * 1.4)
    else:
        temp_score = max(0.0, 8.0 - (soil_t - 16.0) * 1.2)

    frost_penalty = 0.0
    if tmin < 2.0:
        frost_penalty += min(4.0, (2.0 - tmin) * 0.9)
    if frost_cnt > 0:
        frost_penalty += min(3.0, frost_cnt * 1.0)

    # 4. Potencjał leśny BDL/GUS (0..10 szt./h)
    forest_score = min(10.0, (fc / 55.0) * 10.0)

    base_yield = float(np.clip(rain_score + moisture_score + temp_score + forest_score - frost_penalty, 0.0, 45.0))

    mult, sp_note = compute_species_habitat_multiplier(
        species_filter=species_filter,
        forest_cover_pct=row.get("forest_cover_pct"),
        pine_share_pct=row.get("pine_share_pct"),
        oak_beech_share_pct=row.get("oak_beech_share_pct"),
        spruce_share_pct=row.get("spruce_share_pct"),
        mean_stand_age_years=row.get("mean_stand_age_years"),
    )
    adjusted_yield = round(max(0.0, base_yield * mult), 2)
    ab_cls = str(classify_abundance(adjusted_yield))
    has_bdl = bool(row.get("has_bdl_stand_data", False))
    confidence = round(0.70 if has_bdl else 0.55, 2)

    note = (
        f"[Wskaźnik regułowy] Opad 21d: {p21:.1f} mm (impuls 15-21d: {p_slice_15_21:.1f} mm), "
        f"wilg. gleby: {soil_m:.3f} m³/m³, temp. min: {tmin:.1f}°C (przymrozki 7d: {frost_cnt}). {sp_note}"
    )

    return {
        "is_available": True,
        "availability_status": "AVAILABLE",
        "unavailable_reason": None,
        "predicted_yield_pcs_per_h": adjusted_yield,
        "base_predicted_yield_pcs_per_h": round(base_yield, 2),
        "species_habitat_multiplier": mult,
        "abundance_class": ab_cls,
        "confidence_score": confidence,
        "habitat_match_note": note,
        "model_version": RULE_INDEX_VERSION,
        "prediction_source": "RULE_BASED_INDEX",
    }


def _score_ml_rows_live(
    inputs_df: pd.DataFrame,
    species_filter: str,
    reg_pipe: Any,
    clf_pipe: Any,
) -> dict[str, dict[str, Any]]:
    """Uruchamia wytrenowane potoki ML wyłącznie dla wierszy z wystarczającą historią pogodową."""
    sufficient_mask = inputs_df["history_completeness"].isin(SUFFICIENT_HISTORY_STATUSES)
    sufficient_df = inputs_df[sufficient_mask].copy()
    scored_by_area: dict[str, dict[str, Any]] = {}

    if not sufficient_df.empty:
        raw_preds = np.clip(reg_pipe.predict(sufficient_df), a_min=0.0, a_max=None)
        direct_cls = clf_pipe.predict(sufficient_df)
        probs_cls = clf_pipe.predict_proba(sufficient_df)
        clf_classes = [str(c) for c in clf_pipe.named_steps["model"].classes_]

        for idx, (_, row) in enumerate(sufficient_df.iterrows()):
            area_id = str(row["area_id"]).strip().zfill(4)
            base_yield = float(raw_preds[idx])
            mult, sp_note = compute_species_habitat_multiplier(
                species_filter=species_filter,
                forest_cover_pct=row.get("forest_cover_pct"),
                pine_share_pct=row.get("pine_share_pct"),
                oak_beech_share_pct=row.get("oak_beech_share_pct"),
                spruce_share_pct=row.get("spruce_share_pct"),
                mean_stand_age_years=row.get("mean_stand_age_years"),
            )
            adjusted_yield = round(max(0.0, base_yield * mult), 2)
            ab_cls = str(classify_abundance(adjusted_yield))
            prob_map = {
                cls_name: round(float(probs_cls[idx, c_i]), 4)
                for c_i, cls_name in enumerate(clf_classes)
            }
            max_prob = max(prob_map.values()) if prob_map else 0.5
            has_bdl = bool(row.get("has_bdl_stand_data", False))
            confidence = round(min(0.95, max(0.35, 0.5 * max_prob + (0.35 if has_bdl else 0.20))), 2)

            scored_by_area[area_id] = {
                "is_available": True,
                "availability_status": "AVAILABLE",
                "unavailable_reason": None,
                "predicted_yield_pcs_per_h": adjusted_yield,
                "base_predicted_yield_pcs_per_h": round(base_yield, 2),
                "species_habitat_multiplier": mult,
                "abundance_class": ab_cls,
                "abundance_class_classifier": str(direct_cls[idx]),
                "class_probabilities": prob_map,
                "confidence_score": confidence,
                "habitat_match_note": f"[Model ML {MODEL_VERSION}] {sp_note}",
                "model_version": MODEL_VERSION,
                "prediction_source": "ML_MODEL",
                "execution_mode": "LIVE_PIPELINE",
            }

    return scored_by_area


def generate_predictions_for_date(
    target_date: str,
    horizon_days: int = 1,
    species_filter: str = "all",
    prediction_engine: str = "ML_MODEL",
    areas_by_id: dict[str, dict[str, Any]] | None = None,
    origin_lat: float = 51.1079,
    origin_lon: float = 17.0385,
    radius_km: float = 160.0,
    region_filter: str = "ALL",
    precomputed_df: pd.DataFrame | None = None,
    force_live: bool = False,
    forest_lookup: dict[str, dict[str, Any]] | None = None,
    weather_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Główna funkcja zasilająca mapę prognoz w aplikacji Streamlit ([GRZ-16]).

    Dla wybranej daty docelowej `target_date`, horyzontu `horizon_days` (1..3 dni),
    filtra gatunku oraz obszaru:
    1. Wyznacza `issued_at = target_date - horizon_days` (Point-in-Time Cutoff z [GRZ-15]).
    2. Buduje cechy wejściowe dla wszystkich 30 powiatów za pomocą `build_prediction_inputs_batch`.
    3. Dla obszarów ze statusem `COMPLETE` lub `COMPLETE_WEATHER_NO_BDL_STAND`:
       - w trybie `ML_MODEL`: wybiera wynik z `predictions.csv` (jeśli dostępny dla tej daty i H=1
         oraz `force_live=False`) albo uruchamia bezpośrednio zapisany potok `.joblib`,
       - w trybie `RULE_BASED_INDEX`: oblicza Jawny Wskaźnik Warunków Grzybobrania.
       Dodatkowo zawsze dołącza pomocniczo wynik drugiego silnika do porównania w tabeli
       (bez mieszania ich ze sobą!).
    4. Dla obszarów bez wystarczających danych (`PARTIAL_HISTORY`, `NO_WEATHER_HISTORY`,
       `OUT_OF_COVERAGE`) ustawia jawnie `is_available = False`, `predicted_yield_pcs_per_h = None`
       oraz `abundance_class = "NIEDOSTEPNY"`.
    """
    t_target = date.fromisoformat(str(target_date).strip())
    t_issued = t_target - timedelta(days=int(horizon_days))
    target_iso = t_target.isoformat()
    issued_iso = t_issued.isoformat()

    if forest_lookup is None or weather_lookup is None:
        cached_forest, cached_weather = load_cached_inference_reference()
        forest_lookup = forest_lookup or cached_forest
        weather_lookup = weather_lookup or cached_weather

    inputs_df = build_prediction_inputs_batch(
        issued_at=issued_iso,
        horizons=(int(horizon_days),),
        region_ids=None,
        forest_lookup=forest_lookup,
        weather_lookup=weather_lookup,
    )

    reg_pipe, clf_pipe = load_saved_model_pipelines()

    # Próba użycia gotowego wyniku z data/processed/predictions.csv (gdy H=1, ML_MODEL, force_live=False)
    precomputed_lookup: dict[str, dict[str, Any]] = {}
    used_precomputed = False
    if (
        prediction_engine == "ML_MODEL"
        and not force_live
        and int(horizon_days) == 1
        and precomputed_df is not None
        and not precomputed_df.empty
    ):
        sub_pre = precomputed_df[
            (precomputed_df["target_date"] == target_iso)
            & (precomputed_df["generated_at"] == issued_iso)
            & (precomputed_df["species_filter"] == species_filter)
            & (precomputed_df["is_demo"] == False)  # noqa: E712
        ]
        if len(sub_pre) > 0:
            for _, p_row in sub_pre.iterrows():
                aid = str(p_row["area_id"]).strip().zfill(4)
                precomputed_lookup[aid] = p_row.to_dict()
            used_precomputed = True

    # Policz na żywo brakujące lub wszystkie wiersze ML
    live_ml_by_area = _score_ml_rows_live(
        inputs_df=inputs_df,
        species_filter=species_filter,
        reg_pipe=reg_pipe,
        clf_pipe=clf_pipe,
    )

    records: list[dict[str, Any]] = []
    records_by_area: dict[str, dict[str, Any]] = {}

    for _, row in inputs_df.iterrows():
        area_id = str(row["area_id"]).strip().zfill(4)
        area_meta = (areas_by_id or {}).get(area_id, {})
        area_name = str(row.get("region_name") or area_meta.get("area_name") or area_id)

        c_lat = area_meta.get("centroid_lat")
        c_lon = area_meta.get("centroid_lon")
        if c_lat is not None and c_lon is not None:
            dist_km = haversine_km(origin_lat, origin_lon, float(c_lat), float(c_lon))
        else:
            dist_km = 0.0

        matches_county_filter = region_filter == "ALL" or area_id == str(region_filter).zfill(4)
        in_radius = (dist_km <= float(radius_km)) and matches_county_filter

        hist_status = str(row.get("history_completeness") or "NO_WEATHER_HISTORY")
        missing_days_21d = int(row.get("missing_weather_days_21d") or 0)
        has_sufficient_data = hist_status in SUFFICIENT_HISTORY_STATUSES

        # Wynik regułowy (zawsze liczony jako transparentne porównanie / silnik RULE_BASED_INDEX)
        rule_res = compute_rule_based_suitability_yield(row, species_filter=species_filter)

        # Wynik ML (z prekomputowanego CSV lub na żywo z potoku)
        if has_sufficient_data:
            live_ml = live_ml_by_area[area_id]
            if used_precomputed and area_id in precomputed_lookup and pd.notna(precomputed_lookup[area_id].get("predicted_yield_pcs_per_h")):
                pre_row = precomputed_lookup[area_id]
                ml_res = {
                    "is_available": True,
                    "availability_status": "AVAILABLE",
                    "unavailable_reason": None,
                    "predicted_yield_pcs_per_h": round(float(pre_row["predicted_yield_pcs_per_h"]), 2),
                    "base_predicted_yield_pcs_per_h": live_ml["base_predicted_yield_pcs_per_h"],
                    "species_habitat_multiplier": live_ml["species_habitat_multiplier"],
                    "abundance_class": str(pre_row["abundance_class"]),
                    "abundance_class_classifier": live_ml["abundance_class_classifier"],
                    "class_probabilities": live_ml["class_probabilities"],
                    "confidence_score": (
                        round(float(pre_row["confidence_score"]), 2)
                        if pd.notna(pre_row.get("confidence_score"))
                        else live_ml["confidence_score"]
                    ),
                    "habitat_match_note": str(pre_row.get("habitat_match_note") or live_ml["habitat_match_note"]),
                    "model_version": str(pre_row.get("model_version") or MODEL_VERSION),
                    "prediction_source": "ML_MODEL",
                    "execution_mode": "PRECOMPUTED_CSV",
                }
            else:
                ml_res = live_ml
        else:
            reason = (
                f"Brak wystarczających danych pogodowych: status `{hist_status}` "
                f"(brakuje {missing_days_21d} z 21 dni okna przed {issued_iso})."
            )
            ml_res = {
                "is_available": False,
                "availability_status": "UNAVAILABLE_INSUFFICIENT_DATA",
                "unavailable_reason": reason,
                "predicted_yield_pcs_per_h": None,
                "base_predicted_yield_pcs_per_h": None,
                "species_habitat_multiplier": None,
                "abundance_class": "NIEDOSTEPNY",
                "abundance_class_classifier": "NIEDOSTEPNY",
                "class_probabilities": {},
                "confidence_score": None,
                "habitat_match_note": reason,
                "model_version": MODEL_VERSION,
                "prediction_source": "ML_MODEL",
                "execution_mode": "UNAVAILABLE",
            }

        active = ml_res if prediction_engine == "ML_MODEL" else rule_res

        def _opt_float(val: Any) -> float | None:
            if val is None or pd.isna(val):
                return None
            return float(val)

        record: dict[str, Any] = {
            "area_id": area_id,
            "area_name": area_name,
            "target_date": target_iso,
            "generated_at": issued_iso,
            "feature_cutoff_date": issued_iso,
            "horizon_days": int(horizon_days),
            "species_filter": species_filter,
            "distance_km": dist_km,
            "in_radius": in_radius,
            "history_completeness": hist_status,
            "missing_weather_days_21d": missing_days_21d,
            "is_available": bool(active["is_available"]),
            "availability_status": str(active["availability_status"]),
            "unavailable_reason": active["unavailable_reason"],
            "predicted_yield_pcs_per_h": active["predicted_yield_pcs_per_h"],
            "base_predicted_yield_pcs_per_h": active["base_predicted_yield_pcs_per_h"],
            "species_habitat_multiplier": active["species_habitat_multiplier"],
            "target_unit": TARGET_UNIT,
            "abundance_class": active["abundance_class"],
            "abundance_class_classifier": ml_res.get("abundance_class_classifier", "NIEDOSTEPNY"),
            "confidence_score": active["confidence_score"],
            "habitat_match_note": active["habitat_match_note"],
            "model_version": active["model_version"],
            "prediction_source": active["prediction_source"],
            "execution_mode": ml_res.get("execution_mode", "LIVE_PIPELINE")
            if prediction_engine == "ML_MODEL"
            else "LIVE_RULE_INDEX",
            "ml_yield_pcs_per_h": ml_res["predicted_yield_pcs_per_h"],
            "rule_based_yield_pcs_per_h": rule_res["predicted_yield_pcs_per_h"],
            "is_demo": False,
            # Cechy kontekstowe dla tabeli i tooltipu
            "total_area_km2": _opt_float(row.get("total_area_km2")),
            "forest_cover_pct": _opt_float(row.get("forest_cover_pct")),
            "pine_share_pct": _opt_float(row.get("pine_share_pct")),
            "oak_beech_share_pct": _opt_float(row.get("oak_beech_share_pct")),
            "spruce_share_pct": _opt_float(row.get("spruce_share_pct")),
            "mean_stand_age_years": _opt_float(row.get("mean_stand_age_years")),
            "dominant_moisture_type": (
                None
                if row.get("dominant_moisture_type") is None or pd.isna(row.get("dominant_moisture_type"))
                else str(row.get("dominant_moisture_type"))
            ),
            "has_bdl_stand_data": bool(row.get("has_bdl_stand_data", False)),
            "precip_lag_7d_mm": _opt_float(row.get("precip_lag_7d_mm")),
            "precip_lag_14d_mm": _opt_float(row.get("precip_lag_14d_mm")),
            "precip_lag_21d_mm": _opt_float(row.get("precip_lag_21d_mm")),
            "temp_min_night_c": _opt_float(row.get("temp_min_night_c")),
            "soil_moisture_0_7cm_m3m3": _opt_float(row.get("soil_moisture_0_7cm_m3m3")),
            "soil_temperature_0_7cm_c": _opt_float(row.get("soil_temperature_0_7cm_c")),
        }
        records.append(record)
        records_by_area[area_id] = record

    records.sort(
        key=lambda r: (
            r["in_radius"],
            r["is_available"],
            r["predicted_yield_pcs_per_h"] if r["predicted_yield_pcs_per_h"] is not None else -1.0,
            -r["distance_km"],
        ),
        reverse=True,
    )

    in_filter = [r for r in records if r["in_radius"]]
    available_in_filter = [r for r in in_filter if r["is_available"]]
    unavailable_in_filter = [r for r in in_filter if not r["is_available"]]

    execution_summary = (
        "PRECOMPUTED_CSV (data/processed/predictions.csv)"
        if (prediction_engine == "ML_MODEL" and used_precomputed and available_in_filter)
        else (
            "LIVE_PIPELINE (models/rf_yield_pipeline.joblib)"
            if prediction_engine == "ML_MODEL"
            else "LIVE_RULE_INDEX (Jawny Wskaźnik Warunków)"
        )
    )

    return {
        "target_date": target_iso,
        "issued_at": issued_iso,
        "feature_cutoff_date": issued_iso,
        "horizon_days": int(horizon_days),
        "species_filter": species_filter,
        "prediction_engine": prediction_engine,
        "model_version": MODEL_VERSION if prediction_engine == "ML_MODEL" else RULE_INDEX_VERSION,
        "target_unit": TARGET_UNIT,
        "execution_summary": execution_summary,
        "used_precomputed_csv": used_precomputed,
        "records": records,
        "records_by_area": records_by_area,
        "in_filter_count": len(in_filter),
        "available_count": len(available_in_filter),
        "unavailable_count": len(unavailable_in_filter),
        "is_demo": False,
    }


def verify_map_predictions_against_direct_model(
    enriched_geojson: dict[str, Any],
    target_date: str,
    horizon_days: int = 1,
    species_filter: str = "all",
    sample_area_ids: Sequence[str] = ("0201", "0206", "0208", "0213", "0218", "0217"),
) -> dict[str, Any]:
    """
    Weryfikuje zgodność wartości wyświetlanych na mapie (`enriched_geojson`)
    z bezpośrednim wywołaniem zapisanego modelu `.joblib` ([GRZ-16] Warunek #5).

    Dla każdego wybranego powiatu:
    1. Odczytuje wartość `predicted_yield_pcs_per_h` i `abundance_class` bezpośrednio z `properties`
       poligonu przekazywanego do warstwy PyDeck (`enriched_geojson`).
    2. Niezależnie buduje 1 wiersz cech przez `build_prediction_input_row` i wywołuje wprost
       `joblib.load("models/rf_yield_pipeline.joblib").predict(...)` oraz mnożnik siedliskowy BDL.
    3. Porównuje oba wyniki z tolerancją `1e-6`.
    """
    t_target = date.fromisoformat(target_date)
    issued_iso = (t_target - timedelta(days=int(horizon_days))).isoformat()

    forest_lookup, weather_lookup = load_cached_inference_reference()
    reg_pipe = joblib.load(MODELS_DIR / "rf_yield_pipeline.joblib")
    clf_pipe = joblib.load(MODELS_DIR / "rf_abundance_pipeline.joblib")

    props_by_area = {
        str(feat["properties"]["area_id"]).zfill(4): feat["properties"]
        for feat in enriched_geojson.get("features", [])
    }

    checks: list[dict[str, Any]] = []
    all_match = True

    for aid in sample_area_ids:
        area_id = str(aid).zfill(4)
        map_props = props_by_area.get(area_id, {})

        input_row = build_prediction_input_row(
            region_id=area_id,
            issued_at=issued_iso,
            horizon_days=int(horizon_days),
            forest_lookup=forest_lookup,
            weather_lookup=weather_lookup,
        )
        hist_status = input_row["history_completeness"]

        map_yield = map_props.get("predicted_yield_pcs_per_h")
        map_class = map_props.get("abundance_class")

        if hist_status in SUFFICIENT_HISTORY_STATUSES:
            direct_df = pd.DataFrame([input_row])
            base_direct = float(np.clip(reg_pipe.predict(direct_df)[0], 0.0, None))
            mult, _ = compute_species_habitat_multiplier(
                species_filter=species_filter,
                forest_cover_pct=input_row.get("forest_cover_pct"),
                pine_share_pct=input_row.get("pine_share_pct"),
                oak_beech_share_pct=input_row.get("oak_beech_share_pct"),
                spruce_share_pct=input_row.get("spruce_share_pct"),
                mean_stand_age_years=input_row.get("mean_stand_age_years"),
            )
            direct_yield = round(max(0.0, base_direct * mult), 2)
            direct_class = str(classify_abundance(direct_yield))
            direct_clf_class = str(clf_pipe.predict(direct_df)[0])

            yield_match = (
                map_yield is not None
                and abs(float(map_yield) - direct_yield) < 1e-6
            )
            class_match = str(map_class) == direct_class
            row_ok = yield_match and class_match
        else:
            direct_yield = None
            direct_class = "NIEDOSTEPNY"
            direct_clf_class = "NIEDOSTEPNY"
            row_ok = (map_yield is None) and (str(map_class) == "NIEDOSTEPNY")

        if not row_ok:
            all_match = False

        checks.append(
            {
                "area_id": area_id,
                "area_name": str(map_props.get("area_name") or input_row.get("region_name") or area_id),
                "history_completeness": hist_status,
                "has_bdl_stand_data": bool(input_row.get("has_bdl_stand_data", False)),
                "map_yield_pcs_per_h": map_yield,
                "direct_model_yield_pcs_per_h": direct_yield,
                "abs_diff_pcs_per_h": (
                    round(abs(float(map_yield) - float(direct_yield)), 6)
                    if (map_yield is not None and direct_yield is not None)
                    else 0.0 if (map_yield is None and direct_yield is None) else None
                ),
                "map_abundance_class": map_class,
                "direct_abundance_class": direct_class,
                "direct_classifier_class": direct_clf_class,
                "matches_1_to_1": row_ok,
            }
        )

    return {
        "target_date": target_date,
        "issued_at": issued_iso,
        "horizon_days": int(horizon_days),
        "species_filter": species_filter,
        "model_version": MODEL_VERSION,
        "checked_counties_count": len(checks),
        "all_match": all_match,
        "checks": checks,
    }


def generate_precomputed_predictions_csv(
    start_target_date: str = PRECOMPUTED_START_DATE,
    end_target_date: str = PRECOMPUTED_END_DATE,
    species_list: Sequence[str] | None = None,
    output_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generuje produkcyjny plik `data/processed/predictions.csv` zgodny z kontraktem danych
    (`docs/data_contract.md`, Sekcja 2.4) dla wszystkich 30 powiatów, wybranego zakresu dat
    (`horizon_days = 1`) oraz wszystkich filtrów gatunkowych.
    """
    out_path = output_csv_path or (PROCESSED_DIR / "predictions.csv")
    species_filters = list(species_list) if species_list is not None else list(SPECIES_LABELS.keys())

    forest_lookup, weather_lookup = load_cached_inference_reference()
    reg_pipe, clf_pipe = load_saved_model_pipelines()

    d_start = date.fromisoformat(start_target_date)
    d_end = date.fromisoformat(end_target_date)

    contract_rows: list[dict[str, Any]] = []
    cur = d_start
    while cur <= d_end:
        target_iso = cur.isoformat()
        issued_iso = (cur - timedelta(days=1)).isoformat()

        inputs_df = build_prediction_inputs_batch(
            issued_at=issued_iso,
            horizons=(1,),
            region_ids=None,
            forest_lookup=forest_lookup,
            weather_lookup=weather_lookup,
        )

        # Raz wylicz bazową predykcję ML dla 30 powiatów na dany dzień, a potem zastosuj mnożnik gatunkowy
        sufficient_mask = inputs_df["history_completeness"].isin(SUFFICIENT_HISTORY_STATUSES)
        sufficient_df = inputs_df[sufficient_mask].copy()
        if not sufficient_df.empty:
            base_preds = np.clip(reg_pipe.predict(sufficient_df), a_min=0.0, a_max=None)
            probs_cls = clf_pipe.predict_proba(sufficient_df)

            for idx, (_, row) in enumerate(sufficient_df.iterrows()):
                area_id = str(row["area_id"]).strip().zfill(4)
                base_yield = float(base_preds[idx])
                max_prob = float(np.max(probs_cls[idx])) if len(probs_cls[idx]) > 0 else 0.5
                has_bdl = bool(row.get("has_bdl_stand_data", False))
                confidence = round(min(0.95, max(0.35, 0.5 * max_prob + (0.35 if has_bdl else 0.20))), 2)

                for sp in species_filters:
                    mult, sp_note = compute_species_habitat_multiplier(
                        species_filter=sp,
                        forest_cover_pct=row.get("forest_cover_pct"),
                        pine_share_pct=row.get("pine_share_pct"),
                        oak_beech_share_pct=row.get("oak_beech_share_pct"),
                        spruce_share_pct=row.get("spruce_share_pct"),
                        mean_stand_age_years=row.get("mean_stand_age_years"),
                    )
                    adj_yield = round(max(0.0, base_yield * mult), 2)
                    ab_cls = str(classify_abundance(adj_yield))

                    contract_rows.append(
                        {
                            "target_date": target_iso,
                            "area_id": area_id,
                            "species_filter": sp,
                            "predicted_yield_pcs_per_h": adj_yield,
                            "abundance_class": ab_cls,
                            "confidence_score": confidence,
                            "habitat_match_note": f"[Model ML {MODEL_VERSION}] {sp_note}",
                            "generated_at": issued_iso,
                            "model_version": MODEL_VERSION,
                            "is_demo": False,
                        }
                    )
        cur += timedelta(days=1)

    df_out = pd.DataFrame(contract_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    return df_out
