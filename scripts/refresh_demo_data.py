#!/usr/bin/env python3
"""
Jedno polecenie odświeżające dane, cechy i predykcje przed demonstracją [GRZ-20].

Użycie:
    python scripts/refresh_demo_data.py              # bezpieczne odtworzenie z lokalnego cache (offline)
    python scripts/refresh_demo_data.py --fetch-weather  # próba aktualizacji pogody + pełne przeliczenie
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DIR
from src.features import build_features_dataframe
from src.prediction_service import PRECOMPUTED_END_DATE, PRECOMPUTED_START_DATE
from src.refresh_pipeline import (
    assess_predictions_freshness,
    atomic_write_via_temp,
    refresh_predictions_atomically,
    safe_refresh_weather_step,
    write_refresh_manifest,
)


def _refresh_features_atomically() -> dict:
    reports_spatial = pd.read_parquet(PROCESSED_DIR / "reports_spatial.parquet")
    forest_features = pd.read_parquet(PROCESSED_DIR / "forest_features.parquet")
    weather_daily = pd.read_parquet(PROCESSED_DIR / "weather_daily.parquet")

    df_features = build_features_dataframe(
        reports_spatial_df=reports_spatial,
        forest_features_df=forest_features,
        weather_daily_df=weather_daily,
        horizon_days=1,
        valid_only=True,
    )
    if len(df_features) < 513:
        raise ValueError(f"Zbyt mało wierszy w features: {len(df_features)}")

    feat_parquet = PROCESSED_DIR / "features.parquet"
    feat_csv = PROCESSED_DIR / "features.csv"

    atomic_write_via_temp(
        feat_parquet,
        lambda p: df_features.to_parquet(p, index=False),
        lambda p: None if len(pd.read_parquet(p)) == len(df_features) else (_ for _ in ()).throw(ValueError("Błąd walidacji Parquet")),
    )
    atomic_write_via_temp(
        feat_csv,
        lambda p: df_features.to_csv(p, index=False),
        lambda p: None if len(pd.read_csv(p)) == len(df_features) else (_ for _ in ()).throw(ValueError("Błąd walidacji CSV")),
    )

    return {
        "status": "REBUILT_ATOMICALLY",
        "rows_count": int(len(df_features)),
        "columns_count": int(len(df_features.columns)),
    }


def _run_weather_fetch_subprocess() -> pd.DataFrame:
    res = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "fetch_pilot_weather.py")],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if res.returncode != 0:
        raise RuntimeError(f"fetch_pilot_weather.py zakończony kodem {res.returncode}: {res.stderr.strip()}")
    return pd.read_parquet(PROCESSED_DIR / "weather_daily.parquet")


def run_refresh(
    fetch_weather: bool = False,
    start_date: str = PRECOMPUTED_START_DATE,
    end_date: str = PRECOMPUTED_END_DATE,
    reference_date: str | None = None,
) -> dict:
    # 1. Krok pogodowy (opcjonalnie online z pełną ochroną przed nadpisaniem w razie awarii)
    if fetch_weather:
        weather_status = safe_refresh_weather_step(_run_weather_fetch_subprocess)
    else:
        weather_path = PROCESSED_DIR / "weather_daily.parquet"
        existing_rows = int(len(pd.read_parquet(weather_path))) if weather_path.exists() else 0
        weather_status = {
            "status": "SKIPPED_OFFLINE_USING_CACHE",
            "rows_after": existing_rows,
            "preserved_existing": True,
            "error": None,
        }

    # 2. Atomowe odtworzenie cech (features.parquet & features.csv)
    features_status = _refresh_features_atomically()

    # 3. Atomowe odtworzenie prognoz (predictions.csv)
    predictions_status = refresh_predictions_atomically(
        start_target_date=start_date,
        end_target_date=end_date,
    )

    # 4. Ocena aktualności / oznaczenie wyników archiwalnych
    freshness = assess_predictions_freshness(
        PROCESSED_DIR / "predictions.csv",
        reference_date=reference_date,
    )

    manifest = {
        "mode": "online_weather_and_recompute" if fetch_weather else "offline_recompute",
        "weather_step": weather_status,
        "features_step": features_status,
        "predictions_step": predictions_status,
        "freshness": freshness,
    }
    manifest_path = write_refresh_manifest(manifest)

    print("============================================================")
    print(f"[GRZ-20] Odświeżanie zakończone ({manifest['mode']})")
    print(f"  Pogoda:     {weather_status['status']} ({weather_status['rows_after']} wierszy)")
    print(f"  Cechy:      {features_status['status']} ({features_status['rows_count']} wierszy)")
    print(f"  Predykcje:  {predictions_status['status']} ({predictions_status['rows_count']} wierszy, model={predictions_status['model_version']})")
    print(f"  Aktualność: {freshness['freshness_status']} (is_stale={freshness['is_stale']})")
    if freshness.get("warning_message"):
        print(f"  ! {freshness['warning_message']}")
    print(f"  Manifest:   {manifest_path}")
    print("============================================================")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Odświeżanie danych i wyników przed demonstracją [GRZ-20].")
    parser.add_argument(
        "--fetch-weather",
        action="store_true",
        help="Spróbuj pobrać najnowszą pogodę z Open-Meteo (w razie awarii zachowuje istniejące pliki).",
    )
    parser.add_argument("--start-date", default=PRECOMPUTED_START_DATE, help="Początkowa data docelowa predictions.csv")
    parser.add_argument("--end-date", default=PRECOMPUTED_END_DATE, help="Końcowa data docelowa predictions.csv")
    parser.add_argument("--reference-date", default=None, help="Opcjonalna data odniesienia (YYYY-MM-DD) do testu aktualności")
    args = parser.parse_args()

    run_refresh(
        fetch_weather=args.fetch_weather,
        start_date=args.start_date,
        end_date=args.end_date,
        reference_date=args.reference_date,
    )


if __name__ == "__main__":
    main()
