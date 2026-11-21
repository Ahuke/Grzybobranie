"""
Bezpieczny potok odświeżania danych i wyników przed demonstracją [GRZ-20].

Zapewnia:
1. Atomowe zapisywanie i podmianę plików wyjściowych (awaria pobierania lub walidacji
   NIGDY nie nadpisuje poprawnych plików pustymi ani uszkodzonymi danymi).
2. Jawną kontrolę aktualności wyników (`assess_predictions_freshness`): stare wyniki
   archiwalne są oznaczane jako `STALE_ARCHIVE`, a nie przedstawiane jako bieżące.
3. Jednoznaczny manifest odświeżania (`data/processed/refresh_manifest.json`) z datą
   wystawienia (`generated_at` / `issued_at`), datą docelową (`target_date`) i wersją modelu.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import pandas as pd

from src.config import PROCESSED_DIR
from src.data_loader import load_predictions_csv
from src.prediction_service import (
    MODEL_VERSION,
    PRECOMPUTED_END_DATE,
    PRECOMPUTED_START_DATE,
    generate_precomputed_predictions_csv,
)


def atomic_write_via_temp(
    target_path: Path,
    writer_fn: Callable[[Path], None],
    validator_fn: Callable[[Path], None] | None = None,
) -> None:
    """
    Zapisuje plik najpierw do pliku tymczasowego w tym samym katalogu, uruchamia opcjonalną
    walidację (`validator_fn`), a dopiero po sukcesie wykonuje atomową podmianę `os.replace`.
    W razie błędu plik docelowy pozostaje nienaruszony.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".tmp_{target_path.stem}_",
        suffix=target_path.suffix,
        dir=str(target_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer_fn(tmp_path)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise ValueError(f"Wygenerowany plik tymczasowy jest pusty: {tmp_path}")
        if validator_fn is not None:
            validator_fn(tmp_path)
        os.replace(str(tmp_path), str(target_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def assess_predictions_freshness(
    predictions_csv_path: Path = PROCESSED_DIR / "predictions.csv",
    reference_date: str | date | None = None,
) -> dict[str, Any]:
    """
    Ocenia, czy zapisane w `predictions.csv` prognozy są bieżące względem `reference_date`
    (domyślnie dzisiejsza data), czy też są wynikami historycznymi/archiwalnymi (`STALE_ARCHIVE`).
    """
    if not predictions_csv_path.exists():
        return {
            "exists": False,
            "is_stale": True,
            "freshness_status": "MISSING",
            "warning_message": "Brak pliku data/processed/predictions.csv.",
        }

    df = pd.read_csv(predictions_csv_path)
    if df.empty or "target_date" not in df.columns or "generated_at" not in df.columns:
        return {
            "exists": True,
            "is_stale": True,
            "freshness_status": "INVALID_OR_EMPTY",
            "warning_message": "Plik predictions.csv jest pusty lub niekompletny.",
        }

    ref_dt = (
        date.fromisoformat(reference_date)
        if isinstance(reference_date, str)
        else (reference_date or date.today())
    )
    max_target_str = str(df["target_date"].astype(str).max())
    min_target_str = str(df["target_date"].astype(str).min())
    max_issued_str = str(df["generated_at"].astype(str).max())
    min_issued_str = str(df["generated_at"].astype(str).min())
    model_versions = sorted(df["model_version"].dropna().astype(str).unique().tolist())

    max_target_dt = date.fromisoformat(max_target_str)
    age_days = (ref_dt - max_target_dt).days
    is_stale = bool(age_days > 0)

    if is_stale:
        status = "STALE_ARCHIVE"
        warning = (
            f"UWAGA: Zapisane prognozy są przestarzałe / archiwalne "
            f"(ostatnia data docelowa: {max_target_str}, sprzed {age_days} dni względem {ref_dt.isoformat()}). "
            f"Nie przedstawiać ich jako prognoz bieżących bez aktualizacji pogody."
        )
    else:
        status = "CURRENT"
        warning = None

    return {
        "exists": True,
        "rows_count": int(len(df)),
        "reference_date": ref_dt.isoformat(),
        "target_date_min": min_target_str,
        "target_date_max": max_target_str,
        "issued_at_min": min_issued_str,
        "issued_at_max": max_issued_str,
        "model_versions": model_versions,
        "age_days_since_max_target": int(age_days),
        "is_stale": is_stale,
        "freshness_status": status,
        "warning_message": warning,
    }


def safe_refresh_weather_step(
    fetcher_fn: Callable[[], pd.DataFrame],
    weather_daily_path: Path = PROCESSED_DIR / "weather_daily.parquet",
    min_expected_rows: int = 2910,
) -> dict[str, Any]:
    """
    Wykonuje bezpieczne odświeżenie danych pogodowych z ochroną przed nadpisaniem
    istniejących poprawnych plików w razie awarii sieci, błędu API lub pustych danych.
    """
    had_existing = weather_daily_path.exists()
    existing_rows = int(len(pd.read_parquet(weather_daily_path))) if had_existing else 0

    try:
        new_df = fetcher_fn()
        if new_df is None or not isinstance(new_df, pd.DataFrame) or new_df.empty:
            raise ValueError("Pobrany zbiór pogodowy jest pusty.")
        if len(new_df) < min_expected_rows:
            raise ValueError(
                f"Pobrany zbiór ma zbyt mało wierszy ({len(new_df)} < {min_expected_rows})."
            )

        def _write_parquet(p: Path) -> None:
            new_df.to_parquet(p, index=False)

        def _validate_parquet(p: Path) -> None:
            check_df = pd.read_parquet(p)
            if len(check_df) < min_expected_rows:
                raise ValueError("Walidacja zapisanego Parquet nie powiodła się.")

        atomic_write_via_temp(weather_daily_path, _write_parquet, _validate_parquet)
        return {
            "status": "UPDATED",
            "rows_after": int(len(new_df)),
            "preserved_existing": False,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "FAILED_PRESERVED_EXISTING" if had_existing else "FAILED_NO_EXISTING",
            "rows_after": existing_rows,
            "preserved_existing": had_existing,
            "error": str(exc),
        }


def refresh_predictions_atomically(
    start_target_date: str = PRECOMPUTED_START_DATE,
    end_target_date: str = PRECOMPUTED_END_DATE,
    output_csv_path: Path = PROCESSED_DIR / "predictions.csv",
) -> dict[str, Any]:
    """
    Atomowo przelicza i waliduje `predictions.csv` (zapis przez plik tymczasowy + `load_predictions_csv`).
    """
    def _writer(tmp_csv: Path) -> None:
        generate_precomputed_predictions_csv(
            start_target_date=start_target_date,
            end_target_date=end_target_date,
            output_csv_path=tmp_csv,
        )

    def _validator(tmp_csv: Path) -> None:
        loaded = load_predictions_csv(tmp_csv)
        if not loaded:
            raise ValueError("Wygenerowany plik predictions.csv jest pusty!")
        if any(r.get("is_demo") for r in loaded):
            raise ValueError("Wykryto niedozwolony rekord is_demo=True w predictions.csv!")
        if any(not r.get("generated_at") or not r.get("target_date") or not r.get("model_version") for r in loaded):
            raise ValueError("Brak wymaganych pól (generated_at / target_date / model_version)!")

    atomic_write_via_temp(output_csv_path, _writer, _validator)
    df_out = pd.read_csv(output_csv_path)
    return {
        "status": "REBUILT_ATOMICALLY",
        "rows_count": int(len(df_out)),
        "model_version": MODEL_VERSION,
        "target_date_min": str(df_out["target_date"].min()),
        "target_date_max": str(df_out["target_date"].max()),
        "issued_at_min": str(df_out["generated_at"].min()),
        "issued_at_max": str(df_out["generated_at"].max()),
    }


def write_refresh_manifest(
    manifest_data: dict[str, Any],
    manifest_path: Path = PROCESSED_DIR / "refresh_manifest.json",
) -> Path:
    """Zapisuje manifest odświeżania atomowo do JSON."""
    payload = {
        "task_id": "GRZ-20",
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **manifest_data,
    }

    def _writer(p: Path) -> None:
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    atomic_write_via_temp(manifest_path, _writer)
    return manifest_path
