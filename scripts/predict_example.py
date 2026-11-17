#!/usr/bin/env python3
"""
Skrypt demonstracyjny wykonujący predykcję z zapisanego potoku ML (`.joblib`)
bez konieczności uruchamiania treningu ([GRZ-14] Warunek #5).

Uruchomienie domyślne (wczytuje models/example_input.json i zapisuje models/example_output.json/.csv):
    python scripts/predict_example.py

Uruchomienie dla własnego pliku CSV lub JSON:
    python scripts/predict_example.py --input models/example_input.csv --output models/example_output.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MODELS_DIR
from src.model_pipeline import predict_with_saved_pipelines


def load_input_records(input_path: Path) -> list[dict[str, Any]] | pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego do predykcji: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        return list(data)
    elif suffix == ".csv":
        return pd.read_csv(input_path, dtype={"area_id": str, "region_id": str})
    elif suffix == ".parquet":
        return pd.read_parquet(input_path)
    else:
        raise ValueError(f"Nieobsługiwany format pliku wejściowego: {suffix} (oczekiwano .json, .csv lub .parquet)")


def run_prediction_example(
    input_path: Path | None = None,
    output_json_path: Path | None = None,
    output_csv_path: Path | None = None,
    reg_pipeline_path: Path | None = None,
    clf_pipeline_path: Path | None = None,
) -> list[dict[str, Any]]:
    in_path = input_path or (MODELS_DIR / "example_input.json")
    out_json = output_json_path or (MODELS_DIR / "example_output.json")
    out_csv = output_csv_path or (MODELS_DIR / "example_output.csv")

    records = load_input_records(in_path)
    predictions = predict_with_saved_pipelines(
        records,
        reg_pipeline_path=reg_pipeline_path,
        clf_pipeline_path=clf_pipeline_path,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    flat_rows = []
    for item in predictions:
        flat_item = {k: v for k, v in item.items() if k != "class_probabilities"}
        probs = item.get("class_probabilities", {})
        flat_item["prob_BRAK"] = probs.get("BRAK")
        flat_item["prob_SREDNIO"] = probs.get("SREDNIO")
        flat_item["prob_WYSYP"] = probs.get("WYSYP")
        flat_rows.append(flat_item)
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False, encoding="utf-8")

    print("=" * 88)
    print(f"[GRZ-14] Predykcja z gotowego potoku ML (bez uruchamiania treningu)")
    print(f"Wejście: {in_path} (liczba rekordów: {len(predictions)})")
    print("=" * 88)
    print(
        f"{'#':<3} {'Powiat (TERYT)':<30} {'Data':<12} {'Gatunek':<20} "
        f"{'Wynik [szt./h]':<16} {'Klasa (Reg/Clf)':<16} {'BDL'}"
    )
    print("-" * 88)
    for item in predictions:
        pow_label = f"{item['area_id']} ({item['region_name'][:20]})"
        bdl_flag = "TAK" if item["has_bdl_stand_data"] else "NULL (brak BDL)"
        cls_pair = f"{item['abundance_class']}/{item['abundance_class_classifier']}"
        print(
            f"{item['record_index']+1:<3} {pow_label:<30} {item['target_date']:<12} "
            f"{item['species_filter']:<20} {item['predicted_yield_pcs_per_h']:>6.2f} {item['target_unit']:<8} "
            f"{cls_pair:<16} {bdl_flag}"
        )
    print("-" * 88)
    print(f"Zapisano wyjście JSON: {out_json}")
    print(f"Zapisano wyjście CSV:  {out_csv}")
    print("=" * 88)

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wykonuje przykładową predykcję z wytrenowanego potoku ML bez trenowania [GRZ-14]."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=MODELS_DIR / "example_input.json",
        help="Ścieżka do pliku wejściowego (.json, .csv lub .parquet)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=MODELS_DIR / "example_output.json",
        help="Ścieżka do wyjściowego pliku JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=MODELS_DIR / "example_output.csv",
        help="Ścieżka do wyjściowego pliku CSV",
    )
    args = parser.parse_args()
    run_prediction_example(
        input_path=args.input,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
    )


if __name__ == "__main__":
    main()
