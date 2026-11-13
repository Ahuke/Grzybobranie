#!/usr/bin/env python3
"""
Skrypt CLI generujący dane wejściowe do prognoz dla wybranych dat i obszarów [GRZ-15].

Przykładowe użycie:
    PYTHONPATH=. python scripts/build_prediction_inputs.py --issued-at 2025-10-18 --horizons 1 2 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import FEATURE_DICTIONARY
from src.inference_features import (
    MODEL_INPUT_FEATURE_COLUMNS,
    RegionOutOfCoverageError,
    build_prediction_input_row,
    build_prediction_inputs_batch,
    load_inference_reference_data,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def run_edge_case_checks(
    forest_lookup: dict,
    weather_lookup: dict,
    issued_at: str,
) -> dict:
    """Weryfikuje zachowanie przy niepełnej historii, braku BDL, obszarze spoza pokrycia i braku wycieku."""
    # 1. Powiat bez profilu taksacyjnego BDL (0218 - średzki)
    row_no_bdl = build_prediction_input_row(
        region_id="0218",
        issued_at=issued_at,
        horizon_days=1,
        forest_lookup=forest_lookup,
        weather_lookup=weather_lookup,
    )

    # 2. Niepełna historia pogodowa (2025-08-10: archiwum zaczyna się 2025-08-01, więc 21d ma lukę 11 dni)
    row_partial = build_prediction_input_row(
        region_id="0201",
        issued_at="2025-08-10",
        horizon_days=1,
        forest_lookup=forest_lookup,
        weather_lookup=weather_lookup,
    )

    # 3. Obszar spoza pokrycia (9999)
    strict_error_raised = False
    try:
        build_prediction_input_row(
            region_id="9999",
            issued_at=issued_at,
            horizon_days=1,
            forest_lookup=forest_lookup,
            weather_lookup=weather_lookup,
            allow_out_of_coverage=False,
        )
    except RegionOutOfCoverageError:
        strict_error_raised = True

    row_ooc = build_prediction_input_row(
        region_id="9999",
        issued_at=issued_at,
        horizon_days=1,
        forest_lookup=forest_lookup,
        weather_lookup=weather_lookup,
        allow_out_of_coverage=True,
    )

    # 4. Test braku wycieku przyszłej pogody (usunięcie dni > issued_at nie zmienia żadnej cechy)
    truncated_weather = {k: v for k, v in weather_lookup.items() if k[1] <= issued_at}
    row_full = build_prediction_input_row(
        region_id="0201",
        issued_at=issued_at,
        horizon_days=3,
        forest_lookup=forest_lookup,
        weather_lookup=weather_lookup,
    )
    row_trunc = build_prediction_input_row(
        region_id="0201",
        issued_at=issued_at,
        horizon_days=3,
        forest_lookup=forest_lookup,
        weather_lookup=truncated_weather,
    )
    leak_free = all(row_full[c] == row_trunc[c] for c in MODEL_INPUT_FEATURE_COLUMNS)

    return {
        "schema_alignment": {
            "model_input_feature_count": len(MODEL_INPUT_FEATURE_COLUMNS),
            "all_features_present_in_training_dictionary": all(
                c in FEATURE_DICTIONARY for c in MODEL_INPUT_FEATURE_COLUMNS
            ),
        },
        "no_bdl_county_0218": {
            "history_completeness": row_no_bdl["history_completeness"],
            "pine_share_pct_is_null": row_no_bdl["pine_share_pct"] is None,
            "precip_lag_21d_mm": row_no_bdl["precip_lag_21d_mm"],
        },
        "incomplete_history_2025_08_10": {
            "history_completeness": row_partial["history_completeness"],
            "missing_weather_days_21d": row_partial["missing_weather_days_21d"],
            "precip_lag_7d_mm": row_partial["precip_lag_7d_mm"],
            "precip_lag_14d_mm_is_null": row_partial["precip_lag_14d_mm"] is None,
            "precip_lag_21d_mm_is_null": row_partial["precip_lag_21d_mm"] is None,
        },
        "out_of_coverage_9999": {
            "strict_mode_raised_error": strict_error_raised,
            "graceful_status": row_ooc["history_completeness"],
            "is_region_covered": row_ooc["is_region_covered"],
        },
        "future_weather_leakage_test": {
            "issued_at": issued_at,
            "target_date_horizon_3": row_full["target_date"],
            "identical_features_without_future_weather": leak_free,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator cech wejściowych do prognoz [GRZ-15].")
    parser.add_argument("--issued-at", default="2025-10-18", help="Data wystawienia prognozy (YYYY-MM-DD)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3], help="Horyzonty w dniach (np. 1 2 3)")
    parser.add_argument("--region-ids", nargs="*", default=None, help="Opcjonalna lista region_id (domyślnie wszystkie 30)")
    parser.add_argument(
        "--output-parquet",
        default=str(PROCESSED_DIR / "prediction_inputs_sample.parquet"),
        help="Ścieżka wyjściowa Parquet",
    )
    args = parser.parse_args()

    forest_lookup, weather_lookup = load_inference_reference_data()
    df = build_prediction_inputs_batch(
        issued_at=args.issued_at,
        horizons=args.horizons,
        region_ids=args.region_ids,
        forest_lookup=forest_lookup,
        weather_lookup=weather_lookup,
    )

    # Zapis próbki wejścia (bez kolumny listowej missing_feature_columns lub z konwersją do JSON string dla Parquet/CSV)
    out_df = df.copy()
    if "missing_feature_columns" in out_df.columns:
        out_df["missing_feature_columns"] = out_df["missing_feature_columns"].apply(json.dumps)

    out_parquet = Path(args.output_parquet)
    out_df.to_parquet(out_parquet, index=False)

    edge_checks = run_edge_case_checks(forest_lookup, weather_lookup, args.issued_at)
    report = {
        "task": "GRZ-15",
        "issued_at": args.issued_at,
        "horizons": args.horizons,
        "rows_generated": len(out_df),
        "regions_count": int(out_df["region_id"].nunique()),
        "history_completeness_counts": {
            str(k): int(v) for k, v in out_df["history_completeness"].value_counts().items()
        },
        "edge_case_checks": edge_checks,
    }

    report_path = PROCESSED_DIR / "prediction_inputs_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wygenerowano {len(out_df)} wierszy wejściowych -> {out_parquet}")
    print(f"Zapisano raport kontroli -> {report_path}")


if __name__ == "__main__":
    main()
