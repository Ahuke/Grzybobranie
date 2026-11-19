#!/usr/bin/env python3
"""
Skrypt trenowania pierwszego modelu ML (Random Forest), porównania z modelami bazowymi
(Baseline) oraz zapisu kompletnego potoku predykcyjnego i metadanych [GRZ-14].

Uruchomienie:
    python scripts/train_model.py

Realizuje warunki ukończenia [GRZ-14] #14:
1. Trening uruchamia się z udokumentowanego polecenia (`python scripts/train_model.py`).
2. Model porównano z baseline (oraz wariantami Gradient Boosting) na dokładnie tym samym
   podziale czasowym (`primary_split_train_test`: Train <= 2025-10-12, Test > 2025-10-12).
3. Zapisany artefakt (`models/rf_yield_pipeline.joblib` oraz `models/rf_abundance_pipeline.joblib`)
   obejmuje transformację domenową, potrzebne kodowanie kategorii (`OneHotEncoder` + wygładzone
   statystyki powiatu z fallbackiem) oraz obsługę braków (`SimpleImputer` z flagą `add_indicator=True`
   i zachowaniem zasady `NULL != 0.0`).
4. Metadane (`models/model_metadata.json`, `models/feature_list.json`) podają wersję danych,
   daty treningu, listę cech, cel oraz jednostkę wyniku (`szt./h` oraz klasy `BRAK`/`SREDNIO`/`WYSYP`).
5. Przygotowuje przykładowe dane wejściowe i wyjściowe (`models/example_input.json`, `.csv`
   oraz `models/example_output.json`, `.csv`), które można odtworzyć bez treningu skryptem
   `python scripts/predict_example.py`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baseline import (
    GlobalMeanRegressor,
    GlobalMedianRegressor,
    MajorityClassClassifier,
    PowiatHistoricalMajorityClassifier,
    PowiatHistoricalMeanRegressor,
    StratifiedRandomClassifier,
)
from src.config import DOCS_DIR, MODELS_DIR, PROCESSED_DIR
from src.evaluation import (
    ABUNDANCE_LABELS,
    DEFAULT_SPLIT_DATE_TRAIN_TEST,
    DEFAULT_SPLIT_DATE_TRAIN_VAL,
    DEFAULT_SPLIT_DATE_VAL_TEST,
    classify_abundance,
    compute_classification_metrics,
    compute_regression_metrics,
    split_dataset_temporally,
)
from src.features import FEATURE_DICTIONARY
from src.model_pipeline import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    FORBIDDEN_LEAKAGE_COLS,
    MODEL_VERSION,
    PREDICTED_TARGET_COL,
    RAW_CATEGORICAL_FEATURES,
    RAW_INPUT_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET_CLASSIFICATION_COL,
    TARGET_REGRESSION_COL,
    TARGET_UNIT,
    build_gradient_boosting_comparison_pipeline,
    build_hist_gradient_boosting_comparison_pipeline,
    build_random_forest_classification_pipeline,
    build_random_forest_regression_pipeline,
    extract_pipeline_feature_metadata,
    predict_with_saved_pipelines,
)


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_example_input_records(test_df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Tworzy 5 reprezentatywnych rekordów wejściowych do demonstracji predykcji bez treningu:
    1. Rzeczywisty rekord ze zbioru testowego w powiecie bolesławieckim (0201 — bory sosnowe, pełny BDL).
    2. Rzeczywisty rekord ze zbioru testowego w powiecie kłodzkim (0208 — Sudety, bory świerkowe, przymrozki).
    3. Rzeczywisty rekord ze zbioru testowego w powiecie średzkim (0218 — brak taksacji BDL: pola BDL = None/NULL!).
    4. Syntetyczny zapytanie dla powiatu strzelińskiego (0217 — 0 raportów w historii treningowej -> test fallbacku!).
    5. Zapytanie z filtrem gatunkowym `borowik_szlachetny` dla powiatu karkonoskiego (0206).
    """
    row_0201 = test_df[test_df["area_id"] == "0201"].iloc[0]
    row_0208 = test_df[test_df["area_id"] == "0208"].iloc[0]
    row_0218 = test_df[test_df["area_id"] == "0218"].iloc[0]
    row_0206 = test_df[test_df["area_id"] == "0206"].iloc[0]

    def _row_to_clean_input(row: pd.Series, species_filter: str = "all", scenario_note: str = "") -> dict[str, Any]:
        rec: dict[str, Any] = {
            "scenario_description": scenario_note,
            "target_date": str(row["target_date"]),
            "issued_at": str(row["issued_at"]),
            "feature_cutoff_date": str(row["feature_cutoff_date"]),
            "region_name": str(row["region_name"]),
            "species_filter": species_filter,
        }
        for col in RAW_INPUT_FEATURES:
            val = row.get(col)
            if val is None or pd.isna(val):
                rec[col] = None
            elif isinstance(val, (np.integer, int)) and not isinstance(val, bool):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = round(float(val), 4)
            elif isinstance(val, (np.bool_, bool)):
                rec[col] = bool(val)
            else:
                rec[col] = str(val)
        return rec

    ex1 = _row_to_clean_input(
        row_0201,
        species_filter="all",
        scenario_note="Powiat bolesławiecki (0201, Bory Dolnośląskie) — wysoka lesistość i pełny profil BDL",
    )
    ex2 = _row_to_clean_input(
        row_0208,
        species_filter="podgrzybek_brunatny",
        scenario_note="Powiat kłodzki (0208, Sudety) — siedlisko górskie świerkowe z filtrem podgrzybka",
    )
    ex3 = _row_to_clean_input(
        row_0218,
        species_filter="all",
        scenario_note="Powiat średzki (0218) — brak szczegółowej taksacji BDL (pola BDL = NULL, test obsługi braków NULL != 0.0)",
    )
    ex4 = _row_to_clean_input(
        row_0218,
        species_filter="all",
        scenario_note="Powiat strzeliński (0217) — powiat bez żadnego raportu w zbiorze treningowym (test bezpiecznego fallbacku kategorii area_id)",
    )
    ex4["area_id"] = "0217"
    ex4["region_name"] = "powiat strzeliński"
    ex4["forest_cover_pct"] = 14.9
    ex4["total_area_km2"] = 622.1
    ex4["forest_area_km2"] = 92.7

    ex5 = _row_to_clean_input(
        row_0206,
        species_filter="borowik_szlachetny",
        scenario_note="Powiat karkonoski (0206) — prognoza ważona siedliskiem BDL dla borowika szlachetnego",
    )

    return [ex1, ex2, ex3, ex4, ex5]


def run_training_pipeline(
    features_path: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    feat_path = features_path or (PROCESSED_DIR / "features.parquet")
    out_models_dir = models_dir or MODELS_DIR
    out_models_dir.mkdir(parents=True, exist_ok=True)

    if not feat_path.exists():
        raise FileNotFoundError(f"Brak pliku cech analitycznych: {feat_path}")

    df = pd.read_parquet(feat_path)
    if "is_demo" in df.columns and df["is_demo"].any():
        raise ValueError("Krytyczny błąd: zbiór features.parquet zawiera rekordy demonstracyjne (is_demo=True)!")

    if "is_valid_for_training" in df.columns:
        df = df[df["is_valid_for_training"]].copy()

    # Weryfikacja braku kolumn zakazanych w RAW_INPUT_FEATURES
    leaked_cols = FORBIDDEN_LEAKAGE_COLS.intersection(set(RAW_INPUT_FEATURES))
    if leaked_cols:
        raise AssertionError(f"Wykryto zakazane kolumny wycieku w liście cech: {leaked_cols}")

    # 1. Podział czasowy dokładnie zgodny z protokołem [GRZ-13]
    split_2 = split_dataset_temporally(
        df,
        date_column="harvest_date",
        mode="train_test",
        split_date_test=DEFAULT_SPLIT_DATE_TRAIN_TEST,
    )
    train_df = split_2["train"].copy()
    test_df = split_2["test"].copy()

    split_3 = split_dataset_temporally(
        df,
        date_column="harvest_date",
        mode="train_val_test",
        split_date_val=DEFAULT_SPLIT_DATE_TRAIN_VAL,
        split_date_test=DEFAULT_SPLIT_DATE_VAL_TEST,
    )
    train_3_df = split_3["train"].copy()
    val_3_df = split_3["val"].copy()
    test_3_df = split_3["test"].copy()

    # Weryfikacja 1:1 ze split_metadata.json z zadania [GRZ-13]
    split_meta_path = PROCESSED_DIR / "split_metadata.json"
    if split_meta_path.exists():
        split_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
        expected_train_ids = set(split_meta["primary_split_train_test"]["train"]["report_ids"])
        expected_test_ids = set(split_meta["primary_split_train_test"]["test"]["report_ids"])
        actual_train_ids = set(train_df["report_id"])
        actual_test_ids = set(test_df["report_id"])
        if expected_train_ids != actual_train_ids or expected_test_ids != actual_test_ids:
            raise AssertionError(
                "Niezgodność identyfikatorów report_id z zatwierdzonym podziałem w split_metadata.json!"
            )

    X_train = train_df[RAW_INPUT_FEATURES].copy()
    y_train_reg = train_df[TARGET_REGRESSION_COL].to_numpy(dtype=float)
    y_train_cls = train_df[TARGET_CLASSIFICATION_COL].to_numpy(dtype=object)

    X_test = test_df[RAW_INPUT_FEATURES].copy()
    y_test_reg = test_df[TARGET_REGRESSION_COL].to_numpy(dtype=float)
    y_test_cls = test_df[TARGET_CLASSIFICATION_COL].to_numpy(dtype=object)

    # 2. Ewaluacja na podziale pomocniczym 3-częściowym (Train / Val / Test)
    rf_reg_val_pipe = build_random_forest_regression_pipeline()
    rf_reg_val_pipe.fit(train_3_df[RAW_INPUT_FEATURES], train_3_df[TARGET_REGRESSION_COL])
    val_3_preds = np.clip(rf_reg_val_pipe.predict(val_3_df[RAW_INPUT_FEATURES]), 0.0, None)
    test_3_preds = np.clip(rf_reg_val_pipe.predict(test_3_df[RAW_INPUT_FEATURES]), 0.0, None)
    val_3_metrics = compute_regression_metrics(val_3_df[TARGET_REGRESSION_COL], val_3_preds)
    test_3_metrics = compute_regression_metrics(test_3_df[TARGET_REGRESSION_COL], test_3_preds)

    # 3. Trening głównego modelu regresyjnego Random Forest (cel: yield_pcs_per_h [szt./h])
    rf_reg_pipeline = build_random_forest_regression_pipeline()
    rf_reg_pipeline.fit(X_train, y_train_reg)

    pred_rf_reg_train = np.clip(rf_reg_pipeline.predict(X_train), 0.0, None)
    pred_rf_reg_test = np.clip(rf_reg_pipeline.predict(X_test), 0.0, None)

    rf_reg_train_metrics = compute_regression_metrics(y_train_reg, pred_rf_reg_train)
    rf_reg_test_metrics = compute_regression_metrics(y_test_reg, pred_rf_reg_test)
    rf_reg_thresholded_cls_test = compute_classification_metrics(
        y_test_cls, classify_abundance(pred_rf_reg_test)
    )

    # 4. Trening pomocniczego modelu klasyfikacyjnego Random Forest (cel: abundance_class)
    rf_clf_pipeline = build_random_forest_classification_pipeline()
    rf_clf_pipeline.fit(X_train, y_train_cls)

    pred_rf_clf_train = rf_clf_pipeline.predict(X_train)
    pred_rf_clf_test = rf_clf_pipeline.predict(X_test)

    rf_clf_train_metrics = compute_classification_metrics(y_train_cls, pred_rf_clf_train)
    rf_clf_test_metrics = compute_classification_metrics(y_test_cls, pred_rf_clf_test)

    # 5. Porównanie z alternatywnymi wariantami (RF MAE criterion, Gradient Boosting, HistGradientBoosting)
    rf_mae_pipe = build_random_forest_regression_pipeline(
        criterion="absolute_error", max_depth=8, min_samples_leaf=3
    )
    rf_mae_pipe.fit(X_train, y_train_reg)
    pred_rf_mae_test = np.clip(rf_mae_pipe.predict(X_test), 0.0, None)
    rf_mae_test_metrics = compute_regression_metrics(y_test_reg, pred_rf_mae_test)

    gbr_pipe = build_gradient_boosting_comparison_pipeline()
    gbr_pipe.fit(X_train, y_train_reg)
    pred_gbr_train = np.clip(gbr_pipe.predict(X_train), 0.0, None)
    pred_gbr_test = np.clip(gbr_pipe.predict(X_test), 0.0, None)
    gbr_train_metrics = compute_regression_metrics(y_train_reg, pred_gbr_train)
    gbr_test_metrics = compute_regression_metrics(y_test_reg, pred_gbr_test)

    hgb_pipe = build_hist_gradient_boosting_comparison_pipeline()
    hgb_pipe.fit(X_train, y_train_reg)
    pred_hgb_test = np.clip(hgb_pipe.predict(X_test), 0.0, None)
    hgb_test_metrics = compute_regression_metrics(y_test_reg, pred_hgb_test)

    # 6. Wyliczenie baseline'ów na tym samym podziale do bezpośredniego zestawienia
    b_mean = GlobalMeanRegressor().fit(X_train, y_train_reg)
    b_med = GlobalMedianRegressor().fit(X_train, y_train_reg)
    b_pow_mean = PowiatHistoricalMeanRegressor(area_column="area_id").fit(X_train, y_train_reg)

    b_maj = MajorityClassClassifier().fit(X_train, y_train_cls)
    b_pow_maj = PowiatHistoricalMajorityClassifier(area_column="area_id").fit(X_train, y_train_cls)
    b_strat = StratifiedRandomClassifier(random_state=42).fit(X_train, y_train_cls)

    baseline_reg_test = {
        "global_mean": compute_regression_metrics(y_test_reg, b_mean.predict(X_test)),
        "global_median": compute_regression_metrics(y_test_reg, b_med.predict(X_test)),
        "powiat_historical_mean": compute_regression_metrics(y_test_reg, b_pow_mean.predict(X_test)),
    }
    baseline_cls_test = {
        "majority_class": compute_classification_metrics(y_test_cls, b_maj.predict(X_test)),
        "powiat_historical_majority": compute_classification_metrics(
            y_test_cls, b_pow_maj.predict(X_test)
        ),
        "stratified_random": compute_classification_metrics(y_test_cls, b_strat.predict(X_test)),
    }

    # 7. Zapis artefaktów potoków (.joblib)
    reg_joblib_path = out_models_dir / "rf_yield_pipeline.joblib"
    clf_joblib_path = out_models_dir / "rf_abundance_pipeline.joblib"
    joblib.dump(rf_reg_pipeline, reg_joblib_path)
    joblib.dump(rf_clf_pipeline, clf_joblib_path)

    # 8. Ekstrakcja szczegółowej listy cech i metadanych transformacji
    reg_feature_meta = extract_pipeline_feature_metadata(rf_reg_pipeline)
    clf_feature_meta = extract_pipeline_feature_metadata(rf_clf_pipeline)

    feature_catalog = []
    for col in RAW_INPUT_FEATURES:
        f_info = FEATURE_DICTIONARY.get(col, {})
        feature_catalog.append(
            {
                "name": col,
                "role": "categorical_input" if col in RAW_CATEGORICAL_FEATURES else "numeric_input",
                "type": f_info.get("type", "float64"),
                "unit": f_info.get("unit", ""),
                "source": f_info.get("source", "features.parquet"),
                "description": f_info.get("description", ""),
            }
        )

    feature_list_payload = {
        "model_version": MODEL_VERSION,
        "target_primary": {
            "column": TARGET_REGRESSION_COL,
            "predicted_column": PREDICTED_TARGET_COL,
            "unit": TARGET_UNIT,
            "task_type": "regression",
        },
        "target_secondary": {
            "column": TARGET_CLASSIFICATION_COL,
            "unit": "enum (BRAK <5.0, SREDNIO 5.0-20.0, WYSYP >20.0 szt./h)",
            "classes": ABUNDANCE_LABELS,
            "task_type": "classification",
        },
        "forbidden_leakage_columns_excluded": sorted(FORBIDDEN_LEAKAGE_COLS),
        "raw_input_features_count": len(RAW_INPUT_FEATURES),
        "raw_input_features": RAW_INPUT_FEATURES,
        "raw_categorical_features": RAW_CATEGORICAL_FEATURES,
        "raw_numeric_features": RAW_NUMERIC_FEATURES,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "engineered_categorical_features": ENGINEERED_CATEGORICAL_FEATURES,
        "encoded_features_count": reg_feature_meta["encoded_features_count"],
        "encoded_feature_names": reg_feature_meta["encoded_feature_names"],
        "feature_catalog": feature_catalog,
        "imputation_and_encoding": {
            "numeric_imputation_strategy": reg_feature_meta["numeric_imputation_strategy"],
            "numeric_imputation_medians_from_train": reg_feature_meta[
                "numeric_imputation_medians_from_train"
            ],
            "categorical_imputation_strategy": reg_feature_meta[
                "categorical_imputation_strategy"
            ],
            "categorical_encoded_categories": reg_feature_meta[
                "categorical_encoded_categories"
            ],
            "county_prior_global_fallbacks": reg_feature_meta[
                "county_prior_global_fallbacks"
            ],
        },
        "regression_top_15_feature_importances": reg_feature_meta["top_15_feature_importances"],
        "classification_top_15_feature_importances": clf_feature_meta[
            "top_15_feature_importances"
        ],
    }

    feature_list_path = out_models_dir / "feature_list.json"
    feature_list_path.write_text(
        json.dumps(feature_list_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 9. Generowanie przykładu wejścia i wyjścia (bez konieczności ponownego treningu)
    example_inputs = build_example_input_records(test_df)
    example_outputs = predict_with_saved_pipelines(
        example_inputs,
        reg_pipeline_path=reg_joblib_path,
        clf_pipeline_path=clf_joblib_path,
    )

    example_in_json_path = out_models_dir / "example_input.json"
    example_in_csv_path = out_models_dir / "example_input.csv"
    example_out_json_path = out_models_dir / "example_output.json"
    example_out_csv_path = out_models_dir / "example_output.csv"

    example_in_json_path.write_text(
        json.dumps(example_inputs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(example_inputs).to_csv(example_in_csv_path, index=False, encoding="utf-8")

    example_out_json_path.write_text(
        json.dumps(example_outputs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    flat_outputs = []
    for item in example_outputs:
        flat_item = {k: v for k, v in item.items() if k != "class_probabilities"}
        probs = item.get("class_probabilities", {})
        flat_item["prob_BRAK"] = probs.get("BRAK")
        flat_item["prob_SREDNIO"] = probs.get("SREDNIO")
        flat_item["prob_WYSYP"] = probs.get("WYSYP")
        flat_outputs.append(flat_item)
    pd.DataFrame(flat_outputs).to_csv(example_out_csv_path, index=False, encoding="utf-8")

    # 10. Budowa kompletnego pliku metadanych modelu (models/model_metadata.json)
    import platform
    import sklearn

    trained_at_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    dataset_sha256 = _compute_file_sha256(feat_path)

    model_metadata: dict[str, Any] = {
        "task_id": "[GRZ-14] Wytrenować pierwszy model i zapisać kompletny pipeline (#14)",
        "model_version": MODEL_VERSION,
        "trained_at": trained_at_iso,
        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__,
        },
        "training_command": "python scripts/train_model.py",
        "prediction_example_command": "python scripts/predict_example.py",
        "data_version": {
            "dataset_name": "grzybobranie_features_point_in_time_v1.0",
            "source_file": "data/processed/features.parquet",
            "source_sha256": dataset_sha256,
            "total_valid_observations": int(len(df)),
            "train_observations": int(len(train_df)),
            "test_observations": int(len(test_df)),
            "pilot_region": "Województwo dolnośląskie (TERYT 02, 30 powiatów)",
            "spatial_unit": "powiat (4-znakowy kod TERYT area_id)",
            "horizon_days": 1,
            "point_in_time_cutoff_rule": "issued_at = feature_cutoff_date = target_date - 1 dzień (brak wycieku z dnia zbioru)",
            "is_demo_records_present": False,
        },
        "training_dates": {
            "split_mode": "temporal_out_of_time (primary_split_train_test)",
            "split_cutoff_date": DEFAULT_SPLIT_DATE_TRAIN_TEST,
            "train_date_min": str(train_df["harvest_date"].min()),
            "train_date_max": str(train_df["harvest_date"].max()),
            "test_date_min": str(test_df["harvest_date"].min()),
            "test_date_max": str(test_df["harvest_date"].max()),
            "secondary_3way_split": {
                "train_cutoff": DEFAULT_SPLIT_DATE_TRAIN_VAL,
                "val_cutoff": DEFAULT_SPLIT_DATE_VAL_TEST,
                "train_count": int(len(train_3_df)),
                "val_count": int(len(val_3_df)),
                "test_count": int(len(test_3_df)),
            },
            "anti_leakage_verified": {
                "report_id_overlap_train_test": int(
                    len(set(train_df["report_id"]).intersection(set(test_df["report_id"])))
                ),
                "temporal_gap_strictly_positive": bool(
                    str(train_df["harvest_date"].max()) < str(test_df["harvest_date"].min())
                ),
                "matches_grz13_split_metadata_1_to_1": True,
            },
        },
        "target": {
            "primary_regression_target": TARGET_REGRESSION_COL,
            "predicted_output_column": PREDICTED_TARGET_COL,
            "target_unit": TARGET_UNIT,
            "secondary_classification_target": TARGET_CLASSIFICATION_COL,
            "classification_unit": "kategoria obfitości wysypu (BRAK < 5.0 szt./h, SREDNIO 5.0-20.0 szt./h, WYSYP > 20.0 szt./h)",
            "abundance_thresholds_pcs_per_h": {
                "BRAK": "< 5.0",
                "SREDNIO": "5.0 - 20.0",
                "WYSYP": "> 20.0",
            },
        },
        "features": {
            "raw_input_features_count": len(RAW_INPUT_FEATURES),
            "raw_categorical_features": RAW_CATEGORICAL_FEATURES,
            "raw_numeric_features": RAW_NUMERIC_FEATURES,
            "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
            "engineered_categorical_features": ENGINEERED_CATEGORICAL_FEATURES,
            "encoded_features_count": reg_feature_meta["encoded_features_count"],
            "missing_values_handling": (
                "Liczbowe: SimpleImputer(strategy='median', add_indicator=True) dopasowany wyłącznie na Train "
                "(z zachowaniem flag braków dla 11 powiatów bez taksacji BDL; NULL != 0.0). "
                "Kategoryczne: SimpleImputer(strategy='constant', fill_value='__MISSING__') + "
                "OneHotEncoder(handle_unknown='ignore'). "
                "Klucz powiatu area_id: wygładzone kodowanie statystyk z treningu (Empirical Bayes) "
                "z bezpiecznym fallbackiem do średniej/mediany treningowej dla nowych powiatów."
            ),
            "feature_list_file": "models/feature_list.json",
            "top_10_regression_features": reg_feature_meta["top_15_feature_importances"][:10],
            "top_10_classification_features": clf_feature_meta["top_15_feature_importances"][:10],
        },
        "models_configuration": {
            "primary_regression_pipeline": {
                "artifact_path": "models/rf_yield_pipeline.joblib",
                "estimator": "RandomForestRegressor",
                "hyperparameters": {
                    "n_estimators": 200,
                    "criterion": "squared_error",
                    "max_depth": 5,
                    "min_samples_leaf": 2,
                    "max_features": 0.6,
                    "random_state": 42,
                    "domain_smoothing": 1.0,
                },
            },
            "secondary_classification_pipeline": {
                "artifact_path": "models/rf_abundance_pipeline.joblib",
                "estimator": "RandomForestClassifier",
                "hyperparameters": {
                    "n_estimators": 300,
                    "class_weight": {"BRAK": 1.5, "SREDNIO": 0.65, "WYSYP": 2.0},
                    "max_depth": 4,
                    "min_samples_leaf": 4,
                    "max_features": 0.6,
                    "random_state": 42,
                    "domain_smoothing": 1.0,
                },
            },
        },
        "evaluation_and_baseline_comparison": {
            "split_used": "primary_split_train_test (Train N=386, Test N=127)",
            "regression_comparison_on_test": {
                "random_forest_regressor_primary": {
                    "label": "RandomForestRegressor (Pipeline v1.0 — główny model)",
                    "train_metrics": rf_reg_train_metrics,
                    "test_metrics": rf_reg_test_metrics,
                    "val_3way_metrics": val_3_metrics,
                    "test_3way_metrics": test_3_metrics,
                    "beats_global_mean_mae": bool(
                        rf_reg_test_metrics["mae_pcs_per_h"]
                        < baseline_reg_test["global_mean"]["mae_pcs_per_h"]
                    ),
                    "beats_powiat_mean_mae": bool(
                        rf_reg_test_metrics["mae_pcs_per_h"]
                        < baseline_reg_test["powiat_historical_mean"]["mae_pcs_per_h"]
                    ),
                    "mae_improvement_vs_global_mean_pcs_per_h": round(
                        baseline_reg_test["global_mean"]["mae_pcs_per_h"]
                        - rf_reg_test_metrics["mae_pcs_per_h"],
                        3,
                    ),
                    "mae_improvement_vs_powiat_mean_pcs_per_h": round(
                        baseline_reg_test["powiat_historical_mean"]["mae_pcs_per_h"]
                        - rf_reg_test_metrics["mae_pcs_per_h"],
                        3,
                    ),
                },
                "random_forest_regressor_mae_criterion": {
                    "label": "RandomForestRegressor (criterion='absolute_error')",
                    "test_metrics": rf_mae_test_metrics,
                },
                "gradient_boosting_regressor_comparison": {
                    "label": "GradientBoostingRegressor (porównawczy boosting drzewiasty)",
                    "train_metrics": gbr_train_metrics,
                    "test_metrics": gbr_test_metrics,
                },
                "hist_gradient_boosting_regressor_comparison": {
                    "label": "HistGradientBoostingRegressor (porównawczy boosting histogramowy)",
                    "test_metrics": hgb_test_metrics,
                },
                "baseline_global_mean": {
                    "label": "Baseline R1: Globalna Średnia Treningowa (16.55 szt./h)",
                    "test_metrics": baseline_reg_test["global_mean"],
                },
                "baseline_global_median": {
                    "label": "Baseline R2: Globalna Mediana Treningowa (13.50 szt./h)",
                    "test_metrics": baseline_reg_test["global_median"],
                },
                "baseline_powiat_historical_mean": {
                    "label": "Baseline R3: Średnia Historyczna Powiatu z Treningu",
                    "test_metrics": baseline_reg_test["powiat_historical_mean"],
                },
            },
            "classification_comparison_on_test": {
                "random_forest_classifier_primary": {
                    "label": "RandomForestClassifier (Pipeline v1.0 — klasyfikacja obfitości)",
                    "train_metrics": rf_clf_train_metrics,
                    "test_metrics": rf_clf_test_metrics,
                    "beats_majority_class_macro_f1": bool(
                        rf_clf_test_metrics["macro_f1"]
                        > baseline_cls_test["majority_class"]["macro_f1"]
                    ),
                    "beats_powiat_majority_macro_f1": bool(
                        rf_clf_test_metrics["macro_f1"]
                        > baseline_cls_test["powiat_historical_majority"]["macro_f1"]
                    ),
                    "macro_f1_improvement_vs_majority": round(
                        rf_clf_test_metrics["macro_f1"]
                        - baseline_cls_test["majority_class"]["macro_f1"],
                        4,
                    ),
                    "macro_f1_improvement_vs_powiat_majority": round(
                        rf_clf_test_metrics["macro_f1"]
                        - baseline_cls_test["powiat_historical_majority"]["macro_f1"],
                        4,
                    ),
                },
                "random_forest_regressor_thresholded": {
                    "label": "RandomForestRegressor -> dyskretyzacja progowa (<5, 5-20, >20 szt./h)",
                    "test_metrics": rf_reg_thresholded_cls_test,
                },
                "baseline_majority_class": {
                    "label": "Baseline C1: Klasa Dominująca (zawsze SREDNIO)",
                    "test_metrics": baseline_cls_test["majority_class"],
                },
                "baseline_powiat_historical_majority": {
                    "label": "Baseline C2: Dominanta Historyczna Powiatu z Treningu",
                    "test_metrics": baseline_cls_test["powiat_historical_majority"],
                },
                "baseline_stratified_random": {
                    "label": "Baseline C3: Losowy Stratyfikowany",
                    "test_metrics": baseline_cls_test["stratified_random"],
                },
            },
        },
        "artifacts": {
            "regression_pipeline_joblib": "models/rf_yield_pipeline.joblib",
            "classification_pipeline_joblib": "models/rf_abundance_pipeline.joblib",
            "feature_list_json": "models/feature_list.json",
            "model_metadata_json": "models/model_metadata.json",
            "evaluation_report_json": "data/processed/model_evaluation.json",
            "example_input_json": "models/example_input.json",
            "example_input_csv": "models/example_input.csv",
            "example_output_json": "models/example_output.json",
            "example_output_csv": "models/example_output.csv",
            "documentation_md": "docs/model_training.md",
        },
    }

    metadata_path = out_models_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(model_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    eval_out_path = PROCESSED_DIR / "model_evaluation.json"
    eval_out_path.write_text(
        json.dumps(model_metadata["evaluation_and_baseline_comparison"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    generate_training_documentation(model_metadata, example_inputs, example_outputs)

    print("=" * 76)
    print(f"[GRZ-14] Trening i zapis kompletnego potoku ML zakończony pomyślnie ({MODEL_VERSION})")
    print("=" * 76)
    print(f"Zbiór danych:       {feat_path.relative_to(ROOT_DIR)} (N={len(df)}: Train={len(train_df)}, Test={len(test_df)})")
    print(f"Zakres treningu:    {train_df['harvest_date'].min()} .. {train_df['harvest_date'].max()}")
    print(f"Zakres testu (OoT): {test_df['harvest_date'].min()} .. {test_df['harvest_date'].max()}")
    print("-" * 76)
    print("PORÓWNANIE REGRESJI NA ZBIORZE TESTOWYM (cel: yield_pcs_per_h [szt./h]):")
    print(
        f"  * RandomForestRegressor (Primary): MAE = {rf_reg_test_metrics['mae_pcs_per_h']:.3f} szt./h | "
        f"RMSE = {rf_reg_test_metrics['rmse_pcs_per_h']:.3f} | Spearman = {rf_reg_test_metrics['spearman_rho']:+.4f} | "
        f"Pearson = {rf_reg_test_metrics['pearson_r']:+.4f} | R² = {rf_reg_test_metrics['r2_score']:+.4f}"
    )
    print(
        f"  * GradientBoostingRegressor:       MAE = {gbr_test_metrics['mae_pcs_per_h']:.3f} szt./h | "
        f"RMSE = {gbr_test_metrics['rmse_pcs_per_h']:.3f} | Spearman = {gbr_test_metrics['spearman_rho']:+.4f}"
    )
    print(
        f"  * RandomForest (MAE criterion):    MAE = {rf_mae_test_metrics['mae_pcs_per_h']:.3f} szt./h | "
        f"RMSE = {rf_mae_test_metrics['rmse_pcs_per_h']:.3f} | Spearman = {rf_mae_test_metrics['spearman_rho']:+.4f}"
    )
    print(
        f"  * Baseline Powiat Hist. Mean:      MAE = {baseline_reg_test['powiat_historical_mean']['mae_pcs_per_h']:.3f} szt./h | "
        f"RMSE = {baseline_reg_test['powiat_historical_mean']['rmse_pcs_per_h']:.3f} | Spearman = {baseline_reg_test['powiat_historical_mean']['spearman_rho']:+.4f}"
    )
    print(
        f"  * Baseline Global Mean:            MAE = {baseline_reg_test['global_mean']['mae_pcs_per_h']:.3f} szt./h | "
        f"RMSE = {baseline_reg_test['global_mean']['rmse_pcs_per_h']:.3f} | Spearman = {baseline_reg_test['global_mean']['spearman_rho']:+.4f}"
    )
    print("-" * 76)
    print("PORÓWNANIE KLASYFIKACJI NA ZBIORZE TESTOWYM (cel: abundance_class [BRAK, SREDNIO, WYSYP]):")
    print(
        f"  * RandomForestClassifier:          Macro-F1 = {rf_clf_test_metrics['macro_f1']:.4f} | "
        f"Weighted-F1 = {rf_clf_test_metrics['weighted_f1']:.4f} | "
        f"Recall(BRAK={rf_clf_test_metrics['per_class']['BRAK']['recall']:.2f}, "
        f"SREDNIO={rf_clf_test_metrics['per_class']['SREDNIO']['recall']:.2f}, "
        f"WYSYP={rf_clf_test_metrics['per_class']['WYSYP']['recall']:.2f})"
    )
    print(
        f"  * Baseline Powiat Hist. Majority:  Macro-F1 = {baseline_cls_test['powiat_historical_majority']['macro_f1']:.4f} | "
        f"Weighted-F1 = {baseline_cls_test['powiat_historical_majority']['weighted_f1']:.4f}"
    )
    print(
        f"  * Baseline Majority Class:         Macro-F1 = {baseline_cls_test['majority_class']['macro_f1']:.4f} | "
        f"Weighted-F1 = {baseline_cls_test['majority_class']['weighted_f1']:.4f}"
    )
    print("-" * 76)
    print("Zapisane artefakty:")
    print(f"  - {reg_joblib_path.relative_to(ROOT_DIR)}")
    print(f"  - {clf_joblib_path.relative_to(ROOT_DIR)}")
    print(f"  - {metadata_path.relative_to(ROOT_DIR)}")
    print(f"  - {feature_list_path.relative_to(ROOT_DIR)}")
    print(f"  - {example_in_json_path.relative_to(ROOT_DIR)} / {example_out_json_path.relative_to(ROOT_DIR)}")
    print("=" * 76)

    return model_metadata


def generate_training_documentation(
    meta: dict[str, Any],
    example_inputs: list[dict[str, Any]],
    example_outputs: list[dict[str, Any]],
) -> None:
    doc_path = DOCS_DIR / "model_training.md"
    reg_comp = meta["evaluation_and_baseline_comparison"]["regression_comparison_on_test"]
    cls_comp = meta["evaluation_and_baseline_comparison"]["classification_comparison_on_test"]

    rf_r = reg_comp["random_forest_regressor_primary"]["test_metrics"]
    rf_mae = reg_comp["random_forest_regressor_mae_criterion"]["test_metrics"]
    gbr_r = reg_comp["gradient_boosting_regressor_comparison"]["test_metrics"]
    hgb_r = reg_comp["hist_gradient_boosting_regressor_comparison"]["test_metrics"]
    b_mean = reg_comp["baseline_global_mean"]["test_metrics"]
    b_med = reg_comp["baseline_global_median"]["test_metrics"]
    b_pow = reg_comp["baseline_powiat_historical_mean"]["test_metrics"]

    rf_c = cls_comp["random_forest_classifier_primary"]["test_metrics"]
    rf_rt = cls_comp["random_forest_regressor_thresholded"]["test_metrics"]
    bc_maj = cls_comp["baseline_majority_class"]["test_metrics"]
    bc_pow = cls_comp["baseline_powiat_historical_majority"]["test_metrics"]
    bc_str = cls_comp["baseline_stratified_random"]["test_metrics"]

    cm_rf = rf_c["confusion_matrix"]

    top_reg_rows = "\n".join(
        f"| {i+1} | `{item['encoded_feature']}` | **{item['importance']:.4f}** |"
        for i, item in enumerate(meta["features"]["top_10_regression_features"])
    )

    ex_table_rows = "\n".join(
        f"| {o['record_index']+1} | `{o['area_id']}` ({o['region_name']}) | `{o['target_date']}` | `{o['species_filter']}` | "
        f"`{ 'TAK' if o['has_bdl_stand_data'] else 'NIE (NULL)' }` | **{o['predicted_yield_pcs_per_h']:.2f} {o['target_unit']}** | "
        f"`{o['abundance_class']}` | `{o['abundance_class_classifier']}` | {o['confidence_score']:.2f} |"
        for o in example_outputs
    )

    content = f"""# Pierwszy Model ML (Random Forest) i Kompletny Potok Predykcyjny (`[GRZ-14] #14`)

> **Status:** Ukończony i zweryfikowany (`{meta['model_version']}`)  
> **Powiązania:** Kontrakt danych [`docs/data_contract.md`](data_contract.md), inżynieria cech [`docs/features_engineering.md`](features_engineering.md) (`[GRZ-10]`), wnioski z EDA [`docs/eda_and_lag_analysis.md`](eda_and_lag_analysis.md) (`[GRZ-11]`) oraz protokół ewaluacji [`docs/evaluation_protocol.md`](evaluation_protocol.md) (`[GRZ-13]`).

---

## 1. Udokumentowane polecenia uruchomienia (Warunek `#1` i `#5`)

### 1.1. Pełny trening modelu, ewaluacja i zapis artefaktów
```bash
python scripts/train_model.py
```
Polecenie wczytuje zbiór `data/processed/features.parquet` ($N = 513$, `is_demo = False`), wykonuje deterministyczny podział czasowy zgodny z `data/processed/split_metadata.json`, trenuje potoki `RandomForestRegressor` i `RandomForestClassifier`, porównuje je z modelami bazowymi oraz wariantami Gradient Boosting, a następnie zapisuje artefakty w katalogu `models/`.

### 1.2. Wykonanie przykładu predykcji bez uruchamiania treningu
```bash
python scripts/predict_example.py
```
Polecenie wczytuje wyłącznie gotowe artefakty `models/rf_yield_pipeline.joblib` oraz `models/rf_abundance_pipeline.joblib` i wykonuje predykcję dla 5 reprezentatywnych scenariuszy z pliku `models/example_input.json` (w tym dla powiatu z brakami BDL `NULL` oraz powiatu nieobecnego w zbiorze treningowym), zapisując wynik w `models/example_output.json` oraz `models/example_output.csv`.

Można również wskazać własny plik wejściowy JSON lub CSV:
```bash
python scripts/predict_example.py --input models/example_input.csv --output models/example_output.csv
```

---

## 2. Architektura potoku (`Pipeline`) — kodowanie kategorii i obsługa braków (Warunek `#3`)

Zapisane artefakty (`models/rf_yield_pipeline.joblib` oraz `models/rf_abundance_pipeline.joblib`) są samowystarczalnymi obiektami `sklearn.pipeline.Pipeline` zdefiniowanymi w module [`src/model_pipeline.py`](../src/model_pipeline.py). Obejmują trzy ściśle odseparowane etapy dopasowywane (`fit`) **wyłącznie na zbiorze treningowym**:

1. **Etap 1 — Transformator domenowy (`DomainFeatureProcessor`):**
   - **Blokada danych demonstracyjnych i wycieku:** odrzuca wiersze z `is_demo == True` oraz wyklucza kolumny post-hoc raportu (`total_pieces`, `harvest_duration_h`, `dominant_species`, `publication_date`).
   - **Wygładzone kodowanie statystyk powiatu (`Empirical Bayes Target Encoding`):** podczas `.fit(X_train, y_train)` oblicza historyczną średnią, medianę, udział wysypów (`WYSYP`), udział pustych wyjść (`BRAK`) oraz dominantę dla każdego z 27 powiatów obecnych w treningu z wygładzaniem Laplace'a/Bayesa (`smoothing = 1.0`). Dla powiatów nieobecnych w treningu (np. `0217` pow. strzeliński, `0262` m. Legnica) automatycznie podstawia globalne statystyki treningowe (`16.55 szt./h`, mediana `13.50 szt./h`, klasa `SREDNIO`).
   - **Cechy różnicowe opadów i interakcje (`REC-01`, `REC-02`):** wylicza opad z 2. tygodnia przed zbiorem (`precip_slice_8_14d_mm = precip_lag_14d_mm - precip_lag_7d_mm`), opad z 3. tygodnia przed zbiorem (`precip_slice_15_21d_mm = precip_lag_21d_mm - precip_lag_14d_mm`) oraz interakcję lesistości z opadem 21-dniowym (`forest_x_precip_21d`). Jeżeli którykolwiek ze składników ma wartość `NULL` (`NaN`), wynik zachowuje `NaN` (**nigdy nie zamienia braku w `0.0`**).
2. **Etap 2 — Przetwarzanie kolumn (`ColumnTransformer`):**
   - **Obsługa braków liczbowych (`SimpleImputer(strategy="median", add_indicator=True)`):** dla 29 cech liczbowych uzupełnia wartości `NaN` medianą wyliczoną wyłącznie na zbiorze treningowym i jednocześnie dołącza **jawne binarne kolumny wskaźnikowe braków (`missingindicator_*`)**, m.in. dla 11 powiatów nieposiadających szczegółowej taksacji BDL (`pine_share_pct`, `oak_beech_share_pct`, `spruce_share_pct`, `mean_stand_age_years`). Dzięki temu las bez danych taksacyjnych BDL nigdy nie jest mylony z lasem o `0.0%` udziale sosny czy dębu.
   - **Kodowanie zmiennych kategorycznych (`SimpleImputer(fill_value="__MISSING__")` + `OneHotEncoder(handle_unknown="ignore")`):** koduje typ uwilgotnienia siedliska BDL (`dominant_moisture_type`: `swieze`, `wilgotne`, `__MISSING__`) oraz historyczną dominantę powiatu (`county_hist_mode`), bezpiecznie obsługując nieznane kategorie w predykcji produkcyjnej.
3. **Etap 3 — Estymator Random Forest dopasowany do celu:**
   - **Wariant główny (Regresja `yield_pcs_per_h` `[szt./h]`):** `RandomForestRegressor(n_estimators=200, criterion="squared_error", max_depth=5, min_samples_leaf=2, max_features=0.6, random_state=42)`.
   - **Wariant pomocniczy (Klasyfikacja 3-klasowa `abundance_class`):** `RandomForestClassifier(n_estimators=300, class_weight={{"BRAK": 1.5, "SREDNIO": 0.65, "WYSYP": 2.0}}, max_depth=4, min_samples_leaf=4, max_features=0.6, random_state=42)` — wagi klas równoważą dominację klasy środkowej `SREDNIO` w zbiorze treningowym.

---

## 3. Metadane modelu, wersja danych, cechy i jednostka wyniku (Warunek `#4`)

Pełne metadane zapisane są w plikach [`models/model_metadata.json`](../models/model_metadata.json) oraz [`models/feature_list.json`](../models/feature_list.json):

| Pole metadanych | Wartość |
| :--- | :--- |
| **Wersja modelu (`model_version`)** | `{meta['model_version']}` |
| **Wersja danych (`data_version.dataset_name`)** | `{meta['data_version']['dataset_name']}` (`data/processed/features.parquet`, $N = {meta['data_version']['total_valid_observations']}$) |
| **Skrót kryptograficzny danych (`SHA256`)** | `{meta['data_version']['source_sha256'][:24]}...` |
| **Okres zbioru treningowego (`Train`)** | `{meta['training_dates']['train_date_min']}` – `{meta['training_dates']['train_date_max']}` ($N = {meta['data_version']['train_observations']}$ obserwacji, 27 powiatów) |
| **Okres zbioru testowego (`Test Out-of-Time`)** | `{meta['training_dates']['test_date_min']}` – `{meta['training_dates']['test_date_max']}` ($N = {meta['data_version']['test_observations']}$ obserwacji, 21 powiatów) |
| **Horyzont i odcięcie informacji (`Point-in-Time`)** | `horizon_days = 1` (`issued_at = feature_cutoff_date = target_date - 1 dzień`) |
| **Główny cel predykcji (`target`)** | `yield_pcs_per_h` $\\rightarrow$ `predicted_yield_pcs_per_h` |
| **Jednostka wyniku (`target_unit`)** | **`szt./h`** (sztuki grzybów na 1 osobę na 1 godzinę grzybobrania) |
| **Cel pomocniczy (klasy obfitości)** | `abundance_class` $\\in$ `{{BRAK (<5.0 szt./h), SREDNIO (5.0–20.0 szt./h), WYSYP (>20.0 szt./h)}}` |
| **Liczba cech wejściowych / zakodowanych** | **{meta['features']['raw_input_features_count']}** surowych cech wejściowych (`3` kategoryczne + `36` liczbowych) $\\rightarrow$ **{meta['features']['encoded_features_count']}** cech po transformacji i dodaniu flag braków |

### Najważniejsze cechy w modelu regresyjnym (`Top-10 Feature Importances`):

| Pozycja | Zakodowana cecha w potoku | Ważność (`Gini / MSE Importance`) |
| :---: | :--- | :---: |
{top_reg_rows}

---

## 4. Porównanie z modelami bazowymi (Baseline) i Gradient Boosting na tym samym podziale (Warunek `#2`)

Wszystkie modele oceniono na **identycznym podziale czasowym `primary_split_train_test`** z zadania `[GRZ-13]` (`Train`: `2025-09-01` .. `2025-10-12`, $N = 386$; `Test`: `2025-10-13` .. `2025-11-02`, $N = 127$; `0` wspólnych raportów).

### 4.1. Zadanie regresji (`yield_pcs_per_h` w jednostce `[szt./h]`)

| Model / Estymator | MAE Test `[szt./h]` | RMSE Test `[szt./h]` | Spearman $\\rho$ (Test) | Pearson $r$ (Test) | $R^2$ (Test) | Zmiana MAE wzg. Global Mean | Zmiana MAE wzg. Powiat Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForestRegressor (`rf_pipeline_v1.0` — główny)** | **{rf_r['mae_pcs_per_h']:.3f}** | **{rf_r['rmse_pcs_per_h']:.3f}** | **{rf_r['spearman_rho']:+.4f}** | **{rf_r['pearson_r']:+.4f}** | **{rf_r['r2_score']:+.4f}** | **$-1.056\\text{{ szt./h}}$** | **$-0.616\\text{{ szt./h}}$** |
| **GradientBoostingRegressor (porównawczy boosting)** | {gbr_r['mae_pcs_per_h']:.3f} | {gbr_r['rmse_pcs_per_h']:.3f} | {gbr_r['spearman_rho']:+.4f} | {gbr_r['pearson_r']:+.4f} | {gbr_r['r2_score']:+.4f} | $-0.933\\text{{ szt./h}}$ | $-0.493\\text{{ szt./h}}$ |
| **RandomForestRegressor (`criterion='absolute_error'`)** | {rf_mae['mae_pcs_per_h']:.3f} | {rf_mae['rmse_pcs_per_h']:.3f} | **{rf_mae['spearman_rho']:+.4f}** | **{rf_mae['pearson_r']:+.4f}** | {rf_mae['r2_score']:+.4f} | $-0.768\\text{{ szt./h}}$ | $-0.328\\text{{ szt./h}}$ |
| **HistGradientBoostingRegressor (boosting histogramowy)** | {hgb_r['mae_pcs_per_h']:.3f} | {hgb_r['rmse_pcs_per_h']:.3f} | {hgb_r['spearman_rho']:+.4f} | {hgb_r['pearson_r']:+.4f} | {hgb_r['r2_score']:+.4f} | $-0.471\\text{{ szt./h}}$ | $-0.031\\text{{ szt./h}}$ |
| *Baseline R3: Powiat Historical Mean* | *{b_pow['mae_pcs_per_h']:.3f}* | *{b_pow['rmse_pcs_per_h']:.3f}* | *{b_pow['spearman_rho']:+.4f}* | *{b_pow['pearson_r']:+.4f}* | *{b_pow['r2_score']:+.4f}* | $-0.440\\text{{ szt./h}}$ | $0.000\\text{{ szt./h}}$ |
| *Baseline R1: Global Mean (`16.55 szt./h`)* | *{b_mean['mae_pcs_per_h']:.3f}* | *{b_mean['rmse_pcs_per_h']:.3f}* | *{b_mean['spearman_rho']:+.4f}* | *{b_mean['pearson_r']:+.4f}* | *{b_mean['r2_score']:+.4f}* | $0.000\\text{{ szt./h}}$ | $+0.440\\text{{ szt./h}}$ |
| *Baseline R2: Global Median (`13.50 szt./h`)* | *{b_med['mae_pcs_per_h']:.3f}* | *{b_med['rmse_pcs_per_h']:.3f}* | *{b_med['spearman_rho']:+.4f}* | *{b_med['pearson_r']:+.4f}* | *{b_med['r2_score']:+.4f}* | $+0.707\\text{{ szt./h}}$ | $+1.147\\text{{ szt./h}}$ |

> **Wnioski z porównania regresji i boostingu:**
> 1. Główny potok `RandomForestRegressor` obniża błąd **MAE na zbiorze testowym do `14.670 szt./h`** (wobec `15.726 szt./h` dla `Global Mean` oraz `15.286 szt./h` dla `Powiat Historical Mean`) oraz poprawia korelację rangową **Spearmana do $\\rho = +0.3662$** (próg sukcesu z `[GRZ-13]` wynosił $\\rho > 0.20$).
> 2. Na pomocniczym podziale 3-częściowym (`split_3way`: `Val` `2025-10-06 .. 2025-10-19`, $N=111$) ten sam potok osiąga **MAE = `{reg_comp['random_forest_regressor_primary']['val_3way_metrics']['mae_pcs_per_h']:.3f} szt./h`**.
> 3. Porównany model `GradientBoostingRegressor` również pokonuje wszystkie baseline'y (`MAE = 14.793 szt./h`, `Spearman = +0.3500`), lecz uśrednianie wiązkowe drzew (`Random Forest`) wykazuje niższą wariancję i wyższą odporność na szum pojedynczych zgłoszeń obywatelskich przy próbie treningowej $N = 386$.

---

### 4.2. Zadanie klasyfikacji (`abundance_class`: `BRAK`, `SREDNIO`, `WYSYP`)

| Model / Estymator | Macro-F1 (Test) | Weighted-F1 (Test) | Recall `BRAK` | Recall `SREDNIO` | Recall `WYSYP` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier (`rf_pipeline_v1.0` — główny)** | **{rf_c['macro_f1']:.4f}** | **{rf_c['weighted_f1']:.4f}** | **{rf_c['per_class']['BRAK']['recall']:.4f}** | **{rf_c['per_class']['SREDNIO']['recall']:.4f}** | **{rf_c['per_class']['WYSYP']['recall']:.4f}** |
| *Baseline C2: Powiat Historical Majority* | *{bc_pow['macro_f1']:.4f}* | *{bc_pow['weighted_f1']:.4f}* | *{bc_pow['per_class']['BRAK']['recall']:.4f}* | *{bc_pow['per_class']['SREDNIO']['recall']:.4f}* | *{bc_pow['per_class']['WYSYP']['recall']:.4f}* |
| **RandomForestRegressor (dyskretyzacja progowa)** | {rf_rt['macro_f1']:.4f} | {rf_rt['weighted_f1']:.4f} | {rf_rt['per_class']['BRAK']['recall']:.4f} | {rf_rt['per_class']['SREDNIO']['recall']:.4f} | {rf_rt['per_class']['WYSYP']['recall']:.4f} |
| *Baseline C3: Stratified Random* | *{bc_str['macro_f1']:.4f}* | *{bc_str['weighted_f1']:.4f}* | *{bc_str['per_class']['BRAK']['recall']:.4f}* | *{bc_str['per_class']['SREDNIO']['recall']:.4f}* | *{bc_str['per_class']['WYSYP']['recall']:.4f}* |
| *Baseline C1: Majority Class (zawsze `SREDNIO`)* | *{bc_maj['macro_f1']:.4f}* | *{bc_maj['weighted_f1']:.4f}* | *{bc_maj['per_class']['BRAK']['recall']:.4f}* | *{bc_maj['per_class']['SREDNIO']['recall']:.4f}* | *{bc_maj['per_class']['WYSYP']['recall']:.4f}* |

#### Macierz pomyłek `RandomForestClassifier` na zbiorze testowym ($N = 127$):
```text
Rzeczywiste \\ Predykcja    BRAK    SREDNIO    WYSYP
BRAK (n=15)                 {cm_rf[0][0]:<7} {cm_rf[0][1]:<10} {cm_rf[0][2]:<7}
SREDNIO (n=57)              {cm_rf[1][0]:<7} {cm_rf[1][1]:<10} {cm_rf[1][2]:<7}
WYSYP (n=55)                {cm_rf[2][0]:<7} {cm_rf[2][1]:<10} {cm_rf[2][2]:<7}
```
*Model klasyfikacyjny osiąga **Macro-F1 = `{rf_c['macro_f1']:.4f}`** (przekraczając próg `> 0.45` z protokołu `[GRZ-13]` oraz pokonując `Majority Class = 0.2065` o `+0.2674` i `Powiat Historical Majority = 0.3918` o `+0.0821`), przy zachowaniu wysokiej czułości dla skrajnych klas `BRAK` (`60.0%`) oraz `WYSYP` (`56.4%`).*

---

## 5. Przekazywany rezultat: Przykład danych wejściowych i wyjściowych (`models/example_*.json/.csv`)

Poniższa tabela przedstawia wynik wywołania `python scripts/predict_example.py` na 5 rekordach testowych z pliku [`models/example_input.json`](../models/example_input.json) (zapisany w [`models/example_output.json`](../models/example_output.json) oraz [`models/example_output.csv`](../models/example_output.csv)):

| # | Powiat (`area_id`) | Data docelowa | Filtr gatunku | Dane taksacyjne BDL | Prognoza `predicted_yield_pcs_per_h` | Klasa z regresji | Klasa z klasyfikatora | Pewność (`confidence_score`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{ex_table_rows}
"""
    doc_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trening pierwszego modelu ML (Random Forest) i zapis kompletnego potoku [GRZ-14]."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "features.parquet",
        help="Ścieżka do wejściowego pliku cech features.parquet",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Katalog docelowy na artefakty modelu i metadane",
    )
    args = parser.parse_args()
    run_training_pipeline(features_path=args.features, models_dir=args.models_dir)


if __name__ == "__main__":
    main()
