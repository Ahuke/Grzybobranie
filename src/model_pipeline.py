"""
Moduł potoku modelowania uczenia maszynowego (Pipeline) dla projektu „Gdzie na grzyby?” [GRZ-14].

Realizuje założenia architektury zatwierdzonej w docs/data_audit.md, docs/eda_and_lag_analysis.md
oraz docs/evaluation_protocol.md:
1. Pełny, samowystarczalny potok scikit-learn (`Pipeline`) łączący:
   - transformację domenową (`DomainFeatureProcessor`: wygładzone kodowanie statystyk powiatu
     wyliczone wyłącznie na zbiorze treningowym, okna różnicowe opadów 8–14d i 15–21d,
     interakcję lesistości z opadem 21-dniowym),
   - obsługę braków danych (`SimpleImputer(strategy="median", add_indicator=True)` dla cech
     liczbowych z jawnymi flagami braków BDL/pogody oraz `SimpleImputer(fill_value="__MISSING__")`
     dla cech kategorycznych — zgodnie z regułą `NULL != 0.0`),
   - kodowanie zmiennych kategorycznych (`OneHotEncoder(handle_unknown="ignore")`),
   - estymator `RandomForestRegressor` (cel główny: `yield_pcs_per_h` [szt./h]) oraz
     `RandomForestClassifier` (cel pomocniczy: `abundance_class` [`BRAK`, `SREDNIO`, `WYSYP`]).
2. Ochronę przed wyciekiem danych (No Future Leakage & No Target Leakage):
   - jawna blokada kolumn post-hoc raportu (`total_pieces`, `harvest_duration_h`, `dominant_species`)
     oraz danych demonstracyjnych (`is_demo == True`).
3. Możliwość zapisu i odtworzenia artefaktu (`.joblib`) oraz wykonania predykcji poza notebookiem
   bez ponownego uruchamiania treningu.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import MODELS_DIR, PROCESSED_DIR
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

MODEL_VERSION: str = "rf_pipeline_v1.0"
TARGET_REGRESSION_COL: str = "yield_pcs_per_h"
TARGET_CLASSIFICATION_COL: str = "abundance_class"
PREDICTED_TARGET_COL: str = "predicted_yield_pcs_per_h"
TARGET_UNIT: str = "szt./h"

# Kolumny zakazane na wejściu modelu (wyciek etykiety, cechy post-hoc z tego samego grzybobrania lub metadane)
FORBIDDEN_LEAKAGE_COLS: set[str] = {
    "yield_pcs_per_h",
    "abundance_class",
    "total_pieces",
    "harvest_duration_h",
    "dominant_species",
    "worminess_pct",
    "publication_date",
    "split_2way",
    "split_3way",
    "report_id",
    "observation_id",
    "source_url",
    "is_demo",
    "is_valid_for_training",
}

# Surowe cechy kategoryczne wymagane na wejściu potoku
RAW_CATEGORICAL_FEATURES: list[str] = [
    "area_id",
    "dominant_moisture_type",
    "season_phase",
]

# Surowe cechy liczbowe i logiczne wymagane na wejściu potoku (znane w chwili feature_cutoff_date = t - 1)
RAW_NUMERIC_FEATURES: list[str] = [
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
    "total_area_km2",
    "forest_cover_pct",
    "forest_area_km2",
    "pine_share_pct",
    "oak_beech_share_pct",
    "spruce_share_pct",
    "mean_stand_age_years",
    "has_bdl_stand_data",
    "month",
    "day_of_year",
    "week_of_year",
    "day_of_week",
    "is_weekend",
    "day_of_year_sin",
    "day_of_year_cos",
]

RAW_INPUT_FEATURES: list[str] = RAW_CATEGORICAL_FEATURES + RAW_NUMERIC_FEATURES

# Cechy przekazywane z DomainFeatureProcessor do ColumnTransformer (stabilne poza czasem)
ENGINEERED_NUMERIC_FEATURES: list[str] = [
    "county_hist_mean_smoothed",
    "county_hist_median_smoothed",
    "county_hist_wysyp_rate",
    "county_hist_brak_rate",
    "precip_lag_21d_mm",
    "precip_slice_15_21d_mm",
    "precip_lag_14d_mm",
    "precip_lag_7d_mm",
    "precip_cutoff_1d_mm",
    "forest_x_precip_21d",
    "temp_min_night_c",
    "temp_min_lag_3d_c",
    "frost_nights_lag_7d_count",
    "wind_speed_mean_7d_kmh",
    "soil_moisture_0_7cm_m3m3",
    "soil_temperature_0_7cm_c",
    "total_area_km2",
    "forest_cover_pct",
    "forest_area_km2",
    "pine_share_pct",
    "oak_beech_share_pct",
    "spruce_share_pct",
    "mean_stand_age_years",
    "has_bdl_stand_data",
    "is_weekend",
]

ENGINEERED_CATEGORICAL_FEATURES: list[str] = [
    "dominant_moisture_type",
    "county_hist_mode",
]


class DomainFeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Pierwszy etap potoku predykcyjnego scikit-learn:
    1. Waliduje wejście i normalizuje klucz przestrzenny `area_id` (4-znakowy kod TERYT).
    2. W metodzie `.fit(X, y)` oblicza wyłącznie na zbiorze treningowym wygładzone
       statystyki historyczne powiatu (Empirical Bayes Target Encoding z parametrem `smoothing`).
       Dla powiatów bez obserwacji w zbiorze treningowym (np. 0217, 0262) stosuje bezpieczny
       fallback do statystyk globalnych ze zbioru treningowego.
    3. W metodzie `.transform(X)` wylicza cechy pochodne opadów (`precip_slice_8_14d_mm`,
       `precip_slice_15_21d_mm`) oraz interakcję siedliskowo-opadową (`forest_x_precip_21d`),
       zachowując pełną semantykę `NULL != 0.0` (brak składnika daje `np.nan`, a nie `0.0`).
    """

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        if "is_demo" in df.columns and df["is_demo"].fillna(False).astype(bool).any():
            raise ValueError(
                "Naruszenie kontraktu bezpieczeństwa: wykryto rekordy demonstracyjne (is_demo=True)!"
            )

        if "area_id" not in df.columns and "region_id" in df.columns:
            df["area_id"] = df["region_id"]
        elif "area_id" not in df.columns:
            df["area_id"] = "__UNKNOWN__"

        df["area_id"] = (
            df["area_id"]
            .fillna("__UNKNOWN__")
            .astype(str)
            .str.strip()
            .apply(lambda s: s.zfill(4) if s.isdigit() and len(s) <= 4 else s)
        )
        return df

    def fit(self, X: pd.DataFrame | list[dict[str, Any]], y: Any = None) -> DomainFeatureProcessor:
        df = self._ensure_dataframe(X)

        self.global_mean_: float = 16.55
        self.global_median_: float = 13.5
        self.global_wysyp_rate_: float = 0.2772
        self.global_brak_rate_: float = 0.2513
        self.global_mode_: str = "SREDNIO"

        self.county_mean_: dict[str, float] = {}
        self.county_median_: dict[str, float] = {}
        self.county_wysyp_rate_: dict[str, float] = {}
        self.county_brak_rate_: dict[str, float] = {}
        self.county_mode_: dict[str, str] = {}
        self.covered_counties_in_train_: list[str] = []

        if y is not None:
            y_arr = np.asarray(y)
            if len(y_arr) == 0:
                raise ValueError("Wektor docelowy y w zbiorze treningowym nie może być pusty.")

            if np.issubdtype(y_arr.dtype, np.number):
                y_num = y_arr.astype(float)
                y_cls = np.asarray(classify_abundance(y_num), dtype=str)
            else:
                y_cls = y_arr.astype(str)
                class_midpoints = {"BRAK": 2.0, "SREDNIO": 13.5, "WYSYP": 32.0}
                y_num = np.array(
                    [class_midpoints.get(str(val), 13.5) for val in y_cls], dtype=float
                )

            self.global_mean_ = float(np.mean(y_num))
            self.global_median_ = float(np.median(y_num))
            self.global_wysyp_rate_ = float(np.mean(y_cls == "WYSYP"))
            self.global_brak_rate_ = float(np.mean(y_cls == "BRAK"))
            self.global_mode_ = str(pd.Series(y_cls).mode()[0])

            grouped = pd.DataFrame(
                {
                    "area_id": df["area_id"].astype(str),
                    "y_num": y_num,
                    "y_cls": y_cls,
                }
            )
            for area_id, grp in grouped.groupby("area_id"):
                if area_id == "__UNKNOWN__":
                    continue
                n = len(grp)
                weight = float(n) / (float(n) + float(self.smoothing))
                self.county_mean_[area_id] = (
                    weight * float(grp["y_num"].mean()) + (1.0 - weight) * self.global_mean_
                )
                self.county_median_[area_id] = (
                    weight * float(grp["y_num"].median()) + (1.0 - weight) * self.global_median_
                )
                self.county_wysyp_rate_[area_id] = (
                    weight * float((grp["y_cls"] == "WYSYP").mean())
                    + (1.0 - weight) * self.global_wysyp_rate_
                )
                self.county_brak_rate_[area_id] = (
                    weight * float((grp["y_cls"] == "BRAK").mean())
                    + (1.0 - weight) * self.global_brak_rate_
                )
                self.county_mode_[area_id] = str(grp["y_cls"].mode()[0])

            self.covered_counties_in_train_ = sorted(self.county_mean_.keys())

        return self

    def transform(self, X: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        df = self._ensure_dataframe(X)

        # Upewnienie się, że wszystkie oczekiwane surowe kolumny istnieją (brak kolumny -> np.nan, nigdy 0.0!)
        for col in RAW_NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
            else:
                if col in {"has_bdl_stand_data", "is_weekend"}:
                    df[col] = (
                        df[col]
                        .map(
                            lambda v: np.nan
                            if v is None or pd.isna(v)
                            else (1.0 if bool(v) and str(v).lower() not in {"false", "0"} else 0.0)
                        )
                        .astype(float)
                    )
                else:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in RAW_CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = None
            else:
                df[col] = df[col].map(
                    lambda v: None
                    if v is None or pd.isna(v) or str(v).strip() == "" or str(v).strip().lower() in {"nan", "null", "none"}
                    else str(v).strip()
                )

        areas = df["area_id"].astype(str)
        df["county_hist_mean_smoothed"] = (
            areas.map(self.county_mean_).fillna(self.global_mean_).astype(float)
        )
        df["county_hist_median_smoothed"] = (
            areas.map(self.county_median_).fillna(self.global_median_).astype(float)
        )
        df["county_hist_wysyp_rate"] = (
            areas.map(self.county_wysyp_rate_).fillna(self.global_wysyp_rate_).astype(float)
        )
        df["county_hist_brak_rate"] = (
            areas.map(self.county_brak_rate_).fillna(self.global_brak_rate_).astype(float)
        )
        df["county_hist_mode"] = areas.map(self.county_mode_).fillna(self.global_mode_).astype(object)

        # Cechy różnicowe opadów i interakcja z lesistością (jeśli którykolwiek składnik jest NaN, wynik też jest NaN)
        df["precip_slice_8_14d_mm"] = df["precip_lag_14d_mm"] - df["precip_lag_7d_mm"]
        df["precip_slice_15_21d_mm"] = df["precip_lag_21d_mm"] - df["precip_lag_14d_mm"]
        df["forest_x_precip_21d"] = (df["forest_cover_pct"] * df["precip_lag_21d_mm"]) / 100.0

        return df[ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """
    Tworzy etap przetwarzania kolumn (ColumnTransformer):
    - dla cech liczbowych: imputacja medianą ze zbioru treningowego + jawny wskaźnik braków (MissingIndicator),
      dzięki czemu model odróżnia brak danych BDL/pogody (`NULL`) od potwierdzonego zera (`0.0`),
    - dla cech kategorycznych: jawna kategoria `__MISSING__` dla wartości `NULL` oraz kodowanie
      One-Hot z bezpiecznym ignorowaniem nieznanych kategorii (`handle_unknown="ignore"`).
    """
    numeric_transformer = SimpleImputer(strategy="median", add_indicator=True)
    categorical_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ENGINEERED_NUMERIC_FEATURES),
            ("cat", categorical_transformer, ENGINEERED_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_random_forest_regression_pipeline(
    smoothing: float = 1.0,
    n_estimators: int = 200,
    criterion: str = "squared_error",
    max_depth: int | None = 5,
    min_samples_leaf: int = 2,
    max_features: float | str = 0.6,
    random_state: int = 42,
) -> Pipeline:
    """
    Buduje główny potok regresyjny Random Forest dla celu `yield_pcs_per_h` [szt./h].
    """
    return Pipeline(
        steps=[
            ("domain_fe", DomainFeatureProcessor(smoothing=smoothing)),
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_random_forest_classification_pipeline(
    smoothing: float = 1.0,
    n_estimators: int = 100,
    max_depth: int | None = 4,
    min_samples_leaf: int = 4,
    max_features: float | str = 0.6,
    class_weight: dict[str, float] | str | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Buduje pomocniczy potok klasyfikacyjny Random Forest dla 3-klasowego celu
    `abundance_class` (`BRAK`, `SREDNIO`, `WYSYP`).
    Domyślne wagi klas kompensują nadreprezentację klasy środkowej `SREDNIO` w zbiorze treningowym.
    """
    if class_weight is None:
        class_weight = {"BRAK": 1.6, "SREDNIO": 0.7, "WYSYP": 2.1}

    return Pipeline(
        steps=[
            ("domain_fe", DomainFeatureProcessor(smoothing=smoothing)),
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight=class_weight,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_gradient_boosting_comparison_pipeline(
    smoothing: float = 1.0,
    n_estimators: int = 80,
    learning_rate: float = 0.04,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    subsample: float = 0.8,
    random_state: int = 42,
) -> Pipeline:
    """
    Buduje potok porównawczy Gradient Boosting (odpowiednik drzewiastego boostingu gradientowego
    typu XGBoost/GBR) na tym samym zestawie cech i transformacji.
    """
    return Pipeline(
        steps=[
            ("domain_fe", DomainFeatureProcessor(smoothing=smoothing)),
            ("preprocessor", build_preprocessor()),
            (
                "model",
                GradientBoostingRegressor(
                    loss="squared_error",
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    subsample=subsample,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting_comparison_pipeline(
    smoothing: float = 1.0,
    max_iter: int = 80,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> Pipeline:
    """
    Buduje potok porównawczy HistGradientBoostingRegressor (histogramowy boosting gradientowy
    w stylu LightGBM/XGBoost hist).
    """
    return Pipeline(
        steps=[
            ("domain_fe", DomainFeatureProcessor(smoothing=smoothing)),
            ("preprocessor", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )


def compute_species_habitat_multiplier(
    species_filter: str,
    forest_cover_pct: float | None,
    pine_share_pct: float | None,
    oak_beech_share_pct: float | None,
    spruce_share_pct: float | None,
    mean_stand_age_years: float | None,
) -> tuple[float, str]:
    """
    Oblicza jawny mnożnik siedliskowy BDL ($W_{a,s} \\in [0.25, 1.25]$) dla wybranego gatunku
    zgodnie z zatwierdzoną architekturą w docs/data_audit.md (Sekcja 6.1).
    Gdy brak szczegółowej taksacji BDL (`NULL`), zwraca neutralny mnożnik 1.0 z adnotacją o braku danych BDL.
    """
    sp = (species_filter or "all").strip().lower()
    fc_txt = f"{forest_cover_pct:.1f}%" if forest_cover_pct is not None and not pd.isna(forest_cover_pct) else "NULL"

    if sp in {"all", "mieszane", "__null__"}:
        return 1.0, f"Prognoza ogólna dla wszystkich gatunków leśnych (lesistość powiatu: {fc_txt})."

    has_bdl = any(
        v is not None and not pd.isna(v)
        for v in (pine_share_pct, oak_beech_share_pct, spruce_share_pct)
    )
    if not has_bdl:
        return (
            1.0,
            f"Brak szczegółowej taksacji gatunkowej BDL (NULL) — zastosowano bazową prognozę powiatu (lesistość: {fc_txt}).",
        )

    pine = float(pine_share_pct) if pine_share_pct is not None and not pd.isna(pine_share_pct) else 50.0
    oak_beech = (
        float(oak_beech_share_pct)
        if oak_beech_share_pct is not None and not pd.isna(oak_beech_share_pct)
        else 20.0
    )
    spruce = float(spruce_share_pct) if spruce_share_pct is not None and not pd.isna(spruce_share_pct) else 15.0
    age = float(mean_stand_age_years) if mean_stand_age_years is not None and not pd.isna(mean_stand_age_years) else 65.0

    if sp == "podgrzybek_brunatny":
        conifer_share = pine + 0.6 * spruce
        mult = float(np.clip(0.55 + 0.008 * conifer_share, 0.40, 1.25))
        note = f"Podgrzybek brunatny: bory iglaste BDL (sosna {pine:.1f}%, świerk {spruce:.1f}%), mnożnik siedliska {mult:.2f}x."
    elif sp == "borowik_szlachetny":
        noble_share = oak_beech * 1.2 + spruce * 0.8 + pine * 0.3
        mult = float(np.clip(0.55 + 0.008 * noble_share, 0.40, 1.25))
        note = f"Borowik szlachetny: dąb/buk ({oak_beech:.1f}%) i świerk ({spruce:.1f}%), mnożnik siedliska {mult:.2f}x."
    elif sp == "kurka":
        mult = float(np.clip(0.50 + 0.0085 * pine, 0.35, 1.20))
        note = f"Pieprznik jadalny (kurka): bory sosnowe ({pine:.1f}%), mnożnik siedliska {mult:.2f}x."
    elif sp == "maslak_zwyczajny":
        young_bonus = 0.12 if age < 60.0 else 0.0
        mult = float(np.clip(0.45 + 0.008 * pine + young_bonus, 0.35, 1.25))
        note = f"Maślak zwyczajny: udział sosny {pine:.1f}% (wiek drzewostanu {age:.0f} lat), mnożnik {mult:.2f}x."
    elif sp == "kania":
        mixed_share = oak_beech + 0.5 * pine
        mult = float(np.clip(0.50 + 0.008 * mixed_share, 0.40, 1.20))
        note = f"Czubajka kania: lasy mieszane i liściaste (dąb/buk {oak_beech:.1f}%), mnożnik {mult:.2f}x."
    else:
        mult = 1.0
        note = f"Gatunek '{sp}': mnożnik neutralny 1.00x."

    return round(mult, 4), note


def extract_pipeline_feature_metadata(pipeline: Pipeline) -> dict[str, Any]:
    """
    Wyciąga z dopasowanego potoku pełną listę cech po transformacji, mediany imputacji
    oraz ważności cech (Feature Importances) estymatora drzewiastego.
    """
    prep: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    domain_fe: DomainFeatureProcessor = pipeline.named_steps["domain_fe"]

    feature_names_out = [str(f) for f in prep.get_feature_names_out()]
    importances = (
        [round(float(x), 6) for x in model.feature_importances_]
        if hasattr(model, "feature_importances_")
        else []
    )

    importance_ranking = sorted(
        [
            {"encoded_feature": name, "importance": imp}
            for name, imp in zip(feature_names_out, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    num_imputer: SimpleImputer = prep.named_transformers_["num"]
    imputation_medians: dict[str, float] = {}
    if hasattr(num_imputer, "statistics_"):
        for col_name, stat_val in zip(ENGINEERED_NUMERIC_FEATURES, num_imputer.statistics_):
            if not np.isnan(stat_val):
                imputation_medians[col_name] = round(float(stat_val), 4)

    cat_ohe: OneHotEncoder = prep.named_transformers_["cat"].named_steps["ohe"]
    categorical_categories: dict[str, list[str]] = {}
    if hasattr(cat_ohe, "categories_"):
        for col_name, cats in zip(ENGINEERED_CATEGORICAL_FEATURES, cat_ohe.categories_):
            categorical_categories[col_name] = [str(c) for c in cats]

    return {
        "raw_input_features_count": len(RAW_INPUT_FEATURES),
        "raw_categorical_features": RAW_CATEGORICAL_FEATURES,
        "raw_numeric_features": RAW_NUMERIC_FEATURES,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "engineered_categorical_features": ENGINEERED_CATEGORICAL_FEATURES,
        "encoded_features_count": len(feature_names_out),
        "encoded_feature_names": feature_names_out,
        "numeric_imputation_strategy": "median_with_missing_indicator (fitted on train only, NULL != 0.0)",
        "numeric_imputation_medians_from_train": imputation_medians,
        "categorical_imputation_strategy": "constant '__MISSING__' + OneHotEncoder(handle_unknown='ignore')",
        "categorical_encoded_categories": categorical_categories,
        "county_prior_global_fallbacks": {
            "global_mean_pcs_per_h": round(domain_fe.global_mean_, 4),
            "global_median_pcs_per_h": round(domain_fe.global_median_, 4),
            "global_wysyp_rate": round(domain_fe.global_wysyp_rate_, 4),
            "global_brak_rate": round(domain_fe.global_brak_rate_, 4),
            "global_mode": domain_fe.global_mode_,
            "covered_counties_in_train_count": len(domain_fe.covered_counties_in_train_),
            "covered_counties_in_train": domain_fe.covered_counties_in_train_,
        },
        "top_15_feature_importances": importance_ranking[:15],
        "all_feature_importances": importance_ranking,
    }


def predict_with_saved_pipelines(
    inputs: pd.DataFrame | list[dict[str, Any]],
    reg_pipeline_path: Path | None = None,
    clf_pipeline_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Wykonuje predykcję dla nowych rekordów wejściowych przy użyciu zapisanych potoków
    `.joblib` bez konieczności uruchamiania treningu ([GRZ-14] Warunek #5).
    """
    reg_path = reg_pipeline_path or (MODELS_DIR / "rf_yield_pipeline.joblib")
    clf_path = clf_pipeline_path or (MODELS_DIR / "rf_abundance_pipeline.joblib")

    if not reg_path.exists():
        raise FileNotFoundError(
            f"Brak wytrenowanego potoku regresyjnego: {reg_path}. Uruchom: python scripts/train_model.py"
        )
    if not clf_path.exists():
        raise FileNotFoundError(
            f"Brak wytrenowanego potoku klasyfikacyjnego: {clf_path}. Uruchom: python scripts/train_model.py"
        )

    reg_pipe: Pipeline = joblib.load(reg_path)
    clf_pipe: Pipeline = joblib.load(clf_path)

    df_in = inputs.copy() if isinstance(inputs, pd.DataFrame) else pd.DataFrame(inputs)
    if len(df_in) == 0:
        return []

    raw_preds_reg = np.clip(reg_pipe.predict(df_in), a_min=0.0, a_max=None)
    preds_cls_direct = clf_pipe.predict(df_in)
    probs_cls = clf_pipe.predict_proba(df_in)
    clf_classes = [str(c) for c in clf_pipe.named_steps["model"].classes_]

    results: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df_in.iterrows()):
        base_yield = float(raw_preds_reg[idx])
        species_filter = str(row.get("species_filter") or "all").strip()
        mult, habitat_note = compute_species_habitat_multiplier(
            species_filter=species_filter,
            forest_cover_pct=row.get("forest_cover_pct"),
            pine_share_pct=row.get("pine_share_pct"),
            oak_beech_share_pct=row.get("oak_beech_share_pct"),
            spruce_share_pct=row.get("spruce_share_pct"),
            mean_stand_age_years=row.get("mean_stand_age_years"),
        )
        adjusted_yield = round(max(0.0, base_yield * mult), 2)
        abundance_from_reg = str(classify_abundance(adjusted_yield))
        direct_cls = str(preds_cls_direct[idx])

        prob_map = {
            cls_name: round(float(probs_cls[idx, c_i]), 4)
            for c_i, cls_name in enumerate(clf_classes)
        }
        max_prob = max(prob_map.values()) if prob_map else 0.5
        has_bdl = bool(
            row.get("has_bdl_stand_data")
            if row.get("has_bdl_stand_data") is not None and not pd.isna(row.get("has_bdl_stand_data"))
            else (row.get("pine_share_pct") is not None and not pd.isna(row.get("pine_share_pct")))
        )
        confidence = round(min(0.95, max(0.35, 0.5 * max_prob + (0.35 if has_bdl else 0.20))), 2)

        area_id = str(row.get("area_id") or row.get("region_id") or "").strip().zfill(4)
        target_date = str(row.get("target_date") or row.get("harvest_date") or row.get("date") or "")
        issued_at = str(row.get("issued_at") or row.get("feature_cutoff_date") or target_date)

        results.append(
            {
                "record_index": idx,
                "area_id": area_id,
                "region_name": str(row.get("region_name") or row.get("area_name") or area_id),
                "target_date": target_date,
                "generated_at": issued_at,
                "species_filter": species_filter,
                "predicted_yield_pcs_per_h": adjusted_yield,
                "base_predicted_yield_pcs_per_h": round(base_yield, 2),
                "species_habitat_multiplier": mult,
                "target_unit": TARGET_UNIT,
                "abundance_class": abundance_from_reg,
                "abundance_class_classifier": direct_cls,
                "class_probabilities": prob_map,
                "confidence_score": confidence,
                "has_bdl_stand_data": has_bdl,
                "habitat_match_note": habitat_note,
                "model_version": MODEL_VERSION,
                "is_demo": False,
            }
        )

    return results
