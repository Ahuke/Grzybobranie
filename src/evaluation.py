"""
Moduł protokołu ewaluacji i metryk jakości dla projektu „Gdzie na grzyby?” [GRZ-13].

Zapewnia:
1. Sztywny, deterministyczny podział czasowy (Temporal Split) bez wycieku danych (leakage).
2. Spójną klasyfikację obfitości zbioru (`BRAK`, `SREDNIO`, `WYSYP`) zgodną z kontraktem danych v2.0.
3. Obliczanie metryk regresji (MAE, RMSE, Pearson, Spearman, R²) oraz klasyfikacji (Macro-F1, Weighted-F1, Confusion Matrix).
4. Walidację szczelności podziału (brak współdzielonych wycieczek i dat).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from scipy.stats import spearmanr, pearsonr

# Oficjalne progi klas obfitości wysypu (docs/data_contract.md i docs/mvp.md v2.0)
ABUNDANCE_LABELS = ["BRAK", "SREDNIO", "WYSYP"]
THRESH_BRAK_MAX = 5.0
THRESH_SREDNIO_MAX = 20.0

# Domyślne daty odcięcia podziału czasowego (sezon jesień 2025: 01.09 - 02.11)
DEFAULT_SPLIT_DATE_TRAIN_TEST = "2025-10-12"
DEFAULT_SPLIT_DATE_TRAIN_VAL = "2025-10-05"
DEFAULT_SPLIT_DATE_VAL_TEST = "2025-10-19"


def classify_abundance(yield_pcs_per_h: float | np.ndarray | pd.Series) -> str | np.ndarray | pd.Series:
    """
    Klasyfikuje wydajność zbioru [szt./h] do 3 rozłącznych klas:
    - `BRAK`: < 5.0 szt./h
    - `SREDNIO`: 5.0 - 20.0 szt./h
    - `WYSYP`: > 20.0 szt./h
    """
    if isinstance(yield_pcs_per_h, (pd.Series, np.ndarray)):
        arr = np.asarray(yield_pcs_per_h)
        classes = np.empty(arr.shape, dtype=object)
        classes[arr < THRESH_BRAK_MAX] = "BRAK"
        classes[(arr >= THRESH_BRAK_MAX) & (arr <= THRESH_SREDNIO_MAX)] = "SREDNIO"
        classes[arr > THRESH_SREDNIO_MAX] = "WYSYP"
        if isinstance(yield_pcs_per_h, pd.Series):
            return pd.Series(classes, index=yield_pcs_per_h.index)
        return classes
    
    val = float(yield_pcs_per_h)
    if val < THRESH_BRAK_MAX:
        return "BRAK"
    elif val <= THRESH_SREDNIO_MAX:
        return "SREDNIO"
    else:
        return "WYSYP"


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> Dict[str, Any]:
    """
    Oblicza zestaw metryk dla predykcji ciągłej `yield_pcs_per_h` [szt./h]:
    - MAE [szt./h] - błąd bezwzględny
    - RMSE [szt./h] - pierwiastek błędu średniokwadratowego
    - Pearson r - korelacja liniowa
    - Spearman rho - monotoniczna korelacja rangowa
    - R² - współczynnik determinacji (z flagą interpretowalności dla danych czasowych)
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if len(y_true_arr) == 0:
        raise ValueError("y_true nie może być puste")

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Korelacje (jeśli predykcja nie jest stałą wartością)
    if np.all(y_pred_arr == y_pred_arr[0]):
        pearson_val = 0.0
        spearman_val = 0.0
    else:
        p_res = pearsonr(y_true_arr, y_pred_arr)
        s_res = spearmanr(y_true_arr, y_pred_arr)
        pearson_val = float(p_res[0]) if not np.isnan(p_res[0]) else 0.0
        spearman_val = float(s_res[0]) if not np.isnan(s_res[0]) else 0.0

    return {
        "sample_count": len(y_true_arr),
        "mae_pcs_per_h": round(mae, 3),
        "rmse_pcs_per_h": round(rmse, 3),
        "pearson_r": round(pearson_val, 4),
        "spearman_rho": round(spearman_val, 4),
        "r2_score": round(r2, 4),
        "r2_interpretable": bool(r2 > -0.05),  # R² poniżej 0 oznacza model gorszy od średniej testowej
    }


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: List[str] = ABUNDANCE_LABELS,
) -> Dict[str, Any]:
    """
    Oblicza zestaw metryk dla predykcji klasowej (`BRAK`, `SREDNIO`, `WYSYP`):
    - Macro-F1 (główna metryka oceny zbalansowanej między klasami)
    - Weighted-F1
    - Macierz pomyłek (Confusion Matrix)
    - Precyzja, Czułość (Recall) i F1 per klasa
    """
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)

    if len(y_true_arr) == 0:
        raise ValueError("y_true nie może być puste")

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )

    per_class = {}
    for idx, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1_score": round(float(f1[idx]), 4),
            "support": int(sup[idx]),
        }

    return {
        "sample_count": len(y_true_arr),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": per_class,
    }


def split_dataset_temporally(
    df: pd.DataFrame,
    date_column: str = "harvest_date",
    mode: str = "train_test",
    split_date_test: str = DEFAULT_SPLIT_DATE_TRAIN_TEST,
    split_date_val: str = DEFAULT_SPLIT_DATE_TRAIN_VAL,
) -> Dict[str, pd.DataFrame]:
    """
    Dzieli zbiór obserwacji czysto według kryterium czasowego (bez wycieków).
    
    Tryby (mode):
    - 'train_test':
        train: date_column <= split_date_test (np. 2025-09-01 .. 2025-10-12)
        test:  date_column > split_date_test  (np. 2025-10-13 .. 2025-11-02)
    - 'train_val_test':
        train: date_column <= split_date_val
        val:   split_date_val < date_column <= split_date_test
        test:  date_column > split_date_test
    """
    df_sorted = df.sort_values(date_column).copy()

    if mode == "train_test":
        train = df_sorted[df_sorted[date_column] <= split_date_test].copy()
        test = df_sorted[df_sorted[date_column] > split_date_test].copy()
        validate_no_leakage(train, test, date_column=date_column)
        return {"train": train, "test": test}

    elif mode == "train_val_test":
        train = df_sorted[df_sorted[date_column] <= split_date_val].copy()
        val = df_sorted[(df_sorted[date_column] > split_date_val) & (df_sorted[date_column] <= split_date_test)].copy()
        test = df_sorted[df_sorted[date_column] > split_date_test].copy()
        validate_no_leakage(train, test, val_df=val, date_column=date_column)
        return {"train": train, "val": val, "test": test}

    else:
        raise ValueError(f"Nieobsługiwany tryb podziału: {mode}. Dostępne: 'train_test', 'train_val_test'")


def validate_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    date_column: str = "harvest_date",
    id_column: str = "report_id",
) -> None:
    """
    Waliduje brak wycieku danych (leakage):
    1. Rozłączność identyfikatorów `report_id`.
    2. Ścisłe następstwo czasowe (brak nakładania się dat).
    """
    train_ids = set(train_df[id_column])
    test_ids = set(test_df[id_column])
    
    overlap_train_test = train_ids.intersection(test_ids)
    if overlap_train_test:
        raise AssertionError(f"Wyciek ID między train i test: {len(overlap_train_test)} wspólnych ID!")

    max_train_date = train_df[date_column].max()
    min_test_date = test_df[date_column].min()

    if val_df is not None:
        val_ids = set(val_df[id_column])
        overlap_train_val = train_ids.intersection(val_ids)
        overlap_val_test = val_ids.intersection(test_ids)
        if overlap_train_val:
            raise AssertionError(f"Wyciek ID między train i val: {len(overlap_train_val)} wspólnych ID!")
        if overlap_val_test:
            raise AssertionError(f"Wyciek ID między val i test: {len(overlap_val_test)} wspólnych ID!")
        
        min_val_date = val_df[date_column].min()
        max_val_date = val_df[date_column].max()

        if max_train_date >= min_val_date:
            raise AssertionError(f"Wyciek czasowy train/val: max train ({max_train_date}) >= min val ({min_val_date})")
        if max_val_date >= min_test_date:
            raise AssertionError(f"Wyciek czasowy val/test: max val ({max_val_date}) >= min test ({min_test_date})")
    else:
        if max_train_date >= min_test_date:
            raise AssertionError(f"Wyciek czasowy train/test: max train ({max_train_date}) >= min test ({min_test_date})")
