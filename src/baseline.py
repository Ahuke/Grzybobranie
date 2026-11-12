"""
Modele bazowe (Baseline Predictors) dla projektu „Gdzie na grzyby?” [GRZ-13].

Stanowią uczciwy, formalny punkt odniesienia (Benchmark) dla złożonych modeli ML (GRZ-14).
Zgodnie z wymogiem DoD:
- Każdy predyktor jest dopasowywany (fit) WYŁĄCZNIE na danych ze zbioru treningowego.
- Zero wycieku danych z przyszłości (No future leakage).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class GlobalMeanRegressor:
    """
    Predyktor odniesienia przewidujący globalną średnią wydajność zbioru ze zbioru treningowego.
    """
    def __init__(self) -> None:
        self.mean_value_: Optional[float] = None

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> GlobalMeanRegressor:
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Zbiór treningowy nie może być pusty")
        self.mean_value_ = float(np.mean(y_arr))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray | int) -> np.ndarray:
        if self.mean_value_ is None:
            raise RuntimeError("Model nie został dopasowany (.fit)")
        n_samples = len(X) if hasattr(X, "__len__") else int(X)
        return np.full(n_samples, self.mean_value_, dtype=float)


class GlobalMedianRegressor:
    """
    Predyktor odniesienia przewidujący medianę ze zbioru treningowego (minimalizuje MAE).
    """
    def __init__(self) -> None:
        self.median_value_: Optional[float] = None

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> GlobalMedianRegressor:
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Zbiór treningowy nie może być pusty")
        self.median_value_ = float(np.median(y_arr))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray | int) -> np.ndarray:
        if self.median_value_ is None:
            raise RuntimeError("Model nie został dopasowany (.fit)")
        n_samples = len(X) if hasattr(X, "__len__") else int(X)
        return np.full(n_samples, self.median_value_, dtype=float)


class PowiatHistoricalMeanRegressor:
    """
    Predyktor odniesienia przewidujący średnią historyczną danego powiatu (area_id / region_id)
    obliczoną WYŁĄCZNIE na zbiorze treningowym. Dla powiatów bez obserwacji w treningu
    stosuje bezpieczny fallback do średniej globalnej.
    """
    def __init__(self, area_column: str = "area_id") -> None:
        self.area_column = area_column
        self.global_mean_: Optional[float] = None
        self.powiat_means_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> PowiatHistoricalMeanRegressor:
        y_ser = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.global_mean_ = float(y_ser.mean())
        col = self.area_column if self.area_column in X.columns else "region_id"
        self.powiat_means_ = y_ser.groupby(X[col]).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("Model nie został dopasowany (.fit)")
        col = self.area_column if self.area_column in X.columns else "region_id"
        areas = X[col].astype(str)
        preds = areas.map(self.powiat_means_).fillna(self.global_mean_).to_numpy(dtype=float)
        return preds


class MajorityClassClassifier:
    """
    Predyktor odniesienia dla klasyfikacji obfitości (`BRAK`, `SREDNIO`, `WYSYP`),
    który zawsze przewiduje najczęstszą klasę ze zbioru treningowego (Dummy Majority).
    """
    def __init__(self) -> None:
        self.majority_class_: Optional[str] = None

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> MajorityClassClassifier:
        y_ser = pd.Series(y)
        if len(y_ser) == 0:
            raise ValueError("Zbiór treningowy nie może być pusty")
        self.majority_class_ = str(y_ser.mode()[0])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray | int) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("Model nie został dopasowany (.fit)")
        n_samples = len(X) if hasattr(X, "__len__") else int(X)
        return np.full(n_samples, self.majority_class_, dtype=object)


class PowiatHistoricalMajorityClassifier:
    """
    Predyktor odniesienia przewidujący dominującą klasę wysypu dla danego powiatu
    na podstawie historii ze zbioru treningowego. Dla powiatów bez danych w train
    cofa się do globalnej dominującej klasy.
    """
    def __init__(self, area_column: str = "area_id") -> None:
        self.area_column = area_column
        self.global_majority_: Optional[str] = None
        self.powiat_modes_: Dict[str, str] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> PowiatHistoricalMajorityClassifier:
        y_ser = pd.Series(y, index=X.index)
        self.global_majority_ = str(y_ser.mode()[0])
        col = self.area_column if self.area_column in X.columns else "region_id"

        modes = {}
        for area, group in y_ser.groupby(X[col]):
            modes[str(area)] = str(group.mode()[0])
        self.powiat_modes_ = modes
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_majority_ is None:
            raise RuntimeError("Model nie został dopasowany (.fit)")
        col = self.area_column if self.area_column in X.columns else "region_id"
        areas = X[col].astype(str)
        preds = areas.map(self.powiat_modes_).fillna(self.global_majority_).to_numpy(dtype=object)
        return preds


class StratifiedRandomClassifier:
    """
    Predyktor probabilistyczny losujący klasy ze zbioru treningowego zgodnie z ich proporcjami.
    """
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: pd.Series | np.ndarray) -> StratifiedRandomClassifier:
        y_ser = pd.Series(y)
        counts = y_ser.value_counts(normalize=True)
        self.classes_ = counts.index.to_numpy()
        self.probabilities_ = counts.to_numpy(dtype=float)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray | int) -> np.ndarray:
        if self.classes_ is None or self.probabilities_ is None:
            raise RuntimeError("Model nie został dopasowany (.fit)")
        n_samples = len(X) if hasattr(X, "__len__") else int(X)
        rng = np.random.default_rng(self.random_state)
        return rng.choice(self.classes_, size=n_samples, p=self.probabilities_)
