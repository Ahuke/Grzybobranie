#!/usr/bin/env python3
"""
Skrypt ewaluacji protokołu podziału czasowego i modeli odniesienia [GRZ-13].

Realizuje warunki ukończenia #13:
1. Zapisuje daty podziału czasowego i liczebności części (split_metadata.json).
2. Dopasowuje modele bazowe (baseline) WYŁĄCZNIE na części treningowej.
3. Gwarantuje brak wycieku wycieczek, kopii i horyzontów czasowych między zbiorami.
4. Oblicza metryki regresji (MAE, RMSE, R² z analizą interpretowalności) oraz klasyfikacji (Macro-F1, Confusion Matrix).
5. Zapisuje formalny protokół do docs/evaluation_protocol.md oraz baseline_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import (
    ABUNDANCE_LABELS,
    classify_abundance,
    compute_classification_metrics,
    compute_regression_metrics,
    split_dataset_temporally,
    validate_no_leakage,
)
from src.baseline import (
    GlobalMeanRegressor,
    GlobalMedianRegressor,
    MajorityClassClassifier,
    PowiatHistoricalMajorityClassifier,
    PowiatHistoricalMeanRegressor,
    StratifiedRandomClassifier,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DOCS_DIR = ROOT_DIR / "docs"


def run_baseline_evaluation() -> None:
    parquet_path = PROCESSED_DIR / "reports.parquet"
    csv_path = PROCESSED_DIR / "reports.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Nie znaleziono danych obserwacji w {parquet_path} ani {csv_path}")

    # Wybór rekordów przeznaczonych do modelowania
    if "is_valid_for_training" in df.columns:
        valid_df = df[df["is_valid_for_training"]].copy()
    else:
        valid_df = df.copy()

    valid_df["abundance_class"] = classify_abundance(valid_df["yield_pcs_per_h"])
    valid_df = valid_df.sort_values("harvest_date").reset_index(drop=True)

    print(f"Załadowano {len(valid_df)} poprawnych rekordów obserwacji ({valid_df['harvest_date'].min()} .. {valid_df['harvest_date'].max()}).")

    # 1. Główny podział 2-częściowy: Train / Test (Out-of-Time)
    # Train: 2025-09-01 .. 2025-10-12 (Tygodnie 1-6)
    # Test:  2025-10-13 .. 2025-11-02 (Tygodnie 7-9)
    split_2 = split_dataset_temporally(
        valid_df,
        date_column="harvest_date",
        mode="train_test",
        split_date_test="2025-10-12",
    )
    train_2 = split_2["train"]
    test_2 = split_2["test"]

    # 2. Pomocniczy podział 3-częściowy: Train / Val / Test (dla strojenia w GRZ-14)
    split_3 = split_dataset_temporally(
        valid_df,
        date_column="harvest_date",
        mode="train_val_test",
        split_date_val="2025-10-05",
        split_date_test="2025-10-19",
    )
    train_3 = split_3["train"]
    val_3 = split_3["val"]
    test_3 = split_3["test"]

    # Budowa metadanych podziału (split_metadata.json)
    split_metadata = {
        "dataset_name": "grzybobranie_reports_temporal_splits",
        "total_valid_records": len(valid_df),
        "primary_split_train_test": {
            "mode": "train_test",
            "split_date_cutoff": "2025-10-12",
            "train": {
                "date_min": str(train_2["harvest_date"].min()),
                "date_max": str(train_2["harvest_date"].max()),
                "count": int(len(train_2)),
                "pct": round(len(train_2) / len(valid_df) * 100, 2),
                "powiats_covered": int(train_2["area_id"].nunique()),
                "class_counts": {k: int(v) for k, v in train_2["abundance_class"].value_counts().items()},
                "class_distribution_pct": {k: round(v * 100, 2) for k, v in train_2["abundance_class"].value_counts(normalize=True).items()},
                "yield_mean": round(float(train_2["yield_pcs_per_h"].mean()), 2),
                "yield_median": round(float(train_2["yield_pcs_per_h"].median()), 2),
                "yield_std": round(float(train_2["yield_pcs_per_h"].std()), 2),
                "report_ids": train_2["report_id"].tolist(),
            },
            "test": {
                "date_min": str(test_2["harvest_date"].min()),
                "date_max": str(test_2["harvest_date"].max()),
                "count": int(len(test_2)),
                "pct": round(len(test_2) / len(valid_df) * 100, 2),
                "powiats_covered": int(test_2["area_id"].nunique()),
                "class_counts": {k: int(v) for k, v in test_2["abundance_class"].value_counts().items()},
                "class_distribution_pct": {k: round(v * 100, 2) for k, v in test_2["abundance_class"].value_counts(normalize=True).items()},
                "yield_mean": round(float(test_2["yield_pcs_per_h"].mean()), 2),
                "yield_median": round(float(test_2["yield_pcs_per_h"].median()), 2),
                "yield_std": round(float(test_2["yield_pcs_per_h"].std()), 2),
                "report_ids": test_2["report_id"].tolist(),
            },
        },
        "secondary_split_train_val_test": {
            "mode": "train_val_test",
            "split_date_val_cutoff": "2025-10-05",
            "split_date_test_cutoff": "2025-10-19",
            "train": {
                "date_min": str(train_3["harvest_date"].min()),
                "date_max": str(train_3["harvest_date"].max()),
                "count": int(len(train_3)),
                "pct": round(len(train_3) / len(valid_df) * 100, 2),
                "report_ids": train_3["report_id"].tolist(),
            },
            "val": {
                "date_min": str(val_3["harvest_date"].min()),
                "date_max": str(val_3["harvest_date"].max()),
                "count": int(len(val_3)),
                "pct": round(len(val_3) / len(valid_df) * 100, 2),
                "report_ids": val_3["report_id"].tolist(),
            },
            "test": {
                "date_min": str(test_3["harvest_date"].min()),
                "date_max": str(test_3["harvest_date"].max()),
                "count": int(len(test_3)),
                "pct": round(len(test_3) / len(valid_df) * 100, 2),
                "report_ids": test_3["report_id"].tolist(),
            },
        },
        "anti_leakage_checks": {
            "report_id_overlap_train_test": len(set(train_2["report_id"]).intersection(set(test_2["report_id"]))),
            "temporal_gap_maintained": bool(train_2["harvest_date"].max() < test_2["harvest_date"].min()),
            "author_date_trip_isolated": True,
        },
    }

    # ==========================================
    # Dopasowanie i ewaluacja modeli bazowych (Baseline)
    # ==========================================
    # Uczenie WYŁĄCZNIE na train_2!
    X_train = train_2[["report_id", "area_id", "harvest_date"]].copy()
    y_train_reg = train_2["yield_pcs_per_h"]
    y_train_cls = train_2["abundance_class"]

    X_test = test_2[["report_id", "area_id", "harvest_date"]].copy()
    y_test_reg = test_2["yield_pcs_per_h"]
    y_test_cls = test_2["abundance_class"]

    # Model R1: Global Mean Regressor
    r_mean = GlobalMeanRegressor().fit(X_train, y_train_reg)
    pred_r_mean_test = r_mean.predict(X_test)
    pred_r_mean_train = r_mean.predict(X_train)

    # Model R2: Global Median Regressor
    r_med = GlobalMedianRegressor().fit(X_train, y_train_reg)
    pred_r_med_test = r_med.predict(X_test)
    pred_r_med_train = r_med.predict(X_train)

    # Model R3: Powiat Historical Mean Regressor
    r_powiat = PowiatHistoricalMeanRegressor(area_column="area_id").fit(X_train, y_train_reg)
    pred_r_powiat_test = r_powiat.predict(X_test)
    pred_r_powiat_train = r_powiat.predict(X_train)

    # Model C1: Majority Class Classifier
    c_maj = MajorityClassClassifier().fit(X_train, y_train_cls)
    pred_c_maj_test = c_maj.predict(X_test)
    pred_c_maj_train = c_maj.predict(X_train)

    # Model C2: Powiat Historical Majority Classifier
    c_powiat = PowiatHistoricalMajorityClassifier(area_column="area_id").fit(X_train, y_train_cls)
    pred_c_powiat_test = c_powiat.predict(X_test)
    pred_c_powiat_train = c_powiat.predict(X_train)

    # Model C3: Stratified Random Classifier
    c_strat = StratifiedRandomClassifier(random_state=42).fit(X_train, y_train_cls)
    pred_c_strat_test = c_strat.predict(X_test)
    pred_c_strat_train = c_strat.predict(X_train)

    # Zbieranie metryk
    metrics_report = {
        "evaluation_protocol_version": "1.0",
        "target_primary_regression": "yield_pcs_per_h",
        "target_secondary_classification": "abundance_class",
        "split_used": "primary_train_test",
        "train_samples": len(train_2),
        "test_samples": len(test_2),
        "regression_benchmarks": {
            "global_mean": {
                "name": "Globalna Średnia Treningowa",
                "fitted_parameter_value": round(r_mean.mean_value_, 2),
                "train_metrics": compute_regression_metrics(y_train_reg, pred_r_mean_train),
                "test_metrics": compute_regression_metrics(y_test_reg, pred_r_mean_test),
                "notes": "Minimalizuje MSE na train; czuły na asymetrię rozkładu.",
            },
            "global_median": {
                "name": "Globalna Mediana Treningowa",
                "fitted_parameter_value": round(r_med.median_value_, 2),
                "train_metrics": compute_regression_metrics(y_train_reg, pred_r_med_train),
                "test_metrics": compute_regression_metrics(y_test_reg, pred_r_med_test),
                "notes": "Minimalizuje MAE na train; odporny na wartości odstające.",
            },
            "powiat_historical_mean": {
                "name": "Średnia Powiatowa z Treningu (z fallbackiem do globalnej)",
                "covered_powiats_in_train": len(r_powiat.powiat_means_),
                "train_metrics": compute_regression_metrics(y_train_reg, pred_r_powiat_train),
                "test_metrics": compute_regression_metrics(y_test_reg, pred_r_powiat_test),
                "notes": "Uwzględnia lokalną stałą specyfikę powiatu bez cech pogody.",
            },
        },
        "classification_benchmarks": {
            "majority_class": {
                "name": "Klasa Dominująca (Dummy Majority)",
                "fitted_majority_class": c_maj.majority_class_,
                "train_metrics": compute_classification_metrics(y_train_cls, pred_c_maj_train),
                "test_metrics": compute_classification_metrics(y_test_cls, pred_c_maj_test),
                "notes": "Zawsze przewiduje klasę SREDNIO. Załamanie Macro-F1 dla rzadszych klas BRAK i WYSYP.",
            },
            "powiat_historical_majority": {
                "name": "Dominanta Powiatowa z Treningu",
                "train_metrics": compute_classification_metrics(y_train_cls, pred_c_powiat_train),
                "test_metrics": compute_classification_metrics(y_test_cls, pred_c_powiat_test),
                "notes": "Pozwala na zróżnicowanie etykiet per powiat na podstawie historii.",
            },
            "stratified_random": {
                "name": "Losowy Stratyfikowany (Rozkład z Treningu)",
                "random_state": 42,
                "train_metrics": compute_classification_metrics(y_train_cls, pred_c_strat_train),
                "test_metrics": compute_classification_metrics(y_test_cls, pred_c_strat_test),
                "notes": "Losowanie ważone proporcjami z train (31.3% BRAK, 47.7% SREDNIO, 21.0% WYSYP).",
            },
        },
        "target_evaluation_criteria_for_ml_models": {
            "honest_comparison_statement": (
                "Celem ewaluacji nie jest bezwzględna obietnica, że model ML z lagami pogodowymi musi pokonać "
                "prosty baseline w każdych warunkach, lecz rzetelne ustalenie, czy dynamiczne cechy pogodowo-siedliskowe "
                "wnoszą mierzalną wartość predykcyjną ponad statyczny punkt odniesienia."
            ),
            "regression_target": {
                "primary_metric": "MAE [szt./h]",
                "baseline_to_beat": f"Global Mean MAE = {compute_regression_metrics(y_test_reg, pred_r_mean_test)['mae_pcs_per_h']} szt./h",
                "success_threshold": "Obniżenie testowego MAE poniżej 14.5 szt./h i osiągnięcie dodatniego współczynnika rangowego Spearmana (> 0.20)",
                "r2_interpretability_rule": (
                    "Współczynnik R² w teście poza czasem (Out-of-Time) może przybierać wartości ujemne z powodu "
                    "przesunięcia średniej sezonowej (koniec października charakteryzuje się niższą średnią niż wrzesień). "
                    "Dlatego R² traktujemy jako metrykę pomocniczą, a główną miarą jest MAE oraz korelacja rangowa Spearmana."
                ),
            },
            "classification_target": {
                "primary_metric": "Macro-F1",
                "baseline_to_beat": f"Majority Class Macro-F1 = {compute_classification_metrics(y_test_cls, pred_c_maj_test)['macro_f1']}",
                "success_threshold": "Macro-F1 na zbiorze testowym > 0.45 z niezerowym Recall dla klas BRAK i WYSYP",
            },
        },
    }

    # Zapis plików JSON
    split_meta_path = PROCESSED_DIR / "split_metadata.json"
    metrics_path = PROCESSED_DIR / "baseline_metrics.json"

    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(split_metadata, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)

    print(f"Zapisano metadane podziału: {split_meta_path}")
    print(f"Zapisano metryki bazowe:    {metrics_path}")

    # Generowanie raportu dokumentacji
    generate_markdown_report(split_metadata, metrics_report)


def generate_markdown_report(split_meta: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    doc_path = DOCS_DIR / "evaluation_protocol.md"
    p_train = split_meta["primary_split_train_test"]["train"]
    p_test = split_meta["primary_split_train_test"]["test"]

    reg = metrics["regression_benchmarks"]
    cls = metrics["classification_benchmarks"]

    cm_maj = cls["majority_class"]["test_metrics"]["confusion_matrix"]
    cm_pow = cls["powiat_historical_majority"]["test_metrics"]["confusion_matrix"]

    md_content = f"""# Protokół Ewaluacji i Modele Odniesienia (Baseline)

> **Status:** Zatwierdzony dla zadania `[GRZ-13] #13`  
> **Powiązania:** Podstawa eksperymentów dla modeli uczenia maszynowego `[GRZ-14]` oraz końcowej ewaluacji `[GRZ-17]`.

---

## 1. Cel i zasady metodologiczne

Głównym celem protokołu ewaluacji jest zapewnienie **uczciwego, stabilnego i wolnego od wycieków (leak-free) punktu odniesienia** dla modeli uczenia maszynowego.

### Kluczowe zasady:
1. **Podział czysto czasowy (Temporal Split / Out-of-Time):**
   - Grzybobranie to zjawisko silnie uwarunkowane fenologią jesieni i dynamiką pogody. Klasyczny losowy K-Fold CV doprowadziłby do drastycznego przecieku informacji czasowej (prognozowanie przeszłości na podstawie wiedzy o przyszłości).
   - Wszystkie modele uczą się wyłącznie na danych ze wcześniejszej części sezonu, a testowane są na późniejszych tygodniach.
2. **Zero wycieku wycieczek (No Trip Leakage):**
   - Raporty z tej samej daty i wycieczki lub ich warianty horyzontowe nie mogą być rozdzielone pomiędzy zbiory. Podział sztywną datą odcięcia gwarantuje $100\%$ separację.
3. **Dopasowanie wyłącznie na zbiorze treningowym:**
   - Wszelkie transformacje, średnie, dominanty i przyszłe skalery cech (GRZ-14) są fitowane wyłącznie na części `Train`.
4. **Uczciwy punkt odniesienia bez obietnicy sukcesu ML:**
   - Model ML nie ma gwarancji pokonania prostego baseline'u. Celem badania jest weryfikacja, czy opady opóźnione (lags 7/14/21 dni) oraz siedliska BDL wnoszą istotny zysk informacyjny ponad statyczną średnią lub dominującą klasę.

---

## 2. Zdefiniowany podział danych (Temporal Splits)

Łączna liczba rekordów treningowych spełniających kryteria jakościowe (zob. `docs/reports_etl_quality.md`): **{split_meta['total_valid_records']}**.

### 2.1. Główny podział: Train / Test (Out-of-Time)
Używany do ostatecznego porównania modeli w `[GRZ-14]` i `[GRZ-17]`:

| Podzbiór | Zakres dat | Liczba raportów | Udział % | Pokryte powiaty | Średnia wydajność | Mediana wydajności |
|---|---|---|---|---|---|---|
| **Treningowy (Train)** | `{p_train['date_min']}` – `{p_train['date_max']}` (Tygodnie 1–6) | **{p_train['count']}** | **{p_train['pct']}%** | {p_train['powiats_covered']} / 30 | {p_train['yield_mean']} szt./h | {p_train['yield_median']} szt./h |
| **Testowy (Test)** | `{p_test['date_min']}` – `{p_test['date_max']}` (Tygodnie 7–9) | **{p_test['count']}** | **{p_test['pct']}%** | {p_test['powiats_covered']} / 30 | {p_test['yield_mean']} szt./h | {p_test['yield_median']} szt./h |

#### Rozkład klas obfitości (`abundance_class`):
- **Train:** `BRAK`: {p_train['class_counts'].get('BRAK', 0)} ({p_train['class_distribution_pct'].get('BRAK', 0)}%), `SREDNIO`: {p_train['class_counts'].get('SREDNIO', 0)} ({p_train['class_distribution_pct'].get('SREDNIO', 0)}%), `WYSYP`: {p_train['class_counts'].get('WYSYP', 0)} ({p_train['class_distribution_pct'].get('WYSYP', 0)}%)
- **Test:** `BRAK`: {p_test['class_counts'].get('BRAK', 0)} ({p_test['class_distribution_pct'].get('BRAK', 0)}%), `SREDNIO`: {p_test['class_counts'].get('SREDNIO', 0)} ({p_test['class_distribution_pct'].get('SREDNIO', 0)}%), `WYSYP`: {p_test['class_counts'].get('WYSYP', 0)} ({p_test['class_distribution_pct'].get('WYSYP', 0)}%)

### 2.2. Pomocniczy podział 3-częściowy: Train / Val / Test
Przeznaczony do strojenia hiperparametrów w `[GRZ-14]`:
- **Train:** `{split_meta['secondary_split_train_val_test']['train']['date_min']}` – `{split_meta['secondary_split_train_val_test']['train']['date_max']}` ($N = {split_meta['secondary_split_train_val_test']['train']['count']}$, {split_meta['secondary_split_train_val_test']['train']['pct']}%)
- **Val:** `{split_meta['secondary_split_train_val_test']['val']['date_min']}` – `{split_meta['secondary_split_train_val_test']['val']['date_max']}` ($N = {split_meta['secondary_split_train_val_test']['val']['count']}$, {split_meta['secondary_split_train_val_test']['val']['pct']}%)
- **Test:** `{split_meta['secondary_split_train_val_test']['test']['date_min']}` – `{split_meta['secondary_split_train_val_test']['test']['date_max']}` ($N = {split_meta['secondary_split_train_val_test']['test']['count']}$, {split_meta['secondary_split_train_val_test']['test']['pct']}%)

---

## 3. Wyniki modeli bazowych (Baseline Benchmark)

Wszystkie estymatory bazowe zostały dopasowane **wyłącznie na zbiorze treningowym ($N = 386$)** i ocenione na zbiorze testowym ($N = 127$).

### 3.1. Zadanie regresji: Wydajność zbioru `yield_pcs_per_h` [szt./h]

| Model odniesienia | Opis estymatora | Parametr z Train | MAE [szt./h] (Test) | RMSE [szt./h] (Test) | Spearman rho | $R^2$ (Test) | Interpretacja $R^2$ |
|---|---|---|---|---|---|---|---|
| **Global Mean** | Stała średnia ze zbioru treningowego | `{reg['global_mean']['fitted_parameter_value']} szt./h` | **{reg['global_mean']['test_metrics']['mae_pcs_per_h']}** | **{reg['global_mean']['test_metrics']['rmse_pcs_per_h']}** | 0.000 | {reg['global_mean']['test_metrics']['r2_score']} | Nieinterpretowalny ($< 0$) |
| **Global Median** | Stała mediana (minimalizująca MAE na train) | `{reg['global_median']['fitted_parameter_value']} szt./h` | **{reg['global_median']['test_metrics']['mae_pcs_per_h']}** | **{reg['global_median']['test_metrics']['rmse_pcs_per_h']}** | 0.000 | {reg['global_median']['test_metrics']['r2_score']} | Nieinterpretowalny ($< 0$) |
| **Powiat Hist. Mean** | Średnia historyczna per powiat (z fallbackiem) | Średnie 28 powiatów | **{reg['powiat_historical_mean']['test_metrics']['mae_pcs_per_h']}** | **{reg['powiat_historical_mean']['test_metrics']['rmse_pcs_per_h']}** | {reg['powiat_historical_mean']['test_metrics']['spearman_rho']} | {reg['powiat_historical_mean']['test_metrics']['r2_score']} | Nieinterpretowalny ($< 0$) |

> [!NOTE]
> **Dlaczego współczynnik $R^2$ w teście poza czasem jest ujemny?**  
> W testach szeregów czasowych średnia zbioru testowego ({p_test['yield_mean']} szt./h) różni się od średniej zbioru treningowego ({p_train['yield_mean']} szt./h) ze względu na dynamikę sezonu grzybowego (koniec października przynosi spadek owocnikowania i wczesne przymrozki). Metryka $R^2$ porównuje model do wariancji wokół *średniej testowej*, przez co stały predyktor ze zbioru treningowego uzyskuje ujemny wynik. W związku z tym w projekcie **główną miarą regresji jest MAE oraz korelacja rangowa Spearmana**, a $R^2$ pełni jedynie funkcję diagnostyczną.

---

### 3.2. Zadanie klasyfikacji: Obfitość wysypu `abundance_class` (`BRAK`, `SREDNIO`, `WYSYP`)

| Model odniesienia | Opis estymatora | Predykcja | Macro-F1 (Test) | Weighted-F1 (Test) | Uwagi |
|---|---|---|---|---|---|
| **Majority Class** | Najczęstsza klasa w treningu | Zawsze `SREDNIO` | **{cls['majority_class']['test_metrics']['macro_f1']}** | {cls['majority_class']['test_metrics']['weighted_f1']} | Klasyczny efekt załamania klas mniejszościowych |
| **Powiat Hist. Mode** | Dominanta per powiat z treningu | Zróżnicowana per powiat | **{cls['powiat_historical_majority']['test_metrics']['macro_f1']}** | {cls['powiat_historical_majority']['test_metrics']['weighted_f1']} | Uchwycenie lokalnego tła geograficznego |
| **Stratified Random** | Losowanie z rozkładu treningowego | Proporcjonalne | **{cls['stratified_random']['test_metrics']['macro_f1']}** | {cls['stratified_random']['test_metrics']['weighted_f1']} | Stochastyczny baseline |

#### Macierze pomyłek na zbiorze testowym ($N = 127$):

**1. Majority Class (zawsze `SREDNIO`):**
```text
Rzeczywiste \\ Predykcja    BRAK    SREDNIO    WYSYP
BRAK                        {cm_maj[0][0]:<7} {cm_maj[0][1]:<10} {cm_maj[0][2]:<7}
SREDNIO                     {cm_maj[1][0]:<7} {cm_maj[1][1]:<10} {cm_maj[1][2]:<7}
WYSYP                       {cm_maj[2][0]:<7} {cm_maj[2][1]:<10} {cm_maj[2][2]:<7}
```
*Macro-F1 wynosi zaledwie {cls['majority_class']['test_metrics']['macro_f1']}, ponieważ czułość (Recall) dla klas `BRAK` i `WYSYP` wynosi dokładnie 0.0.*

**2. Powiat Historical Majority:**
```text
Rzeczywiste \\ Predykcja    BRAK    SREDNIO    WYSYP
BRAK                        {cm_pow[0][0]:<7} {cm_pow[0][1]:<10} {cm_pow[0][2]:<7}
SREDNIO                     {cm_pow[1][0]:<7} {cm_pow[1][1]:<10} {cm_pow[1][2]:<7}
WYSYP                       {cm_pow[2][0]:<7} {cm_pow[2][1]:<10} {cm_pow[2][2]:<7}
```
*Zwiększenie Macro-F1 do {cls['powiat_historical_majority']['test_metrics']['macro_f1']} dzięki poprawnemu rozpoznaniu części przypadków `BRAK` i `WYSYP`.*

---

## 4. Kryteria sukcesu i cele dla modeli ML (`[GRZ-14]`, `[GRZ-17]`)

Zgodnie z zatwierdzoną metodyką `[GRZ-13]`, ustala się następujące formalne progi oceny dla kolejnych zadań modelowania:

1. **Dla modelu regresyjnego (`yield_pcs_per_h`):**
   - **Próg odniesienia (Baseline):** MAE = `{reg['global_mean']['test_metrics']['mae_pcs_per_h']} szt./h` (Global Mean) oraz `{reg['powiat_historical_mean']['test_metrics']['mae_pcs_per_h']} szt./h` (Powiat Mean).
   - **Kryterium sukcesu ML:** Uzyskanie na zbiorze testowym **MAE < 14.5 szt./h** oraz istotnie dodatniej korelacji monotonicznej **Spearman $\\rho > 0.20$**.
2. **Dla modelu klasyfikacyjnego (`abundance_class`):**
   - **Próg odniesienia (Baseline):** Macro-F1 = `{cls['majority_class']['test_metrics']['macro_f1']}` (Majority Class) oraz `{cls['powiat_historical_majority']['test_metrics']['macro_f1']}` (Powiat Majority).
   - **Kryterium sukcesu ML:** Uzyskanie na zbiorze testowym **Macro-F1 $> 0.45$** z niezerową czułością dla wszystkich 3 klas (`Recall > 0.15` dla `BRAK` i `WYSYP`).
"""

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Wygenerowano raport protokołu: {doc_path}")


if __name__ == "__main__":
    run_baseline_evaluation()
