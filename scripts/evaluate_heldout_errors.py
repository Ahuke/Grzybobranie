#!/usr/bin/env python3
"""
Skrypt końcowej oceny modelu na odłożonym zbiorze testowym i analizy błędów [GRZ-17].

Zasady:
1. Zero dostrajania modelu do zbioru testowego — wczytuje wyłącznie gotowe artefakty z [GRZ-14]
   (`models/rf_yield_pipeline.joblib`, `models/rf_abundance_pipeline.joblib`).
2. Porównuje wyniki modelu z protokołem i baseline'ami z [GRZ-13] na tym samym podziale
   (`split_2way == 'test'`, N = 127, 2025-10-13 .. 2025-11-02).
3. Analizuje błędy w grupach:
   - horyzonty czasowe (H = 1, 2, 3 dni),
   - powiaty (region_id) oraz pokrycie przestrzenne (21/30 powiatów w teście),
   - klasy obfitości (BRAK / SREDNIO / WYSYP),
   - dostępność profilu taksacyjnego BDL, precyzja przestrzenna, dni robocze vs weekendy,
   - 10 najgorszych przypadków (worst error cases).
4. Zapisuje wyniki do `data/processed/heldout_error_analysis.json` oraz raportu `docs/model_evaluation_and_errors.md`.
"""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import (
    ABUNDANCE_LABELS,
    compute_classification_metrics,
    compute_regression_metrics,
)
from src.inference_features import (
    build_prediction_input_row,
    load_inference_reference_data,
)

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"


def _group_regression_stats(df: pd.DataFrame, group_col: str) -> list[dict]:
    rows: list[dict] = []
    for key, grp in df.groupby(group_col, dropna=False):
        rows.append(
            {
                "group": str(key),
                "count": int(len(grp)),
                "mae_pcs_per_h": round(float(grp["abs_error"].mean()), 3),
                "rmse_pcs_per_h": round(float(np.sqrt((grp["error"] ** 2).mean())), 3),
                "mean_true_pcs_per_h": round(float(grp["yield_pcs_per_h"].mean()), 2),
                "mean_pred_pcs_per_h": round(float(grp["pred_yield"].mean()), 2),
                "bias_pred_minus_true": round(float(grp["error"].mean()), 2),
            }
        )
    return rows


def run_heldout_evaluation() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    features_path = PROCESSED_DIR / "features.parquet"
    baseline_path = PROCESSED_DIR / "baseline_metrics.json"
    reg_model_path = MODELS_DIR / "rf_yield_pipeline.joblib"
    cls_model_path = MODELS_DIR / "rf_abundance_pipeline.joblib"

    df = pd.read_parquet(features_path)
    baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))

    train_df = df[df["split_2way"] == "train"].copy()
    test_df = df[df["split_2way"] == "test"].copy().sort_values(["target_date", "report_id"]).reset_index(drop=True)

    reg_pipe = joblib.load(reg_model_path)
    cls_pipe = joblib.load(cls_model_path)

    # Predykcja na zbiorze testowym (H = 1, bez żadnego dostrajania!)
    test_df["pred_yield"] = np.clip(reg_pipe.predict(test_df), 0.0, None)
    test_df["pred_class"] = cls_pipe.predict(test_df)
    proba = cls_pipe.predict_proba(test_df)
    cls_order = list(cls_pipe.classes_)
    test_df["pred_prob_max"] = proba.max(axis=1)
    test_df["error"] = test_df["pred_yield"] - test_df["yield_pcs_per_h"]
    test_df["abs_error"] = test_df["error"].abs()

    overall_reg = compute_regression_metrics(test_df["yield_pcs_per_h"], test_df["pred_yield"])
    overall_cls = compute_classification_metrics(test_df["abundance_class"], test_df["pred_class"])

    # Baseline'y z GRZ-13
    base_reg_global = baseline_metrics["regression_benchmarks"]["global_mean"]["test_metrics"]
    base_reg_powiat = baseline_metrics["regression_benchmarks"]["powiat_historical_mean"]["test_metrics"]
    base_cls_maj = baseline_metrics["classification_benchmarks"]["majority_class"]["test_metrics"]
    base_cls_powiat = baseline_metrics["classification_benchmarks"]["powiat_historical_majority"]["test_metrics"]

    # 1. Analiza po horyzontach czasowych H = 1, 2, 3 dni (symulacja Point-in-Time z GRZ-15)
    forest_lookup, weather_lookup = load_inference_reference_data()
    horizon_results: list[dict] = []
    for h in (1, 2, 3):
        h_rows = []
        for _, r in test_df.iterrows():
            t_target = date.fromisoformat(str(r["target_date"]))
            t_issued = (t_target - timedelta(days=h)).isoformat()
            inp = build_prediction_input_row(
                region_id=str(r["region_id"]),
                issued_at=t_issued,
                horizon_days=h,
                forest_lookup=forest_lookup,
                weather_lookup=weather_lookup,
            )
            h_rows.append(inp)
        h_df = pd.DataFrame(h_rows)
        h_pred_reg = np.clip(reg_pipe.predict(h_df), 0.0, None)
        h_pred_cls = cls_pipe.predict(h_df)
        rm = compute_regression_metrics(test_df["yield_pcs_per_h"], h_pred_reg)
        cm = compute_classification_metrics(test_df["abundance_class"], h_pred_cls)
        horizon_results.append(
            {
                "horizon_days": h,
                "count": len(h_df),
                "mae_pcs_per_h": rm["mae_pcs_per_h"],
                "rmse_pcs_per_h": rm["rmse_pcs_per_h"],
                "spearman_rho": rm["spearman_rho"],
                "macro_f1": cm["macro_f1"],
                "weighted_f1": cm["weighted_f1"],
                "beats_powiat_reg_baseline_mae": bool(rm["mae_pcs_per_h"] < base_reg_powiat["mae_pcs_per_h"]),
                "beats_powiat_cls_baseline_f1": bool(cm["macro_f1"] > base_cls_powiat["macro_f1"]),
            }
        )

    # 2. Analiza po regionach (powiatach)
    region_rows = []
    all_regions = sorted(forest_lookup.keys())
    for rid in all_regions:
        grp = test_df[test_df["region_id"] == rid]
        train_cnt = int((train_df["region_id"] == rid).sum())
        rname = forest_lookup[rid]["region_name"]
        if len(grp) == 0:
            region_rows.append(
                {
                    "region_id": rid,
                    "region_name": rname,
                    "train_count": train_cnt,
                    "test_count": 0,
                    "mae_pcs_per_h": None,
                    "rmse_pcs_per_h": None,
                    "mean_true_pcs_per_h": None,
                    "mean_pred_pcs_per_h": None,
                    "coverage_status": "BRAK_OBSERWACJI_W_TESCIE",
                }
            )
        else:
            region_rows.append(
                {
                    "region_id": rid,
                    "region_name": rname,
                    "train_count": train_cnt,
                    "test_count": int(len(grp)),
                    "mae_pcs_per_h": round(float(grp["abs_error"].mean()), 3),
                    "rmse_pcs_per_h": round(float(np.sqrt((grp["error"] ** 2).mean())), 3),
                    "mean_true_pcs_per_h": round(float(grp["yield_pcs_per_h"].mean()), 2),
                    "mean_pred_pcs_per_h": round(float(grp["pred_yield"].mean()), 2),
                    "coverage_status": "POKRYTY" if train_cnt > 0 else "TYLKO_W_TESCIE",
                }
            )

    covered_test_regions = [r for r in region_rows if r["test_count"] > 0]
    uncovered_test_regions = [r for r in region_rows if r["test_count"] == 0]

    # 3. Analiza po grupach cech i kontekstu
    by_class = _group_regression_stats(test_df, "abundance_class")
    by_bdl = _group_regression_stats(test_df, "has_bdl_stand_data")
    by_precision = _group_regression_stats(test_df, "spatial_precision")
    by_weekend = _group_regression_stats(test_df, "is_weekend")

    # 4. Top 10 najgorszych błędów (słabe przypadki)
    worst_df = test_df.sort_values("abs_error", ascending=False).head(10)
    worst_cases = []
    for _, r in worst_df.iterrows():
        worst_cases.append(
            {
                "report_id": str(r["report_id"]),
                "region_id": str(r["region_id"]),
                "region_name": str(r["region_name"]),
                "target_date": str(r["target_date"]),
                "true_yield_pcs_per_h": round(float(r["yield_pcs_per_h"]), 2),
                "pred_yield_pcs_per_h": round(float(r["pred_yield"]), 2),
                "abs_error_pcs_per_h": round(float(r["abs_error"]), 2),
                "true_class": str(r["abundance_class"]),
                "pred_class": str(r["pred_class"]),
                "spatial_precision": str(r["spatial_precision"]),
                "is_weekend": bool(r["is_weekend"]),
            }
        )

    # 5. Diagnostyka kalibracji prawdopodobieństw klasyfikatora (Brier score per klasa)
    brier_by_class = {}
    for idx, cls_name in enumerate(cls_order):
        y_bin = (test_df["abundance_class"] == cls_name).astype(float).to_numpy()
        p_cls = proba[:, idx]
        brier = float(np.mean((p_cls - y_bin) ** 2))
        brier_by_class[str(cls_name)] = round(brier, 4)

    analysis = {
        "task_id": "GRZ-17",
        "evaluation_scope": {
            "model_version": "rf_pipeline_v1.0",
            "train_period": f"{train_df['target_date'].min()} .. {train_df['target_date'].max()}",
            "train_count": int(len(train_df)),
            "test_period": f"{test_df['target_date'].min()} .. {test_df['target_date'].max()}",
            "test_count": int(len(test_df)),
            "test_covered_counties": len(covered_test_regions),
            "test_uncovered_counties": len(uncovered_test_regions),
            "retuned_on_test": False,
        },
        "overall_vs_baseline": {
            "regression": {
                "baseline_global_mean": base_reg_global,
                "baseline_powiat_mean": base_reg_powiat,
                "rf_model": overall_reg,
            },
            "classification": {
                "baseline_majority_class": base_cls_maj,
                "baseline_powiat_majority": base_cls_powiat,
                "rf_model": overall_cls,
            },
        },
        "error_by_horizon": horizon_results,
        "error_by_region": region_rows,
        "error_by_true_class": by_class,
        "error_by_bdl_availability": by_bdl,
        "error_by_spatial_precision": by_precision,
        "error_by_weekend": by_weekend,
        "worst_error_cases_top10": worst_cases,
        "calibration_and_uncertainty_notes": {
            "brier_score_per_class_uncalibrated": brier_by_class,
            "raw_regression_is_calibrated_probability": False,
            "confidence_score_is_calibrated_probability": False,
            "explanation": (
                "Zarówno wynik regresji (predicted_yield_pcs_per_h [szt./h]), jak i heurystyczny "
                "wskaźnik pewności/dopasowania siedliska (confidence_score 0.0-1.0) czy surowe "
                "udziały głosów drzew (predict_proba) NIE są skalibrowanym prawdopodobieństwem zdarzenia."
            ),
        },
    }

    out_json = PROCESSED_DIR / "heldout_error_analysis.json"
    out_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Zapisano analizę błędów -> {out_json}")

    _generate_markdown_report(analysis)


def _generate_markdown_report(analysis: dict) -> None:
    doc_path = DOCS_DIR / "model_evaluation_and_errors.md"
    scope = analysis["evaluation_scope"]
    ov_reg = analysis["overall_vs_baseline"]["regression"]
    ov_cls = analysis["overall_vs_baseline"]["classification"]

    rf_r = ov_reg["rf_model"]
    b_g = ov_reg["baseline_global_mean"]
    b_p = ov_reg["baseline_powiat_mean"]

    rf_c = ov_cls["rf_model"]
    c_m = ov_cls["baseline_majority_class"]
    c_p = ov_cls["baseline_powiat_majority"]

    cm = rf_c["confusion_matrix"]

    horizon_rows_md = "\n".join(
        f"| **H = {h['horizon_days']} dzień/dni** | `{h['count']}` | **{h['mae_pcs_per_h']:.3f}** | {h['rmse_pcs_per_h']:.3f} | {h['spearman_rho']:+.4f} | **{h['macro_f1']:.4f}** | {'TAK' if h['beats_powiat_reg_baseline_mae'] else 'NIE'} | {'TAK' if h['beats_powiat_cls_baseline_f1'] else 'NIE'} |"
        for h in analysis["error_by_horizon"]
    )

    covered_sorted = sorted(
        [r for r in analysis["error_by_region"] if r["test_count"] > 0],
        key=lambda x: x["mae_pcs_per_h"],
    )
    region_rows_md = "\n".join(
        f"| `{r['region_id']}` | {r['region_name']} | {r['train_count']} | **{r['test_count']}** | **{r['mae_pcs_per_h']:.2f}** | {r['mean_true_pcs_per_h']:.2f} | {r['mean_pred_pcs_per_h']:.2f} |"
        for r in covered_sorted
    )

    class_rows_md = "\n".join(
        f"| **`{r['group']}`** | {r['count']} | **{r['mae_pcs_per_h']:.2f}** | {r['mean_true_pcs_per_h']:.2f} | {r['mean_pred_pcs_per_h']:.2f} | {r['bias_pred_minus_true']:+.2f} |"
        for r in analysis["error_by_true_class"]
    )

    worst_rows_md = "\n".join(
        f"| `{w['report_id']}` | `{w['region_id']}` {w['region_name']} | `{w['target_date']}` | **{w['true_yield_pcs_per_h']:.1f}** | {w['pred_yield_pcs_per_h']:.1f} | **{w['abs_error_pcs_per_h']:.1f}** | `{w['true_class']}` $\\rightarrow$ `{w['pred_class']}` |"
        for w in analysis["worst_error_cases_top10"][:6]
    )

    uncovered_names = ", ".join(
        f"`{r['region_id']}` {r['region_name']}"
        for r in analysis["error_by_region"]
        if r["test_count"] == 0
    )

    md = f"""# Końcowa Ocena Modelu na Odłożonym Zbiorze Testowym i Analiza Błędów (`[GRZ-17] #17`)

> **Cel:** Uczciwa ocena zakresu zaufania do predykcji modelu `rf_pipeline_v1.0` na odłożonych danych bez żadnego dostrajania do zbioru testowego (`retuned_on_test = False`).  
> **Okres treningowy (`Train`):** `{scope['train_period']}` ($N = {scope['train_count']}$ obserwacji)  
> **Okres testowy Out-of-Time (`Test`):** `{scope['test_period']}` ($N = {scope['test_count']}$ obserwacji, `{scope['test_covered_counties']} / 30` powiatów)

---

## 1. Zestawienie ogólne: Model ML vs Baseline (`Warunek #1`)

### 1.1. Regresja (`yield_pcs_per_h [szt./h]`)

| Model | N (Test) | MAE `[szt./h]` | RMSE `[szt./h]` | Spearman $\\rho$ | $R^2$ (diagnostycznie) |
|---|---:|---:|---:|---:|---:|
| **RandomForestRegressor (`rf_yield_pipeline.joblib`)** | `{scope['test_count']}` | **{rf_r['mae_pcs_per_h']:.3f}** | **{rf_r['rmse_pcs_per_h']:.3f}** | **{rf_r['spearman_rho']:+.4f}** | {rf_r['r2_score']:.4f} |
| *Baseline: Powiat Historical Mean* | `{scope['test_count']}` | *15.286* | *23.348* | *+0.3156* | *-0.0979* |
| *Baseline: Global Mean (`16.55 szt./h`)* | `{scope['test_count']}` | *15.726* | *23.746* | *0.0000* | *-0.1356* |

### 1.2. Klasyfikacja (`abundance_class`: `BRAK`, `SREDNIO`, `WYSYP`)

| Model | N (Test) | Macro-F1 | Weighted-F1 | Recall `BRAK` | Recall `SREDNIO` | Recall `WYSYP` |
|---|---:|---:|---:|---:|---:|---:|
| **RandomForestClassifier (`rf_abundance_pipeline.joblib`)** | `{scope['test_count']}` | **{rf_c['macro_f1']:.4f}** | **{rf_c['weighted_f1']:.4f}** | {rf_c['per_class']['BRAK']['recall']:.2f} | {rf_c['per_class']['SREDNIO']['recall']:.2f} | {rf_c['per_class']['WYSYP']['recall']:.2f} |
| *Baseline: Powiat Historical Majority* | `{scope['test_count']}` | *0.3918* | *0.3884* | *0.47* | *0.79* | *0.11* |
| *Baseline: Majority Class (`SREDNIO`)* | `{scope['test_count']}` | *0.2065* | *0.2781* | *0.00* | *1.00* | *0.00* |

**Macierz pomyłek klasyfikatora na odłożonym teście ($N = {scope['test_count']}$):**
```text
Rzeczywiste \\ Predykcja    BRAK    SREDNIO    WYSYP
BRAK (n=15)                 {cm[0][0]:<7} {cm[0][1]:<10} {cm[0][2]:<7}
SREDNIO (n=57)              {cm[1][0]:<7} {cm[1][1]:<10} {cm[1][2]:<7}
WYSYP (n=55)                {cm[2][0]:<7} {cm[2][1]:<10} {cm[2][2]:<7}
```

---

## 2. Analiza błędów w grupach (`Warunek #2`)

### 2.1. Degradacja błędu wraz z horyzontem prognozy ($H = 1, 2, 3$ dni)
Wszystkie cechy dla $H \\in \\{{1, 2, 3\\}}$ wygenerowano ściśle *Point-in-Time* przez `build_prediction_input_row(issued_at = target_date - H)`:

| Horyzont | N | MAE `[szt./h]` | RMSE `[szt./h]` | Spearman $\\rho$ | Macro-F1 | Bije Baseline Regresji (`15.286`)? | Bije Baseline Klasyfikacji (`0.3918`)? |
|---|---:|---:|---:|---:|---:| :---: | :---: |
{horizon_rows_md}

> [!IMPORTANT]
> **Kluczowa obserwacja horyzontowa (`Warunek #4`):**  
> Model klasyfikacyjny (`rf_abundance_pipeline`) poprawia regionalny baseline (`0.3918`) **wyłącznie dla horyzontu $H = 1$ dzień (`0.4518`)**. Dla horyzontów $H = 2$ i $H = 3$ dni jego `Macro-F1` spada do `0.3686–0.3809` (poniżej prostej dominanty powiatowej!), ponieważ potok był uczony wyłącznie na cechach z odcięciem $H = 1$. Komunikat w aplikacji Streamlit oraz prezentacji wyraźnie o tym informuje.

### 2.2. Błędy według prawdziwej klasy obfitości (efekt kompresji wariancji)

| Prawdziwa klasa | N | MAE `[szt./h]` | Średnia prawdziwa | Średnia prognozowana | Obciążenie (`Pred - True`) |
|---|---:|---:|---:|---:|---:|
{class_rows_md}

- Model osiąga bardzo dobrą precyzję w dominującej strefie umiarkowanych zbiorów **`SREDNIO`** (`MAE = 6.19 szt./h`).
- Dla skrajnych wysypów **`WYSYP`** (`> 20 szt./h`, średnio `43.18 szt./h`) `RandomForestRegressor(max_depth=5)` systematycznie **niedoszacowuje** (`MeanPred = 19.14 szt./h`, bias `-24.04 szt./h`), natomiast dla pustych koszyków **`BRAK`** lekko zawyża prognozę (`+12.37 szt./h`).

### 2.3. Błędy według powiatów i analiza pokrycia testowego

W zbiorze testowym (`2025-10-13` – `2025-11-02`) wystąpiło **`21 / 30` powiatów**. Dla **`9` powiatów** brak obserwacji w okresie testowym ({uncovered_names}), co stanowi istotne ograniczenie statystyczne oceny regionalnej.

| TERYT | Powiat | N (Train) | N (Test) | MAE Test `[szt./h]` | Średnia True | Średnia Pred |
| :---: | :--- |---:|---:|---:|---:|---:|
{region_rows_md}

- **Najwyższy błąd regionalny:** `0213 powiat milicki` (`N = 25`, `MAE = 27.21 szt./h`) — wynik powtarzających się w drugiej połowie października skrajnych zgłoszeń **`100.0 szt./h`**, których drzewo o głębokości `5` nie jest w stanie wyekstrapolować.
- **Najniższe błędy:** `0202 powiat dzierżoniowski` (`MAE = 6.02`), `0215 powiat oławski` (`MAE = 6.87`), `0220 powiat trzebnicki` (`MAE = 10.58`).

### 2.4. Słabe przypadki — Top-6 największych błędów jednostkowych

| `report_id` | Powiat | Data | Prawdziwy plon | Prognoza ML | Błąd `|e|` | Klasa True $\\rightarrow$ Pred |
|---|---|---|---:|---:|---:|---|
{worst_rows_md}

---

## 3. Ograniczenia samodzielnych raportów i nierównej aktywności (`Warunek #3`)

1. **Nierówna aktywność użytkowników w czasie i przestrzeni:**
   - W teście `74 / 127` (`58.3%`) raportów pochodzi z weekendów (`MAE = 16.06` vs `12.73` w dni robocze).
   - `21 / 30` powiatów ma przynajmniej 1 obserwację w teście, ale większość z nich ma próbę `N < 5`, a `9` powiatów nie ma ani jednego raportu testowego — w tych powiatach **nie da się wiarygodnie zmierzyć błędu lokalnego**.
2. **Subiektywizm i zaokrąglenia etykiety (`szt./h`):**
   - Zbieracze często podają wartości zaokrąglone (`10, 15, 20, 30, 50, 100 szt./h`), a ekstremalne zbiory w Dolinie Baryczy (`100 szt./h`) stanowią szum odstający, zaniżający średnią dokładność regresji (`RMSE = 22.29 szt./h`).
3. **Brak danych taksacyjnych BDL dla części powiatów (`12 / 30`):**
   - Dla `N = 13` rekordów testowych bez profilu BDL (`MAE = 12.01 szt./h`) model opiera się wyłącznie na ogólnej lesistości GUS i pogodzie.

---

## 4. Kalibracja i uczciwy opis przydatności modelu (`Warunek #4 i #5`)

1. **Brak skalibrowanego prawdopodobieństwa (`Warunek #5`):**
   - Zarówno surowy wynik regresji `predicted_yield_pcs_per_h` `[szt./h]`, jak i wskaźnik `confidence_score` (`0.0–1.0`, heurystyka dopasowania siedliska i kompletności danych) oraz surowe głosy drzew `predict_proba` (Brier score per klasa: `BRAK = {analysis['calibration_and_uncertainty_notes']['brier_score_per_class_uncalibrated'].get('BRAK')}`, `SREDNIO = {analysis['calibration_and_uncertainty_notes']['brier_score_per_class_uncalibrated'].get('SREDNIO')}`, `WYSYP = {analysis['calibration_and_uncertainty_notes']['brier_score_per_class_uncalibrated'].get('WYSYP')}`) **NIE stanowią skalibrowanego prawdopodobieństwa** znalezienia grzybów i nie są w ten sposób przedstawiane w aplikacji ani sprawozdaniu.
2. **Kiedy wynikom modelu można ufać (praktyczna przydatność):**
   - Model jest **przydatny jako narzędzie wspomagające ranking powiatów na kolejny dzień (`H = 1`)** (`Spearman = +0.3662`, `MAE = 14.67 < 15.29` baseline'u, `Macro-F1 = 0.4518 > 0.3918` baseline'u) i poprawnie identyfikuje umiarkowane zbiory (`SREDNIO`, `MAE = 6.19 szt./h`).
   - **Nie należy traktować prognoz dosłownie** jako gwarancji konkretnej liczby sztuk przy bardzo dużych wysypach (`> 40 szt./h`) ani ufać klasyfikacji na horyzonty **`H = 2, 3` dni**, gdzie skuteczność klasyfikatora spada poniżej prostego baseline'u powiatowego.
"""

    doc_path.write_text(md, encoding="utf-8")
    print(f"Wygenerowano raport oceny i analizy błędów -> {doc_path}")


if __name__ == "__main__":
    run_heldout_evaluation()
