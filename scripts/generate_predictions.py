#!/usr/bin/env python3
"""
Skrypt generujący produkcyjny plik predykcji `data/processed/predictions.csv`
zgodny z kontraktem danych (`docs/data_contract.md` Sekcja 2.4) oraz weryfikujący
zgodność wartości wyświetlanych na mapie z bezpośrednim wywołaniem modelu ([GRZ-16]).

Uruchomienie:
    python scripts/generate_predictions.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DOCS_DIR, PROCESSED_DIR
from src.data_loader import load_areas_geojson, load_predictions_csv
from src.geo_utils import enrich_geojson_with_predictions
from src.prediction_service import (
    PRECOMPUTED_END_DATE,
    PRECOMPUTED_START_DATE,
    generate_precomputed_predictions_csv,
    generate_predictions_for_date,
    load_precomputed_predictions_df,
    verify_map_predictions_against_direct_model,
)


def run_generation_and_verification(
    start_date: str = PRECOMPUTED_START_DATE,
    end_date: str = PRECOMPUTED_END_DATE,
) -> dict[str, Any]:
    out_csv = PROCESSED_DIR / "predictions.csv"
    df_preds = generate_precomputed_predictions_csv(
        start_target_date=start_date,
        end_target_date=end_date,
        output_csv_path=out_csv,
    )

    # Weryfikacja wczytywania przez produkcyjny loader kontraktu danych (src/data_loader.py)
    loaded_rows = load_predictions_csv(out_csv)
    assert len(loaded_rows) == len(df_preds), "Niezgodność liczby wierszy po wczytaniu przez load_predictions_csv!"
    assert all(not r["is_demo"] for r in loaded_rows), "Wykryto rekord is_demo=True w predictions.csv!"

    # Załaduj granice powiatów i prekomputowany DataFrame do weryfikacji mapy
    areas_geojson, areas_by_id = load_areas_geojson(PROCESSED_DIR / "areas.geojson")
    precomputed_df = load_precomputed_predictions_df(out_csv)

    sample_areas = ("0201", "0206", "0208", "0213", "0218", "0217")

    # 1. Weryfikacja zgodności mapy z bezpośrednim wywołaniem modelu (2025-10-19, all, H=1 z prekomputowanego CSV)
    res_all = generate_predictions_for_date(
        target_date="2025-10-19",
        horizon_days=1,
        species_filter="all",
        prediction_engine="ML_MODEL",
        areas_by_id=areas_by_id,
        precomputed_df=precomputed_df,
    )
    geo_all = enrich_geojson_with_predictions(
        areas_geojson,
        res_all["records_by_area"],
        color_mode="yield_scale",
    )
    ver_all = verify_map_predictions_against_direct_model(
        enriched_geojson=geo_all,
        target_date="2025-10-19",
        horizon_days=1,
        species_filter="all",
        sample_area_ids=sample_areas,
    )
    assert ver_all["all_match"], f"Rozbieżność mapy z modelem (all): {ver_all}"

    # 2. Weryfikacja zgodności mapy dla filtra gatunkowego (2025-10-25, borowik_szlachetny, H=2 na żywo z potoku)
    res_borowik = generate_predictions_for_date(
        target_date="2025-10-25",
        horizon_days=2,
        species_filter="borowik_szlachetny",
        prediction_engine="ML_MODEL",
        areas_by_id=areas_by_id,
        precomputed_df=precomputed_df,
    )
    geo_borowik = enrich_geojson_with_predictions(
        areas_geojson,
        res_borowik["records_by_area"],
        color_mode="yield_scale",
    )
    ver_borowik = verify_map_predictions_against_direct_model(
        enriched_geojson=geo_borowik,
        target_date="2025-10-25",
        horizon_days=2,
        species_filter="borowik_szlachetny",
        sample_area_ids=sample_areas,
    )
    assert ver_borowik["all_match"], f"Rozbieżność mapy z modelem (borowik_szlachetny): {ver_borowik}"

    # 3. Weryfikacja daty brzegowej z niepełną historią (2025-08-11, issued_at=2025-08-10 -> wszystkie 30 powiatów NIEDOSTĘPNE)
    res_unavail = generate_predictions_for_date(
        target_date="2025-08-11",
        horizon_days=1,
        species_filter="all",
        prediction_engine="ML_MODEL",
        areas_by_id=areas_by_id,
        precomputed_df=precomputed_df,
    )
    geo_unavail = enrich_geojson_with_predictions(
        areas_geojson,
        res_unavail["records_by_area"],
        color_mode="yield_scale",
    )
    ver_unavail = verify_map_predictions_against_direct_model(
        enriched_geojson=geo_unavail,
        target_date="2025-08-11",
        horizon_days=1,
        species_filter="all",
        sample_area_ids=("0201", "0208", "0218"),
    )
    assert res_unavail["available_count"] == 0 and res_unavail["unavailable_count"] == 30
    assert ver_unavail["all_match"]

    # 4. Porównanie źródła ML vs modelu regułowego (brak mylenia źródeł)
    res_rule = generate_predictions_for_date(
        target_date="2025-10-19",
        horizon_days=1,
        species_filter="all",
        prediction_engine="RULE_BASED_INDEX",
        areas_by_id=areas_by_id,
    )
    assert res_rule["prediction_engine"] == "RULE_BASED_INDEX"
    assert res_all["prediction_engine"] == "ML_MODEL"
    assert res_rule["model_version"] != res_all["model_version"]

    report: dict[str, Any] = {
        "task": "[GRZ-16] Połączyć model z mapą prognoz",
        "predictions_csv_path": "data/processed/predictions.csv",
        "predictions_summary": {
            "total_rows": int(len(df_preds)),
            "date_min": str(df_preds["target_date"].min()),
            "date_max": str(df_preds["target_date"].max()),
            "unique_dates_count": int(df_preds["target_date"].nunique()),
            "unique_areas_count": int(df_preds["area_id"].nunique()),
            "species_filters": sorted(df_preds["species_filter"].unique().tolist()),
            "model_version": str(df_preds["model_version"].iloc[0]),
            "target_unit": "szt./h",
            "is_demo_any": bool(df_preds["is_demo"].any()),
            "yield_min_pcs_per_h": round(float(df_preds["predicted_yield_pcs_per_h"].min()), 2),
            "yield_mean_pcs_per_h": round(float(df_preds["predicted_yield_pcs_per_h"].mean()), 2),
            "yield_max_pcs_per_h": round(float(df_preds["predicted_yield_pcs_per_h"].max()), 2),
            "abundance_class_counts": {
                str(k): int(v) for k, v in df_preds["abundance_class"].value_counts().items()
            },
        },
        "verification_map_vs_direct_model_precomputed_2025_10_19_all": ver_all,
        "verification_map_vs_direct_model_live_h2_2025_10_25_borowik": ver_borowik,
        "verification_insufficient_history_2025_08_11": {
            "target_date": "2025-08-11",
            "issued_at": "2025-08-10",
            "available_count": res_unavail["available_count"],
            "unavailable_count": res_unavail["unavailable_count"],
            "verification": ver_unavail,
        },
        "source_separation_check": {
            "ml_model_version": res_all["model_version"],
            "rule_index_version": res_rule["model_version"],
            "ml_0201_yield_pcs_per_h": res_all["records_by_area"]["0201"]["predicted_yield_pcs_per_h"],
            "rule_0201_yield_pcs_per_h": res_rule["records_by_area"]["0201"]["predicted_yield_pcs_per_h"],
            "sources_strictly_separated": True,
        },
    }

    report_path = PROCESSED_DIR / "predictions_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_integration_doc(report)

    print("=" * 82)
    print("[GRZ-16] Wygenerowano data/processed/predictions.csv i zweryfikowano mapę prognoz")
    print("=" * 82)
    print(
        f"Plik: {out_csv.relative_to(ROOT_DIR)} | Wiersze: {len(df_preds)} | "
        f"Daty: {start_date} .. {end_date} | Powiaty: {df_preds['area_id'].nunique()} | "
        f"Gatunki: {df_preds['species_filter'].nunique()}"
    )
    print(f"Zgodność mapy z modelem (2025-10-19, all, H=1):              {ver_all['all_match']} ({ver_all['checked_counties_count']}/6 powiatów)")
    print(f"Zgodność mapy z modelem (2025-10-25, borowik, H=2, live):    {ver_borowik['all_match']} ({ver_borowik['checked_counties_count']}/6 powiatów)")
    print(f"Obsługa braku historii (2025-08-11, PARTIAL_HISTORY):        {res_unavail['unavailable_count']}/30 powiatów NIEDOSTĘPNYCH")
    print(f"Raport weryfikacji: {report_path.relative_to(ROOT_DIR)}")
    print("=" * 82)

    return report


def _write_integration_doc(report: dict[str, Any]) -> None:
    doc_path = DOCS_DIR / "map_predictions_integration.md"
    ps = report["predictions_summary"]
    v1 = report["verification_map_vs_direct_model_precomputed_2025_10_19_all"]
    v2 = report["verification_map_vs_direct_model_live_h2_2025_10_25_borowik"]

    rows_v1 = "\n".join(
        f"| `{c['area_id']}` | {c['area_name']} | `{c['history_completeness']}` | "
        f"**{c['map_yield_pcs_per_h']:.2f} szt./h** | **{c['direct_model_yield_pcs_per_h']:.2f} szt./h** | "
        f"`{c['abs_diff_pcs_per_h']}` | `{c['map_abundance_class']}` | `{'TAK' if c['matches_1_to_1'] else 'NIE'}` |"
        for c in v1["checks"]
    )
    rows_v2 = "\n".join(
        f"| `{c['area_id']}` | {c['area_name']} | `{c['history_completeness']}` | "
        f"**{c['map_yield_pcs_per_h']:.2f} szt./h** | **{c['direct_model_yield_pcs_per_h']:.2f} szt./h** | "
        f"`{c['abs_diff_pcs_per_h']}` | `{c['map_abundance_class']}` | `{'TAK' if c['matches_1_to_1'] else 'NIE'}` |"
        for c in v2["checks"]
    )

    md = f"""# Integracja Modelu ML z Mapą Prognoz w Streamlit (`[GRZ-16] #16`)

> **Status:** Ukończony i zweryfikowany (`{ps['model_version']}`)  
> **Powiązania:** Potok ML [`docs/model_training.md`](model_training.md) (`[GRZ-14]`), kontrakt wejścia Point-in-Time [`docs/prediction_input_contract.md`](prediction_input_contract.md) (`[GRZ-15]`), kontrakt danych [`docs/data_contract.md`](data_contract.md).

---

## 1. Architektura integracji (Pipeline $\\rightarrow$ `predictions.csv` / Live $\\rightarrow$ GeoJSON $\\rightarrow$ PyDeck)

1. **Generator i serwis predykcji ([`src/prediction_service.py`](../src/prediction_service.py)):**
   - Wczytuje zapisane potoki `models/rf_yield_pipeline.joblib` oraz `models/rf_abundance_pipeline.joblib` (`rf_pipeline_v1.0`).
   - Dla wybranej przez użytkownika daty docelowej `target_date`, horyzontu `horizon_days` (`1–3` dni), gatunku oraz obszaru wyznacza `issued_at = feature_cutoff_date = target_date - horizon_days` i buduje cechy przez `build_prediction_inputs_batch` z [`src/inference_features.py`](../src/inference_features.py) (`[GRZ-15]`).
   - Dla zakresu prekomputowanego (`{ps['date_min']} – {ps['date_max']}`, `H = 1`) pobiera gotowy wynik z **`data/processed/predictions.csv`** (`{ps['total_rows']}` wierszy, `is_demo = False`), a dla pozostałych dat archiwum lub horyzontów `H = 2, 3` (albo przy włączonym wymuszeniu trybu na żywo) uruchamia zapisany potok ML bezpośrednio.
2. **Wygenerowany plik predykcji zgodny z kontraktem danych (`data/processed/predictions.csv`):**
   - Odtwarzalny poleceniem:
     ```bash
     python scripts/generate_predictions.py
     ```
   - Zakres dat docelowych: **`{ps['date_min']}` – `{ps['date_max']}`** (`{ps['unique_dates_count']}` dni, `{ps['unique_areas_count']}` powiatów, `{len(ps['species_filters'])}` filtrów gatunkowych $\\rightarrow$ **`{ps['total_rows']}` rekordów**).
   - Zakres prognozowanych wartości: **`{ps['yield_min_pcs_per_h']} – {ps['yield_max_pcs_per_h']} szt./h`** (średnia: `{ps['yield_mean_pcs_per_h']} szt./h`).
3. **Skala kolorów zgodna z jednostką wyniku (`szt./h`):**
   - W [`src/config.py`](../src/config.py) (`YIELD_COLOR_SCALE_BINS`) i [`src/geo_utils.py`](../src/geo_utils.py) (`yield_pcs_per_h_to_rgba`) zdefiniowano 5-stopniową skalę kartograficzną w jednostce **`szt./h`**, w 100% spójną z oficjalnymi progami `5.0` i `20.0 szt./h`:
     - **`> 30 szt./h`** (silny `WYSYP`), **`20 – 30 szt./h`** (`WYSYP`), **`12 – 20 szt./h`** (`ŚREDNIO+`), **`5 – 12 szt./h`** (`ŚREDNIO`), **`< 5 szt./h`** (`BRAK`) oraz neutralną szarość dla obszarów **`NIEDOSTĘPNYCH`** (`NULL`).
4. **Obszary bez wystarczających danych (`Warunek #3`):**
   - Powiaty o statusie `PARTIAL_HISTORY`, `NO_WEATHER_HISTORY` lub `OUT_OF_COVERAGE` nie otrzymują sztucznej predykcji — ich wartość `predicted_yield_pcs_per_h` wynosi `None` (`NULL`), klasa przyjmuje wartość `NIEDOSTEPNY`, a poligon na mapie kolorowany jest neutralną szarością z jawną informacją w legendzie, tooltipie i tabeli (np. dla daty `2025-08-11`, gdy brakuje 11 z 21 dni okna pogodowego).
5. **Ścisłe rozdzielenie źródeł wyników (`Warunek #4`):**
   - **`[WYNIK ML]` (`rf_pipeline_v1.0`, `is_demo = False`)** — domyślny widok mapy prognoz w aplikacji Streamlit.
   - **`[WYNIK REGUŁOWY]` (`RULE_SUITABILITY_INDEX_v1.0`, `is_demo = False`)** — Jawny Wskaźnik Warunków Grzybobrania (Tor A), przełączany jawnie i prezentowany w osobnej kolumnie porównawczej, nigdy nie mylony z modelem ML.
   - **`[OBSERWACJE TERENOWE]` (`grzyby.pl`, `[GRZ-12]`)** — rzeczywiste historyczne raporty grzybiarzy.
   - **`[DANE DEMONSTRACYJNE (MOCK)]` (`DEMO_MOCK_v0`, `is_demo = True`)** — syntetyczna próbka z `data/demo/`.

---

## 2. Weryfikacja zgodności wartości na mapie z bezpośrednim wywołaniem modelu (`Warunek #5`)

### 2.1. Scenariusz 1: `target_date = 2025-10-19`, `horizon = 1 dzień`, `species_filter = all` (z `predictions.csv` $\\rightarrow$ GeoJSON mapy)

| TERYT | Powiat | Status historii (`[GRZ-15]`) | Wartość na mapie | Bezpośredni model `.joblib` | Różnica | Klasa | Zgodność 1:1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{rows_v1}

### 2.2. Scenariusz 2: `target_date = 2025-10-25`, `horizon = 2 dni`, `species_filter = borowik_szlachetny` (na żywo z potoku $\\rightarrow$ GeoJSON mapy)

| TERYT | Powiat | Status historii (`[GRZ-15]`) | Wartość na mapie | Bezpośredni model `.joblib` | Różnica | Klasa | Zgodność 1:1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{rows_v2}

*Pełny zapis maszynowy audytu znajduje się w [`data/processed/predictions_verification_report.json`](../data/processed/predictions_verification_report.json).*
"""
    doc_path.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generuje data/processed/predictions.csv i weryfikuje integrację modelu z mapą [GRZ-16]."
    )
    parser.add_argument("--start-date", default=PRECOMPUTED_START_DATE, help="Początkowa target_date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=PRECOMPUTED_END_DATE, help="Końcowa target_date (YYYY-MM-DD)")
    args = parser.parse_args()
    run_generation_and_verification(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
