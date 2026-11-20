# Integracja Modelu ML z Mapą Prognoz w Streamlit (`[GRZ-16] #16`)

> **Status:** Ukończony i zweryfikowany (`rf_pipeline_v1.0`)  
> **Powiązania:** Potok ML [`docs/model_training.md`](model_training.md) (`[GRZ-14]`), kontrakt wejścia Point-in-Time [`docs/prediction_input_contract.md`](prediction_input_contract.md) (`[GRZ-15]`), kontrakt danych [`docs/data_contract.md`](data_contract.md).

---

## 1. Architektura integracji (Pipeline $\rightarrow$ `predictions.csv` / Live $\rightarrow$ GeoJSON $\rightarrow$ PyDeck)

1. **Generator i serwis predykcji ([`src/prediction_service.py`](../src/prediction_service.py)):**
   - Wczytuje zapisane potoki `models/rf_yield_pipeline.joblib` oraz `models/rf_abundance_pipeline.joblib` (`rf_pipeline_v1.0`).
   - Dla wybranej przez użytkownika daty docelowej `target_date`, horyzontu `horizon_days` (`1–3` dni), gatunku oraz obszaru wyznacza `issued_at = feature_cutoff_date = target_date - horizon_days` i buduje cechy przez `build_prediction_inputs_batch` z [`src/inference_features.py`](../src/inference_features.py) (`[GRZ-15]`).
   - Dla zakresu prekomputowanego (`2025-10-13 – 2025-11-06`, `H = 1`) pobiera gotowy wynik z **`data/processed/predictions.csv`** (`4500` wierszy, `is_demo = False`), a dla pozostałych dat archiwum lub horyzontów `H = 2, 3` (albo przy włączonym wymuszeniu trybu na żywo) uruchamia zapisany potok ML bezpośrednio.
2. **Wygenerowany plik predykcji zgodny z kontraktem danych (`data/processed/predictions.csv`):**
   - Odtwarzalny poleceniem:
     ```bash
     python scripts/generate_predictions.py
     ```
   - Zakres dat docelowych: **`2025-10-13` – `2025-11-06`** (`25` dni, `30` powiatów, `6` filtrów gatunkowych $\rightarrow$ **`4500` rekordów**).
   - Zakres prognozowanych wartości: **`4.0 – 44.74 szt./h`** (średnia: `18.19 szt./h`).
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

### 2.1. Scenariusz 1: `target_date = 2025-10-19`, `horizon = 1 dzień`, `species_filter = all` (z `predictions.csv` $\rightarrow$ GeoJSON mapy)

| TERYT | Powiat | Status historii (`[GRZ-15]`) | Wartość na mapie | Bezpośredni model `.joblib` | Różnica | Klasa | Zgodność 1:1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0201` | powiat bolesławiecki | `COMPLETE` | **17.30 szt./h** | **17.30 szt./h** | `0.0` | `SREDNIO` | `TAK` |
| `0206` | powiat karkonoski | `COMPLETE` | **22.96 szt./h** | **22.96 szt./h** | `0.0` | `WYSYP` | `TAK` |
| `0208` | powiat kłodzki | `COMPLETE` | **9.48 szt./h** | **9.48 szt./h** | `0.0` | `SREDNIO` | `TAK` |
| `0213` | powiat milicki | `COMPLETE` | **20.03 szt./h** | **20.03 szt./h** | `0.0` | `WYSYP` | `TAK` |
| `0218` | powiat średzki | `COMPLETE_WEATHER_NO_BDL_STAND` | **12.76 szt./h** | **12.76 szt./h** | `0.0` | `SREDNIO` | `TAK` |
| `0217` | powiat strzeliński | `COMPLETE_WEATHER_NO_BDL_STAND` | **19.63 szt./h** | **19.63 szt./h** | `0.0` | `SREDNIO` | `TAK` |

### 2.2. Scenariusz 2: `target_date = 2025-10-25`, `horizon = 2 dni`, `species_filter = borowik_szlachetny` (na żywo z potoku $\rightarrow$ GeoJSON mapy)

| TERYT | Powiat | Status historii (`[GRZ-15]`) | Wartość na mapie | Bezpośredni model `.joblib` | Różnica | Klasa | Zgodność 1:1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0201` | powiat bolesławiecki | `COMPLETE` | **16.65 szt./h** | **16.65 szt./h** | `0.0` | `SREDNIO` | `TAK` |
| `0206` | powiat karkonoski | `COMPLETE` | **35.11 szt./h** | **35.11 szt./h** | `0.0` | `WYSYP` | `TAK` |
| `0208` | powiat kłodzki | `COMPLETE` | **27.54 szt./h** | **27.54 szt./h** | `0.0` | `WYSYP` | `TAK` |
| `0213` | powiat milicki | `COMPLETE` | **17.29 szt./h** | **17.29 szt./h** | `0.0` | `SREDNIO` | `TAK` |
| `0218` | powiat średzki | `COMPLETE_WEATHER_NO_BDL_STAND` | **12.75 szt./h** | **12.75 szt./h** | `0.0` | `SREDNIO` | `TAK` |
| `0217` | powiat strzeliński | `COMPLETE_WEATHER_NO_BDL_STAND` | **18.56 szt./h** | **18.56 szt./h** | `0.0` | `SREDNIO` | `TAK` |

*Pełny zapis maszynowy audytu znajduje się w [`data/processed/predictions_verification_report.json`](../data/processed/predictions_verification_report.json).*
