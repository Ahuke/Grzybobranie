# Procedura odświeżania danych i wyników przed demonstracją (`[GRZ-20] #20`)

> **Cel:** Zapewnić spójność daty prezentowanej prognozy z aktualnością danych wejściowych oraz zagwarantować, że awaria pobierania nigdy nie zniszczy poprawnych plików na dysku.

---

## 1. Jedno polecenie odświeżające przed demonstracją (`Warunek #1`)

### Wariant standardowy (szybkie odtworzenie z lokalnego cache — w pełni offline):
```bash
python scripts/refresh_demo_data.py
```

### Wariant z próbą aktualizacji pogody z Open-Meteo:
```bash
python scripts/refresh_demo_data.py --fetch-weather
```

Polecenie wykonuje kolejno w sposób **atomowy**:
1. *(Opcjonalnie `--fetch-weather`)* aktualizuje `data/processed/weather_daily.parquet` z Open-Meteo.
2. Przelicza `data/processed/features.parquet` i `features.csv` (`513` wierszy, `70` kolumn).
3. Generuje `data/processed/predictions.csv` (`4 500` wierszy) z jawnymi polami:
   - `generated_at` (data wystawienia prognozy `issued_at`),
   - `target_date` (data docelowa),
   - `model_version` (`rf_pipeline_v1.0`).
4. Weryfikuje wczytanie przez produkcyjny loader `load_predictions_csv` oraz zapisuje raport stanu w [`data/processed/refresh_manifest.json`](../data/processed/refresh_manifest.json).

---

## 2. Ochrona przed awarią i oznaczanie starych wyników (`Warunek #3 i #4`)

1. **Atomowy zapis (`atomic_write_via_temp` w `src/refresh_pipeline.py`):**
   - Każdy plik wynikowy (`weather_daily.parquet`, `features.parquet`, `predictions.csv`) jest najpierw zapisywany do pliku tymczasowego `.tmp_*`, następnie walidowany (niepusty, minimalna liczba wierszy, brak `is_demo == True`), a dopiero po sukcesie podmieniany atomowo przez `os.replace`.
   - Jeśli pobieranie pogody lub walidacja zawiedzie (błąd sieci, timeout, pusta odpowiedź API), skrypt zwraca status `FAILED_PRESERVED_EXISTING` i **nie nadpisuje** poprawnych plików z dysku.
2. **Oznaczanie starych wyników (`STALE_ARCHIVE` vs `CURRENT`):**
   - Funkcja `assess_predictions_freshness` porównuje `max(target_date)` z datą bieżącą (`reference_date`).
   - Gdy `reference_date > max(target_date)` (np. demonstracja odbywa się po zakończeniu sezonu archiwalnego `2025-11-06`), manifest oznacza wyniki jako **`is_stale = true` / `STALE_ARCHIVE`** wraz z jawnym komunikatem ostrzegawczym — stare wyniki nigdy nie są przedstawiane jako prognozy bieżące.

---

## 3. Ręczna procedura aktualizacji raportów i modelu (poza automatycznym odświeżaniem)

W MVP raporty `grzyby.pl` oraz wagi modelu ML nie są aktualizowane automatycznie przed każdym demo. W przypadku dodania nowych tygodni zrzutów HTML należy ręcznie wykonać sekwencję:
1. `python scripts/build_reports_dataset.py` (aktualizacja `reports.parquet` / `reports.csv`)
2. `python scripts/build_spatial_features.py` (aktualizacja `reports_spatial.parquet`)
3. `python scripts/refresh_demo_data.py --fetch-weather` (aktualizacja pogody i `features.parquet`)
4. `python scripts/train_model.py` (przetrenowanie `models/*.joblib` przy zachowaniu podziału czasowego)
5. `python scripts/refresh_demo_data.py` (regeneracja `predictions.csv` z nową wersją modelu)

---

## 4. Checklista weryfikacji w aplikacji przed prezentacją (`Warunek #5`)
- [ ] Uruchomiono `python scripts/refresh_demo_data.py` (lub `--fetch-weather`) i potwierdzono `REBUILT_ATOMICALLY` w `data/processed/refresh_manifest.json`.
- [ ] Uruchomiono `streamlit run app.py` i zweryfikowano, że mapa prognoz wczytuje odświeżony plik `predictions.csv` bez błędów.
- [ ] Sprawdzono, że w interfejsie widoczna jest data wystawienia, data docelowa, wersja modelu (`rf_pipeline_v1.0`) oraz status aktualności danych.
