# Artefakty Modeli ML (`models/`)

Katalog zawiera wytrenowane potoki predykcyjne (`Pipeline` scikit-learn / Random Forest), pliki metadanych JSON oraz przykłady danych wejściowych i wyjściowych (`[GRZ-14] #14`):

## 1. Zawartość katalogu
- `rf_yield_pipeline.joblib` — kompletny potok regresyjny (`DomainFeatureProcessor` + `ColumnTransformer` + `RandomForestRegressor`) przewidujący wydajność grzybobrania `yield_pcs_per_h` w jednostce **`szt./h`** (*wersjonowany w Git dla odtwarzalności poza treningiem i integracji w `[GRZ-16]`*).
- `rf_abundance_pipeline.joblib` — kompletny potok klasyfikacyjny (`RandomForestClassifier`) przewidujący 3-klasową obfitość wysypu `abundance_class` (`BRAK`, `SREDNIO`, `WYSYP`) (*wersjonowany w Git*).
- `model_metadata.json` — pełne metadane modelu: wersja danych i skrót SHA256, daty zbioru treningowego (`2025-09-01 .. 2025-10-12`) i testowego (`2025-10-13 .. 2025-11-02`), lista cech, zmienna docelowa, jednostka wyniku (`szt./h`) oraz zestawienie wyników z modelami bazowymi (`Baseline`) i `GradientBoosting` (*wersjonowany w Git*).
- `feature_list.json` — specyfikacja wszystkich 39 surowych cech wejściowych, cech inżynieryjnych, słowników kategorycznych, median imputacji ze zbioru treningowego oraz ranking ważności cech (`Feature Importances`) (*wersjonowany w Git*).
- `example_input.json` / `example_input.csv` — 5 reprezentatywnych rekordów wejściowych (w tym powiat z brakami danych taksacyjnych BDL `NULL` oraz powiat `0217` bez historii w zbiorze treningowym) (*wersjonowane w Git*).
- `example_output.json` / `example_output.csv` — wynik predykcji dla przykładów wejściowych wygenerowany bez uruchamiania treningu (*wersjonowane w Git*).

## 2. Polecenia uruchomienia (`[GRZ-14]`)
1. **Odtworzenie treningu i zapis artefaktów `.joblib` oraz metadanych:**
   ```bash
   python scripts/train_model.py
   ```
2. **Wykonanie przykładu predykcji bez uruchamiania całego treningu:**
   ```bash
   python scripts/predict_example.py
   ```

## 3. Zasada bezpieczeństwa
Żaden model nie może być trenowany na danych z katalogu `data/demo/` (`is_demo == True`). Transformator `DomainFeatureProcessor` weryfikuje ten warunek i zgłasza wyjątek `ValueError` w przypadku wykrycia rekordu demonstracyjnego.
