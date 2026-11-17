# Pierwszy Model ML (Random Forest) i Kompletny Potok Predykcyjny (`[GRZ-14] #14`)

> **Status:** Ukończony i zweryfikowany (`rf_pipeline_v1.0`)  
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
   - **Wariant pomocniczy (Klasyfikacja 3-klasowa `abundance_class`):** `RandomForestClassifier(n_estimators=100, class_weight={"BRAK": 1.6, "SREDNIO": 0.7, "WYSYP": 2.1}, max_depth=4, min_samples_leaf=4, max_features=0.6, random_state=42)` — wagi klas równoważą dominację klasy środkowej `SREDNIO` w zbiorze treningowym.

---

## 3. Metadane modelu, wersja danych, cechy i jednostka wyniku (Warunek `#4`)

Pełne metadane zapisane są w plikach [`models/model_metadata.json`](../models/model_metadata.json) oraz [`models/feature_list.json`](../models/feature_list.json):

| Pole metadanych | Wartość |
| :--- | :--- |
| **Wersja modelu (`model_version`)** | `rf_pipeline_v1.0` |
| **Wersja danych (`data_version.dataset_name`)** | `grzybobranie_features_point_in_time_v1.0` (`data/processed/features.parquet`, $N = 513$) |
| **Skrót kryptograficzny danych (`SHA256`)** | `78e83a3079191f8057042dfb...` |
| **Okres zbioru treningowego (`Train`)** | `2025-09-01` – `2025-10-12` ($N = 386$ obserwacji, 27 powiatów) |
| **Okres zbioru testowego (`Test Out-of-Time`)** | `2025-10-13` – `2025-11-02` ($N = 127$ obserwacji, 21 powiatów) |
| **Horyzont i odcięcie informacji (`Point-in-Time`)** | `horizon_days = 1` (`issued_at = feature_cutoff_date = target_date - 1 dzień`) |
| **Główny cel predykcji (`target`)** | `yield_pcs_per_h` $\rightarrow$ `predicted_yield_pcs_per_h` |
| **Jednostka wyniku (`target_unit`)** | **`szt./h`** (sztuki grzybów na 1 osobę na 1 godzinę grzybobrania) |
| **Cel pomocniczy (klasy obfitości)** | `abundance_class` $\in$ `{BRAK (<5.0 szt./h), SREDNIO (5.0–20.0 szt./h), WYSYP (>20.0 szt./h)}` |
| **Liczba cech wejściowych / zakodowanych** | **40** surowych cech wejściowych (`3` kategoryczne + `36` liczbowych) $\rightarrow$ **35** cech po transformacji i dodaniu flag braków |

### Najważniejsze cechy w modelu regresyjnym (`Top-10 Feature Importances`):

| Pozycja | Zakodowana cecha w potoku | Ważność (`Gini / MSE Importance`) |
| :---: | :--- | :---: |
| 1 | `num__wind_speed_mean_7d_kmh` | **0.1719** |
| 2 | `num__county_hist_mean_smoothed` | **0.1240** |
| 3 | `num__county_hist_wysyp_rate` | **0.0831** |
| 4 | `num__county_hist_median_smoothed` | **0.0784** |
| 5 | `num__precip_slice_15_21d_mm` | **0.0716** |
| 6 | `num__precip_lag_14d_mm` | **0.0694** |
| 7 | `num__temp_min_lag_3d_c` | **0.0570** |
| 8 | `num__temp_min_night_c` | **0.0522** |
| 9 | `num__soil_temperature_0_7cm_c` | **0.0481** |
| 10 | `num__precip_cutoff_1d_mm` | **0.0475** |

---

## 4. Porównanie z modelami bazowymi (Baseline) i Gradient Boosting na tym samym podziale (Warunek `#2`)

Wszystkie modele oceniono na **identycznym podziale czasowym `primary_split_train_test`** z zadania `[GRZ-13]` (`Train`: `2025-09-01` .. `2025-10-12`, $N = 386$; `Test`: `2025-10-13` .. `2025-11-02`, $N = 127$; `0` wspólnych raportów).

### 4.1. Zadanie regresji (`yield_pcs_per_h` w jednostce `[szt./h]`)

| Model / Estymator | MAE Test `[szt./h]` | RMSE Test `[szt./h]` | Spearman $\rho$ (Test) | Pearson $r$ (Test) | $R^2$ (Test) | Zmiana MAE wzg. Global Mean | Zmiana MAE wzg. Powiat Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForestRegressor (`rf_pipeline_v1.0` — główny)** | **14.670** | **22.293** | **+0.3662** | **+0.3345** | **-0.0009** | **$-1.056\text{ szt./h}$** | **$-0.616\text{ szt./h}$** |
| **GradientBoostingRegressor (porównawczy boosting)** | 14.793 | 22.536 | +0.3500 | +0.3057 | -0.0228 | $-0.933\text{ szt./h}$ | $-0.493\text{ szt./h}$ |
| **RandomForestRegressor (`criterion='absolute_error'`)** | 15.111 | 23.118 | **+0.3599** | **+0.3317** | -0.0764 | $-0.768\text{ szt./h}$ | $-0.328\text{ szt./h}$ |
| **HistGradientBoostingRegressor (boosting histogramowy)** | 15.179 | 22.857 | +0.2451 | +0.2326 | -0.0522 | $-0.471\text{ szt./h}$ | $-0.031\text{ szt./h}$ |
| *Baseline R3: Powiat Historical Mean* | *15.286* | *23.348* | *+0.3156* | *+0.2745* | *-0.0979* | $-0.440\text{ szt./h}$ | $0.000\text{ szt./h}$ |
| *Baseline R1: Global Mean (`16.55 szt./h`)* | *15.726* | *23.746* | *+0.0000* | *+0.0000* | *-0.1356* | $0.000\text{ szt./h}$ | $+0.440\text{ szt./h}$ |
| *Baseline R2: Global Median (`13.50 szt./h`)* | *16.433* | *24.966* | *+0.0000* | *+0.0000* | *-0.2553* | $+0.707\text{ szt./h}$ | $+1.147\text{ szt./h}$ |

> **Wnioski z porównania regresji i boostingu:**
> 1. Główny potok `RandomForestRegressor` obniża błąd **MAE na zbiorze testowym do `14.670 szt./h`** (wobec `15.726 szt./h` dla `Global Mean` oraz `15.286 szt./h` dla `Powiat Historical Mean`) oraz poprawia korelację rangową **Spearmana do $\rho = +0.3662$** (próg sukcesu z `[GRZ-13]` wynosił $\rho > 0.20$).
> 2. Na pomocniczym podziale 3-częściowym (`split_3way`: `Val` `2025-10-06 .. 2025-10-19`, $N=111$) ten sam potok osiąga **MAE = `13.259 szt./h`**.
> 3. Porównany model `GradientBoostingRegressor` również pokonuje wszystkie baseline'y (`MAE = 14.793 szt./h`, `Spearman = +0.3500`), lecz uśrednianie wiązkowe drzew (`Random Forest`) wykazuje niższą wariancję i wyższą odporność na szum pojedynczych zgłoszeń obywatelskich przy próbie treningowej $N = 386$.

---

### 4.2. Zadanie klasyfikacji (`abundance_class`: `BRAK`, `SREDNIO`, `WYSYP`)

| Model / Estymator | Macro-F1 (Test) | Weighted-F1 (Test) | Recall `BRAK` | Recall `SREDNIO` | Recall `WYSYP` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier (`rf_pipeline_v1.0` — główny)** | **0.4739** | **0.4877** | **0.6000** | **0.3860** | **0.5636** |
| *Baseline C2: Powiat Historical Majority* | *0.3918* | *0.3884* | *0.4667* | *0.7895* | *0.1091* |
| **RandomForestRegressor (dyskretyzacja progowa)** | 0.3558 | 0.4709 | 0.0000 | 0.6491 | 0.4909 |
| *Baseline C3: Stratified Random* | *0.3272* | *0.3608* | *0.3333* | *0.4912* | *0.2364* |
| *Baseline C1: Majority Class (zawsze `SREDNIO`)* | *0.2065* | *0.2781* | *0.0000* | *1.0000* | *0.0000* |

#### Macierz pomyłek `RandomForestClassifier` na zbiorze testowym ($N = 127$):
```text
Rzeczywiste \ Predykcja    BRAK    SREDNIO    WYSYP
BRAK (n=15)                 9       1          5      
SREDNIO (n=57)              13      22         22     
WYSYP (n=55)                5       19         31     
```
*Model klasyfikacyjny osiąga **Macro-F1 = `0.4739`** (przekraczając próg `> 0.45` z protokołu `[GRZ-13]` oraz pokonując `Majority Class = 0.2065` o `+0.2674` i `Powiat Historical Majority = 0.3918` o `+0.0821`), przy zachowaniu wysokiej czułości dla skrajnych klas `BRAK` (`60.0%`) oraz `WYSYP` (`56.4%`).*

---

## 5. Przekazywany rezultat: Przykład danych wejściowych i wyjściowych (`models/example_*.json/.csv`)

Poniższa tabela przedstawia wynik wywołania `python scripts/predict_example.py` na 5 rekordach testowych z pliku [`models/example_input.json`](../models/example_input.json) (zapisany w [`models/example_output.json`](../models/example_output.json) oraz [`models/example_output.csv`](../models/example_output.csv)):

| # | Powiat (`area_id`) | Data docelowa | Filtr gatunku | Dane taksacyjne BDL | Prognoza `predicted_yield_pcs_per_h` | Klasa z regresji | Klasa z klasyfikatora | Pewność (`confidence_score`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `0201` (powiat bolesławiecki) | `2025-10-14` | `all` | `TAK` | **19.85 szt./h** | `SREDNIO` | `WYSYP` | 0.62 |
| 2 | `0208` (powiat kłodzki) | `2025-10-18` | `podgrzybek_brunatny` | `TAK` | **9.27 szt./h** | `SREDNIO` | `BRAK` | 0.72 |
| 3 | `0218` (powiat średzki) | `2025-10-18` | `all` | `NIE (NULL)` | **11.75 szt./h** | `SREDNIO` | `SREDNIO` | 0.40 |
| 4 | `0217` (powiat strzeliński) | `2025-10-18` | `all` | `NIE (NULL)` | **19.83 szt./h** | `SREDNIO` | `WYSYP` | 0.46 |
| 5 | `0206` (powiat karkonoski) | `2025-10-25` | `borowik_szlachetny` | `TAK` | **32.27 szt./h** | `WYSYP` | `WYSYP` | 0.65 |
