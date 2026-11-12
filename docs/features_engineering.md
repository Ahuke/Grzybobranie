# Tabela analityczna i inżynieria cech (`[GRZ-10] #10`)

> **Cel zadania (`[GRZ-10]`):** Przygotować jedną tabelę analityczną (`data/processed/features.parquet`) zgodną z momentem wystawienia prognozy (*Point-in-Time Correctness*), łączącą raporty o zbiorach (`[GRZ-07]`), cechy leśne i przestrzenne (`[GRZ-08]`) oraz historię pogody (`[GRZ-09]`), wyliczyć okna opadów 7/14/21 dni, dostępne cechy temperatury, wilgotności gleby oraz sezonowości, a także udokumentować ręczną weryfikację okna 7-dniowego, braków danych (`NULL` vs `0.0`) i złączenia (`join`).

---

## 1. Wygenerowane artefakty i odtwarzalność potoku

| Plik w repozytorium | Format i klucz główny | Liczba wierszy / kolumn | Rola w projekcie |
| :--- | :---: | :---: | :--- |
| [`data/processed/features.parquet`](../data/processed/features.parquet)<br/>[`data/processed/features.csv`](../data/processed/features.csv) | `Parquet` / `CSV`<br/>PK: **`observation_id`** (`== report_id`) | **`513` wierszy**<br/>**`70` kolumn** | **Główna tabela analityczna** dla eksploracyjnej analizy danych (`[GRZ-11]`), treningu modeli ML (`[GRZ-14]`) i końcowej ewaluacji (`[GRZ-17]`). |
| [`data/processed/features_quality_report.json`](../data/processed/features_quality_report.json) | `JSON` | — | Maszynowy raport spójności: liczebność, zakresy dat, rozkład braków danych, wyniki 3 ręcznych weryfikacji oraz pełny słownik 70 kolumn. |
| [`src/features.py`](../src/features.py) | Moduł Python | — | Reużywalne funkcje wyliczania okien pogodowych z odcięciem czasowym (`compute_point_in_time_weather_features`), sezonowości (`compute_seasonality_features`) oraz łączenia tabel (`build_features_dataframe`). |
| [`scripts/build_features_dataset.py`](../scripts/build_features_dataset.py) | Skrypt CLI | — | Powtarzalny skrypt odtwarzający `features.parquet`, `features.csv` i raport jakości wraz z weryfikacją asercji. |

### Polecenie odtwarzające zbiór cech:
```bash
PYTHONPATH=. python scripts/build_features_dataset.py
```

---

## 2. Separacja dat i ochrona przed przeciekiem informacji (`Warunek #3`)

W każdym wierszu zbioru `features.parquet` jawnie rozdzielono cztery pojęcia czasowe:

1. **`observation_date` (aliasy: `harvest_date`, `date`):**
   - Data faktycznego grzybobrania w lesie (np. `"2025-10-05"`), pochodząca z raportu grzybiarza (nigdy spóźniona `publication_date`).
2. **`target_date` ($t$):**
   - Data docelowa, na którą wystawiana i oceniana jest prognoza ($t = \text{observation\_date}$).
3. **`issued_at` ($t_0$):**
   - Data wystawienia prognozy przy zadanym horyzoncie wyprzedzenia $H = \text{horizon\_days}$ (domyślnie $H = 1\text{ dzień}$, czyli $t_0 = t - 1$, np. `"2025-10-04"`).
4. **`feature_cutoff_date` ($t_{\text{cutoff}}$):**
   - Sztywny moment odcięcia dostępności cech pogodowych i glebowych ($t_{\text{cutoff}} = \text{issued\_at} = t - H$, czyli doba poprzedzająca wyjście do lasu `23:59 Europe/Warsaw`).

### Gwarancja braku wycieku z przyszłości (*No Future Weather Leakage*):
- Wszystkie okna opadowe, temperaturowe i glebowe o długości $k \in \{3, 7, 14, 21\}$ dni wyliczane są na domkniętym przedziale:
  $$[\text{window\_}k\text{d\_start\_date},\; \text{window\_end\_date}] = [t_{\text{cutoff}} - (k - 1),\; t_{\text{cutoff}}]$$
- Dla bazowego horyzontu $H = 1\text{ dzień}$ ($t_{\text{cutoff}} = t - 1$) przedział ten wynosi dokładnie **$[t - k,\; t - 1]$**.
- Również zmienne jednodniowe (`precip_cutoff_1d_mm`, `temp_min_night_c`, `temp_mean_cutoff_1d_c`, `soil_moisture_0_7cm_m3m3`, `soil_temperature_0_7cm_c`) pobierane są z doby odcięcia **`feature_cutoff_date` ($t - 1$)**, a **nigdy z samego dnia grzybobrania $t$** (opad lub temperatura popołudniowa z dnia $t$ nie są znane w chwili wystawienia prognozy ani rano przed wejściem do lasu).

---

## 3. Liczebność, zakres danych i integralność 1:1 (`Warunek #2 i #6`)

| Parametr zbioru `features.parquet` | Wartość po połączeniu | Komentarz / Weryfikacja |
| :--- | :---: | :--- |
| **Wejściowe raporty ogółem (`reports_spatial.parquet`)** | `518` | Pełny zbiór z 9 tygodni jesieni 2025 (`5` rekordów oznaczonych w `[GRZ-07]` jako odrzucone: 4 miejskie trawniki + 1 outlier `250 szt./h`). |
| **Wiersze w `features.parquet` (`is_valid_for_training == True`)** | **`513`** | **Dokładnie 1:1** z ważnymi raportami (`unique_report_ids_count = 513`, `multiplication_factor = 1.0`). |
| **Liczba kolumn** | **`70`** | Zgodna ze słownikiem `FEATURE_DICTIONARY` w `src/features.py`. |
| **Zakres dat obserwacji / docelowych (`target_date`)** | **`2025-09-01` – `2025-11-02`** | `62` unikalne dni grzybobrania (9 pełnych tygodni sezonu jesiennego). |
| **Zakres dat wystawienia i odcięcia (`feature_cutoff_date`)** | **`2025-08-31` – `2025-11-01`** | Dla najwcześniejszego raportu (`2025-09-01`) okno 21-dniowe sięga do `2025-08-11` (szereg pogodowy dostępny od `2025-08-01`). |
| **Pokrycie przestrzenne (`region_id`)** | **`28 / 30` powiatów (`93,33%`)** | Nieobserwowane w raportach jedynie 2 mało leśne powiaty: `0217` (strzeliński) i `0262` (m. Legnica). |
| **Podział czasowy 2-częściowy (`split_2way`, `[GRZ-13]`)** | **Train:** `386` (`75,24%`)<br/>**Test:** `127` (`24,76%`) | Train: `2025-09-01 .. 2025-10-12`; Test: `2025-10-13 .. 2025-11-02`. |
| **Podział czasowy 3-częściowy (`split_3way`, `[GRZ-13]`)** | **Train:** `333` (`64,91%`)<br/>**Val:** `111` (`21,64%`)<br/>**Test:** `69` (`13,45%`) | Train: `<= 2025-10-05`; Val: `2025-10-07 .. 2025-10-19`; Test: `>= 2025-10-20`. |
| **Rozkład klas obfitości (`abundance_class`)** | `BRAK`: `112` (`21,83%`)<br/>`SREDNIO`: `239` (`46,59%`)<br/>`WYSYP`: `162` (`31,58%`) | W klasie `BRAK` znajduje się `45` jawnych raportów zerowych (`yield_pcs_per_h == 0.0`). |

---

## 4. Ręczna weryfikacja obliczeń i złączeń (`Warunek #4 i #5`)

Zgodnie z warunkiem ukończenia `#4` (*„Sprawdzono ręcznie obliczenie jednego okna 7-dniowego, jeden brak danych i jeden join”*), poniżej udokumentowano trzy szczegółowe kontrole (automatycznie weryfikowane także przy każdym uruchomieniu `scripts/build_features_dataset.py`):

### 4.1. Ręczne sprawdzenie jednego okna 7-dniowego (`report_id = DG174806`)
- **Obserwacja:** `report_id = "DG174806"`, powiat bolesławiecki (`region_id = "0201"`), data zbioru `observation_date = "2025-10-05"`, horyzont `horizon_days = 1`.
- **Odcięcie informacji:** `issued_at = feature_cutoff_date = "2025-10-04"`.
- **Przedział 7-dniowego okna pogodowego (`[t-7, t-1]`):** od **`2025-09-28`** do **`2025-10-04`** włącznie:

| Dzień w oknie | Data (`YYYY-MM-DD`) | Dobowy opad `precip_mm_1d` (`[mm]`) | Średnia temp. dobowa `temp_mean_day_c` (`[°C]`) | Czy wchodzi do okna 7d? |
| :---: | :---: | :---: | :---: | :---: |
| $t - 7$ | `2025-09-28` | `0.0` | `11.8` | **TAK** |
| $t - 6$ | `2025-09-29` | `0.3` | `9.0` | **TAK** |
| $t - 5$ | `2025-09-30` | `1.6` | `7.2` | **TAK** |
| $t - 4$ | `2025-10-01` | `2.8` | `8.1` | **TAK** |
| $t - 3$ | `2025-10-02` | `0.4` | `6.9` | **TAK** |
| $t - 2$ | `2025-10-03` | `0.0` | `7.3` | **TAK** |
| $t - 1$ *(cutoff)* | `2025-10-04` | `7.9` | `8.1` | **TAK** |
| **Dzień zbioru $t$** | **`2025-10-05`** | **`0.9`** | **`10.5`** | **NIE (po odcięciu!)** |

- **Ręczny rachunek sumy opadów (`precip_lag_7d_mm`):**
  $$0{,}0 + 0{,}3 + 1{,}6 + 2{,}8 + 0{,}4 + 0{,}0 + 7{,}9 = \mathbf{13{,}0\text{ mm}}$$
  Wartość w `features.parquet`: **`13.0 mm`** (zgodność 100%; opad `0.9 mm` z dnia `2025-10-05` został poprawnie wykluczony).
- **Ręczny rachunek średniej temperatury 7-dniowej (`temp_mean_7d_c`):**
  $$\frac{11{,}8 + 9{,}0 + 7{,}2 + 8{,}1 + 6{,}9 + 7{,}3 + 8{,}1}{7} = \frac{58{,}4}{7} = 8{,}3428\dots \approx \mathbf{8{,}3^\circ\text{C}}$$
  Wartość w `features.parquet`: **`8.3 °C`** (zgodność 100%).

---

### 4.2. Ręczne sprawdzenie jednego braku danych (`NULL` vs `0.0` – `Warunek #4 i #5`)
Zweryfikowano trzy aspekty ochrony przed zamianą braków danych w zera:

1. **Przypadek rzeczywisty w zbiorze (`report_id = DG177228`, `2025-11-02`, `region_id = 0218` powiat średzki):**
   - Powiat średzki (`0218`) posiada ogólną lesistość z GUS (`forest_cover_pct = 11.8%`), ale nie posiada szczegółowego profilu taksacyjnego z próbki BDL (`has_bdl_stand_data = False`). Ponadto autor raportu nie podał w treści czasu zbierania ani liczby sztuk.
   - W wierszu `DG177228` w `features.parquet`:
     - `forest_cover_pct = 11.8`, `yield_pcs_per_h = 10.0`,
     - `pine_share_pct = NULL` (`NaN`), `oak_beech_share_pct = NULL`, `spruce_share_pct = NULL`, `mean_stand_age_years = NULL`, `dominant_moisture_type = NULL`,
     - `harvest_duration_h = NULL`, `total_pieces = NULL`.
   - **Wniosek:** Żaden brakujący atrybut leśny ani raportowy nie został zastąpiony wartością `0.0` (łącznie `80` wierszy z powiatów bez profilu BDL zachowuje jawne `NULL`).
2. **Odróżnienie jawnego zera od `NULL` (`report_id = DG174777`, `2025-10-05`, `region_id = 0222` powiat wołowski):**
   - Autor zgłosił pusty koszyk `(0/h)` i brak gatunków. W `features.parquet`: `yield_pcs_per_h = 0.0` (potwierdzony zerowy zbiór), natomiast `dominant_species = NULL` (`None`).
3. **Test luki w szeregu pogodowym (`missing_weather_day_in_window_test`):**
   - Po celowym usunięciu jednego dnia (`2025-10-02`) z okna `[2025-09-28 .. 2025-10-04]` dla powiatu `0201`, funkcja `compute_point_in_time_weather_features` zwraca `precip_lag_7d_mm = None` (`NULL`), zamiast zsumować 6 pozostałych dni lub zwrócić `0.0`.

---

### 4.3. Ręczne sprawdzenie jednego złączenia 3 tabel (`JOIN: report_id = DG174714`)
Prześledzono złączenie rekordu **`DG174714`** (`observation_date = "2025-10-05"`, `region_id = "0208"` powiat kłodzki, `feature_cutoff_date = "2025-10-04"`):

| Grupa danych | Pole | Wartość w tabeli źródłowej | Wartość w `features.parquet` | Zgodność |
| :--- | :--- | :---: | :---: | :---: |
| **1. `reports_spatial.parquet`**<br/>*(PK: `report_id = DG174714`)* | `region_id` / `observation_date`<br/>`yield_pcs_per_h` / `abundance_class`<br/>`dominant_species`<br/>`harvest_duration_h` / `total_pieces`<br/>`spatial_precision` | `"0208"` / `"2025-10-05"`<br/>`20.0 szt./h`<br/>`"podgrzybek_brunatny"`<br/>`3.0 h` / `60.0 szt.`<br/>`"point_precise"` | `"0208"` / `"2025-10-05"`<br/>`20.0` / `"SREDNIO"`<br/>`"podgrzybek_brunatny"`<br/>`3.0` / `60.0`<br/>`"point_precise"` | **TAK** |
| **2. `forest_features.parquet`**<br/>*(PK: `region_id = 0208`)* | `region_name`<br/>`total_area_km2` / `forest_cover_pct`<br/>`pine_share_pct` / `oak_beech_share_pct` / `spruce_share_pct`<br/>`mean_stand_age_years` / `dominant_moisture_type` | `"powiat kłodzki"`<br/>`1640.92 km²` / `48.9%`<br/>`9.8%` / `28.4%` / `58.0%`<br/>`81.2 lat` / `"wilgotne"` | `"powiat kłodzki"`<br/>`1640.92` / `48.9`<br/>`9.8` / `28.4` / `58.0`<br/>`81.2` / `"wilgotne"` | **TAK** |
| **3. `weather_daily.parquet`**<br/>*(PK: `region_id = 0208`, cutoff `2025-10-04`)* | `precip_lag_7d_mm` (`28.09..04.10`)<br/>`precip_lag_14d_mm` (`21.09..04.10`)<br/>`precip_lag_21d_mm` (`14.09..04.10`)<br/>`temp_min_night_c` *(doba `04.10`)*<br/>`soil_moisture_0_7cm_m3m3` *(doba `04.10`)* | `9.8 mm`<br/>`24.8 mm`<br/>`33.5 mm`<br/>`-2.1 °C`<br/>`0.3248 m³/m³` | `9.8`<br/>`24.8`<br/>`33.5`<br/>`-2.1`<br/>`0.3248` | **TAK** |

---

## 5. Uzgodniony słownik kolumn (`features.parquet` – 70 kolumn)

Pełna definicja typów, jednostek i źródeł znajduje się w stałej `FEATURE_DICTIONARY` w [`src/features.py`](../src/features.py) oraz w [`data/processed/features_quality_report.json`](../data/processed/features_quality_report.json):

1. **Identyfikatory i klucze (6 kolumn):** `observation_id`, `report_id`, `region_id`, `area_id`, `region_name`, `place_id`.
2. **Daty, horyzont i granice okien (12 kolumn):** `observation_date`, `harvest_date`, `date`, `publication_date`, `target_date`, `issued_at`, `feature_cutoff_date`, `horizon_days`, `window_7d_start_date`, `window_14d_start_date`, `window_21d_start_date`, `window_end_date`.
3. **Zmienne celu i atrybuty raportu (5 kolumn):** `yield_pcs_per_h`, `abundance_class`, `dominant_species`, `harvest_duration_h`, `total_pieces`.
4. **Cechy opadowe przed odcięciem (7 kolumn):** `precip_lag_7d_mm`, `precip_lag_14d_mm`, `precip_lag_21d_mm`, `precip_cutoff_1d_mm`, `precip_hours_lag_7d_h`, `rainy_days_lag_7d_count`, `rainy_days_lag_14d_count`.
5. **Cechy temperaturowe, przymrozkowe i wiatrowe przed odcięciem (9 kolumn):** `temp_min_night_c`, `temp_min_lag_3d_c`, `temp_min_lag_7d_c`, `frost_nights_lag_7d_count`, `temp_mean_cutoff_1d_c`, `temp_mean_7d_c`, `temp_mean_14d_c`, `temp_max_cutoff_1d_c`, `wind_speed_mean_7d_kmh`.
6. **Cechy glebowe i wysokościowe ERA5-Land przed odcięciem (6 kolumn):** `soil_moisture_0_7cm_m3m3`, `soil_moisture_mean_7d_m3m3`, `soil_moisture_mean_14d_m3m3`, `soil_temperature_0_7cm_c`, `soil_temp_mean_7d_c`, `grid_elevation_m`.
7. **Cechy leśne BDL i przestrzenne (12 kolumn):** `total_area_km2`, `forest_cover_pct`, `forest_area_km2`, `pine_share_pct`, `oak_beech_share_pct`, `spruce_share_pct`, `mean_stand_age_years`, `dominant_moisture_type`, `has_bdl_stand_data`, `spatial_precision`, `point_represents_specific_forest`, `distance_to_county_centroid_km`.
8. **Cechy sezonowości (8 kolumn):** `month`, `day_of_year`, `week_of_year`, `day_of_week`, `is_weekend`, `day_of_year_sin`, `day_of_year_cos`, `season_phase`.
9. **Podziały ewaluacyjne i metadane pochodzenia (5 kolumn):** `split_2way`, `split_3way`, `source_url`, `is_valid_for_training`, `is_demo`.
