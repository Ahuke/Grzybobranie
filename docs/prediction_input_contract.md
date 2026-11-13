# Kontrakt wejścia do prognoz (`[GRZ-15] #15`)

> **Cel zadania (`[GRZ-15]`):** Zapewnić modelowi w aplikacji (`[GRZ-16]`) oraz w historycznych symulacjach prognoz dokładnie takie same cechy (nazwy, jednostki, typy i brak przecieku danych z przyszłości), jakie przewidziano w zbiorze treningowym (`[GRZ-10]` / `[GRZ-14]`).

---

## 1. Jedna funkcja / skrypt generujący wejście do modelu

| Element | Rola | Ścieżka |
|---|---|---|
| **Funkcja pojedynczego rekordu** | Buduje 1 wiersz cech dla `(region_id, issued_at, horizon_days)` | [`src/inference_features.py`](../src/inference_features.py) (`build_prediction_input_row`) |
| **Funkcja wsadowa (Batch)** | Generuje `DataFrame` dla wybranych powiatów (domyślnie wszystkich 30) i horyzontów (`1, 2, 3` dni) | [`src/inference_features.py`](../src/inference_features.py) (`build_prediction_inputs_batch`) |
| **Skrypt CLI** | Generuje próbkę wejścia do prognoz (`90` wierszy dla `issued_at=2025-10-18`, horyzonty `1..3`) i raportuje testy brzegowe | [`scripts/build_prediction_inputs.py`](../scripts/build_prediction_inputs.py) |
| **Próbka wyjściowa + raport** | Zapisany wynik przykładowego uruchomienia i weryfikacja przypadków brzegowych | [`data/processed/prediction_inputs_sample.parquet`](../data/processed/prediction_inputs_sample.parquet), [`data/processed/prediction_inputs_quality_report.json`](../data/processed/prediction_inputs_quality_report.json) |

### Przykład użycia w Pythonie:
```python
from src.inference_features import build_prediction_inputs_batch

df_input = build_prediction_inputs_batch(
    issued_at="2025-10-18",
    horizons=(1, 2, 3),
)
```

---

## 2. Zgodność nazw, jednostek i typów z treningiem (`Warunek #1`)

Wszystkie **42 cechy modelowe** (`MODEL_INPUT_FEATURE_COLUMNS` w `src/inference_features.py`) są wyliczane dokładnie tymi samymi funkcjami z modułu [`src/features.py`](../src/features.py) (`compute_point_in_time_weather_features`, `compute_seasonality_features`), które posłużyły do budowy `data/processed/features.parquet`:

1. **Cechy pogodowe i glebowe (22 kolumny):** `precip_lag_7d_mm`, `precip_lag_14d_mm`, `precip_lag_21d_mm`, `precip_cutoff_1d_mm`, `precip_hours_lag_7d_h`, `rainy_days_lag_7d_count`, `rainy_days_lag_14d_count`, `temp_min_night_c`, `temp_min_lag_3d_c`, `temp_min_lag_7d_c`, `frost_nights_lag_7d_count`, `temp_mean_cutoff_1d_c`, `temp_mean_7d_c`, `temp_mean_14d_c`, `temp_max_cutoff_1d_c`, `wind_speed_mean_7d_kmh`, `soil_moisture_0_7cm_m3m3`, `soil_moisture_mean_7d_m3m3`, `soil_moisture_mean_14d_m3m3`, `soil_temperature_0_7cm_c`, `soil_temp_mean_7d_c`, `grid_elevation_m`.
2. **Cechy leśne BDL (9 kolumn):** `total_area_km2`, `forest_cover_pct`, `forest_area_km2`, `pine_share_pct`, `oak_beech_share_pct`, `spruce_share_pct`, `mean_stand_age_years`, `dominant_moisture_type`, `has_bdl_stand_data`.
3. **Kontekst przestrzenny (3 kolumny):** `spatial_precision = "county_aggregate"`, `point_represents_specific_forest = False`, `distance_to_county_centroid_km = 0.0` (prognoza wystawiana jest dla powiatu jako całości, bez pozornej precyzji punktu).
4. **Cechy sezonowości dla `target_date` (8 kolumn):** `month`, `day_of_year`, `week_of_year`, `day_of_week`, `is_weekend`, `day_of_year_sin`, `day_of_year_cos`, `season_phase`.

---

## 3. Data wystawienia, data docelowa i brak wycieku przyszłej pogody (`Warunek #2, #3 i #4`)

- Każdy wygenerowany rekord zawiera jawnie:
  - `issued_at` ($t_0$) – datę wystawienia prognozy,
  - `feature_cutoff_date` ($t_{\text{cutoff}} = t_0$) – granicę dostępności danych pogodowych,
  - `horizon_days` ($H \in [1, 7]$, domyślnie `1, 2, 3`),
  - `target_date` ($t = t_0 + H$) – datę docelową prognozy.
- **Brak wycieku przyszłej pogody w historycznej symulacji (`Warunek #3`):**
  - Okna opadowe, temperaturowe i glebowe (`[t_0 - (k-1), t_0]`) kończą się w dobie `issued_at`.
  - Nawet dla horyzontu $H = 3$ (`issued_at = 2025-10-18`, `target_date = 2025-10-21`) usunięcie wszystkich rekordów pogodowych po `2025-10-18` daje **identyczne** wartości wszystkich 42 cech (`identical_features_without_future_weather = True`).
- **Dostępność przyszłej prognozy pogody (`Warunek #4`):**
  - Architektura cech w MVP celowo **nie wymaga** przyszłych prognoz numerycznych (NWP) dla dni `(issued_at, target_date]`, ponieważ główny sygnał biologiczny pochodzi ze skumulowanej wilgotności i opóźnionych opadów z minionych `7–21` dni (`rainy_days_lag_14d_count`, `precip_lag_21d_mm`, `soil_moisture_mean_14d_m3m3`). Dzięki temu historyczna symulacja prognoz w dowolnym dniu sezonu jest w 100% wolna od problemu braku archiwalnych prognoz NWP, a dozwolony horyzont ograniczono do `1–7` dni (`MAX_SUPPORTED_HORIZON_DAYS = 7`).

---

## 4. Zachowanie przy niepełnej historii i obszarze spoza pokrycia (`Warunek #5`)

| Przypadek brzegowy | Zachowanie funkcji `build_prediction_input_row` | Oznaczenie statusu (`history_completeness`) |
|---|---|---|
| **Niepełna historia pogodowa** (np. `issued_at = "2025-08-10"`, archiwum od `"2025-08-01"`) | Okna mające pełne pokrycie (np. `7d`: `11.1 mm`) są wyliczane; okna z brakującym choćby 1 dniem (np. `14d`, `21d`) zwracają **`None` (`NaN`)**, nigdy `0.0`. | `"PARTIAL_HISTORY"` (`missing_weather_days_21d = 11`) lub `"NO_WEATHER_HISTORY"` (`21`) |
| **Powiat bez szczegółowego profilu BDL** (np. `0218` średzki, 12 powiatów) | Zachowuje ogólny `forest_cover_pct` z GUS oraz pełne cechy pogodowe; atrybuty gatunkowe BDL pozostają jako **`None` (`NaN`)** (spójnie z treningiem). | `"COMPLETE_WEATHER_NO_BDL_STAND"` |
| **Obszar spoza pokrycia** (np. `region_id = "9999"` spoza Dolnego Śląska) | Domyślnie (`allow_out_of_coverage=False`) rzuca jawny wyjątek `RegionOutOfCoverageError`. W trybie łagodnym (`allow_out_of_coverage=True`) zwraca `is_region_covered = False` i cechy jako `None`. | `"OUT_OF_COVERAGE"` |
