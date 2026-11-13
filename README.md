# Gdzie na grzyby?

**Przestrzenny system predykcji wysypu grzybów w oparciu o raporty zbieraczy (`grzyby.pl`), siedliska leśne (BDL) i meteorologię (Open-Meteo)**

---

## 1. Szybki start — instalacja i uruchomienie aplikacji (`[GRZ-02]`, `[GRZ-05]`, `[GRZ-06]`, `[GRZ-12]`)

### Wymagania wstępne
- **Python 3.11+** (zalecane użycie wirtualnego środowiska `.venv`)

### Krok 1: Utworzenie środowiska wirtualnego i instalacja zależności

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux (bash / zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Krok 2: Konfiguracja zmiennych lokalnych (opcjonalnie)
Skopiuj szablon `.env.example` do `.env` (plik `.env` jest ignorowany przez Git):
```bash
cp .env.example .env
```

### Krok 3: Uruchomienie aplikacji Streamlit
```bash
streamlit run app.py
```
Po uruchomieniu aplikacja jest dostępna w przeglądarce pod adresem **`http://localhost:8501`**.

### Krok 4: Weryfikacja działania (testy dymne)
```bash
pytest -v
```

---

## 2. Wymagane pliki danych i tryby pracy aplikacji

Aplikacja działa w oparciu o zatwierdzony po audycie kontrakt danych opisany w [`docs/data_contract.md`](docs/data_contract.md) oraz [`docs/data_audit.md`](docs/data_audit.md) i obsługuje dwa odseparowane środowiska:

1. **Tryb rzeczywistych obserwacji i pokrycia danych (`data/processed/` — domyślny widok eksploracyjny `[GRZ-12]`):**
   - `data/processed/areas.geojson` — rzeczywiste granice **30 powiatów woj. dolnośląskiego (`TERYT 02`, `EPSG:4326`, `is_demo = False`)** wraz z cechami siedliskowymi BDL,
   - `data/processed/reports.parquet` (`N = 518`) oraz `data/processed/reports_spatial.parquet` (`N = 518`) — rzeczywiste zgłoszenia z portalu `grzyby.pl` po złączeniu przestrzennym z powiatami (`484 point_precise`, `34 county_aggregate`),
   - `data/processed/forest_features.parquet` (`N = 30`) — profil leśny BDL dla 30 powiatów,
   - `data/processed/weather.csv` (`N = 2910`) — szeregi dobowe Open-Meteo ERA5-Land (`2025-08-01 .. 2025-11-05`),
   - `data/processed/features.parquet` (`N = 513`) — zwalidowany zbiór analityczny Point-in-Time (`2025-09-01 .. 2025-11-02`, pokrycie `28 z 30` powiatów).
   - *Kluczowe właściwości widoku `[GRZ-12]`:* interaktywne filtrowanie po zakresie dat / pojedynczym dniu, gatunku obserwacji (`dominant_species`), powiecie i promieniu odległości; pokazywanie powiatów z `0` raportów jako `BRAK DANYCH (NULL)` (nigdy jako `0.0 szt./h`); interaktywny weryfikator zgodności 1:1 pojedynczego rekordu z tabelami źródłowymi.
2. **Tryb demonstracyjny UI (`data/demo/` — wersjonowany w Git, dostępny z przełącznika w panelu bocznym `[GRZ-02]`):**
   - `data/demo/areas_demo.geojson`, `data/demo/predictions_demo.csv`, `data/demo/reports_demo.csv`, `data/demo/weather_demo.csv`.
   - *Bezpieczeństwo:* wszystkie rekordy w `data/demo/` mają flagę `is_demo = True` i są wykluczone z treningu ML.
3. **Rzeczywiste próbki audytowe ze Sprintu 1 (`data/sample/` — wersjonowane w Git):**
   - `data/sample/reports_audit_52_ds.csv` — 52 rzeczywiste raporty z Dolnego Śląska (`[GRZ-03]`),
   - `data/sample/weather_daily_sample_ds.csv` — 210 dobowych rekordów z Open-Meteo ERA5-Land (`[GRZ-04]`),
   - `data/sample/region_powiaty_02_ds.geojson` — granice 30 powiatów woj. dolnośląskiego (`19 941,7 km²`, `EPSG:4326`, `[GRZ-05]`),
   - `data/sample/bdl_stands_sample_ds.geojson` oraz `.csv` — wektorowa próbka 105 wydzieleń leśnych BDL (`Wydzielenia PGL LP` / `G_SUBAREA`, w tym 74 drzewostanów `D-STAN` w 7 klastrach leśnych, `[GRZ-05]`),
   - `data/sample/bdl_powiaty_agg_sample_ds.csv` — wynik przestrzennej agregacji GeoPandas (`gpd.sjoin` w `EPSG:2180`) wydzieleń BDL do poziomu powiatu (`[GRZ-05]`).

---

## 3. Struktura repozytorium

```text
├── app.py                                    # Punkt wejścia aplikacji Streamlit (streamlit run app.py)
├── requirements.txt                          # Lista zależności Pythona
├── .env.example                              # Szablon zmiennych środowiskowych (bez sekretów)
├── .streamlit/
│   └── config.toml                           # Konfiguracja motywu i serwera Streamlit
├── src/                                      # Kod źródłowy aplikacji i modułów wspólnych
│   ├── __init__.py
│   ├── config.py                             # Konfiguracja regionu pilotażowego (Dolny Śląsk, TERYT 02) i słowników
│   ├── data_loader.py                        # Wczytywanie, filtrowanie i weryfikacja 1:1 plików wg kontraktu ([GRZ-02], [GRZ-12])
│   ├── geo_utils.py                          # Obliczenia odległości (Haversine) i wzbogacanie warstw GeoJSON ([GRZ-05], [GRZ-12])
│   ├── features.py                           # Budowa zbioru cech Point-in-Time i słownika kolumn ([GRZ-10])
│   ├── eda.py                                # Analiza eksploracyjna (EDA), opóźnienia 7/14/21d i wykresy ([GRZ-11])
│   ├── evaluation.py                         # Protokół ewaluacji, podział czasowy i metryki ([GRZ-13])
│   └── baseline.py                           # Modele bazowe (GlobalMean, PowiatMean, MajorityClass, Seasonal) ([GRZ-13])
├── scripts/                                  # Skrypty potoku danych, scrapingu, pogody i GIS/BDL
│   ├── audit_grzyby_reports.py               # Skrypt audytowy raportów grzyby.pl ([GRZ-03])
│   ├── fetch_weather_sample.py               # Skrypt pobierania próbki pogody z Open-Meteo z cache'em ([GRZ-04])
│   ├── prepare_region_and_bdl_sample.py      # Skrypt przygotowania granic 30 powiatów i próbki wektorowej BDL ([GRZ-05])
│   ├── build_reports_dataset.py              # Budowa oczyszczonego zbioru obserwacji reports.parquet ([GRZ-07])
│   ├── build_spatial_features.py             # Łączenie przestrzenne raportów z lasami i powiatami ([GRZ-08])
│   ├── build_weather_dataset.py              # Budowa historii pogodowej i okien 7/14/21d dla 30 powiatów ([GRZ-09])
│   ├── build_features_dataset.py             # Łączenie raportów, lasów i pogody w features.parquet Point-in-Time ([GRZ-10])
│   ├── run_eda_and_lag_analysis.py           # Odtwarzanie EDA, 5 wykresów PNG i notatnika Jupyter ([GRZ-11])
│   └── build_split_and_baseline.py           # Podział czasowy train/val/test i benchmarki bazowe ([GRZ-13])
├── notebooks/                                # Notatniki Jupyter i analizy eksploracyjne (EDA, Lag Effect)
│   ├── 01_eda_lag_effect.ipynb               # Notatnik EDA i analizy opóźnienia opad -> zbiory ([GRZ-11])
│   └── README.md
├── models/                                   # Artefakty wytrenowanych modeli ML i metadane (binaria ignorowane w Git)
│   └── README.md
├── tests/                                    # Testy jednostkowe i dymne szkieletu aplikacji, GIS/BDL i kontraktu danych
│   └── test_smoke.py
├── docs/                                     # Dokumentacja projektowa, architektoniczna i raporty z audytów
│   ├── figures/                              # Wykresy analityczne PNG z jednostkami, okresem i próbą n ([GRZ-11])
│   ├── mvp.md                                # Zakres MVP zatwierdzony po audycie ([GRZ-01], [GRZ-06])
│   ├── data_contract.md                      # Kontrakt danych: areas, reports, weather, predictions ([GRZ-01], [GRZ-06])
│   ├── data_audit.md                         # Zbiorczy audyt danych, cel predykcji, odcięcie pogody i plan modelowania ([GRZ-06])
│   ├── reports_audit.md                      # Audyt struktury i kompletności raportów z grzyby.pl ([GRZ-03])
│   ├── weather_audit.md                      # Audyt zmiennych pogodowych i glebowych Open-Meteo ERA5-Land ([GRZ-04])
│   ├── bdl_region_audit.md                   # Audyt granic regionu (PRG) i wektorowej próbki BDL ([GRZ-05])
│   ├── reports_etl.md                        # Dokumentacja potoku ETL obserwacji z grzyby.pl ([GRZ-07])
│   ├── spatial_join_bdl.md                   # Dokumentacja łączenia przestrzennego z powiatami i BDL ([GRZ-08])
│   ├── weather_pipeline.md                   # Dokumentacja potoku pogodowego Open-Meteo dla 30 powiatów ([GRZ-09])
│   ├── features_engineering.md               # Dokumentacja zbioru cech features.parquet i weryfikacji Point-in-Time ([GRZ-10])
│   ├── eda_and_lag_analysis.md               # Raport EDA, efektu opóźnienia opad -> zbiory i rekomendacji ML ([GRZ-11])
│   └── evaluation_protocol.md                # Dokumentacja podziału czasowego i modeli bazowych ([GRZ-13])
└── data/                                     # Katalog danych
    ├── demo/                                 # Próbka demonstracyjna do budowy UI (is_demo = True, 30 powiatów)
    ├── sample/                               # Rzeczywiste próbki audytowe ze Sprintu 1 ([GRZ-03], [GRZ-04], [GRZ-05])
    ├── raw/                                  # Surowe dane (cache GIS/BDL, zrzuty HTML, cache API) — ignorowane w Git
    └── processed/                            # Produkcyjne pliki dla modelu i aplikacji (reports, forest, weather, features)
```

---

## 4. Dokumentacja projektu

- [**Eksploracyjna Analiza Danych (EDA) i Opóźnienie Opad → Zbiory (`docs/eda_and_lag_analysis.md`)**](docs/eda_and_lag_analysis.md) *(`[GRZ-11]`)*
- [**Inżynieria Cech i Zbiór Analityczny Point-in-Time (`docs/features_engineering.md`)**](docs/features_engineering.md) *(`[GRZ-10]`)*
- [**Protokół Ewaluacji, Podział Czasowy i Modele Bazowe (`docs/evaluation_protocol.md`)**](docs/evaluation_protocol.md) *(`[GRZ-13]`)*
- [**Potok Pogodowy dla Regionu Pilotażowego (`docs/weather_pipeline.md`)**](docs/weather_pipeline.md) *(`[GRZ-09]`)*
- [**Łączenie Przestrzenne z Granicami Powiatów i BDL (`docs/spatial_join_bdl.md`)**](docs/spatial_join_bdl.md) *(`[GRZ-08]`)*
- [**Potok ETL Zbioru Obserwacji (`docs/reports_etl.md`)**](docs/reports_etl.md) *(`[GRZ-07]`)*
- [**Zbiorczy Audyt Danych i Zatwierdzenie Celu Predykcji (`docs/data_audit.md`)**](docs/data_audit.md) *(`[GRZ-06]`)*
- [**Zakres MVP i definicja wyniku aplikacji (`docs/mvp.md`)**](docs/mvp.md) *(`[GRZ-01]`, `[GRZ-06]`)*
- [**Kontrakt Danych (`docs/data_contract.md`)**](docs/data_contract.md) *(`[GRZ-01]`, `[GRZ-06]`)*
- [**Audyt raportów z portalu `grzyby.pl` (`docs/reports_audit.md`)**](docs/reports_audit.md) *(`[GRZ-03]`)*
- [**Audyt danych pogodowych Open-Meteo (`docs/weather_audit.md`)**](docs/weather_audit.md) *(`[GRZ-04]`)*
- [**Audyt granic regionu i wektorowej próbki leśnej BDL (`docs/bdl_region_audit.md`)**](docs/bdl_region_audit.md) *(`[GRZ-05]`)*

---

## 5. Źródła danych i klauzula akademicka

- **Raporty o zbiorach grzybów:** Dane o zbiorach pochodzą z portalu `grzyby.pl` – © **Marek Snowarski**, *Grzyby Polski, atlas grzybów* ([www.grzyby.pl](https://www.grzyby.pl)). Projekt ma charakter wyłącznie dydaktyczny (praca studencka na zaliczenie przedmiotu) i niekomercyjny.
- **Dane meteorologiczne i glebowe:** [Open-Meteo Historical & Forecast API](https://open-meteo.com/) (reanaliza ECMWF ERA5-Land, siatka ~9 km; wilgotność i temperatura gleby stanowią wynik modelu numerycznego, a nie pomiaru terenowego *in-situ*).
- **Dane o siedliskach leśnych i granicach:** Bank Danych o Lasach ([BDL](https://www.bdl.lasy.gov.pl/) – Lasy Państwowe / BULiGL, stan `2026`) oraz Państwowy Rejestr Granic (GUGiK PRG).
