# Gdzie na grzyby?

**Przestrzenny system predykcji wysypu grzybów w oparciu o raporty zbieraczy (`grzyby.pl`), siedliska leśne (BDL) i meteorologię (Open-Meteo)**

---

## 1. Szybki start — instalacja i uruchomienie aplikacji (`[GRZ-02]`, `[GRZ-05]`, `[GRZ-06]`)

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

1. **Tryb demonstracyjny UI (`data/demo/` — domyślny na starcie, wersjonowany w Git):**
   - `data/demo/areas_demo.geojson` — rzeczywiste granice wszystkich **30 powiatów woj. dolnośląskiego (`TERYT 02`, `EPSG:4326`)** z Państwowego Rejestru Granic (PRG) wraz z cechami siedlisk leśnych BDL (`[GRZ-05]`),
   - `data/demo/predictions_demo.csv` — przykładowe predykcje wydajności zbioru `[szt./h]` oraz klas (`BRAK`, `SREDNIO`, `WYSYP`),
   - `data/demo/reports_demo.csv` — przykładowe raporty zbieraczy,
   - `data/demo/weather_demo.csv` — przykładowe szeregi pogodowe z oknami opóźnień (`lag 7d/14d/21d`).
   - *Bezpieczeństwo:* wszystkie rekordy w `data/demo/` mają flagę `is_demo = True` i są wykluczone z treningu ML.
2. **Tryb produkcyjny (`data/processed/` — generowany przez potok danych):**
   - `data/processed/areas.geojson` (wersjonowany plik rzeczywistych granic 30 powiatów Dolnego Śląska w `EPSG:4326`, `is_demo = False`), `data/processed/reports.csv`, `data/processed/weather.csv`, `data/processed/predictions.csv`.
   - Jeżeli wszystkie 4 pliki w `data/processed/` nie zostały jeszcze wygenerowane lokalnie, aplikacja wyświetla informację i bezpiecznie przełącza się na `data/demo/`.
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
│   ├── data_loader.py                        # Wczytywanie i walidacja plików wg kontraktu danych (NULL vs 0.0)
│   └── geo_utils.py                          # Obliczenia odległości (Haversine) i wzbogacanie warstw GeoJSON (powiaty + BDL)
├── scripts/                                  # Skrypty potoku danych, scrapingu, pogody i GIS/BDL
│   ├── audit_grzyby_reports.py               # Skrypt audytowy raportów grzyby.pl ([GRZ-03])
│   ├── fetch_weather_sample.py               # Skrypt pobierania próbki pogody z Open-Meteo z cache'em ([GRZ-04])
│   └── prepare_region_and_bdl_sample.py      # Skrypt przygotowania granic 30 powiatów i próbki wektorowej BDL ([GRZ-05])
├── notebooks/                                # Notatniki Jupyter i analizy eksploracyjne (EDA, Lag Effect)
│   └── README.md
├── models/                                   # Artefakty wytrenowanych modeli ML i metadane (binaria ignorowane w Git)
│   └── README.md
├── tests/                                    # Testy jednostkowe i dymne szkieletu aplikacji, GIS/BDL i kontraktu danych
│   └── test_smoke.py
├── docs/                                     # Dokumentacja projektowa, architektoniczna i raporty z audytów
│   ├── mvp.md                                # Zakres MVP zatwierdzony po audycie ([GRZ-01], [GRZ-06])
│   ├── data_contract.md                      # Kontrakt danych: areas, reports, weather, predictions ([GRZ-01], [GRZ-06])
│   ├── data_audit.md                         # Zbiorczy audyt danych, cel predykcji, odcięcie pogody i plan modelowania ([GRZ-06])
│   ├── reports_audit.md                      # Audyt struktury i kompletności raportów z grzyby.pl ([GRZ-03])
│   ├── weather_audit.md                      # Audyt zmiennych pogodowych i glebowych Open-Meteo ERA5-Land ([GRZ-04])
│   └── bdl_region_audit.md                   # Audyt granic regionu (PRG) i wektorowej próbki BDL ([GRZ-05])
└── data/                                     # Katalog danych
    ├── demo/                                 # Próbka demonstracyjna do budowy UI (is_demo = True, 30 powiatów)
    ├── sample/                               # Rzeczywiste próbki audytowe ze Sprintu 1 ([GRZ-03], [GRZ-04], [GRZ-05])
    ├── raw/                                  # Surowe dane (cache GIS/BDL, zrzuty HTML, cache API) — ignorowane w Git
    └── processed/                            # Produkcyjne pliki dla modelu i aplikacji (areas.geojson śledzony w Git)
```

---

## 4. Dokumentacja projektu

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
