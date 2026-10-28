# Gdzie na grzyby?

**Przestrzenny system predykcji wysypu grzybów w oparciu o raporty zbieraczy (`grzyby.pl`), siedliska leśne (BDL) i meteorologię (Open-Meteo)**

## Dokumentacja etapu 1 (Ustalenia MVP & Kontrakt Danych)

- [**Zakres MVP i definicja wyniku aplikacji (`docs/mvp.md`)**](docs/mvp.md) – wybrany region pilotażowy (woj. dolnośląskie, jednostka: powiat `TERYT`), podział prac, rzeczy odłożone na później oraz lista kontrolna `[DO WERYFIKACJI]` dla drugiej osoby z zespołu.
- [**Kontrakt Danych (`docs/data_contract.md`)**](docs/data_contract.md) – specyfikacja plików `areas`, `reports`, `weather` i `predictions`, klucze (`area_id`), format dat (`YYYY-MM-DD`), jednostki oraz zasady obsługi braków danych (`NULL` vs `0.0`).
- [**Próbka demonstracyjna dla interfejsu (`data/demo/`)**](data/demo/README.md) – jawnie oznaczone pliki mock (`is_demo = True`), przeznaczone wyłącznie do równoległej budowy aplikacji Streamlit (wykluczone ze zbioru treningowego ML).

## Struktura repozytorium

```text
├── docs/
│   ├── mvp.md                  # Zakres MVP, definicja wyniku, lista [DO WERYFIKACJI]
│   └── data_contract.md        # Kontrakt danych: areas, reports, weather, predictions
├── data/
│   ├── demo/                   # Próbka demonstracyjna do budowy UI (is_demo = True)
│   │   ├── README.md
│   │   ├── areas_demo.geojson
│   │   ├── reports_demo.csv
│   │   ├── weather_demo.csv
│   │   └── predictions_demo.csv
│   ├── raw/                    # Surowe dane (BDL, scraping, cache API) - ignorowane w Git
│   └── processed/              # Produkcyjne pliki dla modelu i aplikacji (is_demo = False)
└── README.md
```
