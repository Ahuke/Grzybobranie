# Kontrakt Danych (Data Contract) – „Gdzie na grzyby?”

> **Status dokumentu:** **Wersja 2.0 (Zatwierdzona i zamrożona po audycie danych `[GRZ-06] #6`)**  
> Niniejszy dokument definiuje zatwierdzony przez obie osoby w zespole interfejs wymiany danych pomiędzy **Modułem Danych & ML (Osoba A)** a **Modułem Aplikacji Streamlit (Osoba B)**.  
> Wszystkie punkty badawcze ze Sprintu 1 (`[GRZ-03]`, `[GRZ-04]`, `[GRZ-05]`) zostały zweryfikowane i udokumentowane w [`docs/data_audit.md`](data_audit.md).

---

## 1. Konwencje ogólne

### 1.1. Lokalizacja i nazewnictwo plików
Wszystkie dane dzielą się na dwa całkowicie odseparowane środowiska:

| Środowisko | Katalog | Przeznaczenie | Czy trafia do treningu ML? |
| :--- | :--- | :--- | :--- |
| **Demonstracyjne (UI Mock)** | `data/demo/` | Budowa i testowanie interfejsu Streamlit przed ukończeniem scrapingu i modelu ML | **NIE (Zakaz!)** – każdy wiersz ma `is_demo = True` |
| **Produkcyjne / Badawcze** | `data/processed/` | Rzeczywiste dane przetworzone przez potok GeoPandas oraz wyjście wytrenowanego modelu / wskaźnika | **TAK** – każdy wiersz ma `is_demo = False` |

Nazwy plików w obu katalogach:
1. **Obszary i siedliska (`areas`):**
   - Demo: `data/demo/areas_demo.geojson`
   - Prod: `data/processed/areas.geojson`
2. **Raporty grzybiarzy (`reports`):**
   - Demo: `data/demo/reports_demo.csv`
   - Prod: `data/processed/reports.csv`
3. **Dane pogodowe (`weather`):**
   - Demo: `data/demo/weather_demo.csv`
   - Prod: `data/processed/weather.csv`
4. **Predykcje modelu (`predictions`):**
   - Demo: `data/demo/predictions_demo.csv`
   - Prod: `data/processed/predictions.csv`

### 1.2. Identyfikator obszaru (`area_id`)
- Podstawową jednostką przestrzenną w MVP jest **powiat** (30 jednostek w woj. dolnośląskim: `0201`–`0226` oraz miasta na prawach powiatu `0261`, `0262`, `0264`, `0265`).
- Klucz `area_id` to **4-znakowy tekstowy kod TERYT powiatu (`JPT_KOD_JE`)** z zachowaniem wiodącego zera, np.:
  - `"0201"` – powiat bolesławiecki,
  - `"0208"` – powiat kłodzki,
  - `"0264"` – powiat m. Wrocław.
- **Uwaga przy wczytywaniu w Pandas:** Należy zawsze wymuszać typ tekstowy (`dtype={"area_id": str}`), aby Pandas nie obciął wiodącego zera (`"0201"` $\rightarrow$ `201`).

### 1.3. Format dat, strefa czasowa i moment odcięcia (Point-in-Time Cutoff)
- Wszystkie daty zapisywane są w standardzie **ISO 8601 (`YYYY-MM-DD`)**, np. `"2026-10-31"`.
- Strefa czasowa odniesienia dla agregacji dobowych (`00:00–23:59`): **`Europe/Warsaw`**.
- **Reguła odcięcia czasowego (`[GRZ-06]`):**
  - Dla prognozy wystawianej w chwili $t_0$ na dzień docelowy $t \in [t_0,\; t_0 + 7]$, wszystkie skumulowane okna opadowe i temperaturowe (`precip_lag_7d_mm`, `precip_lag_14d_mm`, `precip_lag_21d_mm`, `temp_mean_7d_c`) są liczone wyłącznie z zamkniętego przedziału **$[t-k,\; t-1]$** (dni poprzedzających dzień grzybobrania $t$).
  - Rzeczywista pogoda z archiwum (`is_forecast = False`) może pochodzić wyłącznie z dni $\tau \le t_0 - 1$. Dla dni $\tau \in [t_0,\; t-1]$ wolno używać wyłącznie prognozy meteorologicznej dostępnej w chwili $t_0$ (`is_forecast = True`). **Przyszła rzeczywista pogoda nigdy nie trafia do cech prognozy.**

### 1.4. Kodowanie znaków i układ współrzędnych (CRS)
- Kodowanie plików: **`UTF-8`** (bez BOM).
- Separator kolumn w plikach CSV: przecinek (`,`), separator dziesiętny: kropka (`.`).
- Układ współrzędnych (CRS) dla plików `GeoJSON` przekazywanych do aplikacji: **`EPSG:4326` (WGS84 – długość/szerokość geograficzna)**, wymagany przez PyDeck. Obliczenia powierzchniowe i złączenia przestrzenne (`gpd.sjoin`) w GeoPandas wewnątrz potoku danych wykonujemy w państwowym układzie metrycznym **`EPSG:2180` (`PUWG 1992`)**.

### 1.5. Znaczenie i obsługa braków danych (`NULL` vs `0.0`)
- W plikach CSV brak wartości jest reprezentowany wyłącznie jako **pusty ciąg między przecinkami** (np. `12.4,,0.21`), w `GeoJSON` jako `null`.
- **Zabronione** jest stosowanie wartości zastępczych typu `-999`, `-1`, `"brak"`, `"N/A"` czy `"unknown"`.
- **Semantyka `0.0` vs `NULL` (`[GRZ-06]`):**
  - `0.0` oznacza **potwierdzony pomiar zerowy** (np. `precip_mm_1d = 0.0` $\rightarrow$ w danym dniu nie padało; `yield_pcs_per_h = 0.0` $\rightarrow$ grzybiarz był w lesie i opublikował jawny raport zerowy `(0/h)`).
  - `NULL` (`NaN` / brak wiersza w `reports.csv`) oznacza **brak informacji / brak obserwacji**. **Brak raportu w danym powiecie i dniu nigdy nie jest traktowany jako zerowy zbiór (`0.0`).**

---

## 2. Specyfikacja plików (Zatwierdzona w `[GRZ-06]`)

### 2.1. Plik `areas` (`areas.geojson` / `areas_demo.geojson`)
Zawiera rzeczywiste granice 30 powiatów woj. dolnośląskiego (GUGiK PRG) oraz cechy siedlisk leśnych zagregowane z Banku Danych o Lasach (BDL, dla wydzieleń drzewostanowych `area_type_cd == 'D-STAN'`) i GUS BDL.

- **Klucz główny (PK):** `area_id`
- **Format:** `GeoJSON` (`FeatureCollection`, `CRS: EPSG:4326`)

| Pole (w `properties`) | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Unikalny identyfikator powiatu (`"0201"`–`"0265"`). |
| `area_name` | `string` | tekst | NIE | Nazwa powiatu (np. `"powiat bolesławiecki"`, `"powiat karkonoski"`). |
| `voivodeship_id` | `string` | kod TERYT (2 znaki) | NIE | Kod województwa (`"02"` dla dolnośląskiego). |
| `centroid_lon` | `float` | stopnie (`°E`) | NIE | Długość geograficzna środka ciężkości powiatu (wyliczonego w `EPSG:2180`). |
| `centroid_lat` | `float` | stopnie (`°N`) | NIE | Szerokość geograficzna środka ciężkości powiatu. |
| `total_area_km2` | `float` | $\text{km}^2$ | NIE | Całkowita powierzchnia powiatu wyliczona w `EPSG:2180`. |
| `forest_cover_pct` | `float` | `%` (`0–100`) | TAK | Lesistość powiatu (udział powierzchni leśnej w powierzchni całkowitej). |
| `pine_share_pct` | `float` | `%` (`0–100`) | TAK | Udział powierzchniowy drzewostanów (`D-STAN`) z panującą **sosną (`SO`)**. |
| `oak_beech_share_pct` | `float` | `%` (`0–100`) | TAK | Udział powierzchniowy drzewostanów (`D-STAN`) z panującym **dębem (`DB`) lub bukiem (`BK`)**. |
| `spruce_share_pct` | `float` | `%` (`0–100`) | TAK | Udział powierzchniowy drzewostanów (`D-STAN`) z panującym **świerkiem (`ŚW`)**. |
| `mean_stand_age_years`| `float` | lata | TAK | Średni ważony powierzchniowo wiek drzewostanów (`D-STAN`) z BDL. |
| `dominant_moisture_type`| `string`| kategoria BDL | TAK | Dominujący stopień uwilgotnienia siedliska z kodu `TSL`: `"suche"`, `"swieze"`, `"wilgotne"`, `"bagienne"`. |
| `is_demo` | `bool` | `true`/`false` | NIE | Flaga bezpieczeństwa (`true` dla danych demonstracyjnych). |
| `geometry` | `Polygon` / `MultiPolygon` | `EPSG:4326` | NIE | Rzeczywista geometria poligonu powiatu. |

*(Pola opcjonalne metadanych w `properties`: `has_bdl_stand_data`, `bdl_sample_stands_count`, `data_source_boundaries`, `data_source_bdl`, `data_date`, `crs`).*

---

### 2.2. Plik `reports` (`reports.csv` / `reports_demo.csv`)
Zawiera zeskrapowane doniesienia grzybiarzy z portalu `grzyby.pl` przypisane do powiatów.

- **Klucz główny (PK):** `report_id`
- **Klucz obcy (FK):** `area_id` $\rightarrow$ `areas.area_id`

| Kolumna | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `report_id` | `string` | tekst | NIE | Unikalny identyfikator raportu (np. `"DG174888"`, w demo prefiks `"DEMO_REP_..."`). |
| `date` | `string` | `YYYY-MM-DD` | NIE | **Data faktycznego grzybobrania (`harvest_date`)** – nigdy data spóźnionej publikacji! |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Kod powiatu przypisany po `place_id` / `(lat, lon)`. |
| `location_raw` | `string` | tekst | TAK | Opis miejsca z raportu (np. `"Krępnica pow. bolesławiecki"`). |
| `dominant_species` | `string` | słownik gatunków | TAK | Główny zgłaszany gatunek grzyba: `"borowik_szlachetny"`, `"podgrzybek_brunatny"`, `"kurka"`, `"maslak_zwyczajny"`, `"kania"`, `"mieszane"`. `NULL` gdy nie określono. |
| `yield_pcs_per_h` | `float` | `szt./h` | NIE | **Główna zmienna docelowa (Target):** znormalizowana wydajność zbioru z nagłówka `(X/h)` ($\ge 0.0$, 100% wypełnienia). |
| `harvest_duration_h` | `float` | godziny (`h`) | TAK | Czas spędzony w lesie (dostępny w ok. 44% raportów). |
| `worminess_pct` | `float` | `%` (`0–100`) | TAK | Opcjonalny odsetek grzybów zaczerwionych (`NULL` w >95% wpisów, wyłączony z cech/celu ML). |
| `is_demo` | `bool` | `True`/`False` | NIE | `True` dla próbki UI, `False` dla rzeczywistego scrapingu. |

---

### 2.3. Plik `weather` (`weather.csv` / `weather_demo.csv`)
Zawiera dobowe wskaźniki meteorologiczne i glebowe (model reanalizy **ECMWF ERA5-Land**, oczko siatki ~9 km, doba `00:00–23:59 Europe/Warsaw`) oraz cechy opóźnione (`[t-k, t-1]`) dla każdego powiatu.

- **Klucz główny złożony (PK):** `(date, area_id)`
- **Klucz obcy (FK):** `area_id` $\rightarrow$ `areas.area_id`

| Kolumna | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `string` | `YYYY-MM-DD` | NIE | Data doby meteorologicznej w strefie `Europe/Warsaw`. |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Identyfikator powiatu. |
| `precip_mm_1d` | `float` | `mm` ($\text{l/m}^2$) | NIE | Dobowa suma opadów w danym dniu ($\ge 0.0$). *(Nie wchodzi do cech `lag_k` z tego samego dnia!)* |
| `precip_lag_7d_mm` | `float` | `mm` | TAK *(gdy brak pełnego okna)* | Skumulowana suma opadów z pełnych 7 dni poprzedzających (`[t-7, t-1]`). |
| `precip_lag_14d_mm` | `float` | `mm` | TAK *(gdy brak pełnego okna)* | Skumulowana suma opadów z pełnych 14 dni poprzedzających (`[t-14, t-1]`). |
| `precip_lag_21d_mm` | `float` | `mm` | TAK *(gdy brak pełnego okna)* | Skumulowana suma opadów z pełnych 21 dni poprzedzających (`[t-21, t-1]`). |
| `temp_min_night_c` | `float` | `°C` | NIE | Minimalna temperatura nocno-poranna z doby (wykrywa przymrozki hamujące grzybnię). |
| `temp_mean_7d_c` | `float` | `°C` | TAK | Średnia temperatura dobowa z ostatnich 7 dni (`[t-7, t-1]`). |
| `soil_moisture_0_7cm_m3m3` | `float` | $\text{m}^3/\text{m}^3$ | TAK | Modelowana objętościowa wilgotność gleby w warstwie `0–7 cm` (średnia dobowa z danych godzinowych ERA5-Land / IFS). |
| `is_forecast` | `bool` | `True`/`False` | NIE | `False` dla danych historycznych (znanych przed chwilą wystawienia prognozy $t_0$), `True` dla prognozy NWP. |
| `is_demo` | `bool` | `True`/`False` | NIE | Flaga danych demonstracyjnych. |

---

### 2.4. Plik `predictions` (`predictions.csv` / `predictions_demo.csv`)
Główny plik wynikowy generowany przez potok predykcyjny i konsumowany przez aplikację Streamlit do kolorowania mapy oraz rekomendacji powiatów.

- **Klucz główny złożony (PK):** `(target_date, area_id, species_filter)`
- **Klucz obcy (FK):** `area_id` $\rightarrow$ `areas.area_id`

| Kolumna | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `target_date` | `string` | `YYYY-MM-DD` | NIE | Data docelowa prognozy grzybobrania ($t \in [t_0,\; t_0+7]$). |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Identyfikator powiatu (`"0201"`–`"0265"`). |
| `species_filter` | `string` | słownik gatunków | NIE | Filtr gatunku w UI: `"all"` (główny cel modelu) lub gatunek ważony siedliskiem BDL (`"borowik_szlachetny"`, `"podgrzybek_brunatny"`, `"kurka"`, `"maslak_zwyczajny"`, `"kania"`). |
| `predicted_yield_pcs_per_h`| `float` | `szt./h` | NIE | **Główny wynik liczbowy:** prognozowana wydajność zbioru w sztukach na godzinę ($\ge 0.0$). |
| `abundance_class` | `string` | enum | NIE | Kategoria obfitości wysypu: `"BRAK"` (`< 5.0`), `"SREDNIO"` (`5.0 – 20.0`), `"WYSYP"` (`> 20.0`). |
| `confidence_score` | `float` | `0.0 – 1.0` | TAK | Wskaźnik pewności predykcji / dopasowania siedliska leśnego BDL do wybranego gatunku. |
| `habitat_match_note` | `string` | tekst | TAK | Krótki opis wyjaśniający czynniki pogodowo-siedliskowe dla UI. |
| `generated_at` | `string` | `YYYY-MM-DD` | NIE | Data wystawienia prognozy ($t_0$ – moment odcięcia danych historycznych). |
| `model_version` | `string` | tekst | NIE | Identyfikator modelu / wskaźnika (w próbce demo zawsze `"DEMO_MOCK_v0"`). |
| `is_demo` | `bool` | `True`/`False` | NIE | Flaga bezpieczeństwa (`True` dla próbki interfejsu). |

---

## 3. Słownik wartości kontrolowanych (Zatwierdzony w `[GRZ-06]`)

### 3.1. Gatunki grzybów (`species_filter` / `dominant_species`)
| Kod w plikach danych | Etykieta w interfejsie Streamlit | Preferowany drzewostan (BDL) |
| :--- | :--- | :--- |
| `"all"` | Ogółem (wszystkie gatunki) | Wszystkie typy lasu (`forest_cover_pct`) |
| `"borowik_szlachetny"` | Borowik szlachetny | Lasy dębowe, bukowe, mieszane i górskie bory świerkowe (`oak_beech_share_pct`, `spruce_share_pct`) |
| `"podgrzybek_brunatny"` | Podgrzybek brunatny | Bory sosnowe i świerkowe na siedliskach świeżych i wilgotnych (`pine_share_pct`, `spruce_share_pct`) |
| `"kurka"` | Pieprznik jadalny (kurka) | Bory sosnowe i mieszane na siedliskach świeżych (`pine_share_pct`) |
| `"maslak_zwyczajny"` | Maślak zwyczajny | Młodniki sosnowe i obrzeża borów sosnowych (`pine_share_pct`, niższy `mean_stand_age_years`) |
| `"kania"` | Czubajka kania | Obrzeża lasów mieszanych i liściastych (`oak_beech_share_pct`, `pine_share_pct`) |

---

## 4. Zatwierdzenie kontraktu po audycie (`[GRZ-06] #6`)

- **Osoba A (Moduł Danych, GIS & ML):** Kontrakt zatwierdzony (`2026-11-09`).
- **Osoba B (Moduł Aplikacji Streamlit & Wizualizacji):** Kontrakt zatwierdzony (`2026-11-09`).
