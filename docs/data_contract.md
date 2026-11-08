# Kontrakt Danych (Data Contract) – „Gdzie na grzyby?”

> **Status dokumentu:** Wersja 1.0  
> Niniejszy dokument definiuje interfejs wymiany danych pomiędzy **Modułem Danych & ML** a **Modułem Aplikacji Streamlit**.  
> Pola oznaczone tagiem **`[DO WERYFIKACJI]`** wymagają potwierdzenia po pobraniu pierwszej surowej paczki z BDL / `grzyby.pl`.

---

## 1. Konwencje ogólne

### 1.1. Lokalizacja i nazewnictwo plików
Wszystkie dane dzielą się na dwa całkowicie odseparowane środowiska:

| Środowisko | Katalog | Przeznaczenie | Czy trafia do treningu ML? |
| :--- | :--- | :--- | :--- |
| **Demonstracyjne (UI Mock)** | `data/demo/` | Budowa i testowanie interfejsu Streamlit przed ukończeniem scrapingu i modelu ML | **NIE (Zakaz!)** – każdy wiersz ma `is_demo = True` |
| **Produkcyjne / Badawcze** | `data/processed/` | Rzeczywiste dane przetworzone przez potok GeoPandas oraz wyjście wytrenowanego modelu | **TAK** – każdy wiersz ma `is_demo = False` |

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
- Podstawową jednostką przestrzenną w MVP jest **powiat**.
- Klucz `area_id` to **4-znakowy tekstowy kod TERYT powiatu (JPT_KOD_JE)** z zachowaniem wiodącego zera, np.:
  - `"0201"` – powiat bolesławiecki,
  - `"0208"` – powiat kłodzki,
  - `"0264"` – m. Wrocław.
- **Uwaga przy wczytywaniu w Pandas:** Należy zawsze wymuszać typ tekstowy (`dtype={"area_id": str}`), aby Pandas nie obciął wiodącego zera (`"0201"` $\rightarrow$ `201`).

### 1.3. Format dat i czasu
- Wszystkie daty zapisywane są w standardzie **ISO 8601 (`YYYY-MM-DD`)**, np. `"2026-10-24"`.
- Strefa czasowa odniesienia dla agregacji dobowych opadów i temperatur nocnych: **`Europe/Warsaw`**.

### 1.4. Kodowanie znaków i separatory
- Kodowanie plików: **`UTF-8`** (bez BOM).
- Separator kolumn w plikach CSV: przecinek (`,`), separator dziesiętny: kropka (`.`).
- Układ współrzędnych (CRS) dla plików `GeoJSON` przekazywanych do aplikacji: **`EPSG:4326` (WGS84 – długość/szerokość geograficzna)**, wymagany przez PyDeck / Folium. *(Obliczenia powierzchniowe w GeoPandas wewnątrz potoku danych wykonujemy w układzie metrycznym `EPSG:2180`).*

### 1.5. Znaczenie i obsługa braków danych (`NULL` / `NaN`)
- W plikach CSV brak wartości jest reprezentowany wyłącznie jako **pusty ciąg między przecinkami** (np. `12.4,,0.21`), w `GeoJSON` jako `null`.
- **Zabronione** jest stosowanie wartości zastępczych typu `-999`, `-1`, `"brak"`, `"N/A"` czy `"unknown"`.
- **Semantyka `0.0` vs `NULL`:**
  - `0.0` oznacza **potwierdzony pomiar zerowy** (np. `precip_mm_1d = 0.0` $\rightarrow$ w danym dniu nie padało; `yield_pcs_per_h = 0.0` $\rightarrow$ grzybiarz był w lesie i nic nie znalazł).
  - `NULL` (`NaN`) oznacza **brak informacji / brak pomiaru** (np. w danym dniu nikt nie dodał raportu z powiatu, lub w raporcie nie podano dominującego gatunku).

---

## 2. Specyfikacja plików

### 2.1. Plik `areas` (`areas.geojson` / `areas_demo.geojson`)
Zawiera granice powiatów oraz statyczne (wolnozmienne) cechy siedlisk leśnych zagregowane z Banku Danych o Lasach (BDL) za pomocą GeoPandas.

- **Klucz główny (PK):** `area_id`
- **Format:** `GeoJSON` (`FeatureCollection`, `CRS: EPSG:4326`)

| Pole (w `properties`) | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Unikalny identyfikator powiatu (np. `"0201"`). |
| `area_name` | `string` | tekst | NIE | Nazwa powiatu (np. `"powiat bolesławiecki"`). |
| `voivodeship_id` | `string` | kod TERYT (2 znaki) | NIE | Kod województwa (np. `"02"` dla dolnośląskiego). |
| `centroid_lon` | `float` | stopnie (`°E`) | NIE | Długość geograficzna centroidu leśnego powiatu (do obliczeń odległości i zapytań pogodowych). |
| `centroid_lat` | `float` | stopnie (`°N`) | NIE | Szerokość geograficzna centroidu leśnego powiatu. |
| `total_area_km2` | `float` | $\text{km}^2$ | NIE | Całkowita powierzchnia powiatu. |
| `forest_cover_pct` | `float` | `%` (`0–100`) | NIE | Lesistość powiatu (udział powierzchni leśnej w powierzchni całkowitej). |
| `pine_share_pct` | `float` | `%` (`0–100`) | TAK `[DO WERYFIKACJI]` | Udział drzewostanów z panującą **sosną** (bory sosnowe – podgrzybek, maślak, kurka). |
| `oak_beech_share_pct` | `float` | `%` (`0–100`) | TAK `[DO WERYFIKACJI]` | Udział drzewostanów liściastych z panującym **dębem lub bukiem** (borowik szlachetny). |
| `spruce_share_pct` | `float` | `%` (`0–100`) | TAK `[DO WERYFIKACJI]` | Udział drzewostanów z panującym **świerkiem** (borowik górski, podgrzybek). |
| `mean_stand_age_years`| `float` | lata | TAK `[DO WERYFIKACJI]` | Średni ważony powierzchniowo wiek drzewostanu z BDL. |
| `dominant_moisture_type`| `string`| kategoria BDL | TAK `[DO WERYFIKACJI]` | Dominujący stopień uwilgotnienia siedliska wg BDL: `"suche"`, `"swieze"`, `"wilgotne"`, `"bagienne"`. |
| `is_demo` | `bool` | `true`/`false` | NIE | Flaga bezpieczeństwa (`true` dla danych demonstracyjnych). |
| `geometry` | `Polygon` / `MultiPolygon` | `EPSG:4326` | NIE | Geometria poligonu powiatu. |

---

### 2.2. Plik `reports` (`reports.csv` / `reports_demo.csv`)
Zawiera zeskrapowane i zgeokodowane pojedyncze doniesienia grzybiarzy z portalu `grzyby.pl` (Ground Truth).

- **Klucz główny (PK):** `report_id`
- **Klucz obcy (FK):** `area_id` $\rightarrow$ `areas.area_id`

| Kolumna | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `report_id` | `string` | tekst | NIE | Unikalny identyfikator raportu (w demo prefiks `"DEMO_REP_..."`). |
| `date` | `string` | `YYYY-MM-DD` | NIE | Data grzybobrania (lub publikacji raportu, jeśli brak daty wyjścia). |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Przypisany przez GeoPandas kod powiatu. |
| `location_raw` | `string` | tekst | TAK | Oryginalny opis miejsca z raportu (np. `"okolice Świętoszowa, Bory Dolnośląskie"`). |
| `dominant_species` | `string` | słownik gatunków | TAK `[DO WERYFIKACJI]` | Główny zgłaszany gatunek grzyba: `"borowik_szlachetny"`, `"podgrzybek_brunatny"`, `"kurka"`, `"maslak_zwyczajny"`, `"kania"`, `"mieszane"`. `NULL` gdy nie określono. |
| `yield_pcs_per_h` | `float` | `szt./h` | NIE | **Kluczowa zmienna celu (Target):** deklarowana wydajność grzybobrania w sztukach na godzinę ($\ge 0.0$). |
| `harvest_duration_h` | `float` | godziny (`h`) | TAK | Czas spędzony w lesie (jeśli podany w raporcie). |
| `worminess_pct` | `float` | `%` (`0–100`) | TAK `[DO WERYFIKACJI]` | Szacowany odsetek grzybów zaczerwionych („robaczywych”), jeśli raportowany. |
| `is_demo` | `bool` | `True`/`False` | NIE | `True` dla próbki UI, `False` dla rzeczywistego scrapingu. |

---

### 2.3. Plik `weather` (`weather.csv` / `weather_demo.csv`)
Zawiera dzienne wskaźniki meteorologiczne oraz przeliczone cechy opóźnione (Lag Features) dla każdego powiatu.

- **Klucz główny złożony (PK):** `(date, area_id)`
- **Klucz obcy (FK):** `area_id` $\rightarrow$ `areas.area_id`

| Kolumna | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `string` | `YYYY-MM-DD` | NIE | Data pomiaru lub prognozy meteorologicznej. |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Identyfikator powiatu. |
| `precip_mm_1d` | `float` | `mm` ($\text{l/m}^2$) | NIE | Dobowa suma opadów w danym dniu ($\ge 0.0$). |
| `precip_lag_7d_mm` | `float` | `mm` | NIE | Skumulowana suma opadów z ostatnich 7 dni (`[t-7, t-1]`). |
| `precip_lag_14d_mm` | `float` | `mm` | NIE | Skumulowana suma opadów z ostatnich 14 dni (`[t-14, t-1]`). |
| `precip_lag_21d_mm` | `float` | `mm` | NIE | Skumulowana suma opadów z ostatnich 21 dni (`[t-21, t-1]`). |
| `temp_min_night_c` | `float` | `°C` | NIE | Minimalna temperatura nocna/poranna z doby (kluczowy wyzwalacz dla grzybni). |
| `temp_mean_7d_c` | `float` | `°C` | TAK | Średnia temperatura dobowa z ostatnich 7 dni. |
| `soil_moisture_0_7cm_m3m3` | `float` | $\text{m}^3/\text{m}^3$ | TAK `[DO WERYFIKACJI]` | Objętościowa wilgotność gleby w warstwie płytkiej `0–7 cm` z Open-Meteo (typowo `0.05 – 0.45`). |
| `is_forecast` | `bool` | `True`/`False` | NIE | `False` dla danych historycznych (obserwacji), `True` dla prognozy pogodowej na kolejne dni. |
| `is_demo` | `bool` | `True`/`False` | NIE | Flaga danych demonstracyjnych. |

---

### 2.4. Plik `predictions` (`predictions.csv` / `predictions_demo.csv`)
Główny plik wynikowy generowany przez potok ML i konsumowany przez aplikację Streamlit do kolorowania mapy oraz rekomendacji tras.

- **Klucz główny złożony (PK):** `(target_date, area_id, species_filter)`
- **Klucz obcy (FK):** `area_id` $\rightarrow$ `areas.area_id`

| Kolumna | Typ | Jednostka | Czy NULL? | Opis |
| :--- | :--- | :--- | :--- | :--- |
| `target_date` | `string` | `YYYY-MM-DD` | NIE | Data, której dotyczy prognoza grzybobrania (np. sobota nadchodzącego weekendu). |
| `area_id` | `string` | kod TERYT (4 znaki) | NIE | Identyfikator powiatu. |
| `species_filter` | `string` | słownik gatunków | NIE | Filtr gatunku użyty w UI: `"all"` (ogółem) lub konkretny gatunek: `"borowik_szlachetny"`, `"podgrzybek_brunatny"`, `"kurka"`, `"maslak_zwyczajny"`. |
| `predicted_yield_pcs_per_h`| `float` | `szt./h` | NIE | Prognozowana przez model liczba grzybów na godzinę ($\ge 0.0$, zaokrąglona do 1 miejsca po przecinku). |
| `abundance_class` | `string` | enum | NIE | Kategoria wysypu: `"BRAK"` (`< 5`), `"SREDNIO"` (`5–20`), `"WYSYP"` (`> 20`) `[DO WERYFIKACJI]`. |
| `confidence_score` | `float` | `0.0 – 1.0` | TAK | Wskaźnik pewności predykcji / dopasowania siedliska leśnego do gatunku. |
| `habitat_match_note` | `string` | tekst | TAK | Krótki opis wyjaśniający dla UI (np. `"Wysoki udział borów sosnowych (82%) + opad 38 mm z 14 dni"`). |
| `generated_at` | `string` | `YYYY-MM-DD` | NIE | Data wygenerowania predykcji. |
| `model_version` | `string` | tekst | NIE | Identyfikator modelu (w próbce demo zawsze `"DEMO_MOCK_v0"`). |
| `is_demo` | `bool` | `True`/`False` | NIE | Flaga bezpieczeństwa (`True` dla próbki interfejsu). |

---

## 3. Słownik wartości kontrolowanych (Enums) `[DO WERYFIKACJI]`

### 3.1. Gatunki grzybów (`species_filter` / `dominant_species`)
| Kod w plikach danych | Etykieta w interfejsie Streamlit | Preferowany drzewostan (BDL) |
| :--- | :--- | :--- |
| `"all"` | Wszystkie gatunki (Ogółem) | Wszystkie typy lasu |
| `"borowik_szlachetny"` | Borowik szlachetny (Prawdziwek) | Lasy dębowe, bukowe, mieszane i świerkowe (`oak_beech_share_pct`, `spruce_share_pct`) |
| `"podgrzybek_brunatny"` | Podgrzybek brunatny | Bory sosnowe i świerkowe (`pine_share_pct`, `spruce_share_pct`) |
| `"kurka"` | Pieprznik jadalny (Kurka) | Bory sosnowe i mieszane na siedliskach świeżych (`pine_share_pct`) |
| `"maslak_zwyczajny"` | Maślak zwyczajny | Młodniki sosnowe (`pine_share_pct`, niski `mean_stand_age_years`) |
| `"kania"` | Czubajka kania | Obrzeża lasów mieszanych i liściastych |

---

## 4. Pola wymagające weryfikacji (`[DO WERYFIKACJI]`)

1. **Atrybuty BDL w `areas.geojson`:**
   - Po pobraniu paczki BDL trzeba potwierdzić, czy liczymy udział gatunków panujących powierzchniowo (z pola `pow_wydz` i `gat_pan` w tabeli taksacyjnej BDL), czy bierzemy jedynie typ siedliskowy lasu (`TSL`, np. `Bśw` – bór świeży, `LMśw` – las mieszany świeży, `Lśw` – las świeży). *Propozycja:* wyliczyć zarówno udział głównych gatunków (`pine_share_pct`, `oak_beech_share_pct`, `spruce_share_pct`), jak i `dominant_moisture_type`.
2. **Wyciąganie gatunków z `grzyby.pl` (`dominant_species`):**
   - W tabeli zbiorczej raportów na `grzyby.pl` zawsze dostępny jest `yield_pcs_per_h`, natomiast informacja o gatunkach bywa zawarta w tekście doniesienia. Jeśli w części raportów nie uda się wyodrębnić gatunku, pole `dominant_species` przyjmie wartość `NULL` (co jest obsłużone w kontrakcie i pozwala trenować model dla `"all"`, a specyficzne filtry gatunkowe ważyć cechami siedliska BDL).
3. **Dostępność `soil_moisture_0_7cm_m3m3` w Open-Meteo Historical API:**
   - Open-Meteo udostępnia wilgotność gleby z reanalizy ERA5-Land (`soil_moisture_0_to_7cm`). Należy zweryfikować kompletność szeregu dla wszystkich lat historycznych; w razie pojedynczych luk dopuszczamy `NULL` i imputację w potoku cech.
