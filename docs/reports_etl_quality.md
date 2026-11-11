# Raport jakości i potoku budowy zbioru obserwacji (`[GRZ-07]`)

> **Cel zadania (`[GRZ-07]`):** Zbudować kompletny, powtarzalny zbiór obserwacji o zbiorach grzybów zgodny z zatwierdzonym celem predykcji (`yield_pcs_per_h` oraz `abundance_class`) i kontraktem danych (`docs/data_contract.md`), ujednolicić daty, gatunki, jednostki i identyfikatory, wyeliminować duplikaty, zachować ślad pochodzenia oraz udokumentować lejek jakości po każdym etapie przetwarzania.

---

## 1. Artefakty wygenerowane w zadaniu `[GRZ-07]`

| Plik w repozytorium | Rola i zawartość | Liczba rekordów |
| :--- | :--- | :---: |
| [`scripts/build_reports_dataset.py`](../scripts/build_reports_dataset.py) | Powtarzalny skrypt ETL parsujący 9 tygodniowych zrzutów z `data/raw/*.html` (lub odtwarzający zbiór ze zanonimizowanego snapshotu `data/sample/reports_raw_518_ds.json`), wykonujący złączenie przestrzenne `gpd.sjoin` (`EPSG:2180`) oraz walidację jakości | — |
| [`data/processed/reports.parquet`](../data/processed/reports.parquet) | **Pełny analityczny zbiór obserwacji (`Parquet`)** zawierający wszystkie `518` wyekstrahowanych rekordów wraz ze śladem pochodzenia (`source_url`, `source_archive_week`, `place_id`, `lat`, `lon`), obiema datami (`harvest_date`, `publication_date`), wyliczeniami `yield_pcs_per_h` oraz flagami jakości (`quality_status`, `is_valid_for_training`, `quality_reason`) | **`518`** (`513` `VALID` + `5` wykluczonych) |
| [`data/processed/reports.csv`](../data/processed/reports.csv) | **Produkcyjny plik kontraktu danych (`CSV`)** konsumowany przez `src/data_loader.py` i potok ML (`is_demo = False`), zawierający wyłącznie rekordy zweryfikowane jako poprawne do treningu (`is_valid_for_training == True`) | **`513`** |
| [`data/processed/reports_quality_report.json`](../data/processed/reports_quality_report.json) | Maszynowy raport jakości z dokładnymi licznikami na każdym etapie lejka ETL, rozkładem zmiennej celu, pokryciem powiatów i listą wykluczonych rekordów | — |
| [`data/sample/reports_raw_518_ds.json`](../data/sample/reports_raw_518_ds.json) | Zanonimizowany, ustrukturyzowany snapshot wejściowy pozwalający na deterministyczne odtworzenie `reports.parquet` i `reports.csv` na maszynie bez dostępu do surowych zrzutów HTML w `data/raw/` | **`518`** |

### Odtworzenie zbioru jednym poleceniem:
```bash
PYTHONPATH=. python scripts/build_reports_dataset.py
```

---

## 2. Lejek jakości danych (Data Quality Funnel – liczba rekordów po każdym etapie)

Źródłem danych jest **9 pełnych tygodni szczytu sezonu jesiennego 2025** (`01.09.2025 – 02.11.2025`) dla województwa dolnośląskiego (`DS`), pobranych po zalogowaniu za pisemną zgodą autora portalu `grzyby.pl` (Marka Snowarskiego).

| Etap potoku ETL | Opis operacji i reguł walidacji | Liczba rekordów | Odrzucone / Oznaczone na etapie |
| :--- | :--- | :---: | :---: |
| **Etap 1: Surowa ekstrakcja z 9 tygodni archiwum (`DS`)** | Parsowanie 9 tygodniowych plików z `data/raw/` (`20250901-20250907` .. `20251027-20251102`): identyfikator `DGxxxxxx`, link GPS, `place_id`, daty, tagi łacińskie gatunków i treść | **`518`** | `0` |
| **Etap 2: Deduplikacja techniczna (`report_id`)** | Eliminacja duplikatów klucza głównego `report_id` na styku sąsiednich tygodniowych zrzutów archiwum | **`518`** | **`0` odrzuconych** (wszystkie `518` ID unikalne: `DG171104`..`DG177253`) |
| **Etap 3: Przypisanie przestrzenne (`GeoPandas sjoin` w `EPSG:2180`)** | Projekcja współrzędnych `(lon, lat)` z `EPSG:4326` do układu metrycznego `EPSG:2180`, złączenie `gpd.sjoin(..., predicate="within")` z poligonami 30 powiatów (`data/processed/areas.geojson`) i weryfikacja z etykietą tekstową | **`518`** | **`0` odrzuconych**, **`3` oznaczone i rozwiązane** (`VALID_BORDER_LOCALITY_RESOLVED`) |
| **Etap 4: Deduplikacja semantyczna (`author_id + harvest_date + place_id`)** | Wykrycie wielokrotnych raportów tego samego autora z tego samego dnia zbioru i tego samego punktu `place_id` | **`516`** | **`2` wykluczone** (`EXCLUDED_SAME_DAY_AUTHOR_PLACE_DUPLICATE`: `DG172071`, `DG174245`) |
| **Etap 5: Walidacja siedliska leśnego i wiarygodności `yield_pcs_per_h`** | Odfiltrowanie miejskich zbiorów trawnikowych poza lasem (`Coprinus comatus` w parkach Wrocławia) oraz niezweryfikowanych ekstremalnych wartości odstających (`> 200 szt./h` bez podanego czasu zbioru) | **`513`** | **`3` wykluczone** (`2` `EXCLUDED_URBAN_NON_FOREST`: `DG174153`, `DG173963` + `1` `EXCLUDED_EXTREME_UNVERIFIED_OUTLIER`: `DG175278`) |

---

## 3. Szczegółowy wykaz wpisów niejednoznacznych i wykluczonych (`[GRZ-07]` Warunek #4)

Wszystkie `518` rekordów (w tym `5` wykluczonych i `3` z rozwiązaną niejednoznacznością graniczną) jest zachowanych w pliku [`data/processed/reports.parquet`](../data/processed/reports.parquet) z jawnymi kolumnami `quality_status`, `is_valid_for_training` oraz `quality_reason`:

| `report_id` | Data zbioru | Lokalizacja (`location_raw`) | `area_id` | `portal_yield_pcs_per_h` | `quality_status` | `is_valid_for_training` | Przyczyna (`quality_reason`) |
| :---: | :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **`DG172071`** | `2025-09-18` | `Rędzin (Wrocław)` | `0264` | `2.0` | `EXCLUDED_SAME_DAY_AUTHOR_PLACE_DUPLICATE` | `False` | Duplikat tego samego autora (`Taka sytuacja`), dnia (`2025-09-18`) i miejsca (`place_id=5041`); zachowano główny raport **`DG172083`** (`15.0 szt./h`). |
| **`DG174245`** | `2025-10-01` | `Jeżów Sudecki pow. karkonoski` | `0206` | `30.0` | `EXCLUDED_SAME_DAY_AUTHOR_PLACE_DUPLICATE` | `False` | Drugi raport tego samego autora (`NorthDevil`) z tego samego dnia (`2025-10-01`) i miejsca (`place_id=8426`); zachowano raport **`DG174288`** z jawnie podanym czasem zbioru (`0.5 h`). |
| **`DG174153`** | `2025-09-30` | `Wrocław` | `0264` | `14.0` | `EXCLUDED_URBAN_NON_FOREST` | `False` | Zbiór czernidłaka kołpakowatego (`Coprinus comatus`) na miejskich trawnikach w Parku Klecińskim we Wrocławiu (poza siedliskami leśnymi BDL). |
| **`DG173963`** | `2025-09-28` | `Wrocław` | `0264` | `30.0` | `EXCLUDED_URBAN_NON_FOREST` | `False` | Zbiór wyłącznie `Coprinus comatus` w mieście (*„Polecam, gdy brak czasu na las”*) – wyłączony ze zbioru uczącego dla lasów. |
| **`DG175278`** | `2025-10-11` | `Kozłów pow. bolesławiecki` | `0201` | `250.0` | `EXCLUDED_EXTREME_UNVERIFIED_OUTLIER` | `False` | Ekstremalna wartość `250.0 szt./h` (*„Zielonki. Bardzo dużo zielonek!”*) bez podanego czasu zbierania ani liczby sztuk – wyłączona przed treningiem regresji. |
| **`DG172141`** | `2025-09-18` | `pow. wrocławski` | **`0223`** | `2.0` | `VALID_BORDER_LOCALITY_RESOLVED` | `True` | Ogólna etykieta `pow. wrocławski` ma w bazie `grzyby.pl` domyślny punkt w centrum Wrocławia (`0264`); nadpisano zgodnie z intencją autora na `0223` (`powiat wrocławski`). |
| **`DG172561`** | `2025-09-20` | `Wilkocin pow. polkowicki` | **`0201`** | `40.0` | `VALID_BORDER_LOCALITY_RESOLVED` | `True` | Kompleks leśny na granicy pow. polkowickiego (`0216`) i bolesławieckiego (`0201`); punkt GPS (`51.48949, 15.81727`) leży wewnątrz poligonu `0201` (`gpd.sjoin` w `EPSG:2180`). |
| **`DG171637`** | `2025-09-14` | `Konradów (Wałbrzych)` | **`0221`** | `5.0` | `VALID_BORDER_LOCALITY_RESOLVED` | `True` | Punkt GPS (`50.79506, 16.23441`) na pograniczu Wałbrzycha (`0265`) leży wewnątrz poligonu leśnego powiatu wałbrzyskiego (`0221`). |

---

## 4. Zasady obliczania i walidacji `yield_pcs_per_h` (`[GRZ-07]` Warunek #3)

Zgodnie z warunkiem **„Szt./h obliczane jest tylko z poprawnej liczby sztuk i dodatniego czasu zbioru”**:
1. W formularzu dodawania doniesienia na `grzyby.pl` wskaźnik `(X/h)` wyliczany jest z zadeklarowanej liczby grzybów oraz dodatniego czasu grzybobrania znormalizowanego na 1 osobę (`total_pieces / (harvest_duration_h * pickers_count)`).
2. Dodatkowo parser w [`scripts/build_reports_dataset.py`](../scripts/build_reports_dataset.py) wyodrębnił bezpośrednio z treści raportów:
   - jawny czas zbierania **`harvest_duration_h > 0`** dla **`116` raportów** (od `0.25 h` do `8.0 h`, **`0` wpisów z czasem $\le 0$**),
   - jawną liczbę zebranych owocników **`total_pieces >= 0`** dla **`185` raportów** (w tym `45` jawnych raportów zerowych `0 szt.` oraz zbiory dodatnie do `275 szt.`),
   - szacowane zaczerwienie **`worminess_pct`** dla **`20` raportów** (od `0.0%` do `90.0%`).
3. W przypadku raportu **`DG176081`** (`2025-10-18`, pow. kłodzki `0208`), gdzie nagłówek portalu zaokrąglił wynik w dół do `(0/h)`, ale autor jawnie podał w treści: *„Jeden borowik. Spacer dwugodzinny po wyższych terenach”* (`total_pieces = 1`, `harvest_duration_h = 2.0`), skrypt wyliczył dokładną dodatnią wydajność **`0.5 szt./h`** (`yield_calc_method = "computed_from_pieces_and_positive_duration_corrected_zero"`), dzięki czemu wpis nie jest błędnie traktowany jako pusty las (`0.0`).

---

## 5. Zakres dat, rozkład zmiennej celu i pokrycie regionu (`[GRZ-07]` Warunek #5)

### 5.1. Zakres dat (`N = 513` rekordów `VALID`)
- **Zakres dat grzybobrania (`harvest_date` / `date`):** od **`2025-09-01`** do **`2025-11-02`** (**`62` unikalne dni** na przestrzeni 9 tygodni).
- **Zakres dat publikacji (`publication_date`):** od **`2025-09-02`** do **`2025-11-03`** (średnie opóźnienie publikacji `pub_delay_days`: `0,24 dnia`, mediana: `0 dni`, maksimum: `5 dni`).

### 5.2. Rozkład zmiennej celu (`yield_pcs_per_h` oraz `abundance_class`)
- **Wartość minimalna:** `0.0 szt./h` (**`45` jawnych raportów o zerowym zbiorze** – kluczowe obserwacje negatywne z początku września przed opadami oraz z końca października po przymrozkach),
- **Mediana:** `15.0 szt./h`, **Średnia:** `18.58 szt./h`, **Wartość maksymalna (po odfiltrowaniu outliera):** `120.0 szt./h`.
- **Rozkład klas obfitości (`abundance_class`):**
  - `BRAK` (`< 5 szt./h`): **`112` raportów (`21,8%`)** *(w tym `45` raportów `0.0 szt./h` oraz `67` raportów `0.5–4.0 szt./h`)*,
  - `SREDNIO` (`5 – 20 szt./h`): **`239` raportów (`46,6%`)**,
  - `WYSYP` (`> 20 szt./h`): **`162` raporty (`31,6%`)**.

### 5.3. Rozkład znormalizowanych gatunków (`dominant_species`)
| Kategoria `dominant_species` | Liczba raportów (`VALID`) | Udział (`%`) |
| :--- | :---: | :---: |
| `"mieszane"` | `194` | `37,8%` |
| `"podgrzybek_brunatny"` | `114` | `22,2%` |
| `NULL` *(w tym 45 raportów zerowych + raporty ogólne bez tagu)* | `89` | `17,3%` |
| `"borowik_szlachetny"` | `78` | `15,2%` |
| `"kania"` | `17` | `3,3%` |
| `"maslak_zwyczajny"` | `11` | `2,1%` |
| `"kurka"` | `10` | `1,9%` |

### 5.4. Pokrycie przestrzenne regionu pilotażowego (`TERYT 02` – woj. dolnośląskie)
- **Pokryte powiaty:** **`28 z 30 powiatów (`93,33%`)** w **`189` unikalnych punktach leśnych (`place_id`)**.
- **Powiaty o największej liczbie obserwacji:**
  1. `0201` (powiat bolesławiecki – Bory Dolnośląskie): **`101` raportów** (śr. `23,24 szt./h`),
  2. `0214` (powiat oleśnicki – Lasy Twardogórskie / Sycowskie): **`59` raportów** (śr. `14,97 szt./h`),
  3. `0213` (powiat milicki – Dolina Baryczy): **`58` raportów** (śr. `26,52 szt./h`),
  4. `0220` (powiat trzebnicki – Wzgórza Trzebnickie): **`38` raportów** (śr. `10,24 szt./h`),
  5. `0208` (powiat kłodzki – Góry Stołowe / Bystrzyckie / Bialskie): **`24` raporty** (śr. `20,08 szt./h`).
- **Jedynie 2 powiaty bez raportów w badanym okresie (`6,67%`):**
  - `0217` (powiat strzeliński – powiat wybitnie rolniczy o niskiej lesistości),
  - `0262` (m. Legnica – powiat miejski o lesistości zaledwie `4,2%`).
