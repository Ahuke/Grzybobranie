# Przetwarzanie przestrzenne i tabela cech leśnych (`[GRZ-08]`)

> **Cel zadania (`[GRZ-08]`):** Połączyć obserwacje o zbiorach grzybów ze strukturą przestrzenną regionu (30 powiatów woj. dolnośląskiego, `TERYT 02`) bez nadawania im pozornej dokładności (eliminacja *false precision*), przygotować tabelę cech leśnych (`forest_features`) oraz warstwę obszarów (`areas`) z kluczem `region_id`, udokumentować sposoby agregacji i jednostki oraz zapewnić brak multiplikacji rekordów przy złączeniu przestrzennym.

---

## 1. Wygenerowane artefakty przestrzenne

| Plik w repozytorium | Format i klucz | Liczba wierszy | Opis i zawartość |
| :--- | :---: | :---: | :--- |
| [`data/processed/forest_features.parquet`](../data/processed/forest_features.parquet)<br/>[`data/processed/forest_features.csv`](../data/processed/forest_features.csv) | `Parquet` / `CSV`<br/>PK: **`region_id`** (alias: `area_id`) | **`30`** (wszystkie powiaty Dolnego Śląska) | Tabela cech leśnych zagregowanych na poziomie powiatu: lesistość (`%`), szacowana pow. lasu (`km²`), udział sosny, dębu/buka i świerka (`%`), średni wiek (`lata`), dominujące uwilgotnienie (`enum`) i metadane agregacji. |
| [`data/processed/reports_spatial.parquet`](../data/processed/reports_spatial.parquet)<br/>[`data/processed/reports_spatial.csv`](../data/processed/reports_spatial.csv) | `Parquet` / `CSV`<br/>PK: **`report_id`**<br/>FK: `region_id` | **`518`** (pełny zbiór)<br/>**`513`** (`VALID`) | Tabela obserwacji połączona z geometrią i cechami leśnymi powiatu. Zawiera jawną klasyfikację precyzji przestrzennej (`spatial_precision`) oraz metryczną odległość od centroidu powiatu (`EPSG:2180`). |
| [`data/processed/areas.geojson`](../data/processed/areas.geojson) | `GeoJSON` (`EPSG:4326`)<br/>PK: **`region_id`** (alias: `area_id`) | **`30`** cech (`Feature`) | Geometrie granic 30 powiatów PRG GUGiK z jawnym kluczem `region_id` w sekcji `properties`. |
| [`data/processed/spatial_quality_report.json`](../data/processed/spatial_quality_report.json) | `JSON` | — | Podsumowanie wskaźników jakościowych: rozkład precyzji, współczynnik multiplikacji, weryfikacja CRS i zliczenie lokalizacji granicznych. |
| [`scripts/build_spatial_features.py`](../scripts/build_spatial_features.py) | Skrypt Python (CLI) | — | Powtarzalny potok generujący powyższe zbiory w oparciu o GeoPandas. |

### Polecenie do odtworzenia potoku:
```bash
PYTHONPATH=. python scripts/build_spatial_features.py
```

---

## 2. Eliminacja pozornej dokładności (False Precision – Warunek #1)

W portalach społecznościowych takich jak `grzyby.pl` część użytkowników zgłasza grzybobranie podając dokładną miejscowość leśną (np. *„Krępnica pow. bolesławiecki”*), a część podaje jedynie sam powiat (np. *„pow. trzebnicki”*, *„pow. milicki”*). 

Portal przy zapisie przypisuje takiemu ogólnemu zgłoszeniu umowny punkt współrzędnych (często środek miasta powiatowego lub arbitralny punkt w regionie). 

> [!CAUTION]
> **Zagrożenie błędem pozornej dokładności (*False Precision*):**  
> Gdybyśmy wykonali naiwne złączenie punktowe (`gpd.sjoin`) takiego umownego punktu z warstwą mikro-wydzieleń leśnych BDL (`bdl_stands_sample_ds.geojson`), raport zostałby przypisany do losowego kawałka lasu, który akurat znalazł się pod arbitralnym punktem, co zafałszowałoby cechy siedliskowe modelu predykcyjnego.

### Zastosowane rozwiązanie architektoniczne:
Każdy rekord w [`data/processed/reports_spatial.parquet`](../data/processed/reports_spatial.parquet) posiada jednoznaczną klasyfikację precyzji:

1. **`spatial_precision = "point_precise"` (`484` raporty, `93,44%`):**
   - Raport wskazuje konkretną miejscowość, leśnictwo lub zweryfikowany punkt w lesie (`point_represents_specific_forest = True`).
   - Punkt reprezentuje rzeczywistą lokalizację eksploracji leśnej.
2. **`spatial_precision = "county_aggregate"` (`34` raporty, `6,56%`):**
   - Raport ma charakter ogólny, zgłoszony dla całego powiatu (`point_represents_specific_forest = False`).
   - Punkt współrzędnych traktowany jest wyłącznie jako znacznik pomocniczy powiatu.
   - **Właściwą geometrią odniesienia dla tego raportu jest poligon całego powiatu (`region_id`)**, a cechy środowiskowe przypisywane są z ważonej agregacji powiatowej (`forest_features`), bez odpytywania pojedynczych mikro-wydzieleń BDL.

---

## 3. Zliczenie i opis niejednoznacznych oraz granicznych lokalizacji (Warunek #2)

| Kategoria lokalizacji | Liczba rekordów | Udział | Sposób rozstrzygnięcia |
| :--- | :---: | :---: | :--- |
| **Precyzyjne lokalizacje leśne (`point_precise`)** | **`484`** | `93,44%` | Przypisanie do powiatu i geometrii na podstawie punktu `place_id` i współrzędnych GPS. |
| **Zgłoszenia ogólnopowiatowe (`county_aggregate`)** | **`34`** | `6,56%` | Powiązanie z poligonem powiatu `region_id` bez pozornej dokładności punktowej. |
| **Lokalizacje graniczne rozwiązane (`VALID_BORDER_LOCALITY_RESOLVED`)** | **`3`** | `0,58%` | • **`DG172141`** (`pow. wrocławski`): etykieta nadpisała domyślny miejski centroid Wrocławia (`0264`) na kod powiatu ziemskiego **`0223`**.<br/>• **`DG172561`** (`Wilkocin pow. polkowicki`): punkt leży wewnątrz poligonu Borów Dolnośląskich w powiecie bolesławieckim (**`0201`**) w `EPSG:2180`.<br/>• **`DG171637`** (`Konradów (Wałbrzych)`): punkt GPS leży w poligonie leśnym powiatu wałbrzyskiego (**`0221`**). |
| **Lokalizacje nieprzypisane (poza granicami regionu)** | **`0`** | `0,00%` | **100% obserwacji** mieści się w granicach administracyjnych 30 powiatów Dolnego Śląska (`0201`–`0265`). |

---

## 4. Tabela cech leśnych (`forest_features`) – sposób agregacji i jednostki (Warunek #3)

W plikach [`data/processed/forest_features.parquet`](../data/processed/forest_features.parquet) i [`.csv`](../data/processed/forest_features.csv) zestawiono charakterystyki siedliskowe dla wszystkich 30 jednostek powiatowych:

| Nazwa kolumny | Typ | Jednostka | Sposób wyznaczenia i źródło danych |
| :--- | :---: | :---: | :--- |
| **`region_id`** / **`area_id`** | `string` | 4 znaki TERYT | Identyfikator jednostki przestrzennej (`0201`–`0265`). |
| **`region_name`** | `string` | tekst | Oficjalna nazwa powiatu z Państwowego Rejestru Granic (PRG GUGiK). |
| **`total_area_km2`** | `float` | $\text{km}^2$ | Powierzchnia poligonu powiatu obliczona bezpośrednio z geometrii PRG w układzie metrycznym `EPSG:2180`. |
| **`forest_cover_pct`** | `float` | `%` (`0–100`) | Oficjalna lesistość powiatu według Banku Danych o Lasach / GUS BDL. |
| **`forest_area_km2`** | `float` | $\text{km}^2$ | Szacowana bezwzględna powierzchnia lasów w powiecie: $\text{total\_area\_km2} \times \frac{\text{forest\_cover\_pct}}{100}$. |
| **`pine_share_pct`** | `float` | `%` (`0–100`) | Udział powierzchniowy drzewostanów z panującą sosną zwyczajną (`SO`) ważony powierzchnią taksacyjną wydzieleń `D-STAN` w BDL. |
| **`oak_beech_share_pct`** | `float` | `%` (`0–100`) | Udział powierzchniowy drzewostanów z panującym dębem (`DB`) lub bukiem (`BK`) ważony powierzchnią `D-STAN` w BDL. |
| **`spruce_share_pct`** | `float` | `%` (`0–100`) | Udział powierzchniowy drzewostanów ze świerkiem pospolitym (`ŚW`) ważony powierzchnią `D-STAN` w BDL. |
| **`mean_stand_age_years`** | `float` | lata | Średni ważony wiek drzewostanów w zarządzie Lasów Państwowych: $\frac{\sum (\text{species\_age} \times \text{sub\_area})}{\sum \text{sub\_area}}$. |
| **`dominant_moisture_type`** | `string` | enum | Dominująca kategoria uwilgotnienia siedliska wyznaczona z typów siedliskowych lasu (`TSL`): `"suche"`, `"swieze"`, `"wilgotne"`, `"bagienne"`. |
| **`has_bdl_stand_data`** | `bool` | `true`/`false` | Flaga wskazująca, czy powiat posiada wyliczony profil drzewostanowy BDL (`true` dla 18 powiatów), czy bazuje na ogólnej lesistości GUS (`false` dla 12 powiatów rolniczych/miejskich). |
| **`aggregation_method`** | `string` | tekst | Metoda obliczeń: `"bdl_stand_weighted_mean_epsg2180"` lub `"gus_bdl_official_forestry_prg"`. |

---

## 5. Zgodność układów współrzędnych (CRS – Warunek #4)

W potoku przestrzennym ściśle rozdzielono dwa układy współrzędnych:

1. **`EPSG:2180` (Układ współrzędnych płaskich prostokątnych PUWG 1992):**
   - Jednostka: **metr (`[m]`)**.
   - Wszystkie geometryczne obliczenia powierzchni (`total_area_km2`), centroidów oraz metrycznych odległości obserwacji od środka powiatu (`distance_to_county_centroid_km`) wykonywane są w układzie `EPSG:2180`.
2. **`EPSG:4326` (WGS84 / OGC:CRS84):**
   - Jednostka: **stopnie dziesiętne (`[°E, °N]`)**.
   - Wykorzystywany do zapisu i wymiany plików `GeoJSON` ([`data/processed/areas.geojson`](../data/processed/areas.geojson)) oraz wyświetlania mapy w bibliotece PyDeck (`GeoJsonLayer`).

---

## 6. Niezmiennik braku multiplikacji rekordów (Warunek #5)

W analizach GIS z użyciem `gpd.sjoin` częstym błędem jest przecięcie punktu z wieloma zachodzącymi na siebie poligonami (relacja $1:N$), co sztucznie mnoży liczbę wierszy i zafałszowuje model uczenia maszynowego.

W zadaniu `[GRZ-08]` zagwarantowano ścisły niezmiennik multiplikacji:
- **Liczba obserwacji wejściowych z `[GRZ-07]`:** **`518`**
- **Liczba obserwacji w tabeli przestrzennej `reports_spatial`:** **`518`**
- **Współczynnik multiplikacji:** **`1,0000` (dokładnie $1:1$)**
- **Liczba rekordów zakwalifikowanych do treningu (`is_valid_for_training == True`):** **`513`** (dokładnie tyle samo co w `reports.csv`).

Połączenie z cechami leśnymi następuje po unikalnym kluczu jednostki administracyjnej **`region_id`** (1 wiersz na powiat), co wyklucza jakąkolwiek duplikację obserwacji.
