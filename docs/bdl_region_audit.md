# Audyt granic regionu pilotażowego i próbki danych leśnych BDL (`[GRZ-05] #5`)

> **Cel zadania:** Potwierdzić dostęp do rzeczywistych granic uzgodnionych jednostek przestrzennych (30 powiatów woj. dolnośląskiego, `TERYT 02`) oraz wektorowej próbki danych siedliskowych i drzewostanowych z Banku Danych o Lasach (BDL), zweryfikować przydatność analityczną danych wektorowych w GeoPandas (w opozycji do usług wyłącznie rastrowych), udokumentować źródła, daty, identyfikatory, atrybuty i układy współrzędnych oraz wyświetlić region i próbkę w aplikacji Streamlit z gwarancją działania także przy braku danych BDL.

---

## 1. Źródła danych, daty aktualności i przygotowane pliki

| Zbiór danych | Źródło danych | Data aktualności / pobrania | Układ obliczeń (metryczny) | Układ zapisu i wizualizacji | Zapisane pliki w repozytorium |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Granice jednostek przestrzennych (`powiaty`, `TERYT 02`)** | **Państwowy Rejestr Granic (GUGiK PRG)** – warstwa `A02_Granice_powiatow` (`JPT_KOD_JE 02xx`, zasób zgeneralizowany topologicznie `polska-geojson` + wskaźniki lesistości GUS BDL) | Stan PRG/TERYT: **2026** (pobrano: **`2026-11-08`**) | **`EPSG:2180`** *(PUWG 1992)* | **`EPSG:4326`** *(WGS84 / CRS84)* | • [`data/processed/areas.geojson`](../data/processed/areas.geojson) *(uzgodniony plik kontraktu)*<br/>• [`data/sample/region_powiaty_02_ds.geojson`](../data/sample/region_powiaty_02_ds.geojson)<br/>• [`data/demo/areas_demo.geojson`](../data/demo/areas_demo.geojson) *(30 powiatów, `is_demo=True`)* |
| **Próbka wektorowa wydzieleń leśnych BDL (`N = 105` poligonów)** | **Bank Danych o Lasach (BDL – Lasy Państwowe / BULiGL)** – usługa ArcGIS REST Feature Layer `WMS_BDL/MapServer/5` (*Wydzielenia PGL LP* / odpowiednik warstwy `G_SUBAREA` w paczkach nadleśnictw RDLP Wrocław `13`) | Rok stanu w BDL: **`a_year = 2026`** (pobrano: **`2026-11-08`**) | **`EPSG:2180`** *(PUWG 1992)* | **`EPSG:4326`** *(WGS84 / CRS84)* | • [`data/sample/bdl_stands_sample_ds.geojson`](../data/sample/bdl_stands_sample_ds.geojson)<br/>• [`data/sample/bdl_stands_sample_ds.csv`](../data/sample/bdl_stands_sample_ds.csv)<br/>• [`data/sample/bdl_powiaty_agg_sample_ds.csv`](../data/sample/bdl_powiaty_agg_sample_ds.csv) |

- **Powtarzalny skrypt generujący (z obsługą lokalnego cache’u w `data/raw/gis_cache/`):**  
  [`scripts/prepare_region_and_bdl_sample.py`](../scripts/prepare_region_and_bdl_sample.py)

---

## 2. Weryfikacja: Dane wektorowe BDL (do analiz GeoPandas) vs. Usługi rastrowe (tylko do wyświetlenia)

Jednym z kluczowych warunków zadania `[GRZ-05]` było rozstrzygnięcie, które zasoby Banku Danych o Lasach (`bdl.lasy.gov.pl`) nadają się do **przestrzennych analiz i agregacji w GeoPandas**, a które służą wyłącznie do wyświetlania podkładu mapowego:

| Typ usługi / zasobu BDL | Endpoint / format | Co zwraca? | Czy nadaje się do analiz w GeoPandas (`sjoin` / `overlay`)? | Rola w projekcie „Gdzie na grzyby?” |
| :--- | :--- | :--- | :---: | :--- |
| **1. Usługi przeglądowe OGC WMS i kafelkowe WMTS** | `.../WMS_BDL_mapa_drzewostanow/MapServer/WMSServer` oraz `.../WMTS` | Gotowe rastry (`image/png`) pokolorowane wg wieku lub siedliska | **NIE** – wyłącznie obraz rastrowy do wyświetlenia w tle (`GetMap`) lub punktowego kliknięcia (`GetFeatureInfo`). | Nieprzydatne do wyliczania cech tabelarycznych powiatów dla modelu ML. |
| **2. Usługa wektorowa ArcGIS REST Feature Query (`MapServer/<id>/query`) oraz OGC WFS** | `https://mapserver.bdl.lasy.gov.pl/ArcGIS/rest/services/WMS_BDL/MapServer/5/query?f=geojson&outSR=4326` oraz `wfs.bdl.lasy.gov.pl/geoserver/BDL/ows` | **Pełne geometrie poligonowe (`Polygon` / `MultiPolygon`) + tablica atrybutów taksacyjnych** w formacie `GeoJSON` / `GML` | **TAK (Zweryfikowano!)** – pozwala na operacje `gpd.sjoin`, liczenie powierzchni w `EPSG:2180` i ważoną agregację gatunków/siedlisk. | **Wykorzystane do pobrania próbki 105 wydzieleń BDL (`[GRZ-05]`)** oraz szybkiego próbkowania wokół miejsc grzybobrania. |
| **3. Paczki wektorowe BDL dla nadleśnictw (Udostępnianie danych BDL)** | Paczki ZIP (`SHP` / `GML`) dla nadleśnictw RDLP Wrocław (`13`): warstwa `G_SUBAREA.shp` + tabele `f_subarea.txt`, `f_arod_stand.txt`, `f_arod_species.txt` | **Kompletna siatka poligonów wydzieleń nadleśnictwa (`EPSG:2180`)** połączona kluczem `arodes_int_num` z pełnym opisem taksacyjnym PUL | **TAK** – ten sam model danych i te same identyfikatory (`arodes_int_num`, `adress_forest`, `site_type_cd`, `species_cd`) co w wektorowej usłudze `MapServer/5`. | Docelowy wsad hurtowy do offline’owego przeliczenia pełnych statystyk dla całych nadleśnictw. |

### Dowód przydatności analitycznej na pobranej próbce wektorowej (`N = 105` poligonów BDL):
W skrypcie [`scripts/prepare_region_and_bdl_sample.py`](../scripts/prepare_region_and_bdl_sample.py) wykonano pełny potok analityczny w **GeoPandas**:
1. **Złączenie przestrzenne (`gpd.sjoin` w `EPSG:2180`):** Wszystkie `105` pobranych poligonów wydzieleń BDL przypisano automatycznie (`predicate="within"` dla punktów reprezentatywnych) do `7` poligonów powiatów woj. dolnośląskiego (**`0` obiektów nieprzypisanych**).
2. **Zgodność powierzchni geometrycznej (`EPSG:2180`) z powierzchnią taksacyjną (`sub_area`):** Wyliczona z geometrii poligonów powierzchnia w układzie `EPSG:2180` (`geom_area_ha_epsg2180`) pokrywa się z oficjalną powierzchnią ewidencyjną wydzielenia w BDL (`sub_area_ha`) z dokładnością do ułamków hektara (np. wydzielenie `subarea_id=1979341`: taksacyjna `11,61 ha` vs. geometryczna `11,73 ha`; wydzielenie `1980455`: `4,41 ha` vs. `4,40 ha`).
3. **Kluczowe odkrycie z audytu atrybutów (`area_type_cd`):** W warstwie wektorowej wydzieleń BDL (`G_SUBAREA` / `Wydzielenia PGL LP`) oprócz właściwych drzewostanów (**`area_type_cd = 'D-STAN'`** – w próbce `74` z `105` poligonów, łączna pow. `224,96 ha`) znajdują się również wydzielenia nieleśne i pomocnicze w zarządzie PGL LP (`31` poligonów, `19,78 ha`): linie energetyczne (`L ENERG`), grunty rolne i łąki (`R`, `Ł`, `PS`), potoki i urządzenia wodne (`POTOK`, `URZ WOD`, `STAW R-Ś`), zręby (`ZRĄB`), składy drewna (`SKŁAD DR`) oraz tereny przemysłowe (`T PRZEM`), dla których pola gatunku panującego (`species_cd_d`) i wieku (`species_age`) mają wartość **`NULL` (`None`)**.  
   $\rightarrow$ **Wniosek dla potoku ML:** Przed wyliczeniem średniego wieku lasu i udziału gatunków panujących w powiecie **trzeba zawsze filtrować `area_type_cd == 'D-STAN'`**!

---

## 3. Opis dostępnych identyfikatorów i atrybutów

### 3.1. Granice powiatów (`region_powiaty_02_ds.geojson` / `areas.geojson`, `N = 30`)
Województwo dolnośląskie (`voivodeship_id = "02"`) obejmuje dokładnie **30 powiatów** (**26 powiatów ziemskich**: `0201`–`0226` oraz **4 miasta na prawach powiatu**: `0261` Jelenia Góra, `0262` Legnica, `0264` Wrocław, `0265` Wałbrzych) o łącznej powierzchni geometrycznej **`19 941,7 km²`** (oficjalna pow. GUS: `19 946,7 km²`, zgodność `99,97%`).

| Atrybut w pliku `areas` | Typ | Jednostka | Czy może być `NULL`? | Opis i pochodzenie |
| :--- | :---: | :---: | :---: | :--- |
| **`area_id`** | `string` | 4 znaki TERYT | **NIE** | Klucz główny jednostki przestrzennej (`JPT_KOD_JE` z PRG, zawsze z wiodącym zerem, np. `"0201"` – pow. bolesławiecki, `"0206"` – pow. karkonoski, `"0264"` – m. Wrocław). |
| **`area_name`** | `string` | tekst | **NIE** | Oficjalna nazwa powiatu (z uwzględnieniem zmiany nazwy `powiat jeleniogórski` $\rightarrow$ `powiat karkonoski` dla kodu `0206`). |
| **`voivodeship_id`** | `string` | 2 znaki TERYT | **NIE** | Kod województwa: `"02"` (dolnośląskie). |
| **`centroid_lon`**, **`centroid_lat`**| `float` | `°E`, `°N` (`EPSG:4326`) | **NIE** | Współrzędne środka ciężkości powiatu wyliczone w układzie metrycznym `EPSG:2180` i przetransformowane do `EPSG:4326`. |
| **`total_area_km2`** | `float` | $\text{km}^2$ | **NIE** | Powierzchnia poligonu powiatu wyliczona w układzie `EPSG:2180` (od `50,8 km²` dla m. Legnica `0262` do `1640,9 km²` dla pow. kłodzkiego `0208`). |
| **`forest_cover_pct`** | `float` | `%` (`0–100`) | TAK *(w razie braku danych)* | Lesistość powiatu wg GUS BDL / RDLP Wrocław (od `4,2%` w m. Legnica `0262` i `7,6%` w m. Wrocław `0264` do `60,8%` w pow. zgorzeleckim `0225` i `61,4%` w pow. bolesławieckim `0201`). |
| **`pine_share_pct`** | `float` | `%` (`0–100`) | **TAK (`NULL`)** | Udział powierzchniowy drzewostanów z panującą **sosną (`SO`)** wg BDL. |
| **`oak_beech_share_pct`** | `float` | `%` (`0–100`) | **TAK (`NULL`)** | Udział powierzchniowy drzewostanów z panującym **dębem (`DB`) lub bukiem (`BK`)** wg BDL. |
| **`spruce_share_pct`** | `float` | `%` (`0–100`) | **TAK (`NULL`)** | Udział powierzchniowy drzewostanów z panującym **świerkiem (`ŚW`)** wg BDL. |
| **`mean_stand_age_years`** | `float` | lata | **TAK (`NULL`)** | Średni ważony powierzchniowo wiek drzewostanów (`D-STAN`) w powiecie wg BDL. |
| **`dominant_moisture_type`**| `string` | enum kontraktu | **TAK (`NULL`)** | Dominujący stopień uwilgotnienia siedlisk leśnych wyprowadzony z kodu `TSL`: `"suche"`, `"swieze"`, `"wilgotne"`, `"bagienne"`. |
| **`has_bdl_stand_data`** | `bool` | `true`/`false` | **NIE** | Flaga informująca, czy dany powiat posiada wczytany szczegółowy profil drzewostanowy BDL (`true`), czy wyłącznie granice administracyjne PRG i ogólną lesistość (`false`). |
| **`bdl_sample_stands_count`**| `int` | liczba poligonów | **NIE** | Liczba wydzieleń wektorowych z próbki `bdl_stands_sample_ds.geojson` przypisanych przestrzennie (`sjoin`) do powiatu. |

---

### 3.2. Wektorowe wydzielenia leśne BDL (`bdl_stands_sample_ds.geojson` / `.csv`, `N = 105`)
Próbka pochodzi z warstwy wektorowej **`WMS_BDL/MapServer/5` (`Wydzielenia PGL LP`)**, której schemat odpowiada bezpośrednio warstwie `G_SUBAREA` połączonej z tabelą opisową `f_subarea` w paczkach danych BDL:

| Atrybut oryginalny w BDL | Atrybut w naszej próbce | Typ | Opis techniczny i znaczenie domenowe |
| :--- | :--- | :---: | :--- |
| **`subarea_id`** | `subarea_id` | `int` | Unikalny identyfikator geometryczny poligonu wydzielenia na serwerze mapowym BDL (np. `1979341`). |
| **`arodes_int_num`** | `arodes_int_num` | `int` | **Klucz relacyjny SILP / BDL** (np. `1303022540`), za pomocą którego w paczkach nadleśnictw łączy się warstwę poligonową `G_SUBAREA.shp` z tabelami taksacyjnymi `f_subarea.txt`, `f_arod_stand.txt` oraz `f_arod_species.txt`. |
| **`adress_forest`** | `adress_forest` | `string` | **Pełny adres leśny** w standardzie Lasów Państwowych: `RR-NN-O-LL-ODD -POD -00` (np. `13-03-3-05-240 -d -00`, gdzie `13` = RDLP Wrocław, `03` = Nadleśnictwo Bolesławiec, `3` = obręb leśny, `05` = leśnictwo, `240` = oddział, `d` = pododdział/wydzielenie). |
| **`area_type_cd`** | `area_type_cd` / `is_forest_stand` | `string` / `bool` | Kod rodzaju powierzchni: **`D-STAN`** (drzewostan – zalesione wydzielenie leśne) vs. powierzchnie nieleśne/pomocnicze (`SUKCESJA`, `ZRĄB`, `L ENERG`, `POTOK`, `URZ WOD`, `PARK`, `R`, `Ł`, `T PRZEM`, `SKŁAD DR`). |
| **`site_type_cd`** | `site_type_cd` / `moisture_category` | `string` | **Kod Typu Siedliskowego Lasu (TSL):**<br/>• Bory nizinne i górskie: `BŚW` (bór świeży), `BMŚW` (bór mieszany świeży), `BMGŚW` (bór mieszany górski świeży),<br/>• Lasy mieszane i liściaste: `LMŚW` (las mieszany świeży), `LMW` (las mieszany wilgotny), `LŚW` (las świeży), `LW` (las wilgotny), `LMGŚW` (las mieszany górski świeży), `LGŚW` (las górski świeży), `LGW` (las górski wilgotny).<br/>Z końcówki kodu TSL jednoznacznie wynika kategoria uwilgotnienia: `*S` $\rightarrow$ `"suche"`, `*ŚW` $\rightarrow$ `"swieze"`, `*W` $\rightarrow$ `"wilgotne"`, `*B`/`OL*` $\rightarrow$ `"bagienne"`. |
| **`species_cd_d`** | `species_cd_d` / `species_group` | `string` | **Kod gatunku panującego w drzewostanie:**<br/>• `SO` – sosna zwyczajna ($\rightarrow$ `pine_share_pct`),<br/>• `ŚW` – świerk pospolity ($\rightarrow$ `spruce_share_pct`),<br/>• `BK` – buk zwyczajny, `DB` – dąb ($\rightarrow$ `oak_beech_share_pct`),<br/>• pozostałe w próbce: `OL` (olsza czarna), `JW` (klon jawor), `MD` (modrzew). Dla wydzieleń niebędących drzewostanem (`area_type_cd != 'D-STAN'`) przyjmuje wartość `NULL`. |
| **`part_cd`** | `part_cd` | `int` (`1–10`) | Udział gatunku panującego w pierwszym piętrze drzewostanu wyrażony w dziesiątych częściach (`10` = `100%` lity drzewostan, `8` = `80%`, `6` = `60%`). |
| **`species_age`** | `species_age` | `int` `[lata]` | Wiek gatunku panującego w latach (w próbce od `3` lat w uprawach leśnych do `145` lat w starodrzewach świerkowych i bukowych w Sudetach). |
| **`rotation_age`** | `rotation_age` | `int` `[lata]` | Wiek rębności drzewostanu wg Planu Urządzenia Lasu (typowo `100–120` lat). |
| **`sub_area`** | `sub_area_ha` | `float` `[ha]` | Powierzchnia ewidencyjna/taksacyjna wydzielenia w hektarach z bazy BDL. |
| *(geometria)* | `geom_area_ha_epsg2180` | `float` `[ha]` | Powierzchnia poligonu wyliczona bezpośrednio z geometrii wektorowej w GeoPandas po reprojekcji do `EPSG:2180`. |
| **`stand_struct_cd`** | `stand_struct_cd` | `string` | Budowa pionowa drzewostanu (np. `DRZEW` – drzewostan jednopiętrowy/zwarty, `KO` – klasa odnowienia, `KDO` – klasa do odnowienia). |
| **`forest_func_cd`** / **`prot_category_cd`** | `forest_func_cd` / `prot_category_cd` | `string` | Funkcja lasu (`GOSP` – gospodarczy, `OCHR` – ochronny) oraz kategoria ochronności (`OCH WOD` – wodochronne, `OCH GLEB` – glebochronne, `OCH MIAST` – wokół miast, `OCH USZK` – uszkodzone przez przemysł, `OCH CENNE` – cenne przyrodniczo). |
| **`a_year`** | `a_year` | `int` | Rok aktualności danych w Banku Danych o Lasach (`2026`). |

*(Dodatkowo zinwentaryzowano warstwy tematyczne `WMS_BDL_mapa_drzewostanow/MapServer/11` – zawierającą `age_class`, `storey_cd` i `gat_grp` – oraz `WMS_BDL_mapa_siedlisk/MapServer/7` – zawierającą podtyp gleby `soil_subtype_cd`, utwór geologiczny `geol_depos` i uwilgotnienie `moisture_cd`; pełny zrzut metadanych wszystkich warstw zapisano podczas audytu).*

---

## 4. Układy współrzędnych (CRS) i zasady konwersji

1. **Państwowy układ współrzędnych płaskich prostokątnych `EPSG:2180` (`ETRF2000-PL / CS92`, dawniej `PUWG 1992`):**
   - Natywny układ metryczny (jednostka: **metr `[m]`**) dla Państwowego Rejestru Granic (GUGiK PRG), paczek wektorowych BDL (`G_SUBAREA.shp`) oraz usługi `WMS_BDL_mapa_siedlisk`.
   - **Zasada w projekcie:** Wszystkie operacje geometryczne wrażliwe na pole powierzchni i odległość w GeoPandas (`geometry.area`, `geometry.centroid`, `geometry.representative_point()`, `gpd.sjoin`) wykonujemy **wyłącznie po transformacji do `EPSG:2180`** (`gdf.to_crs("EPSG:2180")`).
2. **Układ geograficzny `EPSG:4326` (`WGS84` / `OGC:CRS84` – długość i szerokość geograficzna w stopniach `[lon, lat]`):**
   - Uzgodniony w [`docs/data_contract.md`](data_contract.md) układ zapisu plików `GeoJSON` (`areas.geojson`, `areas_demo.geojson`, `region_powiaty_02_ds.geojson`, `bdl_stands_sample_ds.geojson`), wymagany bezpośrednio przez bibliotekę **PyDeck (`GeoJsonLayer`)** w aplikacji Streamlit.

---

## 5. Wyniki agregacji próbki wektorowej BDL w 7 klastrach leśnych i walidacja z raportami `[GRZ-03]`

Poniższa tabela prezentuje wynik działania potoku GeoPandas ([`data/sample/bdl_powiaty_agg_sample_ds.csv`](../data/sample/bdl_powiaty_agg_sample_ds.csv)) na `105` pobranych poligonach wydzieleń BDL w 7 powiatach Dolnego Śląska:

| Kod powiatu (`area_id`) | Klaster leśny i Nadleśnictwo (RDLP Wrocław `13`) | Liczba poligonów BDL (`D-STAN` / ogółem) | Pow. drzewostanów `D-STAN` `[ha]` | Udział sosny `SO` (`%`) | Udział dębu i buka `DB+BK` (`%`) | Udział świerka `ŚW` (`%`) | Średni ważony wiek `[lat]` | Dominujący TSL (uwilgotnienie) | Spójność z raportami grzybiarzy z `[GRZ-03] #3` |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`0201`** | **Bory Dolnośląskie (Krępnica)** – *Nadl. Bolesławiec (`13-03`)* | `11 / 15` | `44,68 ha` | **100,0%** | `0,0%` | `0,0%` | `66,6` | **`BŚW`** (`swieze`) | Idealna zgodność: monokultury sosnowe na piaskach, gdzie raportowano zbiory podgrzybka brunatnego i maślaka (`DG174563`, `DG174714`). |
| **`0202`** | **Góry Sowie** – *Nadl. Świdnica (`13-26`)* | `11 / 15` | `31,01 ha` | `0,0%` | **21,0%** | **78,8%** | `82,1` | **`LMGŚW`** (`swieze`) | Zgodność z raportem `DG174762` (*„wycieczka w Góry Sowie: 48 prawdziwków, 47 podgrzybków”* – las świerkowo-bukowy regla dolnego). |
| **`0206`** | **Karkonosze** – *Nadl. Szklarska Poręba (`13-24`)* | `7 / 15` | `21,77 ha` | `0,0%` | `9,3%` | **90,7%** | `117,4` | **`LMGŚW`** (`swieze`) | Starodrzewy świerkowe z domieszką buka i jaworu (`JW`) w strefie górskiej. |
| **`0208`** | **Góry Stołowe (Batorów)** – *Nadl. Zdroje (`13-07`)* | `15 / 15` | `43,17 ha` | `0,0%` | `11,4%` | **85,1%** | `77,2` | **`LMGŚW`** (`swieze`) | Zgodność z raportem `DG174435` (*„Las świerkowy”* w pow. kłodzkim). |
| **`0213`** | **Dolina Baryczy (Łazy Wielkie)** – *Nadl. Milicz (`13-18`)* | `11 / 15` | `13,97 ha` | **69,2%** | **26,6%** | `0,0%` | `84,4` | **`LMŚW`** (`swieze`) | Zgodność z raportem `DG174602`: nizinne lasy sosnowo-dębowe z udziałem olszy (`OL`) wokół stawów milickich (podgrzybek, borowik, koźlarz). |
| **`0214`** | **Lasy Twardogórskie (Białe Błoto)** – *Nadl. Oleśnica Śl. (`13-09`)* | `12 / 15` | `39,63 ha` | **91,7%** | `3,1%` | `2,4%` | `54,8` | **`LMŚW`** (`swieze`) | Zgodność z raportem `DG174448` (*„w twardogórskich lasach: maślaki 40, podgrzybki 30, borowiki 12”* – przewaga borów i lasów mieszanych sosnowych). |
| **`0221`** | **Masyw Trójgarbu (Witków)** – *Nadl. Wałbrzych (`13-28`) i Kamienna Góra (`13-12`)* | `7 / 15` | `30,71 ha` | `0,0%` | **53,4%** | **46,6%** | `42,6` | **`LMGŚW`** (`swieze`) | **Bezpośrednie potwierdzenie raportu `DG174833`** (*„masyw Trójgarbu w pobliżu wioski Witków... las bukowy, prawdziwki”*): w BDL dominują tu drzewostany bukowe (`BK 53,4%`) i świerkowe (`ŚW 46,6%`)! |

---

## 6. Odporność aplikacji na brak danych BDL (Warunek ukończenia `#5`)

Zgodnie z wymogiem **„Brak BDL nie blokuje pokazania samego regionu i oceny dalszego zakresu”**, w kodzie i strukturze plików wdrożono pełną separację granic administracyjnych od opcjonalnych atrybutów siedliskowych BDL:
1. **W plikach `areas.geojson` / `areas_demo.geojson` / `region_powiaty_02_ds.geojson`:**
   - Wszystkie **30 powiatów woj. dolnośląskiego (`0201`–`0265`)** posiadają kompletne, rzeczywiste geometrie poligonowe (`EPSG:4326`), powierzchnię (`total_area_km2`), współrzędne centroidu (`centroid_lat`, `centroid_lon`) oraz ogólną lesistość (`forest_cover_pct`).
   - Dla `11` powiatów o mniejszym priorytecie leśnym (np. `0203`, `0204`, `0205`, `0209`, `0217`, `0218`, `0223`, `0224`, `0261`, `0262`, `0265`) oraz dla miejskiego powiatu `0264` (m. Wrocław) szczegółowe atrybuty taksacyjne BDL (`pine_share_pct`, `oak_beech_share_pct`, `spruce_share_pct`, `mean_stand_age_years`, `dominant_moisture_type`) mają jawną wartość **`null` (`None`)**.
2. **W module wczytywania i wzbogacania (`src/data_loader.py`, `src/geo_utils.py`) oraz interfejsie (`app.py`):**
   - Powiaty z wartościami `NULL` w polach BDL są poprawnie wczytywane, uwzględniane w obliczeniach odległości `haversine_km`, wyświetlane w tabeli oraz rysowane na mapie PyDeck z czytelną etykietą `NULL (brak BDL)`.
   - Nawet w przypadku całkowitego usunięcia lub wyłączenia warstwy `data/sample/bdl_stands_sample_ds.geojson` aplikacja wyświetla komunikat informacyjny i bezbłędnie prezentuje mapę 30 powiatów regionu pilotażowego (zweryfikowano testem automatycznym w `tests/test_smoke.py`).
