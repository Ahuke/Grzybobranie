# Zakres MVP – „Gdzie na grzyby?”

> **Status dokumentu:** Wersja 1.0 (Propozycja do przeglądu zespołowego)  
> Pola i decyzje oznaczone tagiem **`[DO WERYFIKACJI]`** wymagają ostatecznego potwierdzenia przez drugą osobę z zespołu przed zamrożeniem kontraktu danych.

---

## 1. Cel pierwszej wersji (MVP)

Celem wersji MVP jest dostarczenie działającego end-to-end prototypu systemu predykcji wysypu grzybów dla **jednego wybranego regionu w Polsce**, łączącego:
1. **Historyczne raporty zbieraczy** (portal `grzyby.pl`),
2. **Charakterystykę siedlisk leśnych** (Bank Danych o Lasach – BDL),
3. **Dane meteorologiczne z uwzględnieniem efektu opóźnienia (Lag Effect)** (Open-Meteo API),

oraz prezentującego wyniki w interaktywnej aplikacji **Streamlit**.

Podział pracy w zespole opiera się na sztywnym kontrakcie danych (`docs/data_contract.md`), dzięki czemu:
- **Moduł Danych & ML (Osoba A)** może niezależnie budować scraper, potok GeoPandas i model regresji/klasyfikacji,
- **Moduł Aplikacji & Wizualizacji (Osoba B)** może równolegle budować interfejs Streamlit na jawnie oznaczonej próbce demonstracyjnej (`data/demo/`).

---

## 2. Wybór regionu i jednostki przestrzennej

### 2.1. Region pilotażowy `[DO WERYFIKACJI]`
- **Wybrany region:** **Województwo dolnośląskie** (kod TERYT województwa: `"02"`).
- **Uzasadnienie wyboru:**
  - Duże zróżnicowanie siedlisk leśnych: od rozległych monokultur sosnowych na północy (Bory Dolnośląskie – idealne pod podgrzybki i maślaki) po lasy mieszane, bukowe i świerkowe na południu (Sudety i Przedgórze Sudeckie – idealne pod borowika szlachetnego).
  - Wysoka aktywność użytkowników na portalu `grzyby.pl` (duża liczba raportów historycznych).
  - *Alternatywa rezerwowa:* Województwo mazowieckie (TERYT `"14"`), jeśli zespół uzna je za wygodniejsze.

### 2.2. Podstawowa jednostka przestrzenna (Spatial Unit)
- **Jednostka w MVP:** **Powiat** (w tym miasta na prawach powiatu), identyfikowany **4-znakowym kodem TERYT** (`area_id`, np. `"0201"` dla powiatu bolesławieckiego).
- **Rola GeoPandas na poziomie powiatu:**
  - Raporty z `grzyby.pl` (często posiadające jedynie nazwę miejscowości, gminy lub powiatu) są geokodowane i przypisywane przestrzennie (`sjoin`) do poligonu powiatu.
  - Wydzielenia leśne z BDL (wielokąty drzewostanów) są przecinane z granicami powiatów i agregowane do cech tabelarycznych powiatu (np. % powierzchni leśnej, % udziału sosny/dębu/buka/świerku, średni wiek lasu, udział siedlisk wilgotnych/świeżych).
  - Dane pogodowe z Open-Meteo są pobierane dla centroidów obszarów leśnych w danym powiecie.

---

## 3. Definicja wyniku aplikacji (Co wylicza i pokazuje system?)

Dla wybranej przez użytkownika **daty docelowej** (np. nadchodzący weekend) oraz **gatunku grzyba** (`all` = ogółem lub konkretny gatunek: borowik szlachetny, podgrzybek brunatny, kurka), system zwraca dla każdego powiatu:

1. **Prognozowaną wydajność zbioru (`predicted_yield_pcs_per_h`)**:
   - Wartość ciągła (float $\ge 0$), wyrażona w **szacowanej liczbie grzybów na godzinę zbierania (`[szt./h]`)** – bezpośrednio odpowiadająca głównemu wskaźnikowi raportowanemu na `grzyby.pl`.
2. **Kategorię obfitości wysypu (`abundance_class`)** `[DO WERYFIKACJI progów]`:
   - `BRAK` – prognoza `< 5 szt./h` (pojedyncze sztuki lub pusto),
   - `SREDNIO` – prognoza `5 – 20 szt./h` (umiarkowany zbiór),
   - `WYSYP` – prognoza `> 20 szt./h` (masowy wysyp grzybów).
3. **Wskaźnik pewności / dopasowania siedliska (`confidence_score`)**:
   - Wartość z przedziału `0.0 – 1.0`, informująca na ile dany powiat posiada odpowiedni drzewostan dla wybranego gatunku oraz pełne pokrycie danymi pogodowymi.
4. **Rekomendację lokalizacji w promieniu (Ranking)**:
   - Po wybraniu punktu startowego (np. miasta w regionie) i promienia (np. `30 km`, `50 km`, `100 km`), aplikacja filtruje powiaty według odległości do centroidu leśnego i sortuje je malejąco według `predicted_yield_pcs_per_h`.

---

## 4. Zakres funkcjonalny MVP (In Scope)

### Moduł 1: Dane, GeoPandas i ML
- [x] Pobranie granic powiatów (PRG / GUGiK) dla wybranego województwa i ustandaryzowanie ich do `areas.geojson`.
- [x] Obliczenie zagregowanych wskaźników drzewostanu z BDL dla każdego powiatu (udział gatunków panujących, wiek, wilgotność siedliska).
- [x] Zeskrapowanie historycznych raportów z `grzyby.pl` dla wybranego regionu i przypisanie ich do `area_id`.
- [x] Pobranie historycznych i bieżących danych pogodowych z Open-Meteo API oraz wyliczenie zmiennych opóźnionych (`precip_lag_7d_mm`, `precip_lag_14d_mm`, `precip_lag_21d_mm`, `temp_min_night_c`, `soil_moisture_0_7cm_m3m3`).
- [x] Eksploracyjna analiza danych (EDA): wykres korelacji opóźnienia deszcz $\rightarrow$ wysyp (Lag Effect).
- [x] Wytrenowanie modelu bazowego (Random Forest / XGBoost) i wygenerowanie pliku `predictions.csv` zgodnego z kontraktem.

### Moduł 2: Aplikacja Streamlit
- [x] Wczytywanie plików zgodnych z `docs/data_contract.md` (w fazie budowy UI z katalogu `data/demo/`, w fazie produkcyjnej z `data/processed/`).
- [x] Interaktywna mapa (PyDeck / Folium) kolorująca powiaty według `predicted_yield_pcs_per_h` lub `abundance_class`.
- [x] Panel filtrów bocznych (Sidebar): wybór daty, wybór gatunku grzyba, wybór miasta startowego i promienia w km.
- [x] Tabela rekomendowanych powiatów w zadanym promieniu wraz z podsumowaniem warunków pogodowych (opad z 7/14 dni, temp. nocna) i dominującego typu lasu.
- [x] Sekcja z wykresami EDA (np. szereg czasowy opadów vs. historyczne i prognozowane zbiory dla klikniętego powiatu).

---

## 5. Rzeczy odkładane na później (Out of Scope / Post-MVP)

Aby dotrzymać terminu i uniknąć blokad technicznych, poniższe elementy **celowo wyłączamy z pierwszej wersji**:

1. **Renderowanie pojedynczych wydzieleń leśnych BDL (dziesiątki tysięcy poligonów) bezpośrednio w przeglądarce:**
   - *W MVP:* Wektory BDL przetwarzamy offline w GeoPandas i agregujemy do poziomu powiatu. Bezpośrednie rysowanie pododdziałów leśnych w Streamlit odkładamy do wersji 2.0 (jako drill-down dla jednego wybranego nadleśnictwa).
2. **Pokrycie całej Polski dla szczegółowych warstw BDL:**
   - *W MVP:* Skupiamy się na jednym województwie (30 powiatów na Dolnym Śląsku).
3. **Dane radarowe IMGW (Polrad) w siatce 1 km × 1 km:**
   - *W MVP:* Korzystamy z otwartego i stabilnego API Open-Meteo (Historical & Forecast API) próbkowanego dla centroidów powiatów.
4. **Wyznaczanie trasy drogowej (OSRM / Google Maps API):**
   - *W MVP:* Odległość „w promieniu X km od miasta” liczymy w linii prostej (odległość geometryczna / Haversine od wybranego miasta do centroidu leśnego powiatu).
5. **System kont użytkowników i dodawanie nowych raportów przez aplikację:**
   - *W MVP:* Aplikacja działa w trybie tylko do odczytu (Read-Only Dashboard).
6. **Zaawansowane NLP na treści postów z forum grzyby.pl:**
   - *W MVP:* Korzystamy z ustrukturyzowanych nagłówków raportów z `grzyby.pl` (data, województwo/miejsce, `szt./h`) oraz słownikowego dopasowania nazw gatunków i powiatów.

---

## 6. Zasady bezpieczeństwa dla próbki demonstracyjnej

1. Próbka do budowy interfejsu znajduje się wyłącznie w katalogu `data/demo/`.
2. Każdy rekord w plikach demonstracyjnych posiada kolumnę `is_demo = True` oraz prefiks `DEMO_` w polach identyfikacyjnych/wersji modelu.
3. Skrypty treningowe ML muszą jawnie odrzucać katalog `data/demo/` oraz odfiltrowywać wiersze z `is_demo == True` (np. `df = df[~df["is_demo"]]`), aby sztuczne dane nigdy nie zanieczyściły zbioru uczącego ani walidacyjnego.

---

## 7. Lista kontrolna do przeglądu przez drugą osobę (`[DO WERYFIKACJI]`)

Przed rozpoczęciem pełnego scrapingu i kodowania prosimy o potwierdzenie poniższych punktów:
- [ ] **Region:** Czy zostajemy przy **województwie dolnośląskim (`TERYT 02`)**, czy wolisz **mazowieckie (`TERYT 14`)**?
- [ ] **Progi klas wysypu (`abundance_class`):** Czy przedziały `< 5` (`BRAK`), `5–20` (`SREDNIO`), `> 20` (`WYSYP`) `[szt./h]` są odpowiednie dla wizualizacji na mapie?
- [ ] **Lista gatunków w filtrze UI:** Czy zestaw `["all", "borowik_szlachetny", "podgrzybek_brunatny", "kurka", "maslak_zwyczajny", "kania"]` pokrywa potrzeby ekranu głównego?
- [ ] **Format plików wymiany:** Czy na etapie MVP zostajemy przy `CSV` + `GeoJSON` (czytelne w Git i łatwe do inspekcji), czy od razu przechodzimy na `Parquet` / `GeoParquet`?
