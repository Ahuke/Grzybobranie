# Zakres MVP – „Gdzie na grzyby?”

> **Status dokumentu:** **Wersja 2.0 (Zatwierdzona po audycie danych `[GRZ-06] #6`)**  
> Wszystkie decyzje architektoniczne, wybór zmiennej docelowej, jednostka przestrzenna, horyzont prognozy i plan modelowania zostały ostatecznie zatwierdzone przez zespół na podstawie audytów `[GRZ-03]`, `[GRZ-04]` i `[GRZ-05]` (zob. [`docs/data_audit.md`](data_audit.md)).

---

## 1. Cel pierwszej wersji (MVP)

Celem wersji MVP jest dostarczenie działającego end-to-end prototypu systemu predykcji wysypu grzybów dla **województwa dolnośląskiego (`TERYT 02`, 30 powiatów)**, łączącego:
1. **Historyczne raporty zbieraczy** (portal `grzyby.pl`),
2. **Charakterystykę siedlisk leśnych** (Bank Danych o Lasach – BDL oraz GUS BDL),
3. **Dane meteorologiczne i glebowe z uwzględnieniem efektu opóźnienia (Lag Effect)** (Open-Meteo API / reanaliza ECMWF ERA5-Land),

oraz prezentującego wyniki w interaktywnej aplikacji **Streamlit**.

Podział pracy w zespole opiera się na zamrożonym kontrakcie danych ([`docs/data_contract.md`](data_contract.md)), dzięki czemu:
- **Moduł Danych & ML (Osoba A)** rozwija potok danych, agregacje GeoPandas oraz dwutorowy moduł predykcyjny (Jawny Wskaźnik Warunków + Model ML),
- **Moduł Aplikacji & Wizualizacji (Osoba B)** rozwija interfejs Streamlit na wspólnej strukturze plików (`data/demo/` oraz `data/processed/`).

---

## 2. Wybór regionu i jednostki przestrzennej (Zatwierdzone w `[GRZ-06]`)

### 2.1. Region pilotażowy
- **Wybrany region:** **Województwo dolnośląskie** (kod TERYT województwa: `"02"`, łączna powierzchnia `19 941,7 km²` w `EPSG:2180`).
- **Potwierdzenie z audytu (`[GRZ-03]`–`[GRZ-05]`):**
  - Wyraźny kontrast siedliskowy potwierdzony w próbce wektorowej BDL: od 100% borów sosnowych na północy (`0201` Bory Dolnośląskie, `BŚW`) po lasy bukowe (`BK 53,4%`, `0221` Masyw Trójgarbu) i górskie bory świerkowe na południu (`ŚW 78,8–90,7%`, Karkonosze `0206`, Góry Stołowe `0208`, Góry Sowie `0202`).
  - Wysokie pokrycie raportami: w jednym badanym tygodniu (`29.09–05.10.2025`) 52 raporty pokryły 18 z 30 powiatów (`60%`).

### 2.2. Podstawowa jednostka przestrzenna (Spatial Unit)
- **Jednostka w MVP:** **Powiat** (26 powiatów ziemskich + 4 miasta na prawach powiatu), identyfikowany **4-znakowym kodem TERYT** (`area_id`, `"0201"`–`"0265"`).
- **Uzasadnienie dopasowania mapy do poziomu powiatu (`[GRZ-06]`):**
  - Współrzędne `(lat, lon)` przypisane do `place_id` na `grzyby.pl` wskazują środek najbliższej miejscowości lub gminy/powiatu, a nie konkretny oddział leśny. Agregacja do poziomu powiatu wiernie odzwierciedla rzeczywistą rozdzielczość danych wejściowych.

---

## 3. Definicja wyniku aplikacji, horyzont prognozy i ewaluacja (`[GRZ-06]`)

Dla wybranej przez użytkownika **daty docelowej $t$** (w horyzoncie $t \in [t_0,\; t_0 + 7\text{ dni}]$ względem dnia wystawienia prognozy $t_0$) oraz **gatunku grzyba** (`all` lub konkretny gatunek), system zwraca dla każdego powiatu:

1. **Główną zmienną docelową – prognozowaną wydajność zbioru (`predicted_yield_pcs_per_h`)**:
   - Wartość ciągła ($\ge 0.0$), wyrażona w **szacowanej liczbie grzybów na godzinę zbierania (`[szt./h]`)**, odpowiadająca obowiązkowemu wskaźnikowi nagłówkowemu `(X/h)` z `grzyby.pl` (`100%` kompletności w audycie).
2. **Kategorię obfitości wysypu (`abundance_class`)**:
   - `BRAK` – prognoza `< 5.0 szt./h` (puste koszyki lub pojedyncze sztuki; `21,2%` wpisów w próbce),
   - `SREDNIO` – prognoza `5.0 – 20.0 szt./h` (umiarkowany, udany zbiór; `67,3%` wpisów w próbce),
   - `WYSYP` – prognoza `> 20.0 szt./h` (masowy wysyp grzybów; `11,5%` wpisów w próbce).
3. **Wskaźnik pewności / dopasowania siedliska (`confidence_score` `0.0 – 1.0`)**:
   - Wyznaczany na podstawie dopasowania składu gatunkowego drzewostanu BDL do wybranego gatunku grzyba oraz dostępności danych.
4. **Zasady metodologiczne zatwierdzone w `[GRZ-06]`:**
   - **Brak wpisu $\neq$ zerowy zbiór:** Brak raportu w danym powiecie i dniu jest traktowany jako `NULL` (brak obserwacji), a nigdy jako `0.0 szt./h`.
   - **Odcięcie czasowe pogody (`[t-k, t-1]`):** Okna opadowe i temperaturowe są liczone wyłącznie z dni poprzedzających dzień zbioru; przyszła rzeczywista pogoda z archiwum nigdy nie trafia do cech prognozy.
   - **Architektura dwutorowa (Model ML + Jawny Wskaźnik Warunków):** Model bazowy uczony jest dla kategorii ogólnej (`"all"`), a filtry gatunkowe ważone są jawnym mnożnikiem siedliskowym BDL; w razie małej liczebności etykiet dla wybranych powiatów gwarantowany wynik zapewnia skalibrowany Jawny Wskaźnik Warunków Grzybobrania.
   - **Ewaluacja:** Podział czasowy (*Temporal Split*), metryki **MAE `[szt./h]`**, **RMSE**, korelacja rangowa **Spearmana** oraz **Macro-F1** dla klas `BRAK / SREDNIO / WYSYP`.

---

## 4. Zakres funkcjonalny MVP (In Scope)

### Moduł 1: Dane, GeoPandas i Predykcja
- [x] Pobranie rzeczywistych granic 30 powiatów (PRG / GUGiK) dla woj. dolnośląskiego i ustandaryzowanie ich do `areas.geojson` (`EPSG:4326`).
- [x] Pobranie wektorowej próbki wydzieleń leśnych BDL (`N = 105`) i wyliczenie wskaźników drzewostanu dla powiatów (`[GRZ-05]`).
- [x] Audyt i próbka 52 raportów z `grzyby.pl` (`[GRZ-03]`) oraz 210 dni historii pogody i gleby z Open-Meteo ERA5-Land (`[GRZ-04]`).
- [x] Zatwierdzenie celu predykcji, odcięcia danych i dwutorowej architektury modelowania w `docs/data_audit.md` (`[GRZ-06]`).
- [ ] Eksploracyjna analiza danych (EDA) i budowa pełnego zestawu cech opóźnionych (Sprint 2).
- [ ] Implementacja Jawnego Wskaźnika Warunków oraz modelu ML, ewaluacja czasowa i wygenerowanie produkcyjnego `predictions.csv` (Sprint 2).

### Moduł 2: Aplikacja Streamlit
- [x] Wczytywanie plików zgodnych z `docs/data_contract.md` (`data/demo/` oraz `data/processed/`) z bezpieczną obsługą `NULL` vs `0.0`.
- [x] Interaktywna mapa PyDeck prezentująca rzeczywiste granice 30 powiatów Dolnego Śląska oraz wektorową warstwę wydzieleń BDL.
- [x] Panel filtrów bocznych (Sidebar): wybór zbioru danych, daty, gatunku grzyba, miasta startowego i promienia w km.
- [x] Tabela rekomendowanych powiatów w zadanym promieniu wraz z podsumowaniem warunków pogodowych i siedliskowych.

---

## 5. Lista kontrolna zatwierdzona w `[GRZ-06]`

- [x] **Region:** Województwo dolnośląskie (`TERYT 02`, 30 powiatów).
- [x] **Progi klas wysypu (`abundance_class`):** `< 5.0` (`BRAK`), `5.0–20.0` (`SREDNIO`), `> 20.0` (`WYSYP`) `[szt./h]`.
- [x] **Lista gatunków w filtrze UI:** `["all", "borowik_szlachetny", "podgrzybek_brunatny", "kurka", "maslak_zwyczajny", "kania"]`.
- [x] **Format plików wymiany:** `CSV` + `GeoJSON` (`UTF-8`, `EPSG:4326`).

---

## 6. Rzeczy odkładane na później (Out of Scope / Post-MVP)

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

## 7. Zasady bezpieczeństwa dla próbki demonstracyjnej

1. Próbka do budowy interfejsu znajduje się wyłącznie w katalogu `data/demo/`.
2. Każdy rekord w plikach demonstracyjnych posiada kolumnę `is_demo = True` oraz prefiks `DEMO_` w polach identyfikacyjnych/wersji modelu.
3. Skrypty treningowe ML muszą jawnie odrzucać katalog `data/demo/` oraz odfiltrowywać wiersze z `is_demo == True` (np. `df = df[~df["is_demo"]]`), aby sztuczne dane nigdy nie zanieczyściły zbioru uczącego ani walidacyjnego.
