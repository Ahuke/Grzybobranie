# Audyt danych pogodowych i próbki `weather_daily` z Open-Meteo (`[GRZ-04] #4`)

> **Cel zadania:** Potwierdzić na rzeczywistej próbce z regionu pilotażowego (Dolny Śląsk), jakie zmienne meteorologiczne i glebowe z Open-Meteo można połączyć z raportami grzybiarzy, zweryfikować faktyczne głębokości warstw gleby i ich modelowy charakter, uzgodnić strefę czasową oraz przygotować skrypt z lokalnym cache’owaniem.

---

## 1. Wybrane źródło, zakres czasowy i lokalizacje w próbce

- **Źródło danych:** **Open-Meteo Historical Weather API** (`https://archive-api.open-meteo.com/v1/archive`), oparte na reanalizie klimatycznej **ECMWF ERA5-Land** (rozdzielczość przestrzenna siatki `0.1° × 0.1°`, tj. ok. **9 km × 9 km**) oraz modelu operacyjnym **ECMWF IFS** dla najświeższych dni.
- **Zapisana próbka wynikowa (`weather_daily`):**  
  [`data/sample/weather_daily_sample_ds.csv`](../data/sample/weather_daily_sample_ds.csv) (`210` rekordów dobowych = `6` lokalizacji × `35` dni).
- **Skrypt pobierający z obsługą cache’u (brak ponawiania zapytań przy kolejnych uruchomieniach):**  
  [`scripts/fetch_weather_sample.py`](../scripts/fetch_weather_sample.py) (surowe odpowiedzi JSON buforowane są lokalnie w `data/raw/weather_cache/`).
- **Zakres dat w próbce:** **`2025-09-01` – `2025-10-05`** (`35 dni`):
  - Obejmuje pełny tydzień rzeczywistych raportów grzybiarzy z zadania `[GRZ-03] #3` (`2025-09-29` – `2025-10-05`),
  - Obejmuje **28 dni wyprzedzenia** przed pierwszym dniem raportów (`2025-09-29`), czyli więcej niż wymagane minimum 21 dni – dzięki temu już dla pierwszego dnia raportów (`2025-09-29`) wszystkie okna skumulowane (`precip_lag_7d_mm`, `precip_lag_14d_mm`, `precip_lag_21d_mm`) są w 100% wypełnione!
- **6 badanych lokalizacji na Dolnym Śląsku (reprezentujących zróżnicowane siedliska i wysokości n.p.m.):**

| Kod powiatu (`area_id`) | Lokalizacja z raportów (`[GRZ-03]`) | Współrzędne zapytania (`lat, lon`) | Środek oczka siatki ERA5-Land (`grid_lat, grid_lon`) | Wysokość oczka siatki (`grid_elevation_m`) | Charakterystyka siedliska |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **`0201`** | Krępnica / Bory Dolnośląskie (pow. bolesławiecki) | `51.32335, 15.56014` | `51.28295, 15.56757` | `165 m n.p.m.` | Nizinne bory sosnowe na piaskach |
| **`0202`** | Góry Sowie (pow. dzierżoniowski) | `50.70015, 16.50078` | `50.72056, 16.46536` | `635 m n.p.m.` | Lasy górskie i mieszane Sudetów Środkowych |
| **`0208`** | Batorów / Góry Stołowe (pow. kłodzki) | `50.43889, 16.42472` | `50.43937, 16.34921` | `604 m n.p.m.` | Górskie bory świerkowe regla dolnego |
| **`0213`** | Łazy Wielkie / Dolina Baryczy (pow. milicki) | `51.41869, 17.29875` | `51.42355, 17.25136` | `190 m n.p.m.` | Nizinne lasy mieszane i wilgotne |
| **`0214`** | Białe Błoto / lasy twardogórskie (pow. oleśnicki) | `51.33312, 17.35364` | `51.35325, 17.38267` | `180 m n.p.m.` | Lasy mieszane Wzgórz Twardogórskich |
| **`0221`** | Witków / masyw Trójgarbu (pow. wałbrzyski) | `50.79697, 16.13085` | `50.79086, 16.17438` | `464 m n.p.m.` | Lasy bukowe i mieszane Przedgórza Sudeckiego |

---

## 2. Dostępność historii, brakujące zmienne i faktyczne głębokości warstw gleby

### 2.1. Zakres dostępnej historii w Open-Meteo
- Endpoint `/v1/archive` udostępnia nieprzerwaną historię od **1940 roku** (dla reanalizy powierzchni lądu **ERA5-Land** od **1950 roku**) aż do **dnia wczorajszego** względem dnia zapytania.
- **Brak luki na ostatnich 5 dniach:** Choć sama reanaliza ERA5-Land ma ok. 5-dniowe opóźnienie przeliczeniowe, domyślny tryb `best_match` w Open-Meteo Archive API automatycznie uzupełnia ostatnie 5 dni z analizy operacyjnej ECMWF IFS (zweryfikowano na datach `2026-10-15 .. 2026-10-28`: `0` wartości `NULL`).

### 2.2. Zmienne niedostępne bezpośrednio w `daily` (wymagające agregacji z `hourly`)
Podczas audytu API zidentyfikowano istotne ograniczenie strukturalne Open-Meteo:
- W parametrze **`daily=`** dostępne są wyłącznie zmienne atmosferyczne na wysokości 2 m oraz opady (`temperature_2m_min/mean/max`, `precipitation_sum`, `rain_sum`, `precipitation_hours`, `et0_fao_evapotranspiration`).
- **Zmienne glebowe (`soil_temperature_*`, `soil_moisture_*`) oraz wilgotność względna powietrza (`relative_humidity_2m`) NIE występują w sekcji `daily`!**
- **Rozwiązanie zastosowane w skrypcie:** Pobieramy je w rozdzielczości godzinowej (`hourly=`) i agregujemy dla każdej doby lokalnej (`24 godziny: 00:00–23:00`) do średniej dobowej (`mean`) oraz minimum dobowego (`min`).

### 2.3. Faktyczne głębokości warstw gleby w Open-Meteo
W Open-Meteo Historical Archive API (`ERA5-Land`) profil glebowy jest podzielony dokładnie na **4 warstwy głębokości**:

| Warstwa w API (`ERA5-Land`) | Zakres głębokości | Temperatura gleby (`[°C]`) | Wilgotność objętościowa gleby (`[m³/m³]`) | Znaczenie biologiczne dla grzybów |
| :--- | :---: | :--- | :--- | :--- |
| **Warstwa 1 (Płytka)** | **`0 – 7 cm`** | `soil_temperature_0_to_7cm` | `soil_moisture_0_to_7cm` | **Kluczowa:** warstwa ściółki leśnej i humusu, gdzie bezpośrednio zawiązują się owocniki grzybów. |
| **Warstwa 2 (Średnia)** | **`7 – 28 cm`** | `soil_temperature_7_to_28cm` | `soil_moisture_7_to_28cm` | **Ważna:** strefa korzeniowa drzew istotna dla grzybów mikoryzowych (borowik, podgrzybek, kurka, rydz) i odporna na jednodniowe przesuszenie wiatrem. |
| **Warstwa 3 (Głęboka)** | **`28 – 100 cm`** | `soil_temperature_28_to_100cm` | `soil_moisture_28_to_100cm` | Pomocnicza: głęboka retencja wodna drzewostanu (wskaźnik długotrwałej suszy hydrologicznej). |
| **Warstwa 4 (Podglebie)**| **`100 – 255 cm`**| `soil_temperature_100_to_255cm`| `soil_moisture_100_to_255cm`| **Pomijana w modelu:** w skali miesiąca niemal stała (np. w Bolesławcu `0.229–0.230 m³/m³`), brak wpływu na bieżący wysyp grzybów. |

> **Uwaga techniczna dot. spójności z endpointem prognozy (`/v1/forecast`):**  
> W endpointzie prognozy `/v1/forecast` domyślne modele (`ICON`/`GFS`) używają innych nazw warstw (`soil_temperature_0cm/6cm/18cm/54cm` oraz `soil_moisture_0_to_1cm/1_to_3cm/3_to_9cm/9_to_27cm`). Jednak `/v1/forecast` wspiera również bezpośrednio parametry **`soil_temperature_0_to_7cm`** oraz **`soil_moisture_0_to_7cm`** z modelu ECMWF IFS. Aby zachować 100% spójności między historią a prognozą, w modelu ML opieramy się na warstwach **`0–7 cm`** oraz **`7–28 cm`**.

---

## 3. Charakter danych glebowych: Model reanalizy vs. Pomiar terenowy (In-Situ)

> **[!IMPORTANT] Ostrzeżenie metodologiczne (Warunek ukończenia `#4`):**  
> Dane o temperaturze i wilgotności gleby z Open-Meteo **NIE są pomiarami terenowymi (in-situ)** z czujników umieszczonych w lesie!

1. **Pochodzenie z modelu numerycznego:**
   - Zmienne `soil_temperature_*` oraz `soil_moisture_*` są wyliczane przez model powierzchni lądu **HTESSEL / ECMWF ERA5-Land** na siatce obliczeniowej o boku ok. **9 km (`0.1° × 0.1°`)**.
   - W pliku [`data/sample/weather_daily_sample_ds.csv`](../data/sample/weather_daily_sample_ds.csv) każdy wiersz posiada jawną kolumnę `soil_data_origin = "model_reanalysis_ERA5_Land_9km"`, a także współrzędne oczka siatki (`grid_lat`, `grid_lon`, `grid_elevation_m`), które różnią się o kilka kilometrów od dokładnych współrzędnych lasu (`request_lat`, `request_lon`).
2. **Konsekwencje dla interpretacji i aplikacji Streamlit:**
   - W interfejsie użytkownika, prezentacji i dokumentacji nigdy nie używamy sformułowania *„pomiar wilgotności gleby w lesie”*, lecz: **„modelowana wilgotność gleby (reanaliza ERA5-Land, warstwa 0–7 cm)”**.
   - Ponieważ oczko siatki `9 × 9 km` uśrednia warunki dla całego terenu (lasów i pól uprawnych), w kolejnym etapie łączymy te wskaźniki z rzeczywistym typem siedliskowym lasu i gatunkiem panującym z **Banku Danych o Lasach (BDL)**.

---

## 4. Strefa czasowa i sposób wyznaczania dnia oraz okien opóźnionych (Lag)

1. **Strefa czasowa (`timezone`):**
   - We wszystkich zapytaniach do Open-Meteo obowiązuje jawnie przekazywana strefa **`Europe/Warsaw`**.
2. **Definicja doby (`date`):**
   - Doba jest wyznaczana jako pełne **24 godziny kalendarzowe czasu lokalnego `00:00 – 23:59` (`Europe/Warsaw`)**, co dokładnie odpowiada sposobowi datowania zbiorów (`YYYY-MM-DD`) przez użytkowników portalu `grzyby.pl`.
3. **Temperatura minimalna (`temp_min_2m_c` vs `temp_min_night_c`):**
   - `temp_min_2m_c` to oficjalne minimum dobowe (`temperature_2m_min` z `00:00–23:59`).
   - Dodatkowo wyliczyliśmy `temp_min_night_c` z godzin nocno-porannych (`00:00–07:00` oraz `21:00–23:00`). W 98% dni w próbce obie wartości są identyczne (minimum dobowe przypada nad ranem ok. `04:00–06:00`).
4. **Sposób wyznaczania okien opóźnionych (`lag_7d`, `lag_14d`, `lag_21d`) – ochrona przed przeciekiem danych (Data Leakage):**
   - Dla dnia grzybobrania $t$ skumulowany opad z $k$ dni jest liczony z przedziału zamkniętego **$[t-k,\; t-1]$** (czyli z pełnych $k$ dni **poprzedzających** dzień zbioru):
     $$\text{precip\_lag\_}k\text{d\_mm}(t) = \sum_{j=1}^{k} \text{precip\_mm\_1d}(t - j)$$
   - **Dlaczego nie wliczamy dnia $t$ do `lag_k`?** Grzybiarz wchodzi do lasu zazwyczaj rano (`06:00–11:00`). Opad, który spadnie w dniu $t$ po południu, nie mógł wywołać wyrośnięcia grzybów zebranych tego samego ranka. Opad z samego dnia $t$ jest dostępny osobno w kolumnie `precip_mm_1d`.
   - Dla pierwszych $k$ dni pobranej serii (gdy nie ma jeszcze pełnych $k$ dni wstecz) w pliku zapisywana jest wartość pusta **`NULL` (`""`)**, a nigdy sztuczne zero ani suma częściowa!

---

## 5. Opis cech w `weather_daily` możliwych do wykorzystania w modelu ML (Przekazywany rezultat)

Poniższa tabela stanowi słownik cech wyjściowych z pliku [`data/sample/weather_daily_sample_ds.csv`](../data/sample/weather_daily_sample_ds.csv) gotowych do połączenia z raportami po kluczu `(date, area_id)`:

| Nazwa kolumny w `weather_daily` | Jednostka | Pochodzenie (Daily / Hourly / Wyliczona) | Zakres w próbce Dolnego Śląska (`09–10.2025`) | Rola w modelu predykcji grzybów |
| :--- | :---: | :--- | :---: | :--- |
| **`precip_mm_1d`** | `mm` | `daily.precipitation_sum` | `0.0 – 27.6 mm` | Dobowa suma opadów (informacja o deszczu w samym dniu wyjścia). |
| **`precip_hours_1d`** | `h` | `daily.precipitation_hours` | `0.0 – 21.0 h` | Czas trwania opadu w ciągu doby (odróżnia krótką ulewę burzową, która spływa po ściółce, od całodniowego deszczu wsiąkającego w mech!). |
| **`precip_lag_7d_mm`** | `mm` | Suma `[t-7, t-1]` | `0.4 – 48.0 mm` | **Cecha główna ML:** skumulowany opad z ostatniego tygodnia (szybko reagujące gatunki: maślaki, kanie, czernidłaki). |
| **`precip_lag_14d_mm`** | `mm` | Suma `[t-14, t-1]` | `8.5 – 70.1 mm` | **Cecha główna ML:** skumulowany opad z ostatnich 2 tygodni (klasyczny wyzwalacz dla borowika szlachetnego i podgrzybka brunatnego). |
| **`precip_lag_21d_mm`** | `mm` | Suma `[t-21, t-1]` | `22.7 – 93.3 mm` | **Cecha główna ML:** długoterminowe nawodnienie grzybni z 3 tygodni. |
| **`temp_min_2m_c`** / **`temp_min_night_c`** | `°C` | `daily.temperature_2m_min` | `-1.9 – +16.9 °C` | **Cecha główna ML:** minimalna temperatura nocna/poranna (wykrywa przymrozki hamujące wzrost grzybni – np. w próbce `2025-10-03` w Górach Stołowych `0208` spadła do `-1.9 °C`, co pokrywa się z raportami grzybiarzy o przymrozkach w `#3`!). |
| **`temp_mean_2m_c`** | `°C` | `daily.temperature_2m_mean` | `2.1 – 22.0 °C` | Średnia dobowa temperatura powietrza na 2 m. |
| **`temp_mean_7d_c`** | `°C` | Średnia z `[t-7, t-1]` | `5.3 – 18.8 °C` | Trend termiczny z ostatniego tygodnia (szok termiczny po upale vs. wychłodzenie jesienne). |
| **`rel_humidity_2m_mean_pct`**| `%` | Średnia z `hourly.relative_humidity_2m` | `59.0 – 97.5 %` | Średnia wilgotność względna powietrza (wysoka wilgotność i mgły sprzyjają utrzymaniu rosy w ściółce mimo niskich opadów). |
| **`et0_evap_mm_1d`** | `mm` | `daily.et0_fao_evapotranspiration` | `0.48 – 4.35 mm` | Ewapotranspiracja referencyjna FAO (wskaźnik tempa wysychania ściółki pod wpływem słońca i wiatru). |
| **`soil_temp_0_7cm_mean_c`** | `°C` | Średnia z `hourly.soil_temperature_0_to_7cm` *(model ERA5-Land)* | `4.4 – 20.6 °C` | **Cecha główna ML:** modelowana średnia temperatura wierzchniej warstwy gleby (`0–7 cm`). |
| **`soil_temp_7_28cm_mean_c`** | `°C` | Średnia z `hourly.soil_temperature_7_to_28cm` *(model ERA5-Land)* | `6.8 – 18.9 °C` | Modelowana temperatura gleby w strefie korzeniowej (`7–28 cm`) – stabilniejsza dobowo niż warstwa `0–7 cm`. |
| **`soil_moisture_0_7cm_m3m3`**| `m³/m³` | Średnia z `hourly.soil_moisture_0_to_7cm` *(model ERA5-Land)* | `0.197 – 0.422` | **Cecha główna ML:** modelowana objętościowa wilgotność gleby w warstwie `0–7 cm`. |
| **`soil_moisture_7_28cm_m3m3`**| `m³/m³` | Średnia z `hourly.soil_moisture_7_to_28cm` *(model ERA5-Land)* | `0.175 – 0.402` | Modelowana wilgotność gleby w strefie korzeniowej (`7–28 cm`). |

---

## 6. Weryfikacja spójności próbki pogodowej z raportami z zadania `[GRZ-03] #3`

Porównanie pobranej próbki `weather_daily_sample_ds.csv` z treścią relacji grzybiarzy z tygodnia `29.09 – 05.10.2025` potwierdziło pełną zgodność fizyczną danych:
1. **Przymrozki nocne 2–4 października 2025:**
   - W raportach `DG174435` (`03.10`, pow. kłodzki `0208`) oraz `DG174563` (`04.10`, pow. bolesławiecki `0201`) grzybiarze pisali o nocnych przymrozkach i skrobaniu szyb.
   - W pobranej próbce Open-Meteo dokładnie w dniach `02.10–04.10.2025` temperatura minimalna `temp_min_2m_c` spadła w pow. kłodzkim (`0208`) do **`-1.9 °C`**, w Górach Sowich (`0202`) do **`-0.3 °C`**, a w pow. wałbrzyskim (`0221`) do **`+0.1 °C`**!
2. **Kontrast wilgotności gleby Góry (Sudety) vs. Niziny:**
   - W Górach Stołowych (`0208`) i Sowich (`0202`) modelowana wilgotność gleby `soil_moisture_0_7cm_m3m3` utrzymywała się na wysokim poziomie **`0.37 – 0.41 m³/m³`** (co odpowiada raportom typu `DG174762`: *„48 prawdziwków, 47 podgrzybków”*), podczas gdy na nizinach w pow. oleśnickim (`0214`) i milickim (`0213`) spadła pod koniec września do **`0.20 – 0.24 m³/m³`** (co odpowiada raportom typu `DG174448`: *„Sucho jak na Saharze”*).
