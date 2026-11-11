# Audyt i pełny zbiór historii pogody dla regionu pilotażowego (`[GRZ-09]`)

> **Cel zadania (`[GRZ-09]`):** Zapewnić spójne szeregi pogodowe dla wszystkich analizowanych obszarów (30 powiatów woj. dolnośląskiego, `TERYT 02`) i całego sezonu obserwacji, rozszerzyć skrypt próbki z `[GRZ-04]` na pełny pilotaż z uwzględnieniem zapasu historii do najdłuższego okna opadów (`precip_lag_21d_mm`), zapisać dane w lokalnym cache oraz scharakteryzować reprezentację punktową jako przybliżenie (eliminacja nadawania pozornej dokładności).

---

## 1. Wygenerowane pliki i artefakty pogody

| Plik w repozytorium | Format i klucz | Liczba wierszy | Rola w projekcie |
| :--- | :---: | :---: | :--- |
| [`data/processed/weather.parquet`](../data/processed/weather.parquet)<br/>[`data/processed/weather.csv`](../data/processed/weather.csv) | `Parquet` / `CSV`<br/>PK: **`(date, area_id)`** | **`2 910`** | **Produkcyjny plik kontraktu danych (`docs/data_contract.md`)** zawierający cechy dobowe i skumulowane opady opóźnione (`precip_lag_7d_mm`, `precip_lag_14d_mm`, `precip_lag_21d_mm`, `temp_min_night_c`, `temp_mean_7d_c`, `soil_moisture_0_7cm_m3m3`), wczytywany przez `src/data_loader.py` i gotowy do treningu ML w `[GRZ-10]`. |
| [`data/processed/weather_daily.parquet`](../data/processed/weather_daily.parquet) | `Parquet`<br/>PK: **`(date, area_id)`** | **`2 910`** | Pełny zbiór surowych i zagregowanych zmiennych meteorologicznych (w tym dobowa suma opadów deszczu `rain_mm_1d`, czas trwania opadów `precipitation_hours`, maksymalna prędkość wiatru `wind_speed_max_kmh`, średnia temperatura gleby `soil_temperature_0_7cm_c`, współrzędne oczka siatki reanalizy `grid_lat`, `grid_lon` oraz wysokość n.p.m. `grid_elevation_m`). |
| [`data/processed/weather_quality_report.json`](../data/processed/weather_quality_report.json) | `JSON` | — | Maszynowy raport kontroli jakości: unikalność kluczy głównych, liczba brakujących dni, rozstępy wartości i rozkłady cech. |
| [`scripts/fetch_pilot_weather.py`](../scripts/fetch_pilot_weather.py) | Skrypt Python (CLI) | — | Powtarzalny potok pobierania z Open-Meteo Archive API z automatycznym buforowaniem w `data/raw/weather_cache/` i mechanizmem retry. |
| `data/raw/weather_cache/*.json` | `JSON` (30 plików) | — | Surowe odpowiedzi API z reanalizy ERA5-Land zabezpieczone w `.gitignore` – ponowne uruchomienie skryptu trwa ułamek sekundy i nie generuje ruchu sieciowego. |

### Polecenie do uruchomienia potoku:
```bash
PYTHONPATH=. python scripts/fetch_pilot_weather.py
```

---

## 2. Zakres czasowy i bufor historii opadów (`Lag Window Buffer`)

Obserwacje grzybobrań z `[GRZ-07]` obejmują okres od **`2025-09-01` do `2025-11-02`**. Najdłuższe wymagane w modelu okno opadów to **21 dni** (`precip_lag_21d_mm` dla okna `[t-21, t-1]`).

Aby każdy dzień obserwacji (w tym pierwszy raport z `2025-09-01`) posiadał kompletną, niepustą historię skumulowanych opadów:
- **Początek szeregu pogodowego:** **`2025-08-01`** (dokładnie **31 dni zapasu historii** przed pierwszym raportem, przewyższające wymagane 21 dni),
- **Koniec szeregu pogodowego:** **`2025-11-05`** (zapewnia pokrycie po ostatnim zgłoszeniu z 2 listopada),
- **Długość szeregu per powiat:** **`97 dni ciągłych`**,
- **Łączna liczba obserwacji:** $30 \text{ powiatów} \times 97 \text{ dni} = \mathbf{2\ 910 \text{ rekordów}}$.

Dzięki temu od dnia `2025-08-22` (22. dzień szeregu) wskaźnik `precip_lag_21d_mm` jest w 100% zdefiniowany, co oznacza, że **żadna obserwacja grzybobrania z września, października ani listopada nie posiada braków w cechach opóźnionych (`NULL`)**.

---

## 3. Reprezentacja punktowa powiatu jako przybliżenie (Warunek metodyczny #4)

Zgodnie z wymaganiem **„Pojedynczy punkt reprezentujący powiat jest opisany jako przybliżenie, a nie średnia powierzchniowa”**:

1. **Charakterystyka punktu zapytania:**
   - Zapytanie do API Open-Meteo kierowane jest na współrzędne leśno-geometrycznego środka ciężkości powiatu (`centroid_lat`, `centroid_lon` wyznaczone w `EPSG:2180`).
   - Open-Meteo mapuje te współrzędne do najbliższego oczka numerycznego modelu reanalizy atmosferycznej **ERA5-Land** o rozdzielczości przestrzennej **`0,1° × 0,1°` (ok. `9 km × 9 km`)**.
2. **Ograniczenia metodyczne (Brak pozornej dokładności):**
   - Wartości meteorologiczne w plikach `weather.parquet` reprezentują warunki w oczku siatki obejmującym centroid leśny danego powiatu.
   - W powiatach górskich o dużych deniwelacjach terenu (np. pow. kłodzki `0208`, pow. karkonoski `0206` czy jeleniogórski `0261`) rzeczywiste sumy opadów na szczytach pasm górskich mogą różnić się od opadów w kotlinach śródgórskich.
   - Dlatego dane te są w potoku analitycznym jawnie traktowane jako **przybliżenie punktowe (`spatial_representation = "point_approximation_centroid"`)**, a nie dokładna średnia całkowa ważona ukształtowaniem terenu całej powierzchni powiatu.
3. **Pochodzenie danych glebowych:**
   - Wilgotność gleby `soil_moisture_0_7cm_m3m3` i temperatura `soil_temperature_0_7cm_c` w warstwie przypowierzchniowej `0–7 cm` pochodzą z bilansu wodno-termicznego modelu fizycznego ERA5-Land, a nie z bezpośrednich sond wkopanych w leśną ściółkę (*in-situ*).

---

## 4. Wyniki kontroli integralności i braków danych (Warunki #1, #3)

| Wskaźnik jakościowy | Wartość | Ocena zgodności |
| :--- | :---: | :---: |
| **Unikalność klucza głównego `(area_id, date)`** | **`100% unikalne`** (`2 910 / 2 910`) | **TAK** – dokładnie jeden rekord dla każdej pary powiat/dzień, brak jakichkolwiek duplikatów. |
| **Liczba brakujących dni w szeregu (`missing_days`)** | **`0 dni`** | **TAK** – ciągły szereg bez ani jednej luki w żadnym z 30 powiatów. |
| **Braki danych w dobowej sumie opadów (`precip_mm_1d`)** | **`0`** (`0,0%`) | **TAK** – pełne pokrycie pomiarowe. |
| **Braki danych w minimalnej temperaturze nocnej (`temp_min_night_c`)** | **`0`** (`0,0%`) | **TAK** – pełne pokrycie pomiarowe. |
| **Braki danych w wilgotności gleby (`soil_moisture_0_7cm_m3m3`)** | **`0`** (`0,0%`) | **TAK** – model reanalizy zwrócił kompletny szereg dla wszystkich 30 lokalizacji. |
| **Strefa czasowa odniesienia dla podziału dób** | **`Europe/Warsaw`** | **TAK** – doba meteorologiczna liczona od `00:00` do `23:59` czasu lokalnego w Polsce. |

---

## 5. Słownik zmiennych i rozkłady statystyczne w okresie `01.08 – 05.11.2025`

| Zmienna | Typ | Jednostka | Wartość min | Średnia | Wartość max | Interpretacja domenowa dla grzybni |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`precip_mm_1d`** | `float` | $\text{mm}$ | `0,0` | `2,34` | `41,7` | Dobowa suma opadów (pomiar zerowy `0.0` ściśle odróżniony od braku danych). W badanym okresie zarejestrowano intensywne opady jesienne do `41,7 mm/dobę`. |
| **`temp_min_night_c`** | `float` | `°C` | `-2,7` | `9,58` | `19,3` | Minimalna temperatura dobowa (kluczowy wskaźnik nocnych spadków temperatury inicjujących owocnikowanie oraz jesiennych przymrozków hamujących wzrost grzybów w listopadzie). |
| **`temp_mean_7d_c`** | `float` | `°C` | `-0,8` | `14,35` | `23,1` | Średnia temperatura z ostatnich 7 dni (`[t-7, t-1]`). |
| **`soil_moisture_0_7cm_m3m3`** | `float` | $\text{m}^3/\text{m}^3$ | `0,0499` | `0,2772` | `0,5097` | Objętościowa wilgotność płytkiej warstwy gleby z reanalizy ERA5-Land (odzwierciedla suszę ściółki na piaskach w sierpniu poniżej `0.06` oraz nasycenie wodą po deszczach we wrześniu i październiku powyżej `0.40`). |
| **`precip_lag_7d_mm`** | `float` | $\text{mm}$ | `0,0` | `16,21` | `78,4` | Skumulowany opad z 7 dni poprzedzających (`[t-7, t-1]`). |
| **`precip_lag_14d_mm`** | `float` | $\text{mm}$ | `0,0` | `31,04` | `98,2` | Skumulowany opad z 14 dni poprzedzających (`[t-14, t-1]`). |
| **`precip_lag_21d_mm`** | `float` | $\text{mm}$ | `0,6` | `43,92` | `112,3` | Główny wskaźnik głębokiej akumulacji wilgoci w ściółce dla borowików i podgrzybków (`[t-21, t-1]`). |
| **`is_forecast`** | `bool` | flag | `False` | — | `False` | Wszystkie rekordy reprezentują zweryfikowane obserwacje historyczne z reanalizy. |
| **`is_demo`** | `bool` | flag | `False` | — | `False` | Rzeczywiste dane produkcyjne, niebędące sztuczną próbką interfejsu. |
