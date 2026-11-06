# Audyt raportów o zbiorach z portalu `grzyby.pl` (`[GRZ-03] #3`)

> **Cel zadania:** Sprawdzić na rzeczywistych raportach z portalu `grzyby.pl`, czy da się zbudować zmienną celu (Target), którą model ML ma przewidywać, określić kompletność atrybutów na próbce ok. 50 wpisów oraz zinwentaryzować możliwości automatyzacji i ograniczenia prawne.

---

## 1. Pochodzenie danych i opis zbadanej próbki

W ramach audytu przejrzano ręcznie i skatalogowano **52 rzeczywiste raporty grzybiarzy** z regionu pilotażowego (**województwo dolnośląskie**, kod `DS` / `TERYT 02`):

- **Główna próbka audytowa ($N = 52$ wpisy):**
  - **Źródło:** Tygodniowe zestawienie archiwalne dla woj. dolnośląskiego:  
    [`https://www.grzyby.pl/wystepowanie/wystepowanie-20250929-20251005-DS.htm`](https://www.grzyby.pl/wystepowanie/wystepowanie-20250929-20251005-DS.htm)
  - **Zakres dat zbioru:** `2025-09-29` – `2025-10-05` (pełny tydzień sezonu jesiennego, od poniedziałku do niedzieli).
  - **Zakres identyfikatorów:** od `#DG174079` do `#DG174888`.
  - **Zapisana próbka (bez chronionych prawem autorskim pełnych tekstów, z bezpośrednimi linkami URL do każdego raportu):**  
    [`data/sample/reports_audit_52_ds.csv`](../data/sample/reports_audit_52_ds.csv) (wygenerowana powtarzalnym skryptem [`scripts/audit_grzyby_reports.py`](../scripts/audit_grzyby_reports.py)).
- **Dodatkowa weryfikacja porównawcza:**
  - Bieżące zestawienie tygodniowe dla Dolnego Śląska (`22–28.10.2026`, $N = 21$ wpisów): [`https://www.grzyby.pl/raporty/wystepowanie-DS.htm`](https://www.grzyby.pl/raporty/wystepowanie-DS.htm)
  - Bieżące zestawienie ogólnopolskie ($N = 56$ najnowszych wpisów): [`https://www.grzyby.pl/raporty/wystepowanie-0.htm`](https://www.grzyby.pl/raporty/wystepowanie-0.htm)
  - Podsumowanie sezonowe z jawnym przypisaniem użytkowników do powiatów: [`https://www.grzyby.pl/grzyby-2025.htm`](https://www.grzyby.pl/grzyby-2025.htm)

---

## 2. Statystyka kompletności pól w próbce ($N = 52$ raporty)

Poniższa tabela podsumowuje, ile spośród 52 rzeczywistych wpisów posiada poszczególne informacje:

| Badane pole / atrybut | Liczba wpisów (z 52) | Udział (%) | Sposób zapisu na portalu `grzyby.pl` | Wnioski dla modelu i potoku danych |
| :--- | :---: | :---: | :--- | :--- |
| **Wydajność zbioru (`yield_pcs_per_h`)** | **52 / 52** | **100,0%** | Obowiązkowy prefiks `(X/h)` na początku każdego wpisu (np. `(20/h)`, `(0/h)`). | **Główna zmienna celu (Target):** 100% kompletności, znormalizowana na 1 osobę i 1 godzinę w lesie. |
| **Data zbioru (`harvest_date`)** | **52 / 52** | **100,0%** | Osobny element DOM (`div.w3-text-align-right`): pełna data w archiwum (np. `5.paź 2025`) lub relatywna w bieżącym tygodniu (`dzisiaj`, `wczoraj`, `3 dni temu`). | **100% kompletności.** Zawsze należy używać tej daty (a nie daty publikacji!) do łączenia z pogodą. |
| **Data i godzina publikacji (`publication_date`)** | **52 / 52** | **100,0%** | Znacznik czasu w stopce wpisu (np. `2025.10.6 22:12`). | Przydatna wyłącznie pomocniczo (do wyliczenia daty zbioru z opisów typu `3 dni temu`). |
| **Identyfikator miejsca (`place_id`)** | **52 / 52** | **100,0%** | Stały numeryczny ID w linku `historyczne_z_miejsca/<id>/1` (44 unikalne miejsca w 52 wpisach). | Pozwala trwale zmapować każde `place_id` do współrzędnych `(lat, lon)` i kodu TERYT powiatu (`area_id`). |
| **Współrzędne GPS `(lat, lon)` oraz powiat (po zalogowaniu)** | **52 / 52** | **100,0%** | W pełnej wersji portalu (po zalogowaniu) lewa kolumna zawiera bezpośredni link `google.pl/maps/place/<lat>+<lon>/...` oraz jawną nazwę miejscowości i powiatu (np. `Witków pow. wałbrzyski`, `Krępnica pow. bolesławiecki`). | **Kluczowe odkrycie zrzutu po zalogowaniu:** Mamy **100% pokrycia współrzędnymi GPS i powiatem (18 powiatów w 1 tygodniu)** bez zgadywania! |
| **Lokalizacja jawna w treści (bez logowania)** | **7 / 52** | **13,5%** | Nazwa wsi, miasta lub pasma górskiego wpisana przez autora w treści relacji (np. *Witków*, *Wołów*, *Góry Sowie*, *Minkowice Oławskie*, *Twardogóra*, *Międzybórz*, *Wrocław*). | **Wersja bez konta NIE wystarczy:** bez zalogowania aż **86,5%** wpisów ma ukryty powiat i współrzędne (co więcej, wzmianka w tekście bywa myląca, np. w `DG174602` autorka wspomina „Gruszeczkę” w pow. milickim dla porównania, podczas gdy zbiór po zalogowaniu miał miejsce w **Grabownie Wielkim, pow. oleśnicki `0214`**!). |
| **Gatunki grzybów (`dominant_species`)** | **44 / 52** | **84,6%** | Portal automatycznie linkuje nazwy grzybów w tekście do haseł łacińskich (np. `<a href=".../Boletus_edulis.htm">prawdziwki</a>`). | Bardzo wysoka dostępność i łatwa ekstrakcja po URL-ach łacińskich (`Boletus_edulis`, `Xerocomus_badius`, `Suillus` itd.). |
| **Liczba sztuk ogółem (`total_pieces`)** | **26 / 52** | **50,0%** | Podawana dobrowolnie słownie lub liczbowo w treści wpisu (np. *„60 podgrzybków”*, *„razem ok. 170 szt.”*, *„4 podgrzybki, 2 maślaki”*). | **Nie nadaje się na główną zmienną celu:** brakuje jej w połowie wpisów i zależy od liczby zbieraczy oraz czasu. |
| **Czas zbierania (`harvest_duration_h`)** | **23 / 52** | **44,2%** | Podawany dobrowolnie w tekście (np. *„3 godziny zbierania”*, *„8-12 w lesie”*, *„1,5 godzinki”*, *„20 min”*). | Dostępny w mniej niż połowie raportów (zakres od `0,33 h` do `8,0 h`, mediana `2,0 h`). |
| **Liczba zbieraczy (`pickers_count`)** | **31 / 52** | **59,6%** | Wynika z tekstu lub formy czasownika (np. *„2 os/3 h”*, *„w dwie osoby”* vs. 1. osoba lp. *„wybrałem się”*). | Wpisy wieloosobowe dzielą łączny zbiór przez osobo-godziny przy wyliczaniu `(X/h)`. |
| **Wskazówka o typie lasu (`forest_type_hint`)** | **21 / 52** | **40,4%** | Wzmianki w tekście (np. *„las bukowy”*, *„las świerkowo-sosnowy”*, *„młode sosny”*). | Doskonałe potwierdzenie walidacyjne dla cech siedliskowych z BDL! |

---

## 3. Kluczowe ustalenia metodologiczne

### 3.1. Jak wyliczany jest wskaźnik `(X/h)` (`yield_pcs_per_h`)?
Porównanie wpisów, w których użytkownicy podali jednocześnie liczbę osób, czas w lesie oraz łączną liczbę grzybów, pozwoliło dokładnie odtworzyć wzór stosowany przez portal `grzyby.pl`:

$$\text{yield\_pcs\_per\_h} \approx \frac{\text{łączna liczba zebranych grzybów}}{\text{liczba osób} \times \text{czas zbierania [h]}}$$

**Dowody z rzeczywistych wpisów w próbce:**
- **`DG174714`:** *„3 godziny zbierania 60 podgrzybków brunatnych”* (1 osoba) $\rightarrow \frac{60}{1 \times 3} = \mathbf{20\text{ szt./h}}$ (w nagłówku: `(20/h)`).
- **`DG174449`:** *„W ciągu godziny w dwie osoby zebraliśmy ok. 30 podgrzybków”* $\rightarrow \frac{30}{2 \times 1} = \mathbf{15\text{ szt./h}}$ (w nagłówku: `(15/h)`).
- **`DG174448`:** *„2 os/3 h. Ok. 100 sztuk”* $\rightarrow \frac{100}{2 \times 3} = 16,7 \approx \mathbf{15\text{ szt./h}}$ (w nagłówku: `(15/h)`).
- **`DG174437`:** *„45 grzybów w dwie osoby przez 4 godziny”* $\rightarrow \frac{45}{2 \times 4} = 5,6 \approx \mathbf{5\text{ szt./h}}$ (w nagłówku: `(5/h)`).
- **`DG174614`:** *„1.5 godziny w lesie. Znalezione: 4 prawdziwki, 3 ceglaki, 2 zajączki i 3 kanie”* (12 szt.) $\rightarrow \frac{12}{1 \times 1,5} = \mathbf{8\text{ szt./h}}$ (w nagłówku: `(8/h)`).

> **Wniosek dla obu osób w zespole:** Zmienna `yield_pcs_per_h` jest już na wejściu znormalizowana względem czasu i liczby zbieraczy, ma 100% wypełnienia i idealnie nadaje się jako zmienna celu modelu regresji (oraz do dyskretyzacji na klasy `BRAK` / `SREDNIO` / `WYSYP`).

---

### 3.2. Oddzielenie daty publikacji od daty zbioru
Na portalu `grzyby.pl` grzybiarz podczas dodawania raportu wybiera dzień, w którym faktycznie był w lesie.
- W zbadanej próbce 52 raportów aż **11 wpisów (21,2%) miało datę publikacji późniejszą niż data zbioru** o **1 do 4 dni**:
  - **4 dni opóźnienia:** `DG174823` – zbiór w czwartek `2025-10-02`, publikacja w poniedziałek `2025-10-06 02:51`.
  - **2 dni opóźnienia:** `DG174845` (zbiór `2025-10-04`, publikacja `2025-10-06`), `DG174663` (zbiór `2025-10-03`, publikacja `2025-10-05`), `DG174380` (zbiór `2025-10-01`, publikacja `2025-10-03`).
  - **1 dzień opóźnienia:** 7 raportów (często grzybobranie niedzielne opisywane jest w poniedziałek rano lub wieczorem, np. `DG174888`, `DG174833`, `DG174825`).
- **Zasada w potoku danych:** Do pola `date` w pliku `reports.csv` trafia wyłącznie **`harvest_date`**. W przypadku parsowania strony bieżącej (`/raporty/wystepowanie-DS.htm`), gdzie zamiast daty bezwzględnej widnieje etykieta `dzisiaj` / `wczoraj` / `N dni temu`, datę zbioru wyliczamy względem daty wygenerowania strony w nagłówku (`TYTUL-RAPORTU-TIME`), a nie z samego znacznika publikacji.

---

### 3.3. Oddzielenie braku wartości (`NULL`) od zera (`0`)
Audyt wykazał konieczność rygorystycznego rozróżniania wartości `0` od `NULL`:
1. **Wartość `0` (Potwierdzony brak zbioru):**
   - Występuje wtedy, gdy w nagłówku widnieje **`(0/h)`** (np. `DG174777`: *„(0/h) Lasy w okolicach Wołowa. Grzybów brak. Sucho.”*). Wówczas `yield_pcs_per_h = 0.0` oraz `total_pieces = 0`.
2. **Wartość `NULL` (Brak informacji w tekście):**
   - Gdy użytkownik podaje np. `(10/h) Trzeba się nachodzić. Sucho.` (`DG174325`), wiemy, że zbierał średnio 10 grzybów na godzinę (`yield_pcs_per_h = 10.0`), ale **nie podał** łącznej liczby sztuk, czasu ani gatunków. W takich przypadkach `total_pieces = NULL`, `harvest_duration_h = NULL` oraz `dominant_species = NULL`. Wstawienie tam `0` byłoby błędem logicznym.

---

### 3.4. Czy występują jawne raporty nieudanych zbiorów?
**TAK.** W zbadanej próbce 52 wpisów z Dolnego Śląska odnotowano:
- **2 jawne raporty całkowicie zerowe (`0 szt./h`, 3,8%):**
  - `DG174777` (`2025-10-05`, okolice Wołowa): *„(0/h) Lasy w okolicach Wołowa. Grzybów brak. Sucho.”*
  - `DG174435` (`2025-10-03`, las świerkowy): *„(0/h) Godzinny rekonesans. Las świerkowy, bez efektów. W lesie wilgotno ale grzybów brak w nocy przymrozki.”*
  - *(W bieżącym tygodniu `22–28.10.2026` na stronie ogólnopolskiej również występują jawne raporty `(0/h)`, np. `DG187140` oraz `DG187133`).*
- **12 raportów o marginalnym/nieudanym zbiorze (`1–5 szt./h`, 23,1%):**
  - np. `DG174567` (`1/h`: *„Bida Panie bida 😬 Im dalej w las, tym gorzej”*), `DG174833` (`2/h`: *„prawie nic nie znalazłem, bo 2 prawdziwki nie można nazwać zbiorem”*), `DG174442` (`2/h`: *„efekt dwa podgrzybki zajączki. W lesie wręcz mokro ale grzybów brak”*).
  - W bieżącym tygodniu na Dolnym Śląsku (`22–28.10.2026`) odpowiednikiem zerowego wyjścia są również wpisy z wartością `1/h` (np. `DG187034`: *„(1/h) grzybobranie masakra jeśli chodzi o wyniki. nie ma nic”* – część użytkowników wybiera z listy `1/h` zamiast `0/h`, gdy znajdzie pojedynczego niejadalnego lub starego grzyba).

**Dlaczego użytkownicy w ogóle raportują puste koszyki?**
Portal `grzyby.pl` stosuje mechanizm premiowania: *„za każde doniesienie po zalogowaniu ekstra 2 dni dostępu do pełnej wersji portalu”*. Dzięki temu stali użytkownicy zgłaszają także nieudane rekonesanse.  
**Ograniczenie dla modelu (Reporting Bias):** Mimo systemu bonusów, w tygodniach bezgrzybnych ogólna liczba raportów jest niższa niż w szczycie wysypu, a część nieudanych wyjść ma etykietę `1/h`–`2/h`. Dlatego przyjęty w `docs/mvp.md` próg klasy **`BRAK` jako `< 5 szt./h`** (zamiast samego `= 0`) jest bardzo dobrze uzasadniony empirycznie!

---

## 4. Rzeczywiste przykłady lokalizacji z próbki (Przekazywany rezultat)

Poniżej zestawiono rzeczywiste przykłady lokalizacji z Dolnego Śląska zidentyfikowane podczas audytu wraz z ich przypisaniem do jednostek przestrzennych (`area_id` – powiat TERYT):

| ID raportu | Data zbioru | Przykład rzeczywistej lokalizacji z raportu | Przypisany powiat (`area_id` TERYT) | `place_id` w bazie `grzyby.pl` | Wydajność (`szt./h`) | Gatunki i siedlisko w raporcie |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **`DG174833`** | `2025-10-05` | *„masyw Trójgarbu w pobliżu wioski Witków”* | **pow. wałbrzyski (`0221`)** | `6005` | `2.0` (`BRAK`) | Borowik szlachetny (`Boletus_edulis`), las bukowy |
| **`DG174777`** | `2025-10-05` | *„Lasy w okolicach Wołowa”* | **pow. wołowski (`0222`)** | `438` | `0.0` (`BRAK`) | Brak grzybów, sucho |
| **`DG174762`** | `2025-10-05` | *„wycieczka w Góry Sowie”* | **pow. dzierżoniowski (`0202`)** | `10784` | `20.0` (`SREDNIO`/`WYSYP`) | Prawdziwek (48), podgrzybek brunatny (47), aksamitki (61), ceglastopory (10) |
| **`DG174686`** | `2025-10-05` | *„Okolice Minkowic Oławskich”* | **pow. oławski (`0215`)** | `4313` | `10.0` (`SREDNIO`) | Podgrzybek, borowik, siedzuń sosnowy, kania |
| **`DG174448`** | `2025-10-03` | *„w twardogórskich lasach”* (Twardogóra) | **pow. oleśnicki (`0214`)** | `1454` | `15.0` (`SREDNIO`) | Maślaki (40), podgrzybki (30), borowiki (12), koźlarze (10), rydze (10), kanie (5) |
| **`DG174246`** | `2025-10-01` | *„Chojnik i Międzybórz”* | **pow. oleśnicki (`0214`)** | `3150` | `10.0` (`SREDNIO`) | Różne gatunki leśne (`6.5 h` w lesie) |
| **`DG174602`** | `2025-10-04` | odniesienie do *„Gruszeczki”* (Dolina Baryczy) | **pow. milicki (`0213`)** | `2403` | `20.0` (`SREDNIO`) | Podgrzybek brunatny, koźlarz babka, borowik szlachetny |
| **`DG174153`** | `2025-09-30` | *„Park Kleciński / Park Krzycki, wzdłuż Ślęzy”* | **m. Wrocław (`0264`)** | `3716` | `14.0` (`SREDNIO`) | Czernidłak kołpakowaty (teren miejski/parkowy – przykład obserwacji odstającej siedliskowo!) |
| **`DG178322`** *(sezon 2025)* | `2025-12` | jawne podsumowanie sezonowe (`grzyby-2025.htm`) | **pow. świdnicki (`0219`)** | — | — | Podsumowanie sezonu dla powiatu świdnickiego |

> **Ważna obserwacja dla zespołu (Case `DG174153` – Park Kleciński we Wrocławiu):**  
> Wpis `DG174153` dotyczy zbierania czernidłaków na miejskich trawnikach we Wrocławiu (`0264`). Pokazuje to, że przy budowie zbioru treningowego warto odfiltrowywać grzyby pozaleśne/trawnikowe lub opierać się na słowniku głównych gatunków leśnych (`borowik_szlachetny`, `podgrzybek_brunatny`, `kurka`, `maslak_zwyczajny`, `kania`, `koźlarz`, `rydz`, `opieńka`).

---

## 5. Możliwości automatyzacji oraz ograniczenia wykorzystania i publikacji danych

### 5.1. Porównanie: Wersja bez logowania vs. Zrzut z konta zalogowanego (`Wnioski dla automatyzacji`)
Porównanie tej samej strony tygodniowej (`29.09–05.10.2025`, $N = 52$ raporty) pobranej bez logowania oraz zapisanej z zalogowanego konta (`data/raw/grzybobrania od 20250929 do 20251005 dla DS.html`) dało jednoznaczną odpowiedź:

1. **Dane bez konta NIE wystarczą do przypisania powiatów:**
   - Bez zalogowania lokalizacja jest ukryta w **86,5% raportów** (45 z 52 wpisów).
   - Co więcej, poleganie na nazwach miejscowości wspomnianych w treści komentarza prowadzi do błędów (np. w `DG174602` autorka pisze w tekście: *„pojawiły się też robaczki – więcej niż w Gruszeczce”* [pow. milicki `0213`], natomiast po zalogowaniu okazuje się, że zbiór miał miejsce w **Grabownie Wielkim, pow. oleśnicki (`0214`)**!).
2. **Co daje strona z konta zalogowanego (100% precyzji przestrzennej):**
   - W 100% raportów (`52 / 52`) lewa kolumna zawiera gotowy link Google Maps ze współrzędnymi geograficznymi `WGS84` oraz nazwą miejscowości i powiatu, np.:  
     `<a href="https://www.google.pl/maps/place/50.796974+16.130848/@50.796974,16.130848,12z">Witków pow. wałbrzyski</a>`
   - W jednym tygodniu zbadanej próbki 52 raporty pokryły aż **18 różnych powiatów Dolnego Śląska** (`0201`, `0202`, `0205`, `0206`, `0207`, `0208`, `0213`, `0214`, `0215`, `0216`, `0218`, `0219`, `0220`, `0221`, `0222`, `0223`, `0264`, `0265`).
   - Posiadanie dokładnych współrzędnych `(lat, lon)` umożliwia w GeoPandas nie tylko bezbłędne przypisanie do powiatu, ale również przecięcie punktu/bufora bezpośrednio z poligonami leśnymi BDL!
3. **Jak zautomatyzować pobieranie z kontem zalogowanym (bez ręcznego klikania `Cmd+S` setki razy):**
   - **Opcja A (Automatyczny skrypt Python z ciasteczkiem sesyjnym):** Po zalogowaniu w przeglądarce wystarczy skopiować ciasteczko sesyjne do lokalnego pliku `.env` (`GRZYBY_SESSION_COOKIE=...`), a skrypt w Pythonie (`requests.Session`) sam pobierze strony tygodniowe dla Dolnego Śląska (`/wystepowanie/wystepowanie-YYYYMMDD-YYYYMMDD-DS.htm` – to tylko ok. **15–20 plików HTML na cały sezon jesienny!**) z grzecznym opóźnieniem `sleep(4)` i zapisze je lokalnie w `data/raw/`.
   - **Opcja B (Słownik `place_id -> (lat, lon, powiat)`):** Ponieważ każdy raport nawet bez logowania posiada stały identyfikator miejsca `place_id` (`historyczne_z_miejsca/<place_id>/1`), każde raz poznane `place_id` z zalogowanego zrzutu automatycznie zapisujemy w lokalnym słowniku `data/raw/place_id_to_locality.json` (z pierwszego pliku mamy już 44 zmapowane lokalizacje!).

### 5.2. Ograniczenia techniczne (`robots.txt`)
Zbadano plik [`https://www.grzyby.pl/robots.txt`](https://www.grzyby.pl/robots.txt):
- Dla `User-agent: *` zablokowane (`Disallow`) są podkatalogi dynamiczne i prywatne, m.in.: `/pelna/`, `/logowanie`, `/cgi-local/`, `/doniesieniawpoblizu/`, `/innedoniesieniainformatora/`, `/szukaj/`, `/wystepowanie_grzybow/`, `/wystepowanie-foto/`.
- Strony raportów tygodniowych `/raporty/wystepowanie-*.htm`, archiwów sezonowych `/grzyby-202*.htm` oraz `/wystepowanie/wystepowanie-*.htm` **nie są zablokowane** w `robots.txt`.
- Mimo to, ze względu na ochronę serwera przed przeciążeniem (na portalu znajduje się osobna strona `/expl.htm` o przeciążeniach), pobieranie stron musi odbywać się z poszanowaniem serwera: **maksymalnie 1 zapytanie na 3–5 sekund (`sleep`) z lokalnym cache’owaniem plików w `data/raw/`**, tak aby żadnej strony nie pobierać dwukrotnie.

### 5.3. Zgoda autora portalu, ograniczenia prawne i zasady publikacji
Zbadano dokumenty [`/copyright.htm`](https://www.grzyby.pl/copyright.htm), [`/regulamin.htm`](https://www.grzyby.pl/regulamin.htm) oraz **uzyskano bezpośrednią zgodę autora portalu `grzyby.pl`, p. Marka Snowarskiego**, na wykorzystanie danych w ramach projektu akademickiego („na zaliczenie”):

> **Stanowisko i zgoda autora portalu (`Marek Snowarski, grzyby.pl`):**  
> *„Jeśli chodzi o pracę typu «na zaliczenie», licencjacką lub temu podobną tj. bez planowanego dalszego bujnego życia «w internecie» to nie ma problemu z mojej strony, aby skorzystać z różnych danych jak Pan potrafi. Jeśli miałoby być to coś promowane na dłuższy dystans w internecie, to siłą rzeczy jest to w mojej niszy i dla mnie potencjalnie szkodliwy konkurent. Wtedy nie wyrażam zgody i mogę utrudniać dostęp do danych.”*

**Zasady obowiązujące w projekcie na podstawie uzyskanej zgody:**
1. **Charakter wyłącznie akademicki / dydaktyczny:** Projekt jest realizowany wyłącznie jako praca zaliczeniowa (studencka) i nie będzie promowany ani utrzymywany jako publiczny, konkurencyjny serwis internetowy po zakończeniu zaliczenia przedmiotu.
2. **Wyraźna atrybucja źródła i klauzula akademicka w aplikacji Streamlit oraz README:** W stopce aplikacji i dokumentacji umieszczamy podpis:  
   `Dane o zbiorach pochodzą z portalu grzyby.pl – (c) Marek Snowarski, Grzyby Polski, atlas grzybów (www.grzyby.pl). Projekt wyłącznie o charakterze dydaktycznym (na zaliczenie przedmiotu), niekomercyjny.`
3. **Higiena repozytorium i poszanowanie serwera:**
   - Surowe pliki HTML pobrane z sesją zalogowanego użytkownika oraz dane logowania (`.env`) przechowujemy lokalnie w katalogu `data/raw/` (zablokowanym w `.gitignore`).
   - W repozytorium GitHub publikujemy wyłącznie ustrukturyzowane, przetworzone tablice cech potrzebne do odtworzenia eksperymentu i demonstracji projektu na zaliczenie.
   - Pobieranie danych odbywa się z lokalnym cache’owaniem i odstępami czasowymi (`sleep`), aby nie obciążać serwera `grzyby.pl`.

---

## 6. Podsumowanie: Ograniczenia etykiet (Labels) dla modelu ML

Informacja zwrotna dla obu osób w zespole przed zadaniem `[GRZ-06]`:
1. **Zmienna celu (`yield_pcs_per_h`) jest dostępna w 100% raportów**, ale jest obarczona subiektywnym zaokrąglaniem przez grzybiarzy (częste wartości okrągłe: `5`, `10`, `15`, `20`, `30`, `50 szt./h`). Dlatego obok regresji (`MAE`/`RMSE`) warto wspierać klasyfikację w 3 przedziałach (`BRAK: < 5`, `SREDNIO: 5–20`, `WYSYP: > 20`).
2. **Zjawisko rzadkości przestrzenno-czasowej (Sparsity):** W pojedynczym tygodniu dla całego Dolnego Śląska pojawia się ok. `20–60` raportów, co oznacza, że nie każdy z 30 powiatów ma raport w każdym dniu. Model ML powinien zatem uczyć się na parach `(powiat, dzień_z_raportem)` z cechami siedliskowymi (BDL) i pogodowymi (`lag 7d/14d/21d`), a następnie dokonywać predykcji dla wszystkich 30 powiatów regionu.
3. **Gatunki vs. Ogółem:** W `84,6%` raportów znane są gatunki, jednak w większości relacji grzybiarze zbierają miks kilku gatunków jednocześnie, a `yield_pcs_per_h` dotyczy całego koszyka. Oznacza to, że bazowy model ML powinien przewidywać wydajność ogólną (`species_filter = "all"`), a szanse dla konkretnego gatunku (np. borowik vs. podgrzybek) wyliczać poprzez ważenie predykcji udziałem preferowanego drzewostanu z BDL (`oak_beech_share_pct` vs. `pine_share_pct`) lub osobny model warunkowy.
