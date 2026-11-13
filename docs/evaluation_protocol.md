# Protokół Ewaluacji i Modele Odniesienia (Baseline)

> **Status:** Zatwierdzony dla zadania `[GRZ-13] #13`  
> **Powiązania:** Podstawa eksperymentów dla modeli uczenia maszynowego `[GRZ-14]` oraz końcowej ewaluacji `[GRZ-17]`.

---

## 1. Cel i zasady metodologiczne

Głównym celem protokołu ewaluacji jest zapewnienie **uczciwego, stabilnego i wolnego od wycieków (leak-free) punktu odniesienia** dla modeli uczenia maszynowego.

### Kluczowe zasady:
1. **Podział czysto czasowy (Temporal Split / Out-of-Time):**
   - Grzybobranie to zjawisko silnie uwarunkowane fenologią jesieni i dynamiką pogody. Klasyczny losowy K-Fold CV doprowadziłby do drastycznego przecieku informacji czasowej (prognozowanie przeszłości na podstawie wiedzy o przyszłości).
   - Wszystkie modele uczą się wyłącznie na danych ze wcześniejszej części sezonu, a testowane są na późniejszych tygodniach.
2. **Zero wycieku wycieczek (No Trip Leakage):**
   - Raporty z tej samej daty i wycieczki lub ich warianty horyzontowe nie mogą być rozdzielone pomiędzy zbiory. Podział sztywną datą odcięcia gwarantuje $100\%$ separację.
3. **Dopasowanie wyłącznie na zbiorze treningowym:**
   - Wszelkie transformacje, średnie, dominanty i przyszłe skalery cech (GRZ-14) są fitowane wyłącznie na części `Train`.
4. **Uczciwy punkt odniesienia bez obietnicy sukcesu ML:**
   - Model ML nie ma gwarancji pokonania prostego baseline'u. Celem badania jest weryfikacja, czy opady opóźnione (lags 7/14/21 dni) oraz siedliska BDL wnoszą istotny zysk informacyjny ponad statyczną średnią lub dominującą klasę.

---

## 2. Zdefiniowany podział danych (Temporal Splits)

Łączna liczba rekordów treningowych spełniających kryteria jakościowe (zob. `docs/reports_etl_quality.md`): **513**.

### 2.1. Główny podział: Train / Test (Out-of-Time)
Używany do ostatecznego porównania modeli w `[GRZ-14]` i `[GRZ-17]`:

| Podzbiór | Zakres dat | Liczba raportów | Udział % | Pokryte powiaty | Średnia wydajność | Mediana wydajności |
|---|---|---|---|---|---|---|
| **Treningowy (Train)** | `2025-09-01` – `2025-10-12` (Tygodnie 1–6) | **386** | **75.24%** | 27 / 30 | 16.55 szt./h | 13.5 szt./h |
| **Testowy (Test)** | `2025-10-13` – `2025-11-02` (Tygodnie 7–9) | **127** | **24.76%** | 21 / 30 | 24.76 szt./h | 20.0 szt./h |

#### Rozkład klas obfitości (`abundance_class`):
- **Train:** `BRAK`: 97 (25.13%), `SREDNIO`: 182 (47.15%), `WYSYP`: 107 (27.72%)
- **Test:** `BRAK`: 15 (11.81%), `SREDNIO`: 57 (44.88%), `WYSYP`: 55 (43.31%)

### 2.2. Pomocniczy podział 3-częściowy: Train / Val / Test
Przeznaczony do strojenia hiperparametrów w `[GRZ-14]`:
- **Train:** `2025-09-01` – `2025-10-05` ($N = 333$, 64.91%)
- **Val:** `2025-10-07` – `2025-10-19` ($N = 111$, 21.64%)
- **Test:** `2025-10-20` – `2025-11-02` ($N = 69$, 13.45%)

---

## 3. Wyniki modeli bazowych (Baseline Benchmark)

Wszystkie estymatory bazowe zostały dopasowane **wyłącznie na zbiorze treningowym ($N = 386$)** i ocenione na zbiorze testowym ($N = 127$).

### 3.1. Zadanie regresji: Wydajność zbioru `yield_pcs_per_h` [szt./h]

| Model odniesienia | Opis estymatora | Parametr z Train | MAE [szt./h] (Test) | RMSE [szt./h] (Test) | Spearman rho | $R^2$ (Test) | Interpretacja $R^2$ |
|---|---|---|---|---|---|---|---|
| **Global Mean** | Stała średnia ze zbioru treningowego | `16.55 szt./h` | **15.726** | **23.746** | 0.000 | -0.1356 | Nieinterpretowalny ($< 0$) |
| **Global Median** | Stała mediana (minimalizująca MAE na train) | `13.5 szt./h` | **16.433** | **24.966** | 0.000 | -0.2553 | Nieinterpretowalny ($< 0$) |
| **Powiat Hist. Mean** | Średnia historyczna per powiat (z fallbackiem) | Średnie 28 powiatów | **15.286** | **23.348** | 0.3156 | -0.0979 | Nieinterpretowalny ($< 0$) |

> [!NOTE]
> **Dlaczego współczynnik $R^2$ w teście poza czasem jest ujemny?**  
> W testach szeregów czasowych średnia zbioru testowego (24.76 szt./h) różni się od średniej zbioru treningowego (16.55 szt./h) ze względu na dynamikę sezonu grzybowego (koniec października przynosi spadek owocnikowania i wczesne przymrozki). Metryka $R^2$ porównuje model do wariancji wokół *średniej testowej*, przez co stały predyktor ze zbioru treningowego uzyskuje ujemny wynik. W związku z tym w projekcie **główną miarą regresji jest MAE oraz korelacja rangowa Spearmana**, a $R^2$ pełni jedynie funkcję diagnostyczną.

---

### 3.2. Zadanie klasyfikacji: Obfitość wysypu `abundance_class` (`BRAK`, `SREDNIO`, `WYSYP`)

| Model odniesienia | Opis estymatora | Predykcja | Macro-F1 (Test) | Weighted-F1 (Test) | Uwagi |
|---|---|---|---|---|---|
| **Majority Class** | Najczęstsza klasa w treningu | Zawsze `SREDNIO` | **0.2065** | 0.2781 | Klasyczny efekt załamania klas mniejszościowych |
| **Powiat Hist. Mode** | Dominanta per powiat z treningu | Zróżnicowana per powiat | **0.3918** | 0.3884 | Uchwycenie lokalnego tła geograficznego |
| **Stratified Random** | Losowanie z rozkładu treningowego | Proporcjonalne | **0.3512** | 0.3927 | Stochastyczny baseline |

#### Macierze pomyłek na zbiorze testowym ($N = 127$):

**1. Majority Class (zawsze `SREDNIO`):**
```text
Rzeczywiste \ Predykcja    BRAK    SREDNIO    WYSYP
BRAK                        0       15         0      
SREDNIO                     0       57         0      
WYSYP                       0       55         0      
```
*Macro-F1 wynosi zaledwie 0.2065, ponieważ czułość (Recall) dla klas `BRAK` i `WYSYP` wynosi dokładnie 0.0.*

**2. Powiat Historical Majority:**
```text
Rzeczywiste \ Predykcja    BRAK    SREDNIO    WYSYP
BRAK                        7       8          0      
SREDNIO                     10      45         2      
WYSYP                       2       47         6      
```
*Zwiększenie Macro-F1 do 0.3918 dzięki poprawnemu rozpoznaniu części przypadków `BRAK` i `WYSYP`.*

---

## 4. Kryteria sukcesu i cele dla modeli ML (`[GRZ-14]`, `[GRZ-17]`)

Zgodnie z zatwierdzoną metodyką `[GRZ-13]`, ustala się następujące formalne progi oceny dla kolejnych zadań modelowania:

1. **Dla modelu regresyjnego (`yield_pcs_per_h`):**
   - **Próg odniesienia (Baseline):** MAE = `15.726 szt./h` (Global Mean) oraz `15.286 szt./h` (Powiat Mean).
   - **Kryterium sukcesu ML:** Uzyskanie na zbiorze testowym **MAE < 14.5 szt./h** oraz istotnie dodatniej korelacji monotonicznej **Spearman $\rho > 0.20$**.
2. **Dla modelu klasyfikacyjnego (`abundance_class`):**
   - **Próg odniesienia (Baseline):** Macro-F1 = `0.2065` (Majority Class) oraz `0.3918` (Powiat Majority).
   - **Kryterium sukcesu ML:** Uzyskanie na zbiorze testowym **Macro-F1 $> 0.45$** z niezerową czułością dla wszystkich 3 klas (`Recall > 0.15` dla `BRAK` i `WYSYP`).
