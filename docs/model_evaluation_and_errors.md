# Końcowa Ocena Modelu na Odłożonym Zbiorze Testowym i Analiza Błędów (`[GRZ-17] #17`)

> **Cel:** Uczciwa ocena zakresu zaufania do predykcji modelu `rf_pipeline_v1.0` na odłożonych danych bez żadnego dostrajania do zbioru testowego (`retuned_on_test = False`).  
> **Okres treningowy (`Train`):** `2025-09-01 .. 2025-10-12` ($N = 386$ obserwacji)  
> **Okres testowy Out-of-Time (`Test`):** `2025-10-13 .. 2025-11-02` ($N = 127$ obserwacji, `21 / 30` powiatów)

---

## 1. Zestawienie ogólne: Model ML vs Baseline (`Warunek #1`)

### 1.1. Regresja (`yield_pcs_per_h [szt./h]`)

| Model | N (Test) | MAE `[szt./h]` | RMSE `[szt./h]` | Spearman $\rho$ | $R^2$ (diagnostycznie) |
|---|---:|---:|---:|---:|---:|
| **RandomForestRegressor (`rf_yield_pipeline.joblib`)** | `127` | **14.669** | **22.292** | **+0.3662** | -0.0008 |
| *Baseline: Powiat Historical Mean* | `127` | *15.286* | *23.348* | *+0.3156* | *-0.0979* |
| *Baseline: Global Mean (`16.55 szt./h`)* | `127` | *15.726* | *23.746* | *0.0000* | *-0.1356* |

### 1.2. Klasyfikacja (`abundance_class`: `BRAK`, `SREDNIO`, `WYSYP`)

| Model | N (Test) | Macro-F1 | Weighted-F1 | Recall `BRAK` | Recall `SREDNIO` | Recall `WYSYP` |
|---|---:|---:|---:|---:|---:|---:|
| **RandomForestClassifier (`rf_abundance_pipeline.joblib`)** | `127` | **0.4518** | **0.4655** | 0.53 | 0.51 | 0.40 |
| *Baseline: Powiat Historical Majority* | `127` | *0.3918* | *0.3884* | *0.47* | *0.79* | *0.11* |
| *Baseline: Majority Class (`SREDNIO`)* | `127` | *0.2065* | *0.2781* | *0.00* | *1.00* | *0.00* |

**Macierz pomyłek klasyfikatora na odłożonym teście ($N = 127$):**
```text
Rzeczywiste \ Predykcja    BRAK    SREDNIO    WYSYP
BRAK (n=15)                 8       3          4      
SREDNIO (n=57)              12      29         16     
WYSYP (n=55)                4       29         22     
```

---

## 2. Analiza błędów w grupach (`Warunek #2`)

### 2.1. Degradacja błędu wraz z horyzontem prognozy ($H = 1, 2, 3$ dni)
Wszystkie cechy dla $H \in \{1, 2, 3\}$ wygenerowano ściśle *Point-in-Time* przez `build_prediction_input_row(issued_at = target_date - H)`:

| Horyzont | N | MAE `[szt./h]` | RMSE `[szt./h]` | Spearman $\rho$ | Macro-F1 | Bije Baseline Regresji (`15.286`)? | Bije Baseline Klasyfikacji (`0.3918`)? |
|---|---:|---:|---:|---:|---:| :---: | :---: |
| **H = 1 dzień/dni** | `127` | **14.669** | 22.292 | +0.3662 | **0.4518** | TAK | TAK |
| **H = 2 dzień/dni** | `127` | **14.707** | 22.222 | +0.3567 | **0.3686** | TAK | NIE |
| **H = 3 dzień/dni** | `127` | **15.161** | 22.541 | +0.2993 | **0.3809** | TAK | NIE |

> [!IMPORTANT]
> **Kluczowa obserwacja horyzontowa (`Warunek #4`):**  
> Model klasyfikacyjny (`rf_abundance_pipeline`) poprawia regionalny baseline (`0.3918`) **wyłącznie dla horyzontu $H = 1$ dzień (`0.4518`)**. Dla horyzontów $H = 2$ i $H = 3$ dni jego `Macro-F1` spada do `0.3686–0.3809` (poniżej prostej dominanty powiatowej!), ponieważ potok był uczony wyłącznie na cechach z odcięciem $H = 1$. Komunikat w aplikacji Streamlit oraz prezentacji wyraźnie o tym informuje.

### 2.2. Błędy według prawdziwej klasy obfitości (efekt kompresji wariancji)

| Prawdziwa klasa | N | MAE `[szt./h]` | Średnia prawdziwa | Średnia prognozowana | Obciążenie (`Pred - True`) |
|---|---:|---:|---:|---:|---:|
| **`BRAK`** | 15 | **12.37** | 1.77 | 14.14 | +12.37 |
| **`SREDNIO`** | 57 | **6.19** | 13.04 | 16.79 | +3.76 |
| **`WYSYP`** | 55 | **24.08** | 43.18 | 19.14 | -24.04 |

- Model osiąga bardzo dobrą precyzję w dominującej strefie umiarkowanych zbiorów **`SREDNIO`** (`MAE = 6.19 szt./h`).
- Dla skrajnych wysypów **`WYSYP`** (`> 20 szt./h`, średnio `43.18 szt./h`) `RandomForestRegressor(max_depth=5)` systematycznie **niedoszacowuje** (`MeanPred = 19.14 szt./h`, bias `-24.04 szt./h`), natomiast dla pustych koszyków **`BRAK`** lekko zawyża prognozę (`+12.37 szt./h`).

### 2.3. Błędy według powiatów i analiza pokrycia testowego

W zbiorze testowym (`2025-10-13` – `2025-11-02`) wystąpiło **`21 / 30` powiatów**. Dla **`9` powiatów** brak obserwacji w okresie testowym (`0203` powiat głogowski, `0209` powiat legnicki, `0217` powiat strzeliński, `0224` powiat ząbkowicki, `0225` powiat zgorzelecki, `0226` powiat złotoryjski, `0261` powiat m. Jelenia Góra, `0262` powiat m. Legnica, `0265` powiat m. Wałbrzych), co stanowi istotne ograniczenie statystyczne oceny regionalnej.

| TERYT | Powiat | N (Train) | N (Test) | MAE Test `[szt./h]` | Średnia True | Średnia Pred |
| :---: | :--- |---:|---:|---:|---:|---:|
| `0212` | powiat lwówecki | 0 | **1** | **0.46** | 20.00 | 20.46 |
| `0264` | powiat m. Wrocław | 6 | **1** | **4.19** | 20.00 | 15.81 |
| `0221` | powiat wałbrzyski | 10 | **3** | **5.45** | 10.00 | 10.30 |
| `0202` | powiat dzierżoniowski | 13 | **6** | **6.02** | 6.17 | 10.79 |
| `0215` | powiat oławski | 6 | **9** | **6.87** | 20.56 | 22.86 |
| `0208` | powiat kłodzki | 23 | **1** | **9.78** | 20.00 | 10.22 |
| `0207` | powiat kamiennogórski | 6 | **3** | **10.06** | 5.17 | 13.14 |
| `0211` | powiat lubiński | 7 | **1** | **10.47** | 0.00 | 10.47 |
| `0220` | powiat trzebnicki | 28 | **10** | **10.58** | 16.10 | 12.97 |
| `0204` | powiat górowski | 9 | **3** | **10.89** | 29.67 | 23.38 |
| `0223` | powiat wrocławski | 14 | **2** | **11.15** | 15.00 | 13.15 |
| `0216` | powiat polkowicki | 12 | **3** | **11.29** | 33.33 | 22.04 |
| `0206` | powiat karkonoski | 13 | **5** | **12.15** | 38.00 | 29.15 |
| `0214` | powiat oleśnicki | 39 | **20** | **12.29** | 20.70 | 13.87 |
| `0218` | powiat średzki | 13 | **4** | **12.79** | 24.50 | 12.89 |
| `0219` | powiat świdnicki | 11 | **3** | **13.04** | 10.33 | 9.79 |
| `0201` | powiat bolesławiecki | 85 | **16** | **13.78** | 24.50 | 21.08 |
| `0205` | powiat jaworski | 10 | **3** | **15.26** | 30.00 | 27.29 |
| `0222` | powiat wołowski | 12 | **7** | **16.66** | 27.86 | 11.20 |
| `0213` | powiat milicki | 33 | **25** | **27.21** | 38.68 | 20.37 |
| `0210` | powiat lubański | 6 | **1** | **38.82** | 60.00 | 21.18 |

- **Najwyższy błąd regionalny:** `0213 powiat milicki` (`N = 25`, `MAE = 27.21 szt./h`) — wynik powtarzających się w drugiej połowie października skrajnych zgłoszeń **`100.0 szt./h`**, których drzewo o głębokości `5` nie jest w stanie wyekstrapolować.
- **Najniższe błędy:** `0202 powiat dzierżoniowski` (`MAE = 6.02`), `0215 powiat oławski` (`MAE = 6.87`), `0220 powiat trzebnicki` (`MAE = 10.58`).

### 2.4. Słabe przypadki — Top-6 największych błędów jednostkowych

| `report_id` | Powiat | Data | Prawdziwy plon | Prognoza ML | Błąd `|e|` | Klasa True $\rightarrow$ Pred |
|---|---|---|---:|---:|---:|---|
| `DG176065` | `0213` powiat milicki | `2025-10-18` | **100.0** | 17.9 | **82.1** | `WYSYP` $\rightarrow$ `WYSYP` |
| `DG176742` | `0213` powiat milicki | `2025-10-26` | **100.0** | 18.0 | **82.0** | `WYSYP` $\rightarrow$ `SREDNIO` |
| `DG176193` | `0213` powiat milicki | `2025-10-19` | **100.0** | 20.0 | **80.0** | `WYSYP` $\rightarrow$ `SREDNIO` |
| `DG177137` | `0213` powiat milicki | `2025-11-02` | **100.0** | 21.9 | **78.1** | `WYSYP` $\rightarrow$ `SREDNIO` |
| `DG176595` | `0213` powiat milicki | `2025-10-25` | **100.0** | 21.9 | **78.0** | `WYSYP` $\rightarrow$ `SREDNIO` |
| `DG177253` | `0201` powiat bolesławiecki | `2025-11-01` | **80.0** | 28.8 | **51.2** | `WYSYP` $\rightarrow$ `WYSYP` |

---

## 3. Ograniczenia samodzielnych raportów i nierównej aktywności (`Warunek #3`)

1. **Nierówna aktywność użytkowników w czasie i przestrzeni:**
   - W teście `74 / 127` (`58.3%`) raportów pochodzi z weekendów (`MAE = 16.06` vs `12.73` w dni robocze).
   - `21 / 30` powiatów ma przynajmniej 1 obserwację w teście, ale większość z nich ma próbę `N < 5`, a `9` powiatów nie ma ani jednego raportu testowego — w tych powiatach **nie da się wiarygodnie zmierzyć błędu lokalnego**.
2. **Subiektywizm i zaokrąglenia etykiety (`szt./h`):**
   - Zbieracze często podają wartości zaokrąglone (`10, 15, 20, 30, 50, 100 szt./h`), a ekstremalne zbiory w Dolinie Baryczy (`100 szt./h`) stanowią szum odstający, zaniżający średnią dokładność regresji (`RMSE = 22.29 szt./h`).
3. **Brak danych taksacyjnych BDL dla części powiatów (`12 / 30`):**
   - Dla `N = 13` rekordów testowych bez profilu BDL (`MAE = 12.01 szt./h`) model opiera się wyłącznie na ogólnej lesistości GUS i pogodzie.

---

## 4. Kalibracja i uczciwy opis przydatności modelu (`Warunek #4 i #5`)

1. **Brak skalibrowanego prawdopodobieństwa (`Warunek #5`):**
   - Zarówno surowy wynik regresji `predicted_yield_pcs_per_h` `[szt./h]`, jak i wskaźnik `confidence_score` (`0.0–1.0`, heurystyka dopasowania siedliska i kompletności danych) oraz surowe głosy drzew `predict_proba` (Brier score per klasa: `BRAK = 0.1116`, `SREDNIO = 0.2558`, `WYSYP = 0.2367`) **NIE stanowią skalibrowanego prawdopodobieństwa** znalezienia grzybów i nie są w ten sposób przedstawiane w aplikacji ani sprawozdaniu.
2. **Kiedy wynikom modelu można ufać (praktyczna przydatność):**
   - Model jest **przydatny jako narzędzie wspomagające ranking powiatów na kolejny dzień (`H = 1`)** (`Spearman = +0.3662`, `MAE = 14.67 < 15.29` baseline'u, `Macro-F1 = 0.4518 > 0.3918` baseline'u) i poprawnie identyfikuje umiarkowane zbiory (`SREDNIO`, `MAE = 6.19 szt./h`).
   - **Nie należy traktować prognoz dosłownie** jako gwarancji konkretnej liczby sztuk przy bardzo dużych wysypach (`> 40 szt./h`) ani ufać klasyfikacji na horyzonty **`H = 2, 3` dni**, gdzie skuteczność klasyfikatora spada poniżej prostego baseline'u powiatowego.
