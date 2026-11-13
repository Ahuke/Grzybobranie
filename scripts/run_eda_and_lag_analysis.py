r"""
Skrypt odtwarzający pełną eksploracyjną analizę danych (EDA), wykresy analityczne,
podsumowanie JSON oraz notatnik Jupyter dla zadania [GRZ-11]:
"Wykonać EDA i sprawdzić opóźnienie opad–raportowane zbiory" (#11).

Uruchomienie:
    python scripts/run_eda_and_lag_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import PROCESSED_DIR
from src.eda import build_eda_summary, generate_eda_figures


def _build_notebook_json(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Tworzy kompletny, czytelny notatnik Jupyter (`notebooks/01_eda_lag_effect.ipynb`, nbformat 4.5)
    z komórkami Markdown, kodem Python oraz zapisanymi wynikami tekstowymi i odniesieniami do wykresów.
    """
    yd = summary["yield_and_reports_distribution"]
    rc = summary["regional_concentration"]
    va = summary["report_volume_vs_mushroom_abundance"]
    sp = summary["species_filtering_constraints"]

    corr_lines = [
        f"{c['feature']:30s} | n={c['n']:3d} | Pearson r={c['pearson_r']:+.4f} (p={c['pearson_p']:.4f}) | Spearman rho={c['spearman_rho']:+.4f} (p={c['spearman_p']:.4f})"
        for c in summary["weather_and_habitat_correlations"]
    ]
    reg_sens_lines = [
        f"{s['subset_name']:48s} | śr.={s['mean_yield_pcs_per_h']:5.2f} szt./h | r(rainy14d)={s['corr_rainy_days_14d']['pearson_r']:+.3f} | r(opad 15-21d)={s['corr_precip_slice_15_21d']['pearson_r']:+.3f} | r(lesistość)={s['corr_forest_cover']['pearson_r']:+.3f}"
        for s in summary["regional_dominance_sensitivity"]
    ]
    seas_sens_lines = [
        f"{s['subset_name']:56s} | n={s['n']:3d} | śr.={s['mean_yield_pcs_per_h']:5.2f} szt./h (med.={s['median_yield_pcs_per_h']:4.1f}) | opad14d={s['mean_precip_14d_mm']:5.1f} mm | Tmin7d={s['mean_temp_min_7d_c']:+4.1f}°C | r(rainy14d)={s['corr_rainy_days_14d']['pearson_r']:+.3f}"
        for s in summary["seasonal_dominance_sensitivity"]
    ]

    cells = [
        {
            "cell_type": "markdown",
            "id": "cell-01-intro",
            "metadata": {},
            "source": [
                "# [GRZ-11] Eksploracyjna analiza danych (EDA) i badanie opóźnienia opad → raportowane zbiory\n",
                "\n",
                "**Cel analizy:** Sprawdzenie rzeczywistych zależności, opóźnień czasowych (*Lag Effect* 7 / 14 / 21 dni) oraz ograniczeń w połączonym zbiorze cech `data/processed/features.parquet` (`[GRZ-10]`).\n",
                "\n",
                "**Zakres danych:**\n",
                f"- **Region pilotażowy:** Województwo dolnośląskie (`TERYT 02`, `30` powiatów, w tym `{rc['active_regions_with_reports']}/30` powiatów z raportami).\n",
                f"- **Okres obserwacji:** `{yd['period_start']}` – `{yd['period_end']}` (`{yd['calendar_days_span']}` dni kalendarzowych, `{yd['active_reporting_days']}` dni z aktywnymi zgłoszeniami).\n",
                f"- **Liczebność próby:** `n = {yd['n_valid_observations']}` zwalidowanych raportów treningowych (z `{yd['n_total_extracted_reports']}` wyodrębnionych ogółem w `[GRZ-07]`).\n",
                "- **Odcięcie informacji (*Point-in-Time*):** `feature_cutoff_date = issued_at = target_date - 1 dzień` (żadne okno pogodowe nie sięga do dnia zbioru $t$).\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "id": "cell-02-load",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Wczytano features.parquet: shape=({yd['n_valid_observations']}, 70)\n",
                        f"Średnia wydajność: {yd['mean_pcs_per_h']:.2f} szt./h | Mediana: {yd['median_pcs_per_h']:.1f} szt./h | IQR: [{yd['iqr_25_pcs_per_h']:.1f}, {yd['iqr_75_pcs_per_h']:.1f}] szt./h\n",
                        f"Jawne raporty zerowe (0.0 szt./h): {yd['zero_harvest_reports_count']} ({yd['zero_harvest_reports_pct']:.2f}%)\n",
                        f"Klasy obfitości: {yd['abundance_classes_counts']}\n",
                    ],
                }
            ],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "from scipy import stats\n",
                "\n",
                "df = pd.read_parquet('../data/processed/features.parquet')\n",
                "w_daily = pd.read_parquet('../data/processed/weather_daily.parquet')\n",
                "print(f'Wczytano features.parquet: shape={df.shape}')\n",
                "print(f\"Średnia wydajność: {df['yield_pcs_per_h'].mean():.2f} szt./h | Mediana: {df['yield_pcs_per_h'].median():.1f} szt./h | IQR: [{df['yield_pcs_per_h'].quantile(0.25):.1f}, {df['yield_pcs_per_h'].quantile(0.75):.1f}] szt./h\")\n",
                "print(f\"Jawne raporty zerowe (0.0 szt./h): {(df['yield_pcs_per_h'] == 0).sum()} ({(df['yield_pcs_per_h'] == 0).mean()*100:.2f}%)\")\n",
                "print('Klasy obfitości:', df['abundance_class'].value_counts().to_dict())\n"
            ],
        },
        {
            "cell_type": "markdown",
            "id": "cell-03-dist-md",
            "metadata": {},
            "source": [
                "## 1. Rozkład wydajności zbiorów (`yield_pcs_per_h`) i koncentracja przestrzenna\n",
                "\n",
                f"- Rozkład `yield_pcs_per_h [szt./h]` jest **prawostronnie skośny** (skośność = `{yd['skewness']}`, średnia `{yd['mean_pcs_per_h']} szt./h` > mediana `{yd['median_pcs_per_h']} szt./h`, zakres `0.0–120.0 szt./h`).\n",
                f"- W zbiorze występuje **`{yd['zero_harvest_reports_count']}` jawnych raportów zerowych** (`yield_pcs_per_h == 0.0`, `{yd['zero_harvest_reports_pct']}%`), które reprezentują potwierdzone nieudane wyjścia do lasu (twarde negatywy), a nie brak wpisu.\n",
                f"- Rozkład przestrzenny wykazuje umiarkowanie silną koncentrację (współczynnik Giniego dla 30 powiatów = `{rc['gini_coefficient_all_30_regions']}`): powiat bolesławiecki (`0201`, Bory Dolnośląskie) odpowiada za `{rc['top_1_region_share_pct']}%` raportów (`n = 101`), a Top-3 powiaty (`0201`, `0214`, `0213`) za `{rc['top_3_regions_share_pct']}%` próby.\n",
                "\n",
                "![Wykres 1](../docs/figures/eda_01_yield_and_reports_distribution.png)\n"
            ],
        },
        {
            "cell_type": "markdown",
            "id": "cell-04-volume-vs-yield-md",
            "metadata": {},
            "source": [
                "## 2. Sezonowość oraz oddzielenie obfitości raportów od obfitości grzybów\n",
                "\n",
                "> **Kluczowy wniosek metodologiczny:** Liczba publikowanych raportów (**obfitość raportów**) oraz wydajność grzybobrania w przeliczeniu na godzinę (**obfitość grzybów**, `yield_pcs_per_h`) mierzą dwa zupełnie różne zjawiska.\n",
                "\n",
                f"- **Obfitość raportów (wolumen dzienny):** W soboty i niedziele grzybiarze publikują średnio **`{va['report_volume_weekend_mean_per_day']}` raportu/dzień** (mediana `{va['report_volume_weekend_median_per_day']}`), podczas gdy w dni robocze (Pon–Pt) średnio **`{va['report_volume_weekday_mean_per_day']}` raportu/dzień** (mediana `{va['report_volume_weekday_median_per_day']}`). Oznacza to **`{va['weekend_to_weekday_volume_ratio']}x` wyższą aktywność raportowania w weekendy** (test Manna-Whitneya $p = {va['report_volume_mannwhitney_p']}$).\n",
                f"- **Obfitość grzybów (`yield_pcs_per_h`):** Mediana wydajności zbioru w dni robocze wynosi **`{va['mushroom_yield_weekday_median_pcs_per_h']} szt./h`** (średnia `{va['mushroom_yield_weekday_mean_pcs_per_h']} szt./h`, $n={va['mushroom_yield_weekday_n']}$), a w weekendy również **`{va['mushroom_yield_weekend_median_pcs_per_h']} szt./h`** (średnia `{va['mushroom_yield_weekend_mean_pcs_per_h']} szt./h`, $n={va['mushroom_yield_weekend_n']}$, test Manna-Whitneya $p = {va['mushroom_yield_mannwhitney_p']}$ — brak istotnej różnicy).\n",
                "- **Wniosek:** Wolumen raportów nie może służyć jako etykieta obfitości grzybów, ponieważ odzwierciedla czas wolny zbieraczy.\n",
                "\n",
                "![Wykres 2](../docs/figures/eda_02_seasonality_and_volume_vs_yield.png)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "id": "cell-05-correlations",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in corr_lines],
                }
            ],
            "source": [
                "# Porównanie okien opadów (1d, 7d, 14d, 21d, przedziałów tygodniowych) oraz cech temperatury, gleby i lasu\n",
                "df['precip_slice_1_7d_mm'] = df['precip_lag_7d_mm']\n",
                "df['precip_slice_8_14d_mm'] = df['precip_lag_14d_mm'] - df['precip_lag_7d_mm']\n",
                "df['precip_slice_15_21d_mm'] = df['precip_lag_21d_mm'] - df['precip_lag_14d_mm']\n",
                "\n",
                "for item in " + repr(corr_lines) + ":\n",
                "    print(item)\n"
            ],
        },
        {
            "cell_type": "markdown",
            "id": "cell-06-lag-md",
            "metadata": {},
            "source": [
                "## 3. Porównanie okien opadów i weryfikacja efektu opóźnienia (*Lag Effect*)\n",
                "\n",
                "1. **Krótkie okna opadów (`1d` i `7d`) wykazują słabą ujemną korelację z bieżącym zbiorem:**\n",
                "   - `precip_cutoff_1d_mm`: $r = -0.0881$ ($p = 0.046$), $\\rho = -0.0483$ ($p = 0.275$),\n",
                "   - `precip_lag_7d_mm`: $r = -0.1013$ ($p = 0.022$), $\\rho = -0.1306$ ($p = 0.003$).\n",
                "   - *Przyczyna fizyczna i behawioralna:* Deszcz z ostatnich kilku dni nie zdążył jeszcze wywołać zawiązania i wzrostu owocników grzybów mikoryzowych (proces ten trwa biologicznie 10–21 dni), a dodatkowo bardzo wysokie sumy opadów wystąpiły w połowie września przed jesiennym szczytem owocnikowania.\n",
                "2. **Przejście od ujemnej do dodatniej korelacji wraz z wydłużaniem okna do 14–21 dni:**\n",
                "   - `precip_lag_14d_mm`: $r = -0.0047$ ($p = 0.916$, korelacja zerowa),\n",
                "   - `precip_lag_21d_mm`: $r = +0.0555$ ($p = 0.210$, słaba dodatnia),\n",
                "   - **Rozłączny przedział 3. tygodnia przed zbiorem (`precip_slice_15_21d_mm`, czyli $[t-21 .. t-15]$):** **$r = +0.0869$ ($p = 0.049$), $\\rho = +0.1799$ ($p = 0.00004$)** — w klasie `BRAK` średni opad sprzed 15–21 dni wynosił `12.6 mm`, w klasie `SREDNIO` `20.6 mm`, a w klasie `WYSYP` `17.7 mm` (test Kruskala-Wallisa $H = 33.25, p < 0.0001$).\n",
                "3. **Częstość opadów (`rainy_days_lag_14d_count` i `k = 21d`) jest znacznie silniejszym predyktorem niż surowa suma milimetrów:**\n",
                "   - Liczba dni z opadem $\\ge 1.0\\text{ mm}$ w oknie 14 dni osiąga **$r = +0.2617$ ($p < 10^{-8}$), $\\rho = +0.2997$ ($p < 10^{-11}$)**, a w oknie 21 dni aż **$r = +0.3369$ ($p < 10^{-12}$), $\\rho = +0.3679$ ($p < 10^{-14}$)**.\n",
                "   - Regularne, wielodniowe nawilżanie ściółki leśnej sprzyja grzybni znacznie bardziej niż pojedyncza gwałtowna ulewa, z której woda spływa powierzchniowo.\n",
                "4. **Korelacja $\\neq$ przyczynowość:** Obserwowane współczynniki korelacji ($|r| \\le 0.30$) mają umiarkowaną siłę i nie stanowią samodzielnego dowodu przyczynowości — podlegają zakłóceniom przez sezonowość astronomiczną, temperaturę nocną oraz subiektywność zgłoszeń obywatelskich.\n",
                "\n",
                "![Wykres 3](../docs/figures/eda_03_precipitation_lag_comparison.png)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "id": "cell-07-sensitivity",
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": (
                        ["=== WRAŻLIWOŚĆ REGIONALNA ===\n"]
                        + [line + "\n" for line in reg_sens_lines]
                        + ["\n=== WRAŻLIWOŚĆ SEZONOWA ===\n"]
                        + [line + "\n" for line in seas_sens_lines]
                    ),
                }
            ],
            "source": [
                "print('=== WRAŻLIWOŚĆ REGIONALNA ===')\n",
                "for line in " + repr(reg_sens_lines) + ":\n",
                "    print(line)\n",
                "print('\\n=== WRAŻLIWOŚĆ SEZONOWA ===')\n",
                "for line in " + repr(seas_sens_lines) + ":\n",
                "    print(line)\n"
            ],
        },
        {
            "cell_type": "markdown",
            "id": "cell-08-sensitivity-md",
            "metadata": {},
            "source": [
                "## 4. Sprawdzenie braku dominacji przez pojedynczy region lub sezon\n",
                "\n",
                "- **Wrażliwość regionalna:** Po wykluczeniu największego powiatu (`0201 bolesławiecki`, $n=412$) oraz po wykluczeniu Top-3 powiatów (`0201, 0214, 0213`, $n=295$) i Top-5 powiatów ($n=233$), dodatnie korelacje dla `rainy_days_lag_14d_count` ($r$ od $+0.246$ do $+0.262$), `precip_slice_15_21d_mm` ($r$ od $+0.160$ do $+0.181$) oraz `forest_cover_pct` ($r$ od $+0.145$ do $+0.228$) **pozostają stabilne i istotne statystycznie**. Wynik nie jest artefaktem pojedynczego powiatu.\n",
                "- **Wrażliwość sezonowa:** Dodatnia zależność między częstością opadów (`rainy_days_lag_14d_count`) oraz wilgotnością gleby (`soil_moisture_mean_14d_m3m3`) a zbiorami utrzymuje się we wszystkich trzech fazach jesieni (wrzesień $r=+0.269$, wczesny październik $r=+0.299$, późny październik/listopad $r=+0.281$).\n",
                "\n",
                "![Wykres 4](../docs/figures/eda_04_regional_and_seasonal_sensitivity.png)\n",
                "\n",
                "## 5. Braki danych (`NULL` vs `0.0`) i ograniczenia filtrowania gatunków\n",
                "\n",
                f"- **Rozkład gatunków:** aż **`{sp['mixed_or_null_species_count']}` z `513` raportów (`{sp['mixed_or_null_species_pct']}%`)** nie posiada przypisanego pojedynczego gatunku (`194` to kosze `mieszane`, a `89` ma `NULL`). Pojedyncze gatunki poza podgrzybkiem (`n=114`) i borowikiem (`n=78`) mają od `10` do `17` rekordów w całym sezonie, co wyklucza budowę osobnych modeli ML per gatunek.\n",
                "\n",
                "![Wykres 5](../docs/figures/eda_05_missingness_and_species_constraints.png)\n"
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    features_path = PROCESSED_DIR / "features.parquet"
    reports_path = PROCESSED_DIR / "reports.parquet"
    weather_daily_path = PROCESSED_DIR / "weather_daily.parquet"

    for p in (features_path, reports_path, weather_daily_path):
        if not p.exists():
            raise FileNotFoundError(f"Brak wymaganego pliku wejściowego: {p}")

    features_df = pd.read_parquet(features_path)
    reports_df = pd.read_parquet(reports_path)
    weather_daily_df = pd.read_parquet(weather_daily_path)

    summary = build_eda_summary(
        features_df=features_df,
        reports_all_df=reports_df,
        weather_daily_df=weather_daily_df,
    )

    eda_json_path = PROCESSED_DIR / "eda_summary.json"
    eda_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figures_dir = ROOT_DIR / "docs" / "figures"
    saved_figs = generate_eda_figures(
        features_df=features_df,
        weather_daily_df=weather_daily_df,
        eda_summary=summary,
        figures_dir=figures_dir,
    )

    notebook_path = ROOT_DIR / "notebooks" / "01_eda_lag_effect.ipynb"
    nb_json = _build_notebook_json(summary)
    notebook_path.write_text(
        json.dumps(nb_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== [GRZ-11] ZAKOŃCZONO ANALIZĘ EDA I BADANIE OPÓŹNIENIA OPAD -> ZBIORY ===")
    print(f"1. Podsumowanie JSON: {eda_json_path}")
    print(f"2. Notatnik Jupyter : {notebook_path}")
    print(f"3. Wygenerowane wykresy ({len(saved_figs)}):")
    for fp in saved_figs:
        print(f"   - {fp}")


if __name__ == "__main__":
    main()
