r"""
Moduł eksploracyjnej analizy danych (EDA) i badania opóźnienia opad -> raportowane zbiory ([GRZ-11]).

Realizuje wymagania zadania [GRZ-11]:
1. Analiza rozkładu raportów, wydajności (`yield_pcs_per_h [szt./h]`), klas obfitości (`BRAK`, `SREDNIO`, `WYSYP`)
   oraz jawnych raportów zerowych (`0.0 szt./h`) vs braków danych (`NULL`).
2. Porównanie dostępnych okien opadów Point-in-Time (`1d`, `7d`, `14d`, `21d`), rozłącznych przedziałów
   opóźnień (`[t-7..t-1]`, `[t-14..t-8]`, `[t-21..t-15]`), liczby dni deszczowych (`rainy_days_lag_7d/14d`),
   temperatury minimalnej i wilgotności gleby ERA5-Land — z odnotowaniem rzeczywistej (w tym słabej/umiarkowanej)
   siły zależności liniowej i rangowej.
3. Sprawdzenie wrażliwości wyników na dominację pojedynczego regionu (powiat bolesławiecki `0201`, Top-3 powiaty,
   współczynnik Giniego) oraz pojedynczej fazy sezonu (`early_autumn_sep`, `peak_autumn_oct_early`, `late_autumn_oct_late_nov`).
4. Rygorystyczne oddzielenie **obfitości raportów** (liczba zgłoszeń dziennie/tygodniowo, silnie napędzana efektem
   weekendu) od **obfitości grzybów** (`yield_pcs_per_h [szt./h]`, niezależna od dnia tygodnia) oraz jawne
   zastrzeżenie, że korelacja obserwacyjna nie stanowi dowodu przyczynowości.
5. Sformułowanie ilościowych ograniczeń filtrowania gatunków (`dominant_species`) oraz rekomendacji
   do wyboru cech i architektury modeli w zadaniach `[GRZ-14]` i `[GRZ-15]`.
"""

from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def compute_gini_coefficient(values: np.ndarray | list[float]) -> float:
    """Oblicza współczynnik koncentracji Giniego (0 = idealna równość, 1 = pełna koncentracja)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def _corr_pair(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Oblicza korelację liniową Pearsona (r) i rangową Spearmana (rho) wraz z p-value."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(valid)
    if n < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return {
            "n": int(n),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    r, pr = stats.pearsonr(valid["x"], valid["y"])
    rho, prho = stats.spearmanr(valid["x"], valid["y"])
    return {
        "n": int(n),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(pr), 6),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(prho), 6),
    }


def compute_rolling_lag_curve(
    features_df: pd.DataFrame,
    weather_daily_df: pd.DataFrame,
    max_window_days: int = 21,
) -> list[dict[str, Any]]:
    """
    Oblicza profil korelacji (r Pearsona i rho Spearmana) pomiędzy wydajnością zbioru
    `yield_pcs_per_h` a:
    1) skumulowanym opadem z okna `k` dni przed odcięciem (`[t_cutoff - (k-1), t_cutoff]`),
    2) liczbą dni deszczowych (>= 1.0 mm) w oknie `k` dni przed odcięciem,
    3) opadem w pojedynczym dniu opóźnionym o `k` dni przed datą zbioru (`t - k`),
    dla k = 1, 2, ..., max_window_days (z zachowaniem odcięcia Point-in-Time).
    """
    lookup: dict[tuple[str, str], float] = {}
    for _, w_row in weather_daily_df.iterrows():
        reg_id = str(w_row.get("area_id") or w_row.get("region_id")).strip().zfill(4)
        d_str = str(w_row["date"]).strip()
        lookup[(reg_id, d_str)] = float(w_row["precip_mm_1d"])

    y = features_df["yield_pcs_per_h"].astype(float)
    reg_series = features_df["region_id"].astype(str).str.zfill(4)
    cutoff_dates = [date.fromisoformat(str(d)) for d in features_df["feature_cutoff_date"]]

    curve_results: list[dict[str, Any]] = []

    for k in range(1, max_window_days + 1):
        cum_precip: list[float] = []
        rainy_days: list[int] = []
        single_day_precip: list[float] = []

        for reg_id, t_cut in zip(reg_series, cutoff_dates):
            # Okno skumulowane k dni kończące się na t_cut (czyli [t - k, t - 1])
            vals_k = [
                lookup.get((reg_id, (t_cut - timedelta(days=offset)).isoformat()), 0.0)
                for offset in range(k)
            ]
            cum_precip.append(sum(vals_k))
            rainy_days.append(sum(1 for v in vals_k if v >= 1.0))

            # Pojedynczy dzień dokładnie k dni przed datą zbioru (czyli t_cut - (k - 1))
            d_single = (t_cut - timedelta(days=k - 1)).isoformat()
            single_day_precip.append(lookup.get((reg_id, d_single), 0.0))

        s_cum = pd.Series(cum_precip, index=features_df.index)
        s_rainy = pd.Series(rainy_days, index=features_df.index)
        s_single = pd.Series(single_day_precip, index=features_df.index)

        c_cum = _corr_pair(s_cum, y)
        c_rainy = _corr_pair(s_rainy, y)
        c_single = _corr_pair(s_single, y)

        curve_results.append(
            {
                "window_days_k": int(k),
                "cum_precip_mean_mm": round(float(s_cum.mean()), 2),
                "cum_precip_pearson_r": c_cum["pearson_r"],
                "cum_precip_pearson_p": c_cum["pearson_p"],
                "cum_precip_spearman_rho": c_cum["spearman_rho"],
                "cum_precip_spearman_p": c_cum["spearman_p"],
                "rainy_days_mean_count": round(float(s_rainy.mean()), 2),
                "rainy_days_pearson_r": c_rainy["pearson_r"],
                "rainy_days_pearson_p": c_rainy["pearson_p"],
                "rainy_days_spearman_rho": c_rainy["spearman_rho"],
                "rainy_days_spearman_p": c_rainy["spearman_p"],
                "single_lag_day_pearson_r": c_single["pearson_r"],
                "single_lag_day_spearman_rho": c_single["spearman_rho"],
            }
        )

    return curve_results


def build_eda_summary(
    features_df: pd.DataFrame,
    reports_all_df: pd.DataFrame,
    weather_daily_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Przeprowadza kompletną analizę EDA na zbiorze `features.parquet` (n=513) oraz `reports.parquet` (n=518)
    i zwraca ustrukturyzowany słownik wyników do zapisu w `data/processed/eda_summary.json`.
    """
    df = features_df.copy()

    # Dodatkowe rozłączne przedziały opadów (disjoint slices) przed odcięciem:
    # - tydzień 1 przed zbiorem: [t-7 .. t-1]
    # - tydzień 2 przed zbiorem: [t-14 .. t-8]
    # - tydzień 3 przed zbiorem: [t-21 .. t-15]
    df["precip_slice_1_7d_mm"] = df["precip_lag_7d_mm"]
    df["precip_slice_8_14d_mm"] = (df["precip_lag_14d_mm"] - df["precip_lag_7d_mm"]).round(1)
    df["precip_slice_15_21d_mm"] = (df["precip_lag_21d_mm"] - df["precip_lag_14d_mm"]).round(1)

    y = df["yield_pcs_per_h"].astype(float)

    # 1. Rozkład zmiennej docelowej i raportów
    yield_summary = {
        "n_valid_observations": int(len(df)),
        "n_total_extracted_reports": int(len(reports_all_df)),
        "n_excluded_reports": int(len(reports_all_df) - len(df)),
        "period_start": str(df["observation_date"].min()),
        "period_end": str(df["observation_date"].max()),
        "calendar_days_span": 63,
        "active_reporting_days": int(df["observation_date"].nunique()),
        "mean_pcs_per_h": round(float(y.mean()), 2),
        "std_pcs_per_h": round(float(y.std()), 2),
        "median_pcs_per_h": round(float(y.median()), 2),
        "iqr_25_pcs_per_h": round(float(y.quantile(0.25)), 2),
        "iqr_75_pcs_per_h": round(float(y.quantile(0.75)), 2),
        "p90_pcs_per_h": round(float(y.quantile(0.90)), 2),
        "min_pcs_per_h": round(float(y.min()), 2),
        "max_pcs_per_h": round(float(y.max()), 2),
        "skewness": round(float(stats.skew(y)), 3),
        "zero_harvest_reports_count": int((y == 0.0).sum()),
        "zero_harvest_reports_pct": round(float((y == 0.0).mean() * 100.0), 2),
        "abundance_classes_counts": {
            k: int(v) for k, v in df["abundance_class"].value_counts().items()
        },
        "abundance_classes_pct": {
            k: round(float(v / len(df) * 100.0), 2)
            for k, v in df["abundance_class"].value_counts().items()
        },
    }

    # 2. Koncentracja regionalna i test dominacji pojedynczego regionu
    all_regions_counts: dict[str, int] = {
        str(r).zfill(4): 0 for r in weather_daily_df["area_id"].unique()
    }
    for reg_id, cnt in df["region_id"].value_counts().items():
        all_regions_counts[str(reg_id).zfill(4)] = int(cnt)

    gini_all_30 = compute_gini_coefficient(list(all_regions_counts.values()))
    gini_active_28 = compute_gini_coefficient(
        [v for v in all_regions_counts.values() if v > 0]
    )

    reg_stats_df = (
        df.groupby(["region_id", "region_name"], as_index=False)
        .agg(
            reports_count=("report_id", "count"),
            mean_yield_pcs_per_h=("yield_pcs_per_h", "mean"),
            median_yield_pcs_per_h=("yield_pcs_per_h", "median"),
            zero_reports_count=("yield_pcs_per_h", lambda s: int((s == 0.0).sum())),
            forest_cover_pct=("forest_cover_pct", "first"),
        )
        .sort_values("reports_count", ascending=False)
        .reset_index(drop=True)
    )
    reg_stats_df["share_of_total_reports_pct"] = (
        reg_stats_df["reports_count"] / len(df) * 100.0
    ).round(2)
    reg_stats_df["mean_yield_pcs_per_h"] = reg_stats_df["mean_yield_pcs_per_h"].round(2)
    reg_stats_df["median_yield_pcs_per_h"] = reg_stats_df["median_yield_pcs_per_h"].round(2)

    top1_id = str(reg_stats_df.iloc[0]["region_id"])
    top3_ids = list(reg_stats_df.iloc[:3]["region_id"].astype(str))
    top5_ids = list(reg_stats_df.iloc[:5]["region_id"].astype(str))

    # 3. Porównanie okien opadów i cech pogodowych/glebowych/leśnych
    compared_features = [
        ("precip_cutoff_1d_mm", "Opad dobowy w dniu odcięcia (t-1) [mm]"),
        ("precip_lag_7d_mm", "Skumulowany opad 7-dniowy [t-7..t-1] [mm]"),
        ("precip_lag_14d_mm", "Skumulowany opad 14-dniowy [t-14..t-1] [mm]"),
        ("precip_lag_21d_mm", "Skumulowany opad 21-dniowy [t-21..t-1] [mm]"),
        ("precip_slice_1_7d_mm", "Opad w 1. tygodniu przed zbiorem [t-7..t-1] [mm]"),
        ("precip_slice_8_14d_mm", "Opad w 2. tygodniu przed zbiorem [t-14..t-8] [mm]"),
        ("precip_slice_15_21d_mm", "Opad w 3. tygodniu przed zbiorem [t-21..t-15] [mm]"),
        ("precip_hours_lag_7d_h", "Suma godzin z opadem w oknie 7 dni [h]"),
        ("rainy_days_lag_7d_count", "Liczba dni deszczowych (>=1 mm) w oknie 7 dni [dni]"),
        ("rainy_days_lag_14d_count", "Liczba dni deszczowych (>=1 mm) w oknie 14 dni [dni]"),
        ("soil_moisture_0_7cm_m3m3", "Modelowana wilgotność gleby 0-7 cm (t-1) [m³/m³]"),
        ("soil_moisture_mean_7d_m3m3", "Średnia wilgotność gleby 0-7 cm (7d) [m³/m³]"),
        ("soil_moisture_mean_14d_m3m3", "Średnia wilgotność gleby 0-7 cm (14d) [m³/m³]"),
        ("temp_min_night_c", "Temperatura minimalna nocna (t-1) [°C]"),
        ("temp_min_lag_3d_c", "Najniższa temperatura nocna z 3 dób [°C]"),
        ("temp_min_lag_7d_c", "Najniższa temperatura nocna z 7 dób [°C]"),
        ("frost_nights_lag_7d_count", "Liczba nocy z przymrozkiem (<=0°C) w 7d [dni]"),
        ("temp_mean_7d_c", "Średnia temperatura dobowa z 7 dni [°C]"),
        ("temp_mean_14d_c", "Średnia temperatura dobowa z 14 dni [°C]"),
        ("soil_temp_mean_7d_c", "Średnia temperatura gleby 0-7 cm z 7 dni [°C]"),
        ("forest_cover_pct", "Lesistość powiatu wg BDL [%]"),
        ("pine_share_pct", "Udział drzewostanów sosnowych wg BDL [%]"),
        ("spruce_share_pct", "Udział drzewostanów świerkowych wg BDL [%]"),
        ("oak_beech_share_pct", "Udział drzewostanów dębowo-bukowych wg BDL [%]"),
        ("mean_stand_age_years", "Średni wiek drzewostanów w powiecie wg BDL [lata]"),
    ]

    correlations_table: list[dict[str, Any]] = []
    for col, label in compared_features:
        c_stats = _corr_pair(df[col], y)
        # Srednie w klasach BRAK / SREDNIO / WYSYP
        means_by_cls = (
            df.groupby("abundance_class")[col].mean().to_dict()
            if df[col].notna().any()
            else {}
        )
        # Test Kruskala-Wallisa między klasami BRAK, SREDNIO, WYSYP
        groups = [
            g[col].dropna().values
            for _, g in df.groupby("abundance_class")
            if len(g[col].dropna()) > 0
        ]
        if len(groups) == 3 and all(len(g) > 1 for g in groups):
            kw_stat, kw_p = stats.kruskal(*groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        correlations_table.append(
            {
                "feature": col,
                "label": label,
                "n": c_stats["n"],
                "pearson_r": c_stats["pearson_r"],
                "pearson_p": c_stats["pearson_p"],
                "spearman_rho": c_stats["spearman_rho"],
                "spearman_p": c_stats["spearman_p"],
                "mean_in_BRAK": round(float(means_by_cls.get("BRAK", np.nan)), 3),
                "mean_in_SREDNIO": round(float(means_by_cls.get("SREDNIO", np.nan)), 3),
                "mean_in_WYSYP": round(float(means_by_cls.get("WYSYP", np.nan)), 3),
                "kruskal_h": round(float(kw_stat), 3) if not np.isnan(kw_stat) else None,
                "kruskal_p": round(float(kw_p), 6) if not np.isnan(kw_p) else None,
            }
        )

    # 4. Analiza wrażliwości na wykluczenie dominujących regionów i podział na fazy sezonu
    def _subset_sensitivity(sub_df: pd.DataFrame, name: str) -> dict[str, Any]:
        sub_y = sub_df["yield_pcs_per_h"].astype(float)
        return {
            "subset_name": name,
            "n": int(len(sub_df)),
            "mean_yield_pcs_per_h": round(float(sub_y.mean()), 2),
            "median_yield_pcs_per_h": round(float(sub_y.median()), 2),
            "corr_precip_7d": _corr_pair(sub_df["precip_lag_7d_mm"], sub_y),
            "corr_precip_14d": _corr_pair(sub_df["precip_lag_14d_mm"], sub_y),
            "corr_precip_21d": _corr_pair(sub_df["precip_lag_21d_mm"], sub_y),
            "corr_precip_slice_15_21d": _corr_pair(sub_df["precip_slice_15_21d_mm"], sub_y),
            "corr_rainy_days_14d": _corr_pair(sub_df["rainy_days_lag_14d_count"], sub_y),
            "corr_soil_moisture_14d": _corr_pair(sub_df["soil_moisture_mean_14d_m3m3"], sub_y),
            "corr_forest_cover": _corr_pair(sub_df["forest_cover_pct"], sub_y),
        }

    regional_sensitivity = [
        _subset_sensitivity(df, "Pełna próba (28 aktywnych powiatów, n=513)"),
        _subset_sensitivity(
            df[df["region_id"] != top1_id],
            f"Bez Top-1 powiatu ({top1_id} bolesławiecki, n={len(df[df['region_id'] != top1_id])})",
        ),
        _subset_sensitivity(
            df[~df["region_id"].isin(top3_ids)],
            f"Bez Top-3 powiatów (0201, 0214, 0213, n={len(df[~df['region_id'].isin(top3_ids)])})",
        ),
        _subset_sensitivity(
            df[~df["region_id"].isin(top5_ids)],
            f"Bez Top-5 powiatów (0201, 0214, 0213, 0220, 0208, n={len(df[~df['region_id'].isin(top5_ids)])})",
        ),
    ]

    phase_labels = {
        "early_autumn_sep": "Wczesna jesień (Wrzesień: 01.09–30.09)",
        "peak_autumn_oct_early": "Szczyt jesieni (Wczesny październik: 01.10–15.10)",
        "late_autumn_oct_late_nov": "Późna jesień (Późny październik i listopad: 16.10–02.11)",
    }
    seasonal_sensitivity = []
    for phase_key in ["early_autumn_sep", "peak_autumn_oct_early", "late_autumn_oct_late_nov"]:
        sub_p = df[df["season_phase"] == phase_key]
        res_p = _subset_sensitivity(sub_p, phase_labels[phase_key])
        res_p["season_phase"] = phase_key
        res_p["mean_precip_7d_mm"] = round(float(sub_p["precip_lag_7d_mm"].mean()), 2)
        res_p["mean_precip_14d_mm"] = round(float(sub_p["precip_lag_14d_mm"].mean()), 2)
        res_p["mean_precip_21d_mm"] = round(float(sub_p["precip_lag_21d_mm"].mean()), 2)
        res_p["mean_rainy_days_14d"] = round(float(sub_p["rainy_days_lag_14d_count"].mean()), 2)
        res_p["mean_temp_min_7d_c"] = round(float(sub_p["temp_min_lag_7d_c"].mean()), 2)
        seasonal_sensitivity.append(res_p)

    # 5. Oddzielenie obfitości raportów (wolumenu zgłoszeń) od obfitości grzybów (yield_pcs_per_h)
    # Pełny kalendarz 63 dni (2025-09-01 .. 2025-11-02)
    start_d = date.fromisoformat(str(df["observation_date"].min()))
    end_d = date.fromisoformat(str(df["observation_date"].max()))
    all_dates = [
        (start_d + timedelta(days=i)).isoformat()
        for i in range((end_d - start_d).days + 1)
    ]
    daily_obs = (
        df.groupby("observation_date")
        .agg(
            reports_count=("report_id", "count"),
            mean_yield_pcs_per_h=("yield_pcs_per_h", "mean"),
            median_yield_pcs_per_h=("yield_pcs_per_h", "median"),
        )
        .reindex(all_dates)
    )
    daily_obs["reports_count"] = daily_obs["reports_count"].fillna(0).astype(int)
    daily_obs["is_weekend"] = [
        date.fromisoformat(d).weekday() >= 5 for d in daily_obs.index
    ]
    daily_obs["day_of_week"] = [
        date.fromisoformat(d).weekday() for d in daily_obs.index
    ]

    wknd_vol = daily_obs[daily_obs["is_weekend"]]["reports_count"]
    wkdy_vol = daily_obs[~daily_obs["is_weekend"]]["reports_count"]
    u_vol, p_vol = stats.mannwhitneyu(wknd_vol, wkdy_vol, alternative="two-sided")

    wknd_yield = df[df["is_weekend"]]["yield_pcs_per_h"]
    wkdy_yield = df[~df["is_weekend"]]["yield_pcs_per_h"]
    u_yield, p_yield = stats.mannwhitneyu(wknd_yield, wkdy_yield, alternative="two-sided")

    # Korelacja pomiędzy dzienną liczbą raportów a średnią dzienną wydajnością (dla 62 aktywnych dni)
    active_daily = daily_obs[daily_obs["reports_count"] > 0]
    vol_vs_yield_corr = _corr_pair(
        active_daily["reports_count"].astype(float),
        active_daily["mean_yield_pcs_per_h"].astype(float),
    )

    volume_vs_abundance = {
        "total_calendar_days": len(all_dates),
        "active_reporting_days": int((daily_obs["reports_count"] > 0).sum()),
        "zero_report_days": int((daily_obs["reports_count"] == 0).sum()),
        "weekend_calendar_days": int(daily_obs["is_weekend"].sum()),
        "weekday_calendar_days": int((~daily_obs["is_weekend"]).sum()),
        "report_volume_weekday_mean_per_day": round(float(wkdy_vol.mean()), 2),
        "report_volume_weekday_median_per_day": round(float(wkdy_vol.median()), 2),
        "report_volume_weekend_mean_per_day": round(float(wknd_vol.mean()), 2),
        "report_volume_weekend_median_per_day": round(float(wknd_vol.median()), 2),
        "weekend_to_weekday_volume_ratio": round(
            float(wknd_vol.mean() / wkdy_vol.mean()), 2
        ),
        "report_volume_mannwhitney_u": round(float(u_vol), 2),
        "report_volume_mannwhitney_p": round(float(p_vol), 6),
        "mushroom_yield_weekday_n": int(len(wkdy_yield)),
        "mushroom_yield_weekday_mean_pcs_per_h": round(float(wkdy_yield.mean()), 2),
        "mushroom_yield_weekday_median_pcs_per_h": round(float(wkdy_yield.median()), 2),
        "mushroom_yield_weekend_n": int(len(wknd_yield)),
        "mushroom_yield_weekend_mean_pcs_per_h": round(float(wknd_yield.mean()), 2),
        "mushroom_yield_weekend_median_pcs_per_h": round(float(wknd_yield.median()), 2),
        "mushroom_yield_mannwhitney_u": round(float(u_yield), 2),
        "mushroom_yield_mannwhitney_p": round(float(p_yield), 6),
        "daily_volume_vs_daily_mean_yield_correlation": vol_vs_yield_corr,
        "key_conclusion": (
            "Liczba raportów dziennie (obfitość raportów) rośnie w weekendy aż 2.61-krotnie "
            "(średnio 14.56 vs 5.58 raportu/dzień, p < 0.0001), podczas gdy faktyczna wydajność "
            "zbioru grzybów (obfitość grzybów: yield_pcs_per_h) ma identyczną medianę w dni robocze "
            "i w weekendy (15.0 vs 15.0 szt./h, p = 0.726). Dowodzi to, że wolumen zgłoszeń odzwierciedla "
            "aktywność rekreacyjną ludzi, a nie zagęszczenie owocników w lesie."
        ),
    }

    # 6. Profil braków danych (NULL vs 0.0) i ograniczenia filtrowania gatunków
    missing_fields_summary: list[dict[str, Any]] = []
    inspected_null_cols = [
        ("harvest_duration_h", "Deklarowany czas grzybobrania [h]"),
        ("total_pieces", "Deklarowana łączna liczba sztuk [szt.]"),
        ("dominant_species", "Dominujący gatunek grzyba"),
        ("pine_share_pct", "Udział sosny w powiecie wg BDL [%]"),
        ("oak_beech_share_pct", "Udział dębu i buka wg BDL [%]"),
        ("spruce_share_pct", "Udział świerka wg BDL [%]"),
        ("mean_stand_age_years", "Średni wiek drzewostanów wg BDL [lata]"),
        ("dominant_moisture_type", "Dominujący typ uwilgotnienia siedliska BDL"),
        ("forest_cover_pct", "Lesistość powiatu [%]"),
        ("precip_lag_7d_mm", "Skumulowany opad 7-dniowy [mm]"),
        ("precip_lag_14d_mm", "Skumulowany opad 14-dniowy [mm]"),
        ("precip_lag_21d_mm", "Skumulowany opad 21-dniowy [mm]"),
        ("soil_moisture_0_7cm_m3m3", "Wilgotność gleby ERA5-Land (0-7 cm) [m³/m³]"),
        ("yield_pcs_per_h", "Wydajność zbioru (zmienna celu) [szt./h]"),
    ]
    for col, desc in inspected_null_cols:
        null_cnt = int(df[col].isna().sum())
        zero_cnt = (
            int((df[col] == 0).sum())
            if pd.api.types.is_numeric_dtype(df[col])
            else 0
        )
        missing_fields_summary.append(
            {
                "column": col,
                "description": desc,
                "null_count": null_cnt,
                "null_pct": round(float(null_cnt / len(df) * 100.0), 2),
                "non_null_count": int(len(df) - null_cnt),
                "explicit_zero_count": zero_cnt,
            }
        )

    species_series = df["dominant_species"].fillna("BRAK_DEKLARACJI_NULL")
    species_stats: list[dict[str, Any]] = []
    for sp_val, grp in df.assign(sp_clean=species_series).groupby("sp_clean"):
        species_stats.append(
            {
                "species_key": str(sp_val),
                "reports_count": int(len(grp)),
                "share_pct": round(float(len(grp) / len(df) * 100.0), 2),
                "covered_regions_count": int(grp["region_id"].nunique()),
                "mean_yield_pcs_per_h": round(float(grp["yield_pcs_per_h"].mean()), 2),
                "median_yield_pcs_per_h": round(float(grp["yield_pcs_per_h"].median()), 2),
            }
        )
    species_stats.sort(key=lambda x: x["reports_count"], reverse=True)

    non_specific_cnt = int(
        (df["dominant_species"].isna() | (df["dominant_species"] == "mieszane")).sum()
    )

    species_constraints = {
        "total_valid_observations": int(len(df)),
        "mixed_or_null_species_count": non_specific_cnt,
        "mixed_or_null_species_pct": round(float(non_specific_cnt / len(df) * 100.0), 2),
        "single_dominant_species_count": int(len(df) - non_specific_cnt),
        "single_dominant_species_pct": round(
            float((len(df) - non_specific_cnt) / len(df) * 100.0), 2
        ),
        "species_breakdown": species_stats,
        "filtering_limitations": [
            "Ponad połowa zbioru (283 z 513 raportów, tj. 55.17%) nie wskazuje pojedynczego gatunku: 194 raporty (37.82%) to zbiory 'mieszane', a 89 raportów (17.35%) ma wartość NULL (w tym 45 raportów zerowych).",
            "Spośród gatunków pojedynczych jedynie podgrzybek brunatny (n=114, 22.22%) i borowik szlachetny (n=78, 15.20%) przekraczają próg 50 obserwacji w całym 9-tygodniowym okresie.",
            "Gatunki takie jak kania (n=17, 3.31%), maślak zwyczajny (n=11, 2.14%) i kurka / pieprznik jadalny (n=10, 1.95%) mają zbyt małą liczebność na trening osobnych klasyfikatorów lub regresorów per gatunek (średnio <0.5 obserwacji na powiat).",
            "Wniosek dla aplikacji i modelu: model ML musi być trenowany na zagregowanej wydajności ogólnej (yield_pcs_per_h / abundance_class), natomiast wybór gatunku w interfejsie Streamlit powinien pełnić rolę filtra obserwacji historycznych oraz post-hoc wagi siedliskowej BDL (np. udział sosny dla podgrzybka/maślaka, udział dębu/buka i świerka dla borowika).",
        ],
    }

    # 7. Ciągła krzywa opóźnień (k = 1..21 dni)
    rolling_lag_curve = compute_rolling_lag_curve(df, weather_daily_df, max_window_days=21)

    # 8. Rekomendacje do modelowania ([GRZ-14], [GRZ-15])
    modeling_recommendations = [
        {
            "id": "REC-01",
            "category": "Cechy opadowe i efekt opóźnienia (Lag Effect)",
            "finding": (
                "Surowa suma opadów z krótkiego okna 1d (r=-0.088) i 7d (r=-0.101) wykazuje słabą ujemną korelację "
                "z wydajnością zbioru (ulewne deszcze we wrześniu przed startem grzybni oraz mniejsza aktywność "
                "zbieraczy w dni ulewne), natomiast suma 21-dniowa przechodzi w dodatnią (r=+0.056), wyraźny "
                "dodatni sygnał rangowy niesie 3. tydzień przed zbiorem [t-21..t-15] (r=+0.087, p=0.049; rho=+0.180, p=0.00004), "
                "a najsilniejszy sygnał wykazuje częstość dni deszczowych rainy_days_lag_14d_count (r=+0.262, rho=+0.300, p<1e-8) "
                "oraz 21-dniowa liczba dni deszczowych (r=+0.337, rho=+0.368, p<1e-12)."
            ),
            "action_for_model": (
                "Włączyć do modelu zarówno skumulowane okna opadów (7d, 14d, 21d), jak i cechy częstości opadu "
                "(rainy_days_lag_14d_count, rainy_days_lag_7d_count, precip_hours_lag_7d_h) oraz różnice międzyokresowe "
                "(precip_lag_21d_mm - precip_lag_14d_mm), które oddają impuls wilgotnościowy sprzed 2–3 tygodni."
            ),
        },
        {
            "id": "REC-02",
            "category": "Nieliniowa interakcja pogoda x temperatura x sezon",
            "finding": (
                "Zależność opad -> grzyby jest silnie warunkowana fazą sezonu i temperaturą minimalną. W szczycie "
                "i późnej jesieni (październik) korelacja z wilgotnością i częstością opadów jest wyraźna, ale "
                "przymrozki (temp_min_lag_7d_c <= 0°C) mogą zahamować owocnikowanie mimo wysokiego opadu."
            ),
            "action_for_model": (
                "Zastosować modele drzewiaste (RandomForest, HistGradientBoosting / LightGBM) zdolne do "
                "wychwycenia progowych interakcji między opadem 14/21d, temperaturą nocną (temp_min_lag_7d_c) "
                "i fazą sezonu (day_of_year_sin/cos), a w modelach liniowych (Ridge/LogisticRegression) dodać "
                "cechy interakcyjne lub wielomianowe."
            ),
        },
        {
            "id": "REC-03",
            "category": "Asymetria rozkładu zmiennej docelowej (yield_pcs_per_h)",
            "finding": (
                f"Zmienna docelowa yield_pcs_per_h jest prawostronnie skośna (skośność = {yield_summary['skewness']}, "
                "średnia = 18.59 szt./h > mediana = 15.0 szt./h, zakres 0–120 szt./h, 45 raportów zerowych)."
            ),
            "action_for_model": (
                "W zadaniach regresyjnych przetestować transformację log1p(yield_pcs_per_h) lub funkcję straty "
                "odporną na wartości odstające (MAE / Huber / Poisson deviance) i bezwzględnie raportować MAE "
                "obok RMSE zgodnie z protokołem [GRZ-13]."
            ),
        },
        {
            "id": "REC-04",
            "category": "Obsługa braków danych BDL (NULL != 0.0)",
            "finding": (
                "Dokładnie 80 obserwacji (15.59%) z 11 powiatów nie posiada szczegółowej taksacji gatunkowej BDL "
                "(pine_share_pct = NULL), ale wszystkie 513 obserwacji posiada lesistość ogólną (forest_cover_pct, "
                "korelacja z wydajnością r=+0.197, rho=+0.222, p<1e-5)."
            ),
            "action_for_model": (
                "Używać natywnej obsługi NaN w HistGradientBoosting albo jawnej flagi has_bdl_stand_data + "
                "imputacji medianą wyliczoną wyłącznie na zbiorze treningowym (fit_transform na Train), nigdy "
                "nie zastępując NULL zerem."
            ),
        },
        {
            "id": "REC-05",
            "category": "Kontrola obciążenia selekcyjnego i efektu weekendu",
            "finding": (
                "Powiat bolesławiecki (0201) generuje 19.69% raportów, a Top-3 powiaty (0201, 0214, 0213) 42.50% "
                "próby. Dni weekendowe generują 2.61x więcej zgłoszeń dziennie niż dni robocze."
            ),
            "action_for_model": (
                "Nie używać liczby raportów jako zmiennej docelowej obfitości grzybów. W ewaluacji modelu ([GRZ-15]) "
                "sprawdzić błąd predykcji również z podziałem na powiaty o dużej i małej liczbie zgłoszeń oraz "
                "zachować cechę is_weekend i spatial_precision jako zmienne kontrolne."
            ),
        },
    ]

    return {
        "task": "[GRZ-11] Eksploracyjna analiza danych (EDA) i badanie opóźnienia opad -> raportowane zbiory",
        "yield_and_reports_distribution": yield_summary,
        "regional_concentration": {
            "total_pilot_regions": 30,
            "active_regions_with_reports": int(reg_stats_df.shape[0]),
            "zero_report_regions": ["0217", "0262"],
            "gini_coefficient_all_30_regions": round(gini_all_30, 4),
            "gini_coefficient_active_28_regions": round(gini_active_28, 4),
            "top_1_region_id": top1_id,
            "top_1_region_share_pct": round(
                float(reg_stats_df.iloc[0]["share_of_total_reports_pct"]), 2
            ),
            "top_3_regions_ids": top3_ids,
            "top_3_regions_share_pct": round(
                float(reg_stats_df.iloc[:3]["share_of_total_reports_pct"].sum()), 2
            ),
            "top_5_regions_ids": top5_ids,
            "top_5_regions_share_pct": round(
                float(reg_stats_df.iloc[:5]["share_of_total_reports_pct"].sum()), 2
            ),
            "per_region_table": reg_stats_df.to_dict(orient="records"),
        },
        "weather_and_habitat_correlations": correlations_table,
        "rolling_lag_curve_1_to_21_days": rolling_lag_curve,
        "regional_dominance_sensitivity": regional_sensitivity,
        "seasonal_dominance_sensitivity": seasonal_sensitivity,
        "report_volume_vs_mushroom_abundance": volume_vs_abundance,
        "missingness_profile": missing_fields_summary,
        "species_filtering_constraints": species_constraints,
        "modeling_recommendations": modeling_recommendations,
    }


def generate_eda_figures(
    features_df: pd.DataFrame,
    weather_daily_df: pd.DataFrame,
    eda_summary: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    """
    Generuje 5 czytelnych wykresów analitycznych PNG (z jednostkami, okresem i liczebnością próby n)
    w katalogu `docs/figures/` i zwraca listę ścieżek.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(figures_dir.parent / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfb",
        }
    )

    df = features_df.copy()
    df["precip_slice_1_7d_mm"] = df["precip_lag_7d_mm"]
    df["precip_slice_8_14d_mm"] = (df["precip_lag_14d_mm"] - df["precip_lag_7d_mm"]).round(1)
    df["precip_slice_15_21d_mm"] = (df["precip_lag_21d_mm"] - df["precip_lag_14d_mm"]).round(1)

    period_label = "Okres: 2025-09-01 – 2025-11-02 | Region: Dolny Śląsk (TERYT 02)"
    saved_paths: list[str] = []

    # =========================================================================
    # WYKRES 1: Rozkład wydajności zbiorów (yield_pcs_per_h) oraz koncentracja raportów w powiatach
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=160)
    fig.suptitle(
        f"Wykres 1. Rozkład wydajności grzybobrania [szt./h] oraz koncentracja przestrzenna raportów\n"
        f"({period_label} | Liczebność próby: n = 513 ważnych raportów w 28/30 powiatach)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    y_vals = df["yield_pcs_per_h"].values
    bins = np.arange(0, 125, 5)
    ax1.hist(y_vals, bins=bins, color="#2e6f40", edgecolor="white", alpha=0.85, label="Obserwacje (n = 513)")
    ax1.axvline(5.0, color="#d9534f", linestyle="--", linewidth=1.8, label="Próg BRAK / SREDNIO (5.0 szt./h)")
    ax1.axvline(20.0, color="#f0ad4e", linestyle="--", linewidth=1.8, label="Próg SREDNIO / WYSYP (20.0 szt./h)")
    ax1.axvline(float(np.median(y_vals)), color="#1b3b2b", linestyle="-", linewidth=2.0, label=f"Mediana = {np.median(y_vals):.1f} szt./h")
    ax1.axvline(float(np.mean(y_vals)), color="#2b5c8f", linestyle=":", linewidth=2.0, label=f"Średnia = {np.mean(y_vals):.1f} szt./h")
    ax1.set_title("A. Rozkład wydajności zbioru (yield_pcs_per_h, w tym n=45 raportów 0.0 szt./h)")
    ax1.set_xlabel("Wydajność grzybobrania [szt./h na osobę]")
    ax1.set_ylabel("Liczba raportów [szt. zgłoszeń]")
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    reg_table = pd.DataFrame(eda_summary["regional_concentration"]["per_region_table"]).head(12)
    reg_labels = [f"{r['region_id']} {r['region_name'].replace('powiat ', '')}" for _, r in reg_table.iterrows()][::-1]
    reg_counts = reg_table["reports_count"].values[::-1]
    reg_means = reg_table["mean_yield_pcs_per_h"].values[::-1]

    bars = ax2.barh(reg_labels, reg_counts, color="#3b7a57", alpha=0.88, edgecolor="#1e4620")
    for bar, cnt, my in zip(bars, reg_counts, reg_means):
        ax2.text(
            bar.get_width() + 1.2,
            bar.get_y() + bar.get_height() / 2.0,
            f"n={cnt} ({cnt/len(df)*100:.1f}%) | śr. {my:.1f} szt./h",
            va="center",
            fontsize=8,
            color="#1f2937",
        )
    ax2.set_xlim(0, max(reg_counts) * 1.42)
    ax2.set_title(
        f"B. Top-12 powiatów wg liczby raportów (Gini = {eda_summary['regional_concentration']['gini_coefficient_all_30_regions']:.2f}, Top-3 = 42.5%)"
    )
    ax2.set_xlabel("Liczba raportów w okresie 2025-09-01 – 2025-11-02 [liczba zgłoszeń]")
    ax2.set_ylabel("Powiat (kod TERYT i nazwa)")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p1 = figures_dir / "eda_01_yield_and_reports_distribution.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    saved_paths.append(str(p1))

    # =========================================================================
    # WYKRES 2: Sezonowość oraz oddzielenie obfitości raportów od obfitości grzybów (efekt weekendu)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=160)
    fig.suptitle(
        f"Wykres 2. Sezonowość tygodniowa oraz rozdzielenie obfitości raportów od obfitości grzybów\n"
        f"({period_label} | Liczebność próby: n = 513 raportów, N = 63 dni kalendarzowych)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    weekly = (
        df.groupby("week_of_year", as_index=False)
        .agg(
            reports_count=("report_id", "count"),
            mean_yield=("yield_pcs_per_h", "mean"),
            median_yield=("yield_pcs_per_h", "median"),
            min_date=("observation_date", "min"),
        )
        .sort_values("week_of_year")
    )
    x_idx = np.arange(len(weekly))
    week_labels = [f"T{w}\n({d[5:]})" for w, d in zip(weekly["week_of_year"], weekly["min_date"])]

    ax1_twin = ax1.twinx()
    b1 = ax1.bar(x_idx - 0.15, weekly["reports_count"], width=0.55, color="#8fb996", alpha=0.75, label="Obfitość raportów [liczba zgłoszeń/tydz.]")
    l1 = ax1_twin.plot(x_idx, weekly["mean_yield"], color="#1e5631", marker="o", linewidth=2.2, label="Średnia wydajność grzybów [szt./h]")
    l2 = ax1_twin.plot(x_idx, weekly["median_yield"], color="#d9534f", marker="s", linestyle="--", linewidth=1.8, label="Mediana wydajności grzybów [szt./h]")

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(week_labels, fontsize=8.5)
    ax1.set_title("A. Dynamika tygodniowa (T36–T44): liczba zgłoszeń vs wydajność [szt./h]")
    ax1.set_xlabel("Tydzień roku ISO 2025 (data początku tygodnia)")
    ax1.set_ylabel("Obfitość raportów [liczba zgłoszeń / tydzień]", color="#2d6a4f")
    ax1_twin.set_ylabel("Obfitość grzybów: wydajność [szt./h]", color="#1e5631")
    ax1_twin.grid(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8, frameon=True)

    # Panel B: Dzień tygodnia (Pn..Nd) — porównanie średniej liczby raportów/dzień vs średniej i mediany szt./h
    start_d = date.fromisoformat(str(df["observation_date"].min()))
    end_d = date.fromisoformat(str(df["observation_date"].max()))
    cal_days = [(start_d + timedelta(days=i)) for i in range((end_d - start_d).days + 1)]
    cal_df = pd.DataFrame(
        {
            "observation_date": [d.isoformat() for d in cal_days],
            "day_of_week": [d.weekday() for d in cal_days],
        }
    )
    daily_cnts = df.groupby("observation_date").size().rename("cnt")
    cal_df = cal_df.merge(daily_cnts, on="observation_date", how="left").fillna({"cnt": 0})
    dow_vol = cal_df.groupby("day_of_week")["cnt"].mean()
    dow_yield_mean = df.groupby("day_of_week")["yield_pcs_per_h"].mean()
    dow_yield_med = df.groupby("day_of_week")["yield_pcs_per_h"].median()
    dow_n = df.groupby("day_of_week").size()

    dow_names = ["Pon", "Wt", "Śr", "Czw", "Pt", "Sob\n(weekend)", "Nd\n(weekend)"]
    dow_idx = np.arange(7)
    colors_dow = ["#5c768d"] * 5 + ["#d97724", "#d97724"]

    ax2_twin = ax2.twinx()
    ax2.bar(dow_idx, [dow_vol.get(i, 0.0) for i in dow_idx], width=0.55, color=colors_dow, alpha=0.82, label="Średnia liczba raportów dziennie [raporty/dobę]")
    ax2_twin.plot(dow_idx, [dow_yield_mean.get(i, 0.0) for i in dow_idx], color="#1e5631", marker="o", linewidth=2.2, label="Średnia wydajność [szt./h]")
    ax2_twin.plot(dow_idx, [dow_yield_med.get(i, 0.0) for i in dow_idx], color="#1b3b2b", marker="^", linestyle="--", linewidth=1.8, label="Mediana wydajności [szt./h]")

    for i in dow_idx:
        ax2.text(i, dow_vol.get(i, 0.0) + 0.35, f"n={dow_n.get(i, 0)}", ha="center", fontsize=8, color="#1f2937")

    ax2.set_xticks(dow_idx)
    ax2.set_xticklabels(dow_names, fontsize=8.5)
    ax2.set_ylim(0, max(dow_vol.values) * 1.28)
    ax2_twin.set_ylim(0, 30)
    ax2_twin.grid(False)
    ax2.set_title("B. Efekt weekendu: wolumen zgłoszeń (+161%, p<0.0001) vs zbiór [szt./h] (p=0.73)")
    ax2.set_xlabel("Dzień tygodnia (n = łączna liczba raportów w danym dniu tygodnia)")
    ax2.set_ylabel("Średnia dzienna liczba raportów [zgłoszeń / dobę]", color="#3b5975")
    ax2_twin.set_ylabel("Wydajność grzybobrania [szt./h]", color="#1e5631")

    l_a, lab_a = ax2.get_legend_handles_labels()
    l_b, lab_b = ax2_twin.get_legend_handles_labels()
    ax2.legend(l_a + l_b, lab_a + lab_b, loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p2 = figures_dir / "eda_02_seasonality_and_volume_vs_yield.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    saved_paths.append(str(p2))

    # =========================================================================
    # WYKRES 3: Porównanie okien opadów i opóźnienia (Lag Effect 1..21 dni)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=160)
    fig.suptitle(
        f"Wykres 3. Porównanie okien opadów Point-in-Time (1d / 7d / 14d / 21d i przedziały tygodniowe)\n"
        f"({period_label} | Odcięcie informacji: feature_cutoff_date = t - 1 | Liczebność próby: n = 513)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    curve_df = pd.DataFrame(eda_summary["rolling_lag_curve_1_to_21_days"])
    k_vals = curve_df["window_days_k"].values

    ax1.plot(k_vals, curve_df["cum_precip_pearson_r"], marker="o", color="#2b5c8f", linewidth=2.0, label="Suma opadu [mm] w oknie k dni (Pearson r)")
    ax1.plot(k_vals, curve_df["cum_precip_spearman_rho"], marker="s", linestyle="--", color="#5a8fc2", linewidth=1.7, label="Suma opadu [mm] w oknie k dni (Spearman ρ)")
    ax1.plot(k_vals, curve_df["rainy_days_pearson_r"], marker="^", color="#2e6f40", linewidth=2.2, label="Liczba dni deszczowych ≥1 mm w oknie k dni (Pearson r)")
    ax1.plot(k_vals, curve_df["rainy_days_spearman_rho"], marker="d", linestyle="--", color="#52b788", linewidth=1.8, label="Liczba dni deszczowych ≥1 mm w oknie k dni (Spearman ρ)")
    ax1.axhline(0.0, color="#6b7280", linestyle="-", linewidth=1.0)
    ax1.axvline(7, color="#9ca3af", linestyle=":", linewidth=1.2)
    ax1.axvline(14, color="#9ca3af", linestyle=":", linewidth=1.2)
    ax1.axvline(21, color="#9ca3af", linestyle=":", linewidth=1.2)

    ax1.set_xticks(range(1, 22, 2))
    ax1.set_title("A. Profil korelacji z wydajnością [szt./h] w funkcji długości okna k = 1..21 dni")
    ax1.set_xlabel("Długość okna wstecznego przed dniem odcięcia k [dni] (od t-k do t-1)")
    ax1.set_ylabel("Współczynnik korelacji z yield_pcs_per_h (r / ρ) [-]")
    ax1.legend(loc="lower right", fontsize=8, frameon=True)

    # Panel B: Średnie wartości kluczowych okien opadów w klasach BRAK (n=112), SREDNIO (n=239), WYSYP (n=162)
    cls_order = ["BRAK", "SREDNIO", "WYSYP"]
    cls_labels = ["BRAK (<5 szt./h)\nn = 112", "SREDNIO (5–20 szt./h)\nn = 239", "WYSYP (>20 szt./h)\nn = 162"]
    x_cls = np.arange(len(cls_order))
    width = 0.20

    m_7d = [df[df["abundance_class"] == c]["precip_lag_7d_mm"].mean() for c in cls_order]
    m_8_14d = [df[df["abundance_class"] == c]["precip_slice_8_14d_mm"].mean() for c in cls_order]
    m_15_21d = [df[df["abundance_class"] == c]["precip_slice_15_21d_mm"].mean() for c in cls_order]
    m_21d = [df[df["abundance_class"] == c]["precip_lag_21d_mm"].mean() for c in cls_order]

    ax2.bar(x_cls - 1.5 * width, m_7d, width, color="#74a9cf", label="Opad [t-7..t-1] (1. tydz., r=-0.10, ρ=-0.13)")
    ax2.bar(x_cls - 0.5 * width, m_8_14d, width, color="#3690c0", label="Opad [t-14..t-8] (2. tydz., r=+0.07, ρ=+0.11)")
    ax2.bar(x_cls + 0.5 * width, m_15_21d, width, color="#0570b0", label="Opad [t-21..t-15] (3. tydz., r=+0.09, ρ=+0.18)")
    ax2.bar(x_cls + 1.5 * width, m_21d, width, color="#1e5631", label="Suma 21-dniowa [t-21..t-1] (r=+0.06, ρ=+0.04)")

    for i, val in enumerate(m_15_21d):
        ax2.text(x_cls[i] + 0.5 * width, val + 0.9, f"{val:.1f} mm", ha="center", fontsize=8, fontweight="bold", color="#034e7b")

    ax2.set_xticks(x_cls)
    ax2.set_xticklabels(cls_labels, fontsize=9)
    ax2.set_title("B. Średnia suma opadów [mm] wg klasy obfitości i opóźnienia tygodniowego")
    ax2.set_xlabel("Klasa obfitości zbiorów (liczba obserwacji n = 513)")
    ax2.set_ylabel("Średnia suma opadów w przedziale czasu [mm]")
    ax2.legend(loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p3 = figures_dir / "eda_03_precipitation_lag_comparison.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    saved_paths.append(str(p3))

    # =========================================================================
    # WYKRES 4: Wrażliwość na dominację regionu i fazy sezonu
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=160)
    fig.suptitle(
        f"Wykres 4. Analiza wrażliwości: wpływ dominujących powiatów oraz fazy sezonu jesiennego\n"
        f"({period_label} | Liczebność próby: n = 513 ważnych obserwacji)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    reg_sens = eda_summary["regional_dominance_sensitivity"]
    sub_names = [
        f"Pełna próba\n(n={reg_sens[0]['n']})",
        f"Bez 0201 Bolesławiec\n(n={reg_sens[1]['n']})",
        f"Bez Top-3 powiatów\n(n={reg_sens[2]['n']})",
        f"Bez Top-5 powiatów\n(n={reg_sens[3]['n']})",
    ]
    r_rainy14 = [s["corr_rainy_days_14d"]["pearson_r"] for s in reg_sens]
    r_slice15_21 = [s["corr_precip_slice_15_21d"]["pearson_r"] for s in reg_sens]
    r_forest = [s["corr_forest_cover"]["pearson_r"] for s in reg_sens]

    x_s = np.arange(len(sub_names))
    w_s = 0.25
    ax1.bar(x_s - w_s, r_rainy14, w_s, color="#2e6f40", label="Dni deszczowe 14d (rainy_days_lag_14d)")
    ax1.bar(x_s, r_slice15_21, w_s, color="#2b5c8f", label="Opad w 3. tyg. [t-21..t-15] [mm]")
    ax1.bar(x_s + w_s, r_forest, w_s, color="#d97724", label="Lesistość powiatu (forest_cover_pct [%])")
    ax1.axhline(0.0, color="#6b7280", linewidth=1.0)

    for i in range(len(x_s)):
        ax1.text(x_s[i] - w_s, r_rainy14[i] + 0.01, f"{r_rainy14[i]:+.2f}", ha="center", fontsize=8)
        ax1.text(x_s[i], r_slice15_21[i] + 0.01, f"{r_slice15_21[i]:+.2f}", ha="center", fontsize=8)
        ax1.text(x_s[i] + w_s, r_forest[i] + 0.01, f"{r_forest[i]:+.2f}", ha="center", fontsize=8)

    ax1.set_xticks(x_s)
    ax1.set_xticklabels(sub_names, fontsize=8.5)
    ax1.set_ylim(-0.08, 0.36)
    ax1.set_title("A. Stabilność korelacji Pearsona r po usunięciu dominujących powiatów")
    ax1.set_xlabel("Podzbiór danych (test braku dominacji pojedynczego regionu)")
    ax1.set_ylabel("Korelacja liniowa Pearsona r z yield_pcs_per_h [-]")
    ax1.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel B: Porównanie 3 faz sezonu jesiennego
    seas_sens = eda_summary["seasonal_dominance_sensitivity"]
    p_names = [
        f"Wrzesień\n(01.09–30.09)\nn = {seas_sens[0]['n']}",
        f"Wczesny paźdz.\n(01.10–15.10)\nn = {seas_sens[1]['n']}",
        f"Późny paźdz./list.\n(16.10–02.11)\nn = {seas_sens[2]['n']}",
    ]
    x_p = np.arange(len(p_names))
    p_yield_mean = [s["mean_yield_pcs_per_h"] for s in seas_sens]
    p_yield_med = [s["median_yield_pcs_per_h"] for s in seas_sens]
    p_p14 = [s["mean_precip_14d_mm"] for s in seas_sens]
    p_tmin = [s["mean_temp_min_7d_c"] for s in seas_sens]

    ax2_twin = ax2.twinx()
    ax2.bar(x_p - 0.18, p_yield_mean, width=0.34, color="#2e6f40", alpha=0.88, label="Średnia wydajność zbioru [szt./h]")
    ax2.bar(x_p + 0.18, p_yield_med, width=0.34, color="#74c69d", alpha=0.88, label="Mediana wydajności zbioru [szt./h]")
    ax2_twin.plot(x_p, p_p14, color="#1d4ed8", marker="o", linewidth=2.2, label="Średni opad 14-dniowy [mm]")
    ax2_twin.plot(x_p, p_tmin, color="#dc2626", marker="s", linestyle="--", linewidth=2.0, label="Średnie minimum nocne z 7 dób [°C]")

    ax2.set_xticks(x_p)
    ax2.set_xticklabels(p_names, fontsize=8.5)
    ax2.set_ylim(0, 32)
    ax2_twin.set_ylim(0, 48)
    ax2_twin.grid(False)
    ax2.set_title("B. Fazy sezonu: opad 14d [mm] i temperatura min. [°C] vs wydajność [szt./h]")
    ax2.set_xlabel("Faza sezonu jesiennego 2025 (podział próby n = 513)")
    ax2.set_ylabel("Wydajność grzybobrania [szt./h]", color="#1e5631")
    ax2_twin.set_ylabel("Opad 14d [mm] / Temperatura minimalna 7d [°C]", color="#1d4ed8")

    la, laba = ax2.get_legend_handles_labels()
    lb, labb = ax2_twin.get_legend_handles_labels()
    ax2.legend(la + lb, laba + labb, loc="upper left", fontsize=8, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p4 = figures_dir / "eda_04_regional_and_seasonal_sensitivity.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    saved_paths.append(str(p4))

    # =========================================================================
    # WYKRES 5: Braki danych (NULL vs 0.0) oraz ograniczenia filtrowania gatunków
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), dpi=160)
    fig.suptitle(
        f"Wykres 5. Profil braków danych (NULL vs 0.0) oraz ograniczenia filtrowania gatunkowego\n"
        f"({period_label} | Liczebność próby: n = 513 ważnych raportów)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    sp_table = pd.DataFrame(eda_summary["species_filtering_constraints"]["species_breakdown"])
    sp_labels_map = {
        "mieszane": "mieszane (wielogatunkowe)",
        "podgrzybek_brunatny": "podgrzybek brunatny",
        "BRAK_DEKLARACJI_NULL": "NULL (brak deklaracji / 0 szt.)",
        "borowik_szlachetny": "borowik szlachetny",
        "kania": "czubajka kania",
        "maslak_zwyczajny": "maślak zwyczajny",
        "kurka": "kurka (pieprznik jadalny)",
    }
    sp_names = [sp_labels_map.get(k, k) for k in sp_table["species_key"]][::-1]
    sp_counts = sp_table["reports_count"].values[::-1]
    sp_pcts = sp_table["share_pct"].values[::-1]
    sp_regs = sp_table["covered_regions_count"].values[::-1]
    sp_colors = [
        "#d9534f" if k in {"mieszane", "BRAK_DEKLARACJI_NULL"} else "#2e6f40"
        for k in sp_table["species_key"][::-1]
    ]

    bars_sp = ax1.barh(sp_names, sp_counts, color=sp_colors, alpha=0.86, edgecolor="#1f2937")
    for bar, cnt, pct, reg_c in zip(bars_sp, sp_counts, sp_pcts, sp_regs):
        ax1.text(
            bar.get_width() + 3.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"n={cnt} ({pct:.1f}%) | {reg_c}/30 pow.",
            va="center",
            fontsize=8.5,
        )
    ax1.set_xlim(0, max(sp_counts) * 1.42)
    ax1.set_title("A. Struktura gatunkowa raportów (55.2% to zbiory mieszane lub NULL)")
    ax1.set_xlabel("Liczba raportów [liczba zgłoszeń z n = 513]")
    ax1.set_ylabel("Deklarowany gatunek dominujący (dominant_species)")

    # Panel B: Kompletność kluczowych kolumn vs jawne wartości zerowe (0.0)
    miss_cols = [
        ("harvest_duration_h", "Czas zbioru [h]"),
        ("total_pieces", "Liczba sztuk [szt.]"),
        ("dominant_species", "Gatunek dominujący"),
        ("pine_share_pct", "Taksacja BDL (sosna/dąb/wiek)"),
        ("yield_pcs_per_h", "Wydajność [szt./h] (etykieta)"),
        ("precip_lag_7d_mm", "Opad 7d [mm]"),
        ("precip_lag_21d_mm", "Opad 21d [mm]"),
        ("soil_moisture_0_7cm_m3m3", "Wilgotność gleby ERA5"),
    ]
    m_map = {m["column"]: m for m in eda_summary["missingness_profile"]}
    y_pos = np.arange(len(miss_cols))
    null_pcts = [m_map[c]["null_pct"] for c, _ in miss_cols]
    zero_pcts = [m_map[c]["explicit_zero_count"] / len(df) * 100.0 for c, _ in miss_cols]
    valid_nonzero_pcts = [100.0 - n_p - z_p for n_p, z_p in zip(null_pcts, zero_pcts)]

    ax2.barh(y_pos, valid_nonzero_pcts, color="#2e6f40", alpha=0.85, label="Wartość dodatnia / niepusta [%]")
    ax2.barh(y_pos, zero_pcts, left=valid_nonzero_pcts, color="#f0ad4e", alpha=0.90, label="Jawne zero (0.0 szt./h lub 0.0 mm) [%]")
    ax2.barh(
        y_pos,
        null_pcts,
        left=[v + z for v, z in zip(valid_nonzero_pcts, zero_pcts)],
        color="#d9534f",
        alpha=0.85,
        label="Brak danych (NULL / NaN — nie zamieniany w 0.0) [%]",
    )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([label for _, label in miss_cols], fontsize=8.5)
    ax2.set_xlim(0, 105)
    ax2.set_title("B. Rozróżnienie braków danych (NULL) od jawnych zer (0.0) w zbiorze n = 513")
    ax2.set_xlabel("Udział w zbiorze cech features.parquet [% z n = 513]")
    ax2.legend(loc="lower left", fontsize=8, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p5 = figures_dir / "eda_05_missingness_and_species_constraints.png"
    fig.savefig(p5, dpi=160)
    plt.close(fig)
    saved_paths.append(str(p5))

    return saved_paths
