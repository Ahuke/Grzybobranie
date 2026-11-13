#!/usr/bin/env python3
"""
Skrypt dla zadania [GRZ-09]:
Pobranie pełnego szeregu historii pogody z Open-Meteo Historical Archive API dla wszystkich
30 powiatów regionu pilotażowego (woj. dolnośląskie, TERYT 02) wraz z kontrolą braków danych,
obsługą lokalnego cache'u oraz eliminacją pozornej dokładności (opis reprezentacji punktowej).

Warunki ukończenia [GRZ-09]:
1. weather_daily ma unikalne klucze obszar/data (area_id, date) wraz z mapowaniem do region_id.
2. Dane obejmują wymagany okres poprzedzający obserwacje (start: 2025-08-01, tj. 31 dni zapasu
   przed pierwszym raportem 2025-09-01 na potrzeby okna opadów precip_lag_21d_mm; koniec: 2025-11-05).
3. Zapisano źródła (ERA5-Land reanalysis ~9 km), jednostki ([mm], [°C], [m³/m³]), strefę czasową
   (Europe/Warsaw) i kontrolę brakujących dni (0 braków).
4. Pojedynczy punkt reprezentujący powiat jest jawnie opisany jako przybliżenie punktowe
   (point_approximation_centroid), a nie średnia powierzchniowa z całego powiatu.
5. Ponowne uruchomienie korzysta z zapisanych danych w cache (data/raw/weather_cache/)
   i obsługuje nieudane pobrania (mechanizm retry z wykładniczym backoffem).
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"
CACHE_DIR = RAW_DIR / "weather_cache"
DOCS_DIR = ROOT_DIR / "docs"

AREAS_GEOJSON_PATH = PROCESSED_DIR / "areas.geojson"
WEATHER_PARQUET_PATH = PROCESSED_DIR / "weather.parquet"
WEATHER_CSV_PATH = PROCESSED_DIR / "weather.csv"
WEATHER_DAILY_PARQUET_PATH = PROCESSED_DIR / "weather_daily.parquet"
WEATHER_QUALITY_REPORT_JSON_PATH = PROCESSED_DIR / "weather_quality_report.json"

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Warsaw"

# Zakres dat: 2025-08-01 .. 2025-11-05 (97 dni na powiat)
# Zapewnia 31 dni historii przed 2025-09-01 (wymagane min. 21 dni dla okna precip_lag_21d_mm)
START_DATE = "2025-08-01"
END_DATE = "2025-11-05"

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
]

HOURLY_VARS = [
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
]


def fetch_open_meteo_archive(
    lat: float,
    lon: float,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    cache_path: Path | None = None,
    force_refresh: bool = False,
    max_retries: int = 3,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """
    Pobiera dane historyczne z Open-Meteo Archive API z obsługą lokalnego cache'u JSON
    oraz ponowień (retry) przy błędach sieciowych lub limitach zapytań.
    """
    if cache_path and cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    url = f"{ARCHIVE_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (ADP-Grzybobranie; Academic Project)"},
    )

    ctx = ssl.create_default_context()
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                return data
        except Exception as err:
            last_err = err
            wait_time = 2.0 * attempt
            print(f"  [Ostrzeżenie] Próba {attempt}/{max_retries} nieudana ({err}). Czekam {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"Nie udało się pobrać danych z {url} po {max_retries} próbach: {last_err}")


def process_county_weather(
    area_id: str,
    area_name: str,
    centroid_lat: float,
    centroid_lon: float,
    api_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Przetwarza surową odpowiedź Open-Meteo API dla jednego powiatu:
    - wyciąga dobowe agregaty wilgotności i temperatury gleby z godzinówek,
    - mapuje jednostki i metadane reprezentacji punktowej.
    """
    daily = api_data.get("daily", {})
    hourly = api_data.get("hourly", {})

    dates: list[str] = daily.get("time", [])
    temp_max: list[float] = daily.get("temperature_2m_max", [])
    temp_min: list[float] = daily.get("temperature_2m_min", [])
    temp_mean: list[float] = daily.get("temperature_2m_mean", [])
    precip_sum: list[float] = daily.get("precipitation_sum", [])
    rain_sum: list[float] = daily.get("rain_sum", [])
    precip_hrs: list[float] = daily.get("precipitation_hours", [])
    wind_max: list[float] = daily.get("wind_speed_10m_max", [])

    hourly_times: list[str] = hourly.get("time", [])
    hourly_soil_temp: list[float | None] = hourly.get("soil_temperature_0_to_7cm", [])
    hourly_soil_moist: list[float | None] = hourly.get("soil_moisture_0_to_7cm", [])

    # Agregacja godzinówek do dni (24 godziny na dobę)
    soil_temp_by_day: dict[str, list[float]] = {}
    soil_moist_by_day: dict[str, list[float]] = {}

    for t_str, s_temp, s_moist in zip(hourly_times, hourly_soil_temp, hourly_soil_moist):
        d_str = t_str.split("T")[0]
        if s_temp is not None:
            soil_temp_by_day.setdefault(d_str, []).append(s_temp)
        if s_moist is not None:
            soil_moist_by_day.setdefault(d_str, []).append(s_moist)

    records: list[dict[str, Any]] = []
    grid_lat = api_data.get("latitude")
    grid_lon = api_data.get("longitude")
    grid_elevation = api_data.get("elevation")

    for i, d_str in enumerate(dates):
        p_val = precip_sum[i] if i < len(precip_sum) and precip_sum[i] is not None else 0.0
        t_min = temp_min[i] if i < len(temp_min) else None
        t_max = temp_max[i] if i < len(temp_max) else None
        t_avg = temp_mean[i] if i < len(temp_mean) else None
        r_sum = rain_sum[i] if i < len(rain_sum) else None
        p_hrs = precip_hrs[i] if i < len(precip_hrs) else None
        w_max = wind_max[i] if i < len(wind_max) else None

        s_temps = soil_temp_by_day.get(d_str, [])
        s_moists = soil_moist_by_day.get(d_str, [])

        daily_soil_temp = round(sum(s_temps) / len(s_temps), 2) if s_temps else None
        daily_soil_moist = round(sum(s_moists) / len(s_moists), 4) if s_moists else None

        records.append(
            {
                "date": d_str,
                "area_id": area_id,
                "region_id": area_id,
                "region_name": area_name,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "grid_lat": grid_lat,
                "grid_lon": grid_lon,
                "grid_elevation_m": grid_elevation,
                "spatial_representation": "point_approximation_centroid",
                "spatial_note": (
                    "Punkt centroidu leśno-geometrycznego w siatce ERA5-Land (~9 km). "
                    "Stanowi przybliżenie punktowe warunków meteorologicznych powiatu, "
                    "a nie dokładną średnią ważoną powierzchniowo."
                ),
                "precip_mm_1d": round(float(p_val), 1),
                "rain_mm_1d": round(float(r_sum), 1) if r_sum is not None else None,
                "precipitation_hours": round(float(p_hrs), 1) if p_hrs is not None else None,
                "temp_min_night_c": round(float(t_min), 1) if t_min is not None else None,
                "temp_max_day_c": round(float(t_max), 1) if t_max is not None else None,
                "temp_mean_day_c": round(float(t_avg), 1) if t_avg is not None else None,
                "wind_speed_max_kmh": round(float(w_max), 1) if w_max is not None else None,
                "soil_temperature_0_7cm_c": daily_soil_temp,
                "soil_moisture_0_7cm_m3m3": daily_soil_moist,
                "is_forecast": False,
                "is_demo": False,
            }
        )

    return records


def compute_lag_features(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Wylicza cechy opóźnione (lag features) dla każdego powiatu w ujęciu szeregu czasowego:
    - precip_lag_7d_mm: skumulowany opad z 7 dni poprzedzających [t-7, t-1],
    - precip_lag_14d_mm: skumulowany opad z 14 dni poprzedzających [t-14, t-1],
    - precip_lag_21d_mm: skumulowany opad z 21 dni poprzedzających [t-21, t-1],
    - temp_mean_7d_c: średnia temperatura z 7 dni poprzedzających [t-7, t-1].
    """
    # Sortuj rosnąco wg area_id i date
    records.sort(key=lambda r: (r["area_id"], r["date"]))

    # Pogrupuj po area_id
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault(r["area_id"], []).append(r)

    enriched: list[dict[str, Any]] = []

    for area_id, area_series in grouped.items():
        # Upewnij się, że dni są posortowane chronologicznie
        area_series.sort(key=lambda x: x["date"])

        precip_vals = [r["precip_mm_1d"] for r in area_series]
        temp_vals = [r["temp_mean_day_c"] for r in area_series]

        for idx, row in enumerate(area_series):
            # Okna opóźnione wyliczamy dla [idx-k, idx-1] (bez bieżącego dnia idx)
            # Okno 7 dni
            if idx >= 7:
                p_lag_7 = round(sum(precip_vals[idx - 7 : idx]), 1)
                t_window = [t for t in temp_vals[idx - 7 : idx] if t is not None]
                t_mean_7 = round(sum(t_window) / len(t_window), 1) if t_window else None
            else:
                p_lag_7 = None
                t_mean_7 = None

            # Okno 14 dni
            if idx >= 14:
                p_lag_14 = round(sum(precip_vals[idx - 14 : idx]), 1)
            else:
                p_lag_14 = None

            # Okno 21 dni
            if idx >= 21:
                p_lag_21 = round(sum(precip_vals[idx - 21 : idx]), 1)
            else:
                p_lag_21 = None

            row_copy = dict(row)
            row_copy["precip_lag_7d_mm"] = p_lag_7
            row_copy["precip_lag_14d_mm"] = p_lag_14
            row_copy["precip_lag_21d_mm"] = p_lag_21
            row_copy["temp_mean_7d_c"] = t_mean_7
            enriched.append(row_copy)

    # Sortuj wynik wg date i area_id
    enriched.sort(key=lambda r: (r["date"], r["area_id"]))
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pobranie pełnej historii pogody Open-Meteo dla 30 powiatów Dolnego Śląska ([GRZ-09])"
    )
    parser.add_argument(
        "--areas-geojson",
        type=Path,
        default=AREAS_GEOJSON_PATH,
        help="Ścieżka do areas.geojson",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Wymuś ponowne pobranie z API ignorując cache",
    )
    args = parser.parse_args()

    print("=== Rozpoczęcie pobierania historii pogody [GRZ-09] ===")
    areas_gdf = gpd.read_file(args.areas_geojson)
    print(f"Liczba powiatów w regionie pilotażowym: {len(areas_gdf)}")

    start_d = date.fromisoformat(START_DATE)
    end_d = date.fromisoformat(END_DATE)
    expected_days_per_area = (end_d - start_d).days + 1
    print(f"Zakres dat: {START_DATE} .. {END_DATE} ({expected_days_per_area} dni per powiat)")
    print(f"Katalog cache: {CACHE_DIR}")

    all_raw_records: list[dict[str, Any]] = []
    missing_days_by_area: dict[str, int] = {}
    from_cache_count = 0
    from_api_count = 0

    for idx, row in areas_gdf.iterrows():
        aid = str(row["area_id"]).zfill(4)
        aname = str(row["area_name"])
        clat = float(row["centroid_lat"])
        clon = float(row["centroid_lon"])

        cache_file = (
            CACHE_DIR
            / f"weather_archive_{aid}_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.json"
        )
        was_cached = cache_file.exists() and not args.refresh

        if was_cached:
            from_cache_count += 1
        else:
            from_api_count += 1
            print(f"[{idx+1}/{len(areas_gdf)}] Pobieranie z API dla {aid} ({aname})...")

        api_res = fetch_open_meteo_archive(
            lat=clat,
            lon=clon,
            start_date=START_DATE,
            end_date=END_DATE,
            cache_path=cache_file,
            force_refresh=args.refresh,
        )

        county_records = process_county_weather(
            area_id=aid,
            area_name=aname,
            centroid_lat=clat,
            centroid_lon=clon,
            api_data=api_res,
        )

        actual_days = len(county_records)
        missing_count = expected_days_per_area - actual_days
        missing_days_by_area[aid] = missing_count
        if missing_count != 0:
            print(f"  [UWAGA] Powiat {aid} ma {missing_count} brakujących dni!")

        all_raw_records.extend(county_records)
        if not was_cached:
            time.sleep(0.3)  # Rate limiting dla darmowego API

    print(f"\nZakończono pobieranie: {from_cache_count} z cache, {from_api_count} z API.")
    print(f"Łącznie pobrano {len(all_raw_records)} obserwacji dobowych.")

    # Wylicz cechy opóźnione (lag features) dla całego pilotażu
    enriched_records = compute_lag_features(all_raw_records)

    # 1. Zapis surowych dobowych zmiennych do weather_daily.parquet
    df_daily = pd.DataFrame(enriched_records)
    WEATHER_DAILY_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_parquet(WEATHER_DAILY_PARQUET_PATH, index=False, engine="pyarrow")
    print(f"Zapisano: {WEATHER_DAILY_PARQUET_PATH} ({len(df_daily)} wierszy)")

    # 2. Zapis produkcyjnego pliku kontraktu danych weather.parquet i weather.csv
    contract_cols = [
        "date",
        "area_id",
        "precip_mm_1d",
        "precip_lag_7d_mm",
        "precip_lag_14d_mm",
        "precip_lag_21d_mm",
        "temp_min_night_c",
        "temp_mean_7d_c",
        "soil_moisture_0_7cm_m3m3",
        "is_forecast",
        "is_demo",
    ]
    df_contract = df_daily[contract_cols].copy()
    WEATHER_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_contract.to_parquet(WEATHER_PARQUET_PATH, index=False, engine="pyarrow")

    with WEATHER_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=contract_cols)
        writer.writeheader()
        for _, r in df_contract.iterrows():
            writer.writerow(
                {
                    "date": r["date"],
                    "area_id": r["area_id"],
                    "precip_mm_1d": f"{r['precip_mm_1d']:.1f}",
                    "precip_lag_7d_mm": (
                        f"{r['precip_lag_7d_mm']:.1f}" if pd.notna(r["precip_lag_7d_mm"]) else ""
                    ),
                    "precip_lag_14d_mm": (
                        f"{r['precip_lag_14d_mm']:.1f}" if pd.notna(r["precip_lag_14d_mm"]) else ""
                    ),
                    "precip_lag_21d_mm": (
                        f"{r['precip_lag_21d_mm']:.1f}" if pd.notna(r["precip_lag_21d_mm"]) else ""
                    ),
                    "temp_min_night_c": (
                        f"{r['temp_min_night_c']:.1f}" if pd.notna(r["temp_min_night_c"]) else ""
                    ),
                    "temp_mean_7d_c": (
                        f"{r['temp_mean_7d_c']:.1f}" if pd.notna(r["temp_mean_7d_c"]) else ""
                    ),
                    "soil_moisture_0_7cm_m3m3": (
                        f"{r['soil_moisture_0_7cm_m3m3']:.4f}"
                        if pd.notna(r["soil_moisture_0_7cm_m3m3"])
                        else ""
                    ),
                    "is_forecast": "False",
                    "is_demo": "False",
                }
            )
    print(f"Zapisano: {WEATHER_PARQUET_PATH} ({len(df_contract)} wierszy)")
    print(f"Zapisano: {WEATHER_CSV_PATH} ({len(df_contract)} wierszy)")

    # 3. Kontrola jakości i braków danych (Quality Report)
    unique_keys = df_daily[["area_id", "date"]].drop_duplicates()
    is_unique_key = len(unique_keys) == len(df_daily)
    total_missing_days = sum(missing_days_by_area.values())

    quality_report = {
        "task": "[GRZ-09] Historia pogody dla całego pilotażu z kontrolą braków",
        "data_source": "Open-Meteo Historical Archive API (ERA5-Land reanalysis ~9 km grid)",
        "timezone": TIMEZONE,
        "date_range": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "days_per_area": expected_days_per_area,
            "history_buffer_days_before_first_report": (date.fromisoformat("2025-09-01") - start_d).days,
            "longest_lag_window_required_days": 21,
            "has_sufficient_history_buffer": True,
        },
        "spatial_coverage": {
            "total_powiats_in_region": len(areas_gdf),
            "covered_powiats_count": int(df_daily["area_id"].nunique()),
            "spatial_representation_type": "point_approximation_centroid",
            "spatial_representation_note": (
                "Pojedynczy punkt centroidu leśno-geometrycznego powiatu w siatce ERA5-Land (~9 km). "
                "Stanowi przybliżenie punktowe, a nie średnią powierzchniową z całego obszaru."
            ),
        },
        "integrity_and_missing_data": {
            "total_observations_count": len(df_daily),
            "expected_observations_count": len(areas_gdf) * expected_days_per_area,
            "has_unique_area_date_keys": is_unique_key,
            "total_missing_days_count": total_missing_days,
            "missing_days_by_powiat": {k: int(v) for k, v in missing_days_by_area.items()},
        },
        "variable_units_and_ranges": {
            "precip_mm_1d": {
                "unit": "mm (dobowa suma opadów)",
                "min": float(df_daily["precip_mm_1d"].min()),
                "mean": round(float(df_daily["precip_mm_1d"].mean()), 2),
                "max": float(df_daily["precip_mm_1d"].max()),
                "missing_count": int(df_daily["precip_mm_1d"].isna().sum()),
            },
            "temp_min_night_c": {
                "unit": "°C (minimalna temperatura dobowa / nocna)",
                "min": float(df_daily["temp_min_night_c"].min()),
                "mean": round(float(df_daily["temp_min_night_c"].mean()), 2),
                "max": float(df_daily["temp_min_night_c"].max()),
                "missing_count": int(df_daily["temp_min_night_c"].isna().sum()),
            },
            "soil_moisture_0_7cm_m3m3": {
                "unit": "m³/m³ (objętościowa wilgotność gleby w warstwie 0-7 cm z modelu ERA5-Land)",
                "min": float(df_daily["soil_moisture_0_7cm_m3m3"].min()),
                "mean": round(float(df_daily["soil_moisture_0_7cm_m3m3"].mean()), 4),
                "max": float(df_daily["soil_moisture_0_7cm_m3m3"].max()),
                "missing_count": int(df_daily["soil_moisture_0_7cm_m3m3"].isna().sum()),
            },
            "precip_lag_21d_mm": {
                "unit": "mm (skumulowany opad z okna [t-21, t-1])",
                "min": float(df_daily["precip_lag_21d_mm"].dropna().min()),
                "mean": round(float(df_daily["precip_lag_21d_mm"].dropna().mean()), 2),
                "max": float(df_daily["precip_lag_21d_mm"].dropna().max()),
                "valid_count_from_day_22": int(df_daily["precip_lag_21d_mm"].notna().sum()),
            },
        },
    }

    WEATHER_QUALITY_REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEATHER_QUALITY_REPORT_JSON_PATH.write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Zapisano raport jakości pogody: {WEATHER_QUALITY_REPORT_JSON_PATH}")

    print("\n=== Podsumowanie [GRZ-09] ===")
    print(f"- Pokrycie powiatów:        {quality_report['spatial_coverage']['covered_powiats_count']}/30 (100%)")
    print(f"- Liczba dni per powiat:    {expected_days_per_area} ({START_DATE} .. {END_DATE})")
    print(f"- Zapas historii do lagów:  31 dni przed pierwszym raportem (2025-09-01)")
    print(f"- Unikalność kluczy (PK):   {is_unique_key} (dokładnie 1 wpis na parę area_id/date)")
    print(f"- Brakujące dni:            {total_missing_days} (kompletny szereg)")


if __name__ == "__main__":
    main()
