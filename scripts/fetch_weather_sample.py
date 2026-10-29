#!/usr/bin/env python3
"""
Skrypt dla zadania [GRZ-04] (#4):
Pobranie próbki historii pogody z Open-Meteo Historical Archive API dla wybranych punktów
w regionie pilotażowym (woj. dolnośląskie) w okresie obejmującym co najmniej 21 dni przed
datami raportów z grzyby.pl (2025-09-01 .. 2025-10-05, tj. 28 dni przed tygodniem raportów
2025-09-29 .. 2025-10-05).

Zasady:
- Skrypt korzysta z lokalnego cache'u (data/raw/weather_cache/) oraz zapisuje gotowy plik
  wynikowy data/sample/weather_daily_sample_ds.csv.
- Jeśli plik wynikowy już istnieje i nie podano flagi --refresh, skrypt nie ponawia zapytań
  do API przy kolejnych uruchomieniach.
- Dane glebowe są jawnie oznaczone jako wynik modelu reanalizy (ERA5-Land, siatka ~9 km),
  a nie pomiar terenowy (in-situ).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from statistics import mean

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Warsaw"
START_DATE = "2025-09-01"
END_DATE = "2025-10-05"

# 6 reprezentatywnych lokalizacji z próbki raportów [GRZ-03] na Dolnym Śląsku
SAMPLE_LOCATIONS = [
    {
        "area_id": "0201",
        "location_name": "Krępnica / Bory Dolnośląskie (pow. bolesławiecki)",
        "lat": 51.323350,
        "lon": 15.560140,
    },
    {
        "area_id": "0202",
        "location_name": "Góry Sowie (pow. dzierżoniowski)",
        "lat": 50.700154,
        "lon": 16.500778,
    },
    {
        "area_id": "0208",
        "location_name": "Batorów / Góry Stołowe (pow. kłodzki)",
        "lat": 50.438889,
        "lon": 16.424722,
    },
    {
        "area_id": "0213",
        "location_name": "Łazy Wielkie / Dolina Baryczy (pow. milicki)",
        "lat": 51.418694,
        "lon": 17.298751,
    },
    {
        "area_id": "0214",
        "location_name": "Białe Błoto / lasy twardogórskie (pow. oleśnicki)",
        "lat": 51.333120,
        "lon": 17.353640,
    },
    {
        "area_id": "0221",
        "location_name": "Witków / masyw Trójgarbu (pow. wałbrzyski)",
        "lat": 50.796974,
        "lon": 16.130848,
    },
]

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "et0_fao_evapotranspiration",
]

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
]


def fetch_location_weather(loc: dict, cache_dir: Path, refresh: bool = False) -> dict:
    """Pobiera dane z Open-Meteo Archive API z zapisem do lokalnego cache'u JSON."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"open_meteo_{loc['area_id']}_{START_DATE}_{END_DATE}.json"

    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    url = f"{ARCHIVE_API_URL}?{urllib.parse.urlencode(params)}"
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Grzybobranie-ADP-Academic-Project/1.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as err:
            last_err = err
            time.sleep(1.5 * attempt)
    else:
        raise RuntimeError(f"Nie udało się pobrać danych dla {loc['area_id']}: {last_err}")

    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(0.5)
    return payload


def aggregate_daily_records(loc: dict, payload: dict) -> list[dict]:
    """
    Łączy zmienne dobowe (daily) ze zagregowanymi zmiennymi godzinowymi (hourly)
    w strefie czasowej Europe/Warsaw (doba 00:00-23:59) oraz wylicza opóźnienia
    opadów (lag 7d, 14d, 21d) bez wycieku danych z dnia bieżącego [t-k, t-1].
    """
    daily = payload["daily"]
    hourly = payload["hourly"]
    dates = daily["time"]
    hourly_times = hourly["time"]

    # Indeksowanie godzin po dacie YYYY-MM-DD (doba lokalna Europe/Warsaw)
    day_to_hour_indices: dict[str, list[int]] = {d: [] for d in dates}
    for idx, ts in enumerate(hourly_times):
        day_str = ts[:10]
        if day_str in day_to_hour_indices:
            day_to_hour_indices[day_str].append(idx)

    rows: list[dict] = []
    precip_series: list[float] = []
    tmean_series: list[float] = []

    for i, d in enumerate(dates):
        h_idxs = day_to_hour_indices[d]

        def h_vals(var_name: str, idxs: list[int] = h_idxs) -> list[float]:
            arr = hourly[var_name]
            return [float(arr[j]) for j in idxs if arr[j] is not None]

        # Godziny nocne/poranne: 00:00-07:00 oraz 21:00-23:00 danej doby
        night_idxs = [
            j for j in h_idxs if int(hourly_times[j][11:13]) in {0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 23}
        ]
        night_temps = h_vals("temperature_2m", night_idxs)

        st_0_7 = h_vals("soil_temperature_0_to_7cm")
        st_7_28 = h_vals("soil_temperature_7_to_28cm")
        st_28_100 = h_vals("soil_temperature_28_to_100cm")
        sm_0_7 = h_vals("soil_moisture_0_to_7cm")
        sm_7_28 = h_vals("soil_moisture_7_to_28cm")
        sm_28_100 = h_vals("soil_moisture_28_to_100cm")
        rh_2m = h_vals("relative_humidity_2m")

        p_1d = float(daily["precipitation_sum"][i]) if daily["precipitation_sum"][i] is not None else 0.0
        t_mean = float(daily["temperature_2m_mean"][i]) if daily["temperature_2m_mean"][i] is not None else None

        # Obliczanie okien opóźnionych [t-k, t-1] (wyłącznie pełne okna; w przeciwnym razie NULL "")
        lag_7d = round(sum(precip_series[i - 7 : i]), 2) if i >= 7 else ""
        lag_14d = round(sum(precip_series[i - 14 : i]), 2) if i >= 14 else ""
        lag_21d = round(sum(precip_series[i - 21 : i]), 2) if i >= 21 else ""
        tmean_7d = (
            round(mean(tmean_series[i - 7 : i]), 2)
            if i >= 7 and all(x is not None for x in tmean_series[i - 7 : i])
            else ""
        )

        precip_series.append(p_1d)
        tmean_series.append(t_mean if t_mean is not None else 0.0)

        rows.append(
            {
                "date": d,
                "area_id": loc["area_id"],
                "location_name": loc["location_name"],
                "request_lat": loc["lat"],
                "request_lon": loc["lon"],
                "grid_lat": round(float(payload.get("latitude", loc["lat"])), 5),
                "grid_lon": round(float(payload.get("longitude", loc["lon"])), 5),
                "grid_elevation_m": payload.get("elevation", ""),
                "timezone": payload.get("timezone", TIMEZONE),
                "source_api": "Open-Meteo Archive API (ERA5-Land)",
                "soil_data_origin": "model_reanalysis_ERA5_Land_9km",
                "precip_mm_1d": round(p_1d, 2),
                "rain_mm_1d": round(float(daily["rain_sum"][i]), 2) if daily["rain_sum"][i] is not None else "",
                "precip_hours_1d": round(float(daily["precipitation_hours"][i]), 1)
                if daily["precipitation_hours"][i] is not None
                else "",
                "precip_lag_7d_mm": lag_7d,
                "precip_lag_14d_mm": lag_14d,
                "precip_lag_21d_mm": lag_21d,
                "temp_min_2m_c": round(float(daily["temperature_2m_min"][i]), 1)
                if daily["temperature_2m_min"][i] is not None
                else "",
                "temp_min_night_c": round(min(night_temps), 1) if night_temps else "",
                "temp_mean_2m_c": round(t_mean, 1) if t_mean is not None else "",
                "temp_max_2m_c": round(float(daily["temperature_2m_max"][i]), 1)
                if daily["temperature_2m_max"][i] is not None
                else "",
                "temp_mean_7d_c": tmean_7d,
                "rel_humidity_2m_mean_pct": round(mean(rh_2m), 1) if rh_2m else "",
                "et0_evap_mm_1d": round(float(daily["et0_fao_evapotranspiration"][i]), 2)
                if daily["et0_fao_evapotranspiration"][i] is not None
                else "",
                "soil_temp_0_7cm_mean_c": round(mean(st_0_7), 1) if st_0_7 else "",
                "soil_temp_0_7cm_min_c": round(min(st_0_7), 1) if st_0_7 else "",
                "soil_temp_7_28cm_mean_c": round(mean(st_7_28), 1) if st_7_28 else "",
                "soil_temp_28_100cm_mean_c": round(mean(st_28_100), 1) if st_28_100 else "",
                "soil_moisture_0_7cm_m3m3": round(mean(sm_0_7), 3) if sm_0_7 else "",
                "soil_moisture_7_28cm_m3m3": round(mean(sm_7_28), 3) if sm_7_28 else "",
                "soil_moisture_28_100cm_m3m3": round(mean(sm_28_100), 3) if sm_28_100 else "",
                "is_forecast": False,
                "is_demo": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pobiera i zapisuje próbkę historii pogody z Open-Meteo ([GRZ-04] #4)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Wymuś ponowne pobranie danych z API zamiast użycia zapisanego pliku/cache'u",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    cache_dir = root_dir / "data" / "raw" / "weather_cache"
    out_csv = root_dir / "data" / "sample" / "weather_daily_sample_ds.csv"

    if out_csv.exists() and not args.refresh:
        with out_csv.open("r", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        print(
            f"[CACHE] Plik próbki już istnieje ({len(existing_rows)} wierszy): {out_csv}\n"
            f"Pominięto zapytania sieciowe (użyj --refresh, aby pobrać ponownie)."
        )
        return

    all_rows: list[dict] = []
    for loc in SAMPLE_LOCATIONS:
        payload = fetch_location_weather(loc, cache_dir=cache_dir, refresh=args.refresh)
        loc_rows = aggregate_daily_records(loc, payload)
        all_rows.extend(loc_rows)
        print(
            f"Pobrano i przetworzono {len(loc_rows)} dni dla: {loc['area_id']} ({loc['location_name']}) "
            f"[siatka ERA5-Land: {loc_rows[0]['grid_lat']}, {loc_rows[0]['grid_lon']}, "
            f"{loc_rows[0]['grid_elevation_m']} m n.p.m.]"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nZapisano łącznie {len(all_rows)} rekordów do: {out_csv}")


if __name__ == "__main__":
    main()
