r"""
Moduł inżynierii cech i łączenia danych (raporty + cechy leśne BDL + historia pogody)
z zachowaniem rygoru czasowego Point-in-Time Correctness ([GRZ-10] #10).

Główne gwarancje metodologiczne:
1. Każda obserwacja (etykieta) pochodzi z konkretnego raportu (`report_id` / `observation_id`);
   złączenie po `region_id` i dacie nigdy nie powiela wierszy (1:1, multiplication_factor == 1.0).
2. Oddzielenie dat:
   - `observation_date` (`harvest_date` / `date`): dzień faktycznego zbioru w lesie,
   - `target_date`: dzień, na który prognozujemy ($t$),
   - `issued_at`: dzień wystawienia prognozy ($t_0 = t - H$, gdzie $H = \text{horizon\_days}$),
   - `feature_cutoff_date`: dzień odcięcia danych pogodowych ($t_{\text{cutoff}} = t_0 = t - H$).
3. Brak wycieku z przyszłości (No Future Leakage):
   - Wszystkie okna pogodowe (7, 14, 21 dni) oraz cechy temperatury i gleby wyliczane są
     wyłącznie z dób $\tau \le \text{feature\_cutoff\_date}$ (czyli kończą się najpóźniej w dniu $t - H$,
     np. $t - 1$ dla prognozy jednodniowej $H = 1$).
   - Pogoda z samego dnia zbioru $t$ ani dni późniejszych nigdy nie wchodzi do wektora cech.
4. Zakaz imputacji braków zerami (`NULL` != `0.0`):
   - Jeżeli w oknie $k$-dniowym brakuje choćby jednego dnia lub pomiaru w tabeli pogodowej,
     okno przyjmuje wartość `None` (`NaN`), a nigdy częściową sumę ani `0.0`.
   - Braki w cechach leśnych BDL (`pine_share_pct` itd.) oraz opcjonalnych polach raportów
     (`dominant_species`, `harvest_duration_h`) pozostają `None` (`NaN`).
   - Brak raportu w powiecie i dniu nigdy nie jest zamieniany w zerowy zbiór (`0.0 szt./h`).
"""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any

import pandas as pd

from src.evaluation import (
    DEFAULT_SPLIT_DATE_TRAIN_TEST,
    DEFAULT_SPLIT_DATE_TRAIN_VAL,
    DEFAULT_SPLIT_DATE_VAL_TEST,
    classify_abundance,
)

# Oficjalny słownik kolumn zbioru analitycznego features.parquet ([GRZ-10] Warunek #1)
FEATURE_DICTIONARY: dict[str, dict[str, str]] = {
    "observation_id": {
        "type": "string",
        "unit": "id",
        "source": "reports_spatial.report_id",
        "description": "Unikalny klucz główny obserwacji (1:1 ze źródłowym report_id).",
    },
    "report_id": {
        "type": "string",
        "unit": "id",
        "source": "reports_spatial.report_id",
        "description": "Identyfikator źródłowego raportu z portalu grzyby.pl (np. DG174806).",
    },
    "region_id": {
        "type": "string",
        "unit": "TERYT (4 znaki)",
        "source": "forest_features.region_id",
        "description": "Kod TERYT powiatu z zachowaniem wiodącego zera (np. 0201).",
    },
    "area_id": {
        "type": "string",
        "unit": "TERYT (4 znaki)",
        "source": "forest_features.area_id",
        "description": "Alias klucza powiatu zgodny z docs/data_contract.md.",
    },
    "region_name": {
        "type": "string",
        "unit": "tekst",
        "source": "forest_features.region_name",
        "description": "Nazwa powiatu (np. powiat bolesławiecki).",
    },
    "place_id": {
        "type": "string",
        "unit": "id",
        "source": "reports_spatial.place_id",
        "description": "Identyfikator lokalizacji z bazy grzyby.pl.",
    },
    "observation_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "reports_spatial.harvest_date",
        "description": "Data faktycznego grzybobrania (obserwacji terenowej w lesie).",
    },
    "harvest_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "reports_spatial.harvest_date",
        "description": "Data zbioru (alias observation_date używany w module ewaluacji).",
    },
    "date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "reports_spatial.date",
        "description": "Data obserwacji (alias zgodny z kontraktem danych).",
    },
    "publication_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "reports_spatial.publication_date",
        "description": "Data publikacji raportu na portalu grzyby.pl.",
    },
    "target_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "wyliczona (t)",
        "description": "Data docelowa, na którą wystawiana jest prognoza (t = observation_date).",
    },
    "issued_at": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "wyliczona (t0 = t - horizon_days)",
        "description": "Data wystawienia prognozy (moment symulacji predykcji).",
    },
    "feature_cutoff_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "wyliczona (t_cutoff = issued_at)",
        "description": "Data odcięcia dostępności cech pogodowych (ostatnia pełna doba znana w chwili wystawienia prognozy).",
    },
    "horizon_days": {
        "type": "int64",
        "unit": "dni",
        "source": "parametr potoku (domyślnie 1)",
        "description": "Horyzont wyprzedzenia prognozy w dniach (target_date - issued_at).",
    },
    "window_7d_start_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "wyliczona (feature_cutoff_date - 6 dni)",
        "description": "Początek zamkniętego 7-dniowego okna pogodowego [t_cutoff-6, t_cutoff].",
    },
    "window_14d_start_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "wyliczona (feature_cutoff_date - 13 dni)",
        "description": "Początek zamkniętego 14-dniowego okna pogodowego [t_cutoff-13, t_cutoff].",
    },
    "window_21d_start_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "wyliczona (feature_cutoff_date - 20 dni)",
        "description": "Początek zamkniętego 21-dniowego okna pogodowego [t_cutoff-20, t_cutoff].",
    },
    "window_end_date": {
        "type": "string",
        "unit": "YYYY-MM-DD",
        "source": "wyliczona (== feature_cutoff_date)",
        "description": "Koniec wszystkich okien pogodowych (nigdy nie przekracza feature_cutoff_date).",
    },
    "yield_pcs_per_h": {
        "type": "float64",
        "unit": "szt./h",
        "source": "reports_spatial.yield_pcs_per_h",
        "description": "Główna zmienna docelowa (Target regresji): wydajność grzybobrania na 1 osobę i 1 godzinę.",
    },
    "abundance_class": {
        "type": "string",
        "unit": "enum (BRAK, SREDNIO, WYSYP)",
        "source": "wyliczona z yield_pcs_per_h",
        "description": "Zmienna docelowa klasyfikacji (<5 BRAK, 5-20 SREDNIO, >20 WYSYP).",
    },
    "dominant_species": {
        "type": "string (nullable)",
        "unit": "słownik gatunków",
        "source": "reports_spatial.dominant_species",
        "description": "Dominujący gatunek grzyba w raporcie (NULL gdy nie podano).",
    },
    "harvest_duration_h": {
        "type": "float64 (nullable)",
        "unit": "h",
        "source": "reports_spatial.harvest_duration_h",
        "description": "Deklarowany czas zbierania w godzinach (>0 lub NULL).",
    },
    "total_pieces": {
        "type": "float64 (nullable)",
        "unit": "szt.",
        "source": "reports_spatial.total_pieces",
        "description": "Deklarowana łączna liczba zebranych grzybów (NULL gdy nie podano).",
    },
    "precip_lag_7d_mm": {
        "type": "float64 (nullable)",
        "unit": "mm",
        "source": "weather_daily (okno 7d do feature_cutoff_date)",
        "description": "Skumulowana suma opadów z pełnych 7 dni kończących się w dniu odcięcia.",
    },
    "precip_lag_14d_mm": {
        "type": "float64 (nullable)",
        "unit": "mm",
        "source": "weather_daily (okno 14d do feature_cutoff_date)",
        "description": "Skumulowana suma opadów z pełnych 14 dni kończących się w dniu odcięcia.",
    },
    "precip_lag_21d_mm": {
        "type": "float64 (nullable)",
        "unit": "mm",
        "source": "weather_daily (okno 21d do feature_cutoff_date)",
        "description": "Skumulowana suma opadów z pełnych 21 dni kończących się w dniu odcięcia.",
    },
    "precip_cutoff_1d_mm": {
        "type": "float64 (nullable)",
        "unit": "mm",
        "source": "weather_daily (w dniu feature_cutoff_date)",
        "description": "Dobowa suma opadów w dniu odcięcia cech (ostatnia doba przed prognozą).",
    },
    "precip_hours_lag_7d_h": {
        "type": "float64 (nullable)",
        "unit": "h",
        "source": "weather_daily (okno 7d do feature_cutoff_date)",
        "description": "Suma godzin z opadem atmosferycznym w oknie 7 dni przed odcięciem.",
    },
    "rainy_days_lag_7d_count": {
        "type": "float64 (nullable)",
        "unit": "dni (0-7)",
        "source": "weather_daily (okno 7d do feature_cutoff_date)",
        "description": "Liczba dni z opadem dobowym >= 1.0 mm w oknie 7 dni przed odcięciem.",
    },
    "rainy_days_lag_14d_count": {
        "type": "float64 (nullable)",
        "unit": "dni (0-14)",
        "source": "weather_daily (okno 14d do feature_cutoff_date)",
        "description": "Liczba dni z opadem dobowym >= 1.0 mm w oknie 14 dni przed odcięciem.",
    },
    "temp_min_night_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (w dniu feature_cutoff_date)",
        "description": "Minimalna temperatura nocna/poranna w dobie odcięcia cech.",
    },
    "temp_min_lag_3d_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (okno 3d do feature_cutoff_date)",
        "description": "Najniższa temperatura minimalna z ostatnich 3 dób przed odcięciem.",
    },
    "temp_min_lag_7d_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (okno 7d do feature_cutoff_date)",
        "description": "Najniższa temperatura minimalna z ostatnich 7 dób przed odcięciem.",
    },
    "frost_nights_lag_7d_count": {
        "type": "float64 (nullable)",
        "unit": "dni (0-7)",
        "source": "weather_daily (okno 7d do feature_cutoff_date)",
        "description": "Liczba dób z przymrozkiem (temp_min_night_c <= 0.0°C) w oknie 7 dni przed odcięciem.",
    },
    "temp_mean_cutoff_1d_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (w dniu feature_cutoff_date)",
        "description": "Średnia temperatura dobowa na 2 m w dniu odcięcia cech.",
    },
    "temp_mean_7d_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (okno 7d do feature_cutoff_date)",
        "description": "Średnia temperatura dobowa z okna 7 dni kończącego się w dniu odcięcia.",
    },
    "temp_mean_14d_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (okno 14d do feature_cutoff_date)",
        "description": "Średnia temperatura dobowa z okna 14 dni kończącego się w dniu odcięcia.",
    },
    "temp_max_cutoff_1d_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (w dniu feature_cutoff_date)",
        "description": "Maksymalna temperatura dobowa w dniu odcięcia cech.",
    },
    "wind_speed_mean_7d_kmh": {
        "type": "float64 (nullable)",
        "unit": "km/h",
        "source": "weather_daily (okno 7d do feature_cutoff_date)",
        "description": "Średnia z dobowych maksimów prędkości wiatru w oknie 7 dni przed odcięciem.",
    },
    "soil_moisture_0_7cm_m3m3": {
        "type": "float64 (nullable)",
        "unit": "m³/m³",
        "source": "weather_daily (ERA5-Land w dniu feature_cutoff_date)",
        "description": "Modelowana objętościowa wilgotność gleby (0-7 cm) w dniu odcięcia cech.",
    },
    "soil_moisture_mean_7d_m3m3": {
        "type": "float64 (nullable)",
        "unit": "m³/m³",
        "source": "weather_daily (okno 7d ERA5-Land do feature_cutoff_date)",
        "description": "Średnia modelowana wilgotność gleby (0-7 cm) z okna 7 dni przed odcięciem.",
    },
    "soil_moisture_mean_14d_m3m3": {
        "type": "float64 (nullable)",
        "unit": "m³/m³",
        "source": "weather_daily (okno 14d ERA5-Land do feature_cutoff_date)",
        "description": "Średnia modelowana wilgotność gleby (0-7 cm) z okna 14 dni przed odcięciem.",
    },
    "soil_temperature_0_7cm_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (ERA5-Land w dniu feature_cutoff_date)",
        "description": "Modelowana temperatura wierzchniej warstwy gleby (0-7 cm) w dniu odcięcia.",
    },
    "soil_temp_mean_7d_c": {
        "type": "float64 (nullable)",
        "unit": "°C",
        "source": "weather_daily (okno 7d ERA5-Land do feature_cutoff_date)",
        "description": "Średnia modelowana temperatura gleby (0-7 cm) z okna 7 dni przed odcięciem.",
    },
    "grid_elevation_m": {
        "type": "float64 (nullable)",
        "unit": "m n.p.m.",
        "source": "weather_daily.grid_elevation_m",
        "description": "Wysokość n.p.m. oczka siatki ERA5-Land reprezentującego powiat.",
    },
    "total_area_km2": {
        "type": "float64",
        "unit": "km²",
        "source": "forest_features.total_area_km2",
        "description": "Całkowita powierzchnia geometryczna powiatu w układzie EPSG:2180.",
    },
    "forest_cover_pct": {
        "type": "float64 (nullable)",
        "unit": "%",
        "source": "forest_features.forest_cover_pct",
        "description": "Lesistość powiatu (udział lasów w powierzchni całkowitej).",
    },
    "forest_area_km2": {
        "type": "float64 (nullable)",
        "unit": "km²",
        "source": "forest_features.forest_area_km2",
        "description": "Szacowana powierzchnia lasów w powiecie.",
    },
    "pine_share_pct": {
        "type": "float64 (nullable)",
        "unit": "%",
        "source": "forest_features.pine_share_pct",
        "description": "Udział drzewostanów z panującą sosną (SO) wg BDL (NULL przy braku danych BDL).",
    },
    "oak_beech_share_pct": {
        "type": "float64 (nullable)",
        "unit": "%",
        "source": "forest_features.oak_beech_share_pct",
        "description": "Udział drzewostanów z panującym dębem (DB) lub bukiem (BK) wg BDL (NULL przy braku BDL).",
    },
    "spruce_share_pct": {
        "type": "float64 (nullable)",
        "unit": "%",
        "source": "forest_features.spruce_share_pct",
        "description": "Udział drzewostanów z panującym świerkiem (ŚW) wg BDL (NULL przy braku BDL).",
    },
    "mean_stand_age_years": {
        "type": "float64 (nullable)",
        "unit": "lata",
        "source": "forest_features.mean_stand_age_years",
        "description": "Średni ważony powierzchniowo wiek drzewostanów D-STAN w BDL (NULL przy braku BDL).",
    },
    "dominant_moisture_type": {
        "type": "string (nullable)",
        "unit": "kategoria BDL",
        "source": "forest_features.dominant_moisture_type",
        "description": "Dominujące uwilgotnienie siedliska (suche, swieze, wilgotne, bagienne lub NULL).",
    },
    "has_bdl_stand_data": {
        "type": "bool",
        "unit": "flaga",
        "source": "forest_features.has_bdl_stand_data",
        "description": "Czy powiat posiada szczegółowy profil taksacyjny drzewostanów BDL.",
    },
    "spatial_precision": {
        "type": "string",
        "unit": "enum",
        "source": "reports_spatial.spatial_precision",
        "description": "Dokładność lokalizacji raportu (point_precise vs county_aggregate).",
    },
    "point_represents_specific_forest": {
        "type": "bool",
        "unit": "flaga",
        "source": "reports_spatial.point_represents_specific_forest",
        "description": "False dla zgłoszeń ogólnopowiatowych (ochrona przed pozorną dokładnością).",
    },
    "distance_to_county_centroid_km": {
        "type": "float64 (nullable)",
        "unit": "km",
        "source": "reports_spatial.distance_to_county_centroid_km",
        "description": "Odległość punktu zgłoszenia od środka ciężkości powiatu w EPSG:2180.",
    },
    "month": {
        "type": "int64",
        "unit": "miesiąc (1-12)",
        "source": "wyliczona z target_date",
        "description": "Miesiąc kalendarzowy daty docelowej (np. 9, 10, 11).",
    },
    "day_of_year": {
        "type": "int64",
        "unit": "dzień roku (1-366)",
        "source": "wyliczona z target_date",
        "description": "Kolejny dzień w roku kalendarzowym (DOY).",
    },
    "week_of_year": {
        "type": "int64",
        "unit": "tydzień ISO (1-53)",
        "source": "wyliczona z target_date",
        "description": "Numer tygodnia roku wg standardu ISO 8601.",
    },
    "day_of_week": {
        "type": "int64",
        "unit": "dzień tyg. (0=Pon..6=Nd)",
        "source": "wyliczona z target_date",
        "description": "Dzień tygodnia dla daty docelowej.",
    },
    "is_weekend": {
        "type": "bool",
        "unit": "flaga",
        "source": "wyliczona z target_date",
        "description": "True dla soboty i niedzieli (uwzględnia efekt weekendowej aktywności grzybiarzy).",
    },
    "day_of_year_sin": {
        "type": "float64",
        "unit": "[-1, 1]",
        "source": "wyliczona z target_date",
        "description": "Cykliczna transformacja sinus dnia roku sin(2*pi*DOY/365.25).",
    },
    "day_of_year_cos": {
        "type": "float64",
        "unit": "[-1, 1]",
        "source": "wyliczona z target_date",
        "description": "Cykliczna transformacja cosinus dnia roku cos(2*pi*DOY/365.25).",
    },
    "season_phase": {
        "type": "string",
        "unit": "enum",
        "source": "wyliczona z target_date",
        "description": "Faza sezonu jesiennego (early_autumn_sep, peak_autumn_oct_early, late_autumn_oct_late_nov).",
    },
    "split_2way": {
        "type": "string",
        "unit": "enum (train, test)",
        "source": "protokół [GRZ-13]",
        "description": "Przynależność do podziału głównego Train (<=2025-10-12) / Test (>2025-10-12).",
    },
    "split_3way": {
        "type": "string",
        "unit": "enum (train, val, test)",
        "source": "protokół [GRZ-13]",
        "description": "Przynależność do podziału 3-częściowego Train / Val / Test.",
    },
    "source_url": {
        "type": "string",
        "unit": "URL",
        "source": "reports_spatial.source_url",
        "description": "Adres źródłowy raportu na portalu grzyby.pl.",
    },
    "is_valid_for_training": {
        "type": "bool",
        "unit": "flaga",
        "source": "reports_spatial.is_valid_for_training",
        "description": "Flaga ważności obserwacji do treningu i ewaluacji.",
    },
    "is_demo": {
        "type": "bool",
        "unit": "flaga",
        "source": "stała False",
        "description": "Flaga bezpieczeństwa odróżniająca dane produkcyjne od próbki UI.",
    },
}


def _clean_nullable_float(val: Any) -> float | None:
    """Zwraca float lub None, nigdy nie zamieniając None/NaN w 0.0."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in {"nan", "null", "none"}:
            return None
        return float(s)
    return float(val)


def _clean_nullable_str(val: Any) -> str | None:
    """Zwraca niepusty string lub None."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "null", "none"}:
        return None
    return s


def compute_seasonality_features(target_date_str: str) -> dict[str, Any]:
    """Wylicza deterministyczne cechy kalendarzowe i sezonowe dla daty docelowej."""
    dt = date.fromisoformat(target_date_str)
    doy = dt.timetuple().tm_yday
    iso_week = dt.isocalendar().week
    dow = dt.weekday()  # 0 = poniedziałek .. 6 = niedziela
    angle = 2.0 * math.pi * float(doy) / 365.25

    if dt.month <= 9:
        phase = "early_autumn_sep"
    elif dt.month == 10 and dt.day <= 15:
        phase = "peak_autumn_oct_early"
    else:
        phase = "late_autumn_oct_late_nov"

    return {
        "month": int(dt.month),
        "day_of_year": int(doy),
        "week_of_year": int(iso_week),
        "day_of_week": int(dow),
        "is_weekend": bool(dow in {5, 6}),
        "day_of_year_sin": round(math.sin(angle), 6),
        "day_of_year_cos": round(math.cos(angle), 6),
        "season_phase": phase,
    }


def index_weather_by_region_and_date(
    weather_df: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Buduje słownik szybkiego dostępu `(region_id, date) -> rekord pogodowy`,
    weryfikując unikalność klucza `(region_id, date)`.
    """
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    region_col = "region_id" if "region_id" in weather_df.columns else "area_id"

    for _, row in weather_df.iterrows():
        reg_id = str(row[region_col]).strip().zfill(4)
        d_str = str(row["date"]).strip()
        key = (reg_id, d_str)
        if key in lookup:
            raise ValueError(f"Duplikat klucza pogodowego (region_id, date): {key}")

        lookup[key] = {
            "date": d_str,
            "region_id": reg_id,
            "precip_mm_1d": _clean_nullable_float(row.get("precip_mm_1d")),
            "precipitation_hours": _clean_nullable_float(row.get("precipitation_hours")),
            "temp_min_night_c": _clean_nullable_float(row.get("temp_min_night_c")),
            "temp_mean_day_c": _clean_nullable_float(
                row.get("temp_mean_day_c")
                if "temp_mean_day_c" in row
                else row.get("temp_mean_2m_c")
            ),
            "temp_max_day_c": _clean_nullable_float(
                row.get("temp_max_day_c")
                if "temp_max_day_c" in row
                else row.get("temp_max_2m_c")
            ),
            "wind_speed_max_kmh": _clean_nullable_float(row.get("wind_speed_max_kmh")),
            "soil_moisture_0_7cm_m3m3": _clean_nullable_float(
                row.get("soil_moisture_0_7cm_m3m3")
            ),
            "soil_temperature_0_7cm_c": _clean_nullable_float(
                row.get("soil_temperature_0_7cm_c")
                if "soil_temperature_0_7cm_c" in row
                else row.get("soil_temp_0_7cm_mean_c")
            ),
            "grid_elevation_m": _clean_nullable_float(row.get("grid_elevation_m")),
        }
    return lookup


def compute_point_in_time_weather_features(
    region_id: str,
    target_date_str: str,
    weather_lookup: dict[tuple[str, str], dict[str, Any]],
    horizon_days: int = 1,
) -> dict[str, Any]:
    """
    Wylicza wszystkie okna opadów (7/14/21 dni) oraz cechy temperatury i gleby
    dla zadanego obszaru `region_id`, daty docelowej `target_date_str` i horyzontu `horizon_days`.

    Zasada odcięcia informacji (Point-in-Time Cutoff):
    - `issued_at = target_date - horizon_days`
    - `feature_cutoff_date = issued_at`
    - Żaden dzień po `feature_cutoff_date` (w tym `target_date` dla `horizon_days >= 1`)
      nie jest odczytywany ze słownika `weather_lookup`.
    - Jeżeli w oknie `k` dni (`[feature_cutoff_date - (k-1), feature_cutoff_date]`) brakuje
      choćby jednego dnia lub wartość pomiaru wynosi `None`, funkcja zwraca dla tego okna `None`
      (nigdy nie podstawia `0.0` za brak danych!).
    """
    if horizon_days < 1:
        raise ValueError("horizon_days musi wynosić co najmniej 1 dzień, aby uniknąć wycieku z dnia zbioru.")

    reg_id = str(region_id).strip().zfill(4)
    t_target = date.fromisoformat(target_date_str)
    t_issued = t_target - timedelta(days=horizon_days)
    t_cutoff = t_issued  # Ostatnia pełna doba znana w momencie wystawienia prognozy

    def _window_dates(k_days: int) -> list[str]:
        start_d = t_cutoff - timedelta(days=k_days - 1)
        return [(start_d + timedelta(days=offset)).isoformat() for offset in range(k_days)]

    dates_3d = _window_dates(3)
    dates_7d = _window_dates(7)
    dates_14d = _window_dates(14)
    dates_21d = _window_dates(21)

    # Upewnienie się, że żaden dzień w żadnym oknie nie wykracza poza feature_cutoff_date
    cutoff_iso = t_cutoff.isoformat()
    for d_check in dates_21d:
        if d_check > cutoff_iso:
            raise AssertionError(
                f"Wyciek z przyszłości: dzień okna {d_check} > feature_cutoff_date {cutoff_iso}"
            )

    def _extract_complete_series(date_list: list[str], field: str) -> list[float] | None:
        vals: list[float] = []
        for d_iso in date_list:
            rec = weather_lookup.get((reg_id, d_iso))
            if rec is None:
                return None
            v = rec.get(field)
            if v is None or pd.isna(v):
                return None
            vals.append(float(v))
        return vals

    cutoff_rec = weather_lookup.get((reg_id, cutoff_iso))

    # 1. Okna opadów 7 / 14 / 21 dni (zamknięte przedziały kończące się na feature_cutoff_date)
    p_7d_vals = _extract_complete_series(dates_7d, "precip_mm_1d")
    p_14d_vals = _extract_complete_series(dates_14d, "precip_mm_1d")
    p_21d_vals = _extract_complete_series(dates_21d, "precip_mm_1d")
    ph_7d_vals = _extract_complete_series(dates_7d, "precipitation_hours")

    precip_lag_7d = round(sum(p_7d_vals), 1) if p_7d_vals is not None else None
    precip_lag_14d = round(sum(p_14d_vals), 1) if p_14d_vals is not None else None
    precip_lag_21d = round(sum(p_21d_vals), 1) if p_21d_vals is not None else None
    precip_hours_7d = round(sum(ph_7d_vals), 1) if ph_7d_vals is not None else None

    rainy_days_7d = (
        int(sum(1 for x in p_7d_vals if x >= 1.0)) if p_7d_vals is not None else None
    )
    rainy_days_14d = (
        int(sum(1 for x in p_14d_vals if x >= 1.0)) if p_14d_vals is not None else None
    )

    # 2. Cechy temperaturowe i przymrozkowe
    tmin_3d_vals = _extract_complete_series(dates_3d, "temp_min_night_c")
    tmin_7d_vals = _extract_complete_series(dates_7d, "temp_min_night_c")
    tmean_7d_vals = _extract_complete_series(dates_7d, "temp_mean_day_c")
    tmean_14d_vals = _extract_complete_series(dates_14d, "temp_mean_day_c")
    wind_7d_vals = _extract_complete_series(dates_7d, "wind_speed_max_kmh")

    temp_min_lag_3d = round(min(tmin_3d_vals), 1) if tmin_3d_vals is not None else None
    temp_min_lag_7d = round(min(tmin_7d_vals), 1) if tmin_7d_vals is not None else None
    frost_nights_7d = (
        int(sum(1 for x in tmin_7d_vals if x <= 0.0)) if tmin_7d_vals is not None else None
    )
    temp_mean_7d = (
        round(sum(tmean_7d_vals) / len(tmean_7d_vals), 1)
        if tmean_7d_vals is not None
        else None
    )
    temp_mean_14d = (
        round(sum(tmean_14d_vals) / len(tmean_14d_vals), 1)
        if tmean_14d_vals is not None
        else None
    )
    wind_mean_7d = (
        round(sum(wind_7d_vals) / len(wind_7d_vals), 1)
        if wind_7d_vals is not None
        else None
    )

    # 3. Cechy glebowe (ERA5-Land 0-7 cm)
    sm_7d_vals = _extract_complete_series(dates_7d, "soil_moisture_0_7cm_m3m3")
    sm_14d_vals = _extract_complete_series(dates_14d, "soil_moisture_0_7cm_m3m3")
    st_7d_vals = _extract_complete_series(dates_7d, "soil_temperature_0_7cm_c")

    soil_moist_mean_7d = (
        round(sum(sm_7d_vals) / len(sm_7d_vals), 4) if sm_7d_vals is not None else None
    )
    soil_moist_mean_14d = (
        round(sum(sm_14d_vals) / len(sm_14d_vals), 4) if sm_14d_vals is not None else None
    )
    soil_temp_mean_7d = (
        round(sum(st_7d_vals) / len(st_7d_vals), 2) if st_7d_vals is not None else None
    )

    return {
        "target_date": target_date_str,
        "issued_at": t_issued.isoformat(),
        "feature_cutoff_date": cutoff_iso,
        "horizon_days": int(horizon_days),
        "window_7d_start_date": dates_7d[0],
        "window_14d_start_date": dates_14d[0],
        "window_21d_start_date": dates_21d[0],
        "window_end_date": cutoff_iso,
        "precip_lag_7d_mm": precip_lag_7d,
        "precip_lag_14d_mm": precip_lag_14d,
        "precip_lag_21d_mm": precip_lag_21d,
        "precip_cutoff_1d_mm": (
            cutoff_rec.get("precip_mm_1d") if cutoff_rec is not None else None
        ),
        "precip_hours_lag_7d_h": precip_hours_7d,
        "rainy_days_lag_7d_count": rainy_days_7d,
        "rainy_days_lag_14d_count": rainy_days_14d,
        "temp_min_night_c": (
            cutoff_rec.get("temp_min_night_c") if cutoff_rec is not None else None
        ),
        "temp_min_lag_3d_c": temp_min_lag_3d,
        "temp_min_lag_7d_c": temp_min_lag_7d,
        "frost_nights_lag_7d_count": frost_nights_7d,
        "temp_mean_cutoff_1d_c": (
            cutoff_rec.get("temp_mean_day_c") if cutoff_rec is not None else None
        ),
        "temp_mean_7d_c": temp_mean_7d,
        "temp_mean_14d_c": temp_mean_14d,
        "temp_max_cutoff_1d_c": (
            cutoff_rec.get("temp_max_day_c") if cutoff_rec is not None else None
        ),
        "wind_speed_mean_7d_kmh": wind_mean_7d,
        "soil_moisture_0_7cm_m3m3": (
            cutoff_rec.get("soil_moisture_0_7cm_m3m3") if cutoff_rec is not None else None
        ),
        "soil_moisture_mean_7d_m3m3": soil_moist_mean_7d,
        "soil_moisture_mean_14d_m3m3": soil_moist_mean_14d,
        "soil_temperature_0_7cm_c": (
            cutoff_rec.get("soil_temperature_0_7cm_c") if cutoff_rec is not None else None
        ),
        "soil_temp_mean_7d_c": soil_temp_mean_7d,
        "grid_elevation_m": (
            cutoff_rec.get("grid_elevation_m") if cutoff_rec is not None else None
        ),
    }


def build_features_dataframe(
    reports_spatial_df: pd.DataFrame,
    forest_features_df: pd.DataFrame,
    weather_daily_df: pd.DataFrame,
    horizon_days: int = 1,
    valid_only: bool = True,
) -> pd.DataFrame:
    """
    Łączy obserwacje z `reports_spatial`, cechy leśne z `forest_features` oraz
    szeregi pogodowe z `weather_daily` w jedną tabelę analityczną `features`.

    Gwarantuje:
    - 1:1 brak multiplikacji rekordów (każdy wiersz odpowiada unikalnemu `report_id`),
    - brak przecieku danych pogodowych po `feature_cutoff_date`,
    - brak automatycznego zamieniania `NULL` w `0.0`.
    """
    if valid_only and "is_valid_for_training" in reports_spatial_df.columns:
        obs_df = reports_spatial_df[reports_spatial_df["is_valid_for_training"]].copy()
    else:
        obs_df = reports_spatial_df.copy()

    # Weryfikacja unikalności report_id na wejściu
    if obs_df["report_id"].nunique() != len(obs_df):
        raise ValueError("Wejściowa tabela raportów zawiera zduplikowane identyfikatory report_id!")

    # Słownik cech leśnych po kluczu głównym region_id (1 wiersz na powiat)
    if forest_features_df["region_id"].nunique() != len(forest_features_df):
        raise ValueError("Tabela forest_features zawiera zduplikowane wartości region_id!")

    forest_lookup: dict[str, dict[str, Any]] = {}
    for _, f_row in forest_features_df.iterrows():
        reg_id = str(f_row["region_id"]).strip().zfill(4)
        forest_lookup[reg_id] = {
            "region_name": str(f_row["region_name"]),
            "total_area_km2": float(f_row["total_area_km2"]),
            "forest_cover_pct": _clean_nullable_float(f_row.get("forest_cover_pct")),
            "forest_area_km2": _clean_nullable_float(f_row.get("forest_area_km2")),
            "pine_share_pct": _clean_nullable_float(f_row.get("pine_share_pct")),
            "oak_beech_share_pct": _clean_nullable_float(f_row.get("oak_beech_share_pct")),
            "spruce_share_pct": _clean_nullable_float(f_row.get("spruce_share_pct")),
            "mean_stand_age_years": _clean_nullable_float(f_row.get("mean_stand_age_years")),
            "dominant_moisture_type": _clean_nullable_str(f_row.get("dominant_moisture_type")),
            "has_bdl_stand_data": bool(f_row.get("has_bdl_stand_data", False)),
        }

    weather_lookup = index_weather_by_region_and_date(weather_daily_df)

    feature_rows: list[dict[str, Any]] = []

    for _, row in obs_df.iterrows():
        rid = str(row["report_id"]).strip()
        reg_id = str(row.get("region_id") or row.get("area_id") or "").strip().zfill(4)
        obs_date = str(row.get("harvest_date") or row.get("date")).strip()
        pub_date = str(row.get("publication_date") or obs_date).strip()
        target_date = obs_date

        if reg_id not in forest_lookup:
            raise KeyError(f"Brak powiatu {reg_id} w tabeli forest_features dla raportu {rid}")

        f_meta = forest_lookup[reg_id]
        w_feats = compute_point_in_time_weather_features(
            region_id=reg_id,
            target_date_str=target_date,
            weather_lookup=weather_lookup,
            horizon_days=horizon_days,
        )
        s_feats = compute_seasonality_features(target_date)

        yield_val = _clean_nullable_float(row.get("yield_pcs_per_h"))
        if yield_val is None:
            raise ValueError(f"Brak wartości etykiety yield_pcs_per_h dla raportu {rid}")

        abundance = classify_abundance(yield_val)

        # Oznaczenie podziału czasowego zgodnie z protokołem [GRZ-13]
        split_2way = "train" if target_date <= DEFAULT_SPLIT_DATE_TRAIN_TEST else "test"
        if target_date <= DEFAULT_SPLIT_DATE_TRAIN_VAL:
            split_3way = "train"
        elif target_date <= DEFAULT_SPLIT_DATE_VAL_TEST:
            split_3way = "val"
        else:
            split_3way = "test"

        feature_rows.append(
            {
                "observation_id": rid,
                "report_id": rid,
                "region_id": reg_id,
                "area_id": reg_id,
                "region_name": f_meta["region_name"],
                "place_id": str(row.get("place_id") or "").strip(),
                "observation_date": obs_date,
                "harvest_date": obs_date,
                "date": obs_date,
                "publication_date": pub_date,
                "target_date": w_feats["target_date"],
                "issued_at": w_feats["issued_at"],
                "feature_cutoff_date": w_feats["feature_cutoff_date"],
                "horizon_days": w_feats["horizon_days"],
                "window_7d_start_date": w_feats["window_7d_start_date"],
                "window_14d_start_date": w_feats["window_14d_start_date"],
                "window_21d_start_date": w_feats["window_21d_start_date"],
                "window_end_date": w_feats["window_end_date"],
                "yield_pcs_per_h": yield_val,
                "abundance_class": abundance,
                "dominant_species": _clean_nullable_str(row.get("dominant_species")),
                "harvest_duration_h": _clean_nullable_float(row.get("harvest_duration_h")),
                "total_pieces": _clean_nullable_float(row.get("total_pieces")),
                "precip_lag_7d_mm": w_feats["precip_lag_7d_mm"],
                "precip_lag_14d_mm": w_feats["precip_lag_14d_mm"],
                "precip_lag_21d_mm": w_feats["precip_lag_21d_mm"],
                "precip_cutoff_1d_mm": w_feats["precip_cutoff_1d_mm"],
                "precip_hours_lag_7d_h": w_feats["precip_hours_lag_7d_h"],
                "rainy_days_lag_7d_count": w_feats["rainy_days_lag_7d_count"],
                "rainy_days_lag_14d_count": w_feats["rainy_days_lag_14d_count"],
                "temp_min_night_c": w_feats["temp_min_night_c"],
                "temp_min_lag_3d_c": w_feats["temp_min_lag_3d_c"],
                "temp_min_lag_7d_c": w_feats["temp_min_lag_7d_c"],
                "frost_nights_lag_7d_count": w_feats["frost_nights_lag_7d_count"],
                "temp_mean_cutoff_1d_c": w_feats["temp_mean_cutoff_1d_c"],
                "temp_mean_7d_c": w_feats["temp_mean_7d_c"],
                "temp_mean_14d_c": w_feats["temp_mean_14d_c"],
                "temp_max_cutoff_1d_c": w_feats["temp_max_cutoff_1d_c"],
                "wind_speed_mean_7d_kmh": w_feats["wind_speed_mean_7d_kmh"],
                "soil_moisture_0_7cm_m3m3": w_feats["soil_moisture_0_7cm_m3m3"],
                "soil_moisture_mean_7d_m3m3": w_feats["soil_moisture_mean_7d_m3m3"],
                "soil_moisture_mean_14d_m3m3": w_feats["soil_moisture_mean_14d_m3m3"],
                "soil_temperature_0_7cm_c": w_feats["soil_temperature_0_7cm_c"],
                "soil_temp_mean_7d_c": w_feats["soil_temp_mean_7d_c"],
                "grid_elevation_m": w_feats["grid_elevation_m"],
                "total_area_km2": f_meta["total_area_km2"],
                "forest_cover_pct": f_meta["forest_cover_pct"],
                "forest_area_km2": f_meta["forest_area_km2"],
                "pine_share_pct": f_meta["pine_share_pct"],
                "oak_beech_share_pct": f_meta["oak_beech_share_pct"],
                "spruce_share_pct": f_meta["spruce_share_pct"],
                "mean_stand_age_years": f_meta["mean_stand_age_years"],
                "dominant_moisture_type": f_meta["dominant_moisture_type"],
                "has_bdl_stand_data": f_meta["has_bdl_stand_data"],
                "spatial_precision": str(row.get("spatial_precision") or "point_precise"),
                "point_represents_specific_forest": bool(
                    row.get("point_represents_specific_forest", True)
                ),
                "distance_to_county_centroid_km": _clean_nullable_float(
                    row.get("distance_to_county_centroid_km")
                ),
                "month": s_feats["month"],
                "day_of_year": s_feats["day_of_year"],
                "week_of_year": s_feats["week_of_year"],
                "day_of_week": s_feats["day_of_week"],
                "is_weekend": s_feats["is_weekend"],
                "day_of_year_sin": s_feats["day_of_year_sin"],
                "day_of_year_cos": s_feats["day_of_year_cos"],
                "season_phase": s_feats["season_phase"],
                "split_2way": split_2way,
                "split_3way": split_3way,
                "source_url": str(row.get("source_url") or ""),
                "is_valid_for_training": bool(row.get("is_valid_for_training", True)),
                "is_demo": False,
            }
        )

    out_df = pd.DataFrame(feature_rows)

    # Sortowanie deterministyczne po dacie docelowej, powiecie i identyfikatorze raportu
    out_df = out_df.sort_values(
        by=["target_date", "region_id", "report_id"], ascending=[True, True, True]
    ).reset_index(drop=True)

    # Twarda weryfikacja niezmienników [GRZ-10]
    assert len(out_df) == len(obs_df), (
        f"Naruszenie niezmiennika 1:1 przy łączeniu! Wejście={len(obs_df)}, Wyjście={len(out_df)}"
    )
    assert out_df["report_id"].nunique() == len(out_df), "Wykryto duplikaty report_id po złączeniu!"
    assert (out_df["window_end_date"] <= out_df["feature_cutoff_date"]).all(), (
        "Wykryto okno pogodowe wykraczające poza datę odcięcia feature_cutoff_date!"
    )
    assert (out_df["feature_cutoff_date"] < out_df["target_date"]).all(), (
        "Data odcięcia cech pogodowych musi poprzedzać dzień docelowy zbioru!"
    )

    return out_df
