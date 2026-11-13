"""
Funkcje pomocnicze dla obliczeń przestrzennych i przygotowania warstw GeoJSON
dla interfejsu Streamlit ([GRZ-02], [GRZ-05], [GRZ-12]).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pandas as pd

from src.config import ABUNDANCE_CLASSES


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Oblicza odległość geodezyjną (w linii prostej, w kilometrach) pomiędzy dwoma
    punktami (lat, lon) w układzie WGS84 (EPSG:4326) przy użyciu wzoru Haversine.
    """
    earth_radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(earth_radius_km * c, 1)


def _forest_cover_rgba(forest_cover_pct: float | None, has_bdl_data: bool) -> list[int]:
    """
    Zwraca kolor RGBA dla widoku granic regionu i lesistości ([GRZ-05]).
    Brak danych BDL (None) nie blokuje narysowania powiatu na mapie.
    """
    if forest_cover_pct is None:
        return [185, 196, 187, 150]
    pct = float(forest_cover_pct)
    alpha = 205 if has_bdl_data else 155
    if pct >= 45.0:
        return [34, 92, 52, alpha]
    if pct >= 30.0:
        return [68, 132, 82, alpha]
    if pct >= 18.0:
        return [122, 168, 116, alpha]
    return [178, 198, 165, alpha]


def _coverage_count_rgba(reports_count: int) -> tuple[list[int], str]:
    """
    Zwraca kolor RGBA oraz etykietę dla trybu pokrycia danymi (liczby raportów w powiecie, [GRZ-12]).
    Powiat z 0 raportami otrzymuje neutralną szarość (BRAK DANYCH — NULL).
    """
    if reports_count <= 0:
        return [180, 180, 180, 150], "Brak danych (0 raportów — NULL)"
    if reports_count >= 20:
        return [29, 78, 137, 215], f"Wysokie pokrycie ({reports_count} rap., ≥ 20)"
    if reports_count >= 5:
        return [58, 124, 165, 205], f"Średnie pokrycie ({reports_count} rap., 5–19)"
    return [129, 178, 205, 195], f"Niskie pokrycie ({reports_count} rap., 1–4)"


def enrich_geojson_with_real_observations(
    geojson_data: dict[str, Any],
    county_by_id: dict[str, dict[str, Any]],
    color_mode: str = "observed_yield",
) -> dict[str, Any]:
    """
    Wzbogaca kopię słownika GeoJSON 30 powiatów woj. dolnośląskiego o zagregowane
    statystyki rzeczywistych raportów z grzyby.pl ([GRZ-12]).

    Kluczowa zasada semantyczna ([GRZ-06], [GRZ-10], [GRZ-12]):
    - Brak raportów w powiecie (`reports_count == 0`) jest oznaczany jednoznacznie jako
      `Brak danych (0 raportów — NULL)` i neutralny szary kolor `[180, 180, 180, 150]`.
    - Nigdy nie jest prezentowany jako `0.0 szt./h`.
    """
    enriched = copy.deepcopy(geojson_data)
    for feature in enriched.get("features", []):
        props = feature.setdefault("properties", {})
        area_id = str(props.get("area_id", "")).zfill(4)
        props["area_id"] = area_id
        area_name = str(props.get("area_name") or area_id)

        rec = county_by_id.get(area_id, {})
        in_radius = bool(rec.get("in_radius", True))
        reports_count = int(rec.get("reports_count") or 0)
        has_reports = bool(rec.get("has_reports", reports_count > 0))
        status_code = str(rec.get("status_code") or "BRAK_DANYCH")

        total_area = props.get("total_area_km2")
        props["total_area_km2_label"] = (
            f"{float(total_area):.1f} km²" if total_area is not None else "—"
        )
        fc_val = props.get("forest_cover_pct")
        props["forest_cover_pct_label"] = (
            f"{float(fc_val):.1f}%" if fc_val is not None else "NULL (brak)"
        )
        for field_key in ("pine_share_pct", "oak_beech_share_pct", "spruce_share_pct"):
            val = props.get(field_key)
            props[f"{field_key}_label"] = (
                f"{float(val):.1f}%" if val is not None else "NULL (brak BDL)"
            )
        age_val = props.get("mean_stand_age_years")
        props["mean_stand_age_label"] = (
            f"{float(age_val):.1f} lat" if age_val is not None else "NULL (brak BDL)"
        )
        moisture = props.get("dominant_moisture_type")
        props["dominant_moisture_label"] = str(moisture) if moisture else "NULL (brak BDL)"

        has_bdl = bool(
            props.get("has_bdl_stand_data")
            or (props.get("pine_share_pct") is not None and props.get("mean_stand_age_years") is not None)
        )

        dist_km = rec.get("distance_km")
        props["distance_label"] = (
            f"{float(dist_km):.1f} km" + (" (poza filtrem)" if not in_radius else "")
            if dist_km is not None
            else "—"
        )

        props["reports_count"] = reports_count
        props["has_reports"] = has_reports
        props["status_code"] = status_code

        if in_radius and has_reports and rec.get("mean_yield_pcs_per_h") is not None:
            mean_y = float(rec["mean_yield_pcs_per_h"])
            med_y = float(rec.get("median_yield_pcs_per_h") or mean_y)
            max_y = float(rec.get("max_yield_pcs_per_h") or mean_y)
            zero_cnt = int(rec.get("zero_harvest_reports_count") or 0)
            color_info = ABUNDANCE_CLASSES.get(status_code, ABUNDANCE_CLASSES["BRAK_DANYCH"])

            props["observed_yield_label"] = (
                f"śr. {mean_y:.1f} szt./h (mediana {med_y:.1f}, maks. {max_y:.1f})"
            )
            props["reports_coverage_label"] = (
                f"{reports_count} raportów (w tym puste koszyki 0/h: {zero_cnt})"
            )
            props["last_report_date_label"] = str(rec.get("last_report_date") or "—")
            yield_fill = color_info["rgba"]
            cov_fill, cov_desc = _coverage_count_rgba(reports_count)
            props["status_label"] = (
                str(color_info["label"]) if color_mode != "report_coverage" else cov_desc
            )
            line_col = [30, 50, 35, 230]
        else:
            color_info = ABUNDANCE_CLASSES["BRAK_DANYCH"]
            if not in_radius:
                props["observed_yield_label"] = "poza aktywnym filtrem przestrzennym"
                props["reports_coverage_label"] = "poza aktywnym filtrem przestrzennym"
                props["status_label"] = "Poza aktywnym filtrem (promień / powiat)"
            else:
                props["observed_yield_label"] = "BRAK DANYCH (NULL — nie 0.0 szt./h)"
                props["reports_coverage_label"] = "0 raportów w wybranym okresie / filtrze"
                props["status_label"] = str(color_info["label"])
            props["last_report_date_label"] = "brak obserwacji"
            yield_fill = [180, 180, 180, 150] if in_radius else [165, 165, 160, 85]
            cov_fill = [180, 180, 180, 150] if in_radius else [165, 165, 160, 85]
            line_col = [90, 105, 95, 195] if in_radius else [125, 130, 125, 140]

        if color_mode == "report_coverage":
            props["fill_color"] = cov_fill
            props["line_color"] = line_col
        elif color_mode == "region_bdl":
            props["fill_color"] = (
                _forest_cover_rgba(fc_val, has_bdl) if in_radius else [165, 165, 160, 95]
            )
            props["line_color"] = [25, 55, 35, 235] if in_radius else [110, 115, 110, 170]
        else:
            props["fill_color"] = yield_fill
            props["line_color"] = line_col

        precip_14d = rec.get("precip_lag_14d_mm")
        soil_m = rec.get("soil_moisture_0_7cm_m3m3")
        weather_line = (
            f"Opad 14d (ERA5-Land): <b>{float(precip_14d):.1f} mm</b> | "
            f"Wilgotność gleby 0–7 cm: <b>{float(soil_m):.3f} m³/m³</b>"
            if precip_14d is not None and soil_m is not None
            else "Pogoda ERA5-Land: brak agregatu dla okresu"
        )

        props["tooltip_header"] = f"{area_name} (TERYT: {area_id})"
        props["tooltip_sub"] = (
            f"Jednostka przestrzenna: <b>powiat ({props['total_area_km2_label']})</b> | "
            f"Odległość: <b>{props['distance_label']}</b><br/>"
            f"Liczba raportów: <b>{props['reports_coverage_label']}</b><br/>"
            f"Obserwowany zbiór: <b>{props['observed_yield_label']}</b><br/>"
            f"Klasyfikacja / status: <b>{props['status_label']}</b> | "
            f"Ostatni raport: <b>{props['last_report_date_label']}</b>"
        )
        props["tooltip_bdl"] = (
            f"{weather_line}<br/>"
            f"Lesistość BDL: <b>{props['forest_cover_pct_label']}</b> | "
            f"Sosna: <b>{props['pine_share_pct_label']}</b> | "
            f"Dąb/Buk: <b>{props['oak_beech_share_pct_label']}</b> | "
            f"Świerk: <b>{props['spruce_share_pct_label']}</b><br/>"
            f"Wiek drzewostanu: <b>{props['mean_stand_age_label']}</b> | "
            f"Uwilgotnienie BDL: <b>{props['dominant_moisture_label']}</b>"
        )

    return enriched


def build_precise_report_points_layer_data(
    filtered_reports_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Przygotowuje listę punktów do warstwy ScatterplotLayer wyłącznie dla raportów
    posiadających potwierdzoną lokalizację miejscowości / lasu
    (`point_represents_specific_forest == True`), aby zgodnie z kontraktem [GRZ-06]/[GRZ-08]
    nie sugerować pozornej dokładności dla wpisów zagregowanych do centroidu powiatu.
    """
    if filtered_reports_df.empty:
        return []

    points: list[dict[str, Any]] = []
    for _, row in filtered_reports_df.iterrows():
        if not bool(row.get("point_represents_specific_forest", False)):
            continue
        lat = row.get("lat")
        lon = row.get("lon")
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            continue
        y_val = float(row.get("yield_pcs_per_h") or 0.0)
        ab_cls = str(row.get("abundance_class") or "BRAK")
        color_info = ABUNDANCE_CLASSES.get(ab_cls, ABUNDANCE_CLASSES["BRAK_DANYCH"])
        sp_raw = row.get("dominant_species")
        sp_label = (
            "NULL (brak deklaracji)"
            if sp_raw is None or pd.isna(sp_raw)
            else str(sp_raw)
        )
        loc_name = row.get("location_raw") or row.get("place_id") or "—"
        obs_date = row.get("observation_date") or row.get("harvest_date") or "—"
        points.append(
            {
                "report_id": str(row.get("report_id", "")),
                "lat": float(lat),
                "lon": float(lon),
                "fill_color": color_info["rgba"],
                "tooltip_header": f"Raport {row.get('report_id', '')} — {loc_name}",
                "tooltip_sub": (
                    f"Powiat: <b>{row.get('region_name', '—')} ({row.get('region_id', '—')})</b> | "
                    f"Data zbioru: <b>{obs_date}</b><br/>"
                    f"Wydajność: <b>{y_val:.1f} szt./h ({ab_cls})</b> | "
                    f"Dominujący gatunek: <b>{sp_label}</b>"
                ),
                "tooltip_bdl": (
                    f"Precyzja geo: <b>{row.get('spatial_precision', 'point_precise')}</b> | "
                    f"Odcięcie cech: <b>{row.get('feature_cutoff_date', '—')}</b><br/>"
                    f"Źródło: <code>{row.get('source_url', 'grzyby.pl')}</code>"
                ),
            }
        )
    return points


def enrich_geojson_with_predictions(
    geojson_data: dict[str, Any],
    records_by_area: dict[str, dict[str, Any]],
    color_mode: str = "predictions",
) -> dict[str, Any]:
    """
    Dołącza atrybuty predykcji, pogody, odległości, cech leśnych BDL oraz kolor RGBA
    do każdego obiektu Feature w kopii słownika GeoJSON na potrzeby renderowania w PyDeck.
    Obsługuje bezpiecznie brak atrybutów BDL (NULL -> czytelna etykieta tekstowa).
    """
    enriched = copy.deepcopy(geojson_data)
    for feature in enriched.get("features", []):
        props = feature.setdefault("properties", {})
        area_id = str(props.get("area_id", "")).zfill(4)
        props["area_id"] = area_id
        area_name = str(props.get("area_name") or area_id)

        rec = records_by_area.get(area_id)
        in_radius = bool(rec.get("in_radius", True)) if rec else True

        total_area = props.get("total_area_km2")
        props["total_area_km2_label"] = (
            f"{float(total_area):.1f} km²" if total_area is not None else "—"
        )

        fc_val = props.get("forest_cover_pct")
        props["forest_cover_pct_label"] = (
            f"{float(fc_val):.1f}%" if fc_val is not None else "NULL (brak)"
        )

        for field_key in ("pine_share_pct", "oak_beech_share_pct", "spruce_share_pct"):
            val = props.get(field_key)
            props[f"{field_key}_label"] = (
                f"{float(val):.1f}%" if val is not None else "NULL (brak BDL)"
            )

        age_val = props.get("mean_stand_age_years")
        props["mean_stand_age_label"] = (
            f"{float(age_val):.1f} lat" if age_val is not None else "NULL (brak BDL)"
        )

        moisture = props.get("dominant_moisture_type")
        props["dominant_moisture_label"] = str(moisture) if moisture else "NULL (brak BDL)"

        has_bdl = bool(
            props.get("has_bdl_stand_data")
            or (props.get("pine_share_pct") is not None and props.get("mean_stand_age_years") is not None)
        )
        bdl_stands_cnt = int(props.get("bdl_sample_stands_count") or 0)
        props["bdl_status_label"] = (
            f"Profil BDL + próbka wektorowa ({bdl_stands_cnt} wydzieleń)"
            if bdl_stands_cnt > 0
            else ("Profil drzewostanowy BDL" if has_bdl else "Tylko granice PRG (brak szczegółów BDL)")
        )

        if rec and in_radius and rec.get("predicted_yield_pcs_per_h") is not None:
            abundance = str(rec.get("abundance_class") or "BRAK_DANYCH")
            color_info = ABUNDANCE_CLASSES.get(abundance, ABUNDANCE_CLASSES["BRAK_DANYCH"])
            props["predicted_yield_label"] = f"{rec['predicted_yield_pcs_per_h']:.1f} szt./h"
            props["abundance_label"] = str(color_info["label"])
            props["confidence_score"] = rec.get("confidence_score")
            props["confidence_label"] = (
                f"{rec['confidence_score']:.2f}"
                if rec.get("confidence_score") is not None
                else "brak danych"
            )
            props["distance_label"] = (
                f"{rec['distance_km']:.1f} km"
                if rec.get("distance_km") is not None
                else "—"
            )
            props["habitat_match_note"] = (
                rec.get("habitat_match_note")
                or "Brak dodatkowego opisu dopasowania siedliska"
            )
            pred_fill = color_info["rgba"]
            pred_line = [30, 50, 35, 230]
        else:
            color_info = ABUNDANCE_CLASSES["BRAK_DANYCH"]
            props["predicted_yield_label"] = (
                "poza promieniem" if (rec and not in_radius) else "brak prognozy (granice regionu)"
            )
            props["abundance_label"] = str(color_info["label"])
            props["confidence_label"] = "—"
            props["distance_label"] = (
                f"{rec['distance_km']:.1f} km"
                + (" (poza promieniem)" if not in_radius else "")
                if rec and rec.get("distance_km") is not None
                else "—"
            )
            props["habitat_match_note"] = (
                f"Powierzchnia powiatu: {props['total_area_km2_label']} | Status BDL: {props['bdl_status_label']}"
            )
            pred_fill = color_info["rgba"]
            pred_line = [90, 105, 95, 195]

        if color_mode == "region_bdl":
            props["fill_color"] = (
                _forest_cover_rgba(fc_val, has_bdl) if in_radius else [165, 165, 160, 95]
            )
            props["line_color"] = [25, 55, 35, 235] if in_radius else [110, 115, 110, 170]
        else:
            props["fill_color"] = pred_fill
            props["line_color"] = pred_line

        props["tooltip_header"] = f"{area_name} (TERYT: {area_id})"
        props["tooltip_sub"] = (
            f"Powierzchnia: {props['total_area_km2_label']} | Odległość: {props['distance_label']}<br/>"
            f"Prognoza: <b>{props['predicted_yield_label']}</b> ({props['abundance_label']}) | "
            f"Dopasowanie: <b>{props['confidence_label']}</b>"
        )
        props["tooltip_bdl"] = (
            f"Lesistość: <b>{props['forest_cover_pct_label']}</b> | "
            f"Sosna: <b>{props['pine_share_pct_label']}</b> | "
            f"Dąb/Buk: <b>{props['oak_beech_share_pct_label']}</b> | "
            f"Świerk: <b>{props['spruce_share_pct_label']}</b><br/>"
            f"Wiek drzewostanu: <b>{props['mean_stand_age_label']}</b> | "
            f"Uwilgotnienie BDL: <b>{props['dominant_moisture_label']}</b><br/>"
            f"<i>{props['habitat_match_note']}</i>"
        )

    return enriched


def enrich_bdl_stands_geojson(bdl_geojson: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Przygotowuje wektorową warstwę wydzieleń leśnych BDL (`bdl_stands_sample_ds.geojson`)
    do wyświetlenia na mapie PyDeck z ujednoliconymi polami tooltipa.
    Zwraca None, jeśli próbka BDL nie jest dostępna.
    """
    if not bdl_geojson or not bdl_geojson.get("features"):
        return None

    enriched = copy.deepcopy(bdl_geojson)
    for feature in enriched.get("features", []):
        props = feature.setdefault("properties", {})
        adr = props.get("adress_forest", "—")
        insp = props.get("inspectorate_name", "—")
        area_type = props.get("area_type_cd", "—")
        is_stand = bool(props.get("is_forest_stand"))
        tsl_cd = props.get("site_type_cd") or "NULL (grunt nieleśny)"
        tsl_name = props.get("site_type_name") or "—"
        sp_cd = props.get("species_cd_d") or "NULL"
        sp_name = props.get("species_name_pl") or "brak (użytek nieleśny)"
        part = props.get("part_cd")
        part_str = f"{part}/10" if part is not None else "—"
        age = props.get("species_age")
        age_str = f"{age} lat" if age is not None else "NULL"
        sub_ha = props.get("sub_area_ha", 0.0)
        geom_ha = props.get("geom_area_ha_epsg2180", 0.0)
        moist = props.get("moisture_category") or "NULL"

        props.setdefault("fill_color", [46, 139, 87, 195] if is_stand else [140, 140, 135, 130])
        props["line_color"] = [20, 40, 25, 240]
        props["tooltip_header"] = f"Wydzielenie BDL: {adr} ({insp})"
        props["tooltip_sub"] = (
            f"Powiat: <b>{props.get('area_name', '—')} ({props.get('area_id', '—')})</b> | "
            f"ID relacyjny SILP: <code>{props.get('arodes_int_num', '—')}</code><br/>"
            f"Typ powierzchni: <b>{area_type}</b> | "
            f"Pow. taksacyjna: <b>{sub_ha:.2f} ha</b> (geom. EPSG:2180: <b>{geom_ha:.2f} ha</b>)"
        )
        props["tooltip_bdl"] = (
            f"Gatunek panujący: <b>{sp_cd} – {sp_name}</b> (udział: <b>{part_str}</b>, wiek: <b>{age_str}</b>)<br/>"
            f"Typ siedliskowy lasu (TSL): <b>{tsl_cd} ({tsl_name})</b> | Uwilgotnienie: <b>{moist}</b><br/>"
            f"<i>Źródło: BDL Lasy Państwowe (a_year={props.get('a_year', 2026)}, CRS: EPSG:4326)</i>"
        )

    return enriched
