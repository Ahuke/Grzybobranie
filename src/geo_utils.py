"""
Funkcje pomocnicze dla obliczeń przestrzennych i przygotowania warstw GeoJSON
dla interfejsu Streamlit ([GRZ-02], [GRZ-05]).
"""

from __future__ import annotations

import copy
import math
from typing import Any

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

        # Formatowanie powierzchni i cech leśnych z pełnym poszanowaniem NULL (brak BDL nie blokuje mapy)
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
