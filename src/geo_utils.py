"""
Funkcje pomocnicze dla obliczeń przestrzennych i przygotowania warstw GeoJSON
dla interfejsu Streamlit ([GRZ-02]).
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


def enrich_geojson_with_predictions(
    geojson_data: dict[str, Any],
    records_by_area: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Dołącza atrybuty predykcji, pogody, odległości oraz kolor RGBA do każdego
    obiektu Feature w kopii słownika GeoJSON na potrzeby renderowania w PyDeck.
    """
    enriched = copy.deepcopy(geojson_data)
    for feature in enriched.get("features", []):
        props = feature.setdefault("properties", {})
        area_id = str(props.get("area_id", "")).zfill(4)
        props["area_id"] = area_id

        rec = records_by_area.get(area_id)
        if rec and rec.get("in_radius", True):
            abundance = str(rec.get("abundance_class") or "BRAK_DANYCH")
            color_info = ABUNDANCE_CLASSES.get(abundance, ABUNDANCE_CLASSES["BRAK_DANYCH"])
            props["fill_color"] = color_info["rgba"]
            props["line_color"] = [30, 50, 35, 230]
            props["predicted_yield_label"] = (
                f"{rec['predicted_yield_pcs_per_h']:.1f} szt./h"
                if rec.get("predicted_yield_pcs_per_h") is not None
                else "brak danych"
            )
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
        else:
            color_info = ABUNDANCE_CLASSES["BRAK_DANYCH"]
            props["fill_color"] = color_info["rgba"]
            props["line_color"] = [120, 120, 120, 160]
            props["predicted_yield_label"] = "poza filtrem / brak danych"
            props["abundance_label"] = str(color_info["label"])
            props["confidence_label"] = "—"
            props["distance_label"] = (
                f"{rec['distance_km']:.1f} km (poza promieniem)"
                if rec and rec.get("distance_km") is not None
                else "—"
            )
            props["habitat_match_note"] = "Obszar poza wybranym filtrem lub brak predykcji"

        # Czytelne etykiety dla cech leśnych BDL (z poszanowaniem NULL)
        for field_key in ("pine_share_pct", "oak_beech_share_pct", "spruce_share_pct"):
            val = props.get(field_key)
            props[f"{field_key}_label"] = f"{float(val):.1f}%" if val is not None else "NULL (brak)"

        moisture = props.get("dominant_moisture_type")
        props["dominant_moisture_label"] = str(moisture) if moisture else "NULL (brak)"

    return enriched
