"""
Testy dymne (smoke tests) szkieletu aplikacji i kontraktu danych ([GRZ-02]).

Uruchomienie:
    pytest
    lub:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from src.config import ABUNDANCE_CLASSES, PILOT_REGION, REFERENCE_CITIES, SPECIES_LABELS
from src.data_loader import load_dataset_bundle, load_sprint1_audit_samples
from src.geo_utils import enrich_geojson_with_predictions, haversine_km


class TestAppSkeletonSmoke(unittest.TestCase):
    def test_region_configuration(self) -> None:
        self.assertEqual(PILOT_REGION.voivodeship_id, "02")
        self.assertIn("dolnośląskie", PILOT_REGION.voivodeship_name.lower())
        self.assertEqual(PILOT_REGION.spatial_key_name, "area_id")
        self.assertGreaterEqual(len(REFERENCE_CITIES), 5)
        self.assertIn("all", SPECIES_LABELS)
        self.assertIn("WYSYP", ABUNDANCE_CLASSES)

    def test_demo_bundle_loading_and_contract_rules(self) -> None:
        bundle = load_dataset_bundle("demo")
        self.assertEqual(bundle.env_mode, "demo")
        self.assertFalse(bundle.is_fallback_to_demo)

        # Sprawdzenie zachowania wiodącego zera w kodach TERYT (4 znaki)
        self.assertIn("0201", bundle.areas_by_id)
        self.assertIn("0264", bundle.areas_by_id)
        for area_id, props in bundle.areas_by_id.items():
            self.assertEqual(len(area_id), 4)
            self.assertTrue(area_id.startswith("02"))
            self.assertTrue(props.get("is_demo"))

        # Sprawdzenie rozróżnienia NULL (None) vs 0.0 w raportach demo
        reports_by_id = {r["report_id"]: r for r in bundle.reports}
        # DEMO_REP_011 ma jawne yield_pcs_per_h = 0.0 oraz dominant_species = None (NULL)
        rep_011 = reports_by_id["DEMO_REP_011"]
        self.assertEqual(rep_011["yield_pcs_per_h"], 0.0)
        self.assertIsNone(rep_011["dominant_species"])
        self.assertIsNone(rep_011["worminess_pct"])

        # Sprawdzenie rozróżnienia NULL vs 0.0 w pogodzie demo
        w_0264 = [
            w for w in bundle.weather if w["area_id"] == "0264" and w["date"] == "2026-10-31"
        ][0]
        self.assertEqual(w_0264["precip_mm_1d"], 0.0)
        self.assertIsNone(w_0264["soil_moisture_0_7cm_m3m3"])

    def test_prod_fallback_to_demo_when_processed_empty(self) -> None:
        bundle = load_dataset_bundle("prod")
        self.assertEqual(bundle.env_mode, "demo")
        self.assertTrue(bundle.is_fallback_to_demo)

    def test_haversine_and_geojson_enrichment(self) -> None:
        wroclaw_lat, wroclaw_lon = REFERENCE_CITIES["Wrocław (centrum regionu)"]
        dist_self = haversine_km(wroclaw_lat, wroclaw_lon, wroclaw_lat, wroclaw_lon)
        self.assertEqual(dist_self, 0.0)

        bundle = load_dataset_bundle("demo")
        mock_records = {
            "0201": {
                "in_radius": True,
                "abundance_class": "WYSYP",
                "predicted_yield_pcs_per_h": 41.2,
                "confidence_score": 0.92,
                "distance_km": 103.5,
                "habitat_match_note": "Test",
            }
        }
        enriched = enrich_geojson_with_predictions(bundle.areas_geojson, mock_records)
        feat_0201 = [
            f
            for f in enriched["features"]
            if f["properties"]["area_id"] == "0201"
        ][0]
        self.assertEqual(
            feat_0201["properties"]["fill_color"],
            ABUNDANCE_CLASSES["WYSYP"]["rgba"],
        )
        self.assertEqual(feat_0201["properties"]["predicted_yield_label"], "41.2 szt./h")

    def test_sprint1_audit_samples_present(self) -> None:
        samples = load_sprint1_audit_samples()
        self.assertEqual(len(samples["reports_audit"]), 52)
        self.assertEqual(len(samples["weather_sample"]), 210)


if __name__ == "__main__":
    unittest.main()
