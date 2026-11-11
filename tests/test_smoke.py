"""
Testy dymne (smoke tests) szkieletu aplikacji, kontraktu danych ([GRZ-02])
oraz granic regionu i wektorowej próbki leśnej BDL ([GRZ-05]).

Uruchomienie:
    pytest -v
    lub:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ABUNDANCE_CLASSES,
    PILOT_REGION,
    PROCESSED_DIR,
    REFERENCE_CITIES,
    SPECIES_LABELS,
)
from src.data_loader import (
    load_areas_geojson,
    load_dataset_bundle,
    load_reports_csv,
    load_sprint1_audit_samples,
)
from src.geo_utils import (
    enrich_bdl_stands_geojson,
    enrich_geojson_with_predictions,
    haversine_km,
)


class TestAppSkeletonAndRegionSmoke(unittest.TestCase):
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

        # Sprawdzenie obecności wszystkich 30 rzeczywistych powiatów woj. dolnośląskiego (TERYT 02)
        self.assertEqual(len(bundle.areas_by_id), 30)
        self.assertIn("0201", bundle.areas_by_id)
        self.assertIn("0264", bundle.areas_by_id)
        for area_id, props in bundle.areas_by_id.items():
            self.assertEqual(len(area_id), 4)
            self.assertTrue(area_id.startswith("02"))
            self.assertTrue(props.get("is_demo"))

        # Sprawdzenie rozróżnienia NULL (None) vs 0.0 w raportach demo
        reports_by_id = {r["report_id"]: r for r in bundle.reports}
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

    def test_prod_fallback_to_demo_when_processed_incomplete(self) -> None:
        bundle = load_dataset_bundle("prod")
        self.assertEqual(bundle.env_mode, "demo")
        self.assertTrue(bundle.is_fallback_to_demo)

    def test_real_region_boundaries_and_crs_grz05(self) -> None:
        """Weryfikuje rzeczywiste granice 30 powiatów w EPSG:4326 ([GRZ-05])."""
        prod_areas_path = PROCESSED_DIR / "areas.geojson"
        self.assertTrue(prod_areas_path.exists())
        geojson_data, areas_by_id = load_areas_geojson(prod_areas_path)

        self.assertEqual(len(areas_by_id), 30)
        meta = geojson_data.get("metadata", {})
        self.assertIn("EPSG:4326", meta.get("crs_storage", ""))
        self.assertIn("EPSG:2180", meta.get("crs_metric_calculations", ""))
        self.assertEqual(meta.get("retrieved_at"), "2026-11-08")

        # Suma powierzchni 30 powiatów Dolnego Śląska (~19 941.7 km2)
        total_km2 = sum(float(p["total_area_km2"]) for p in areas_by_id.values())
        self.assertGreater(total_km2, 19800.0)
        self.assertLess(total_km2, 20100.0)

        # Weryfikacja, że geometrie są rzeczywistymi wielokątami, a nie prostokątami (min. kilkanaście punktów)
        feat_0201 = [
            f for f in geojson_data["features"] if f["properties"]["area_id"] == "0201"
        ][0]
        ring_0201 = feat_0201["geometry"]["coordinates"][0]
        self.assertGreater(len(ring_0201), 15)
        self.assertFalse(feat_0201["properties"]["is_demo"])

    def test_bdl_vector_sample_and_analytical_aggregation_grz05(self) -> None:
        """Weryfikuje wektorową próbkę 105 wydzieleń BDL i tabelę agregacji GeoPandas ([GRZ-05])."""
        samples = load_sprint1_audit_samples()
        self.assertEqual(len(samples["reports_audit"]), 52)
        self.assertEqual(len(samples["weather_sample"]), 210)
        self.assertEqual(len(samples["bdl_stands_sample"]), 105)
        self.assertEqual(len(samples["bdl_powiaty_agg"]), 7)

        bdl_geojson = samples["bdl_stands_geojson"]
        self.assertIsNotNone(bdl_geojson)
        self.assertEqual(len(bdl_geojson["features"]), 105)
        self.assertEqual(bdl_geojson["metadata"]["forest_stands_count"], 74)
        self.assertEqual(bdl_geojson["metadata"]["bdl_validity_year"], 2026)

        enriched_bdl = enrich_bdl_stands_geojson(bdl_geojson)
        self.assertIsNotNone(enriched_bdl)
        first_props = enriched_bdl["features"][0]["properties"]
        self.assertIn("Wydzielenie BDL:", first_props["tooltip_header"])
        self.assertIn("arodes_int_num", first_props)

    def test_missing_bdl_does_not_block_region_display_grz05(self) -> None:
        """
        Weryfikuje warunek #5 [GRZ-05]:
        Brak atrybutów BDL (wszystkie pola BDL = None) oraz brak warstwy wydzieleń BDL
        nie blokują wyświetlenia granic regionu i wyliczenia zestawienia powiatów.
        """
        bundle = load_dataset_bundle("demo")
        stripped_geojson = copy.deepcopy(bundle.areas_geojson)
        for feat in stripped_geojson["features"]:
            props = feat["properties"]
            props["forest_cover_pct"] = None
            props["pine_share_pct"] = None
            props["oak_beech_share_pct"] = None
            props["spruce_share_pct"] = None
            props["mean_stand_age_years"] = None
            props["dominant_moisture_type"] = None
            props["has_bdl_stand_data"] = False
            props["bdl_sample_stands_count"] = 0

        enriched_pred = enrich_geojson_with_predictions(
            stripped_geojson, {}, color_mode="predictions"
        )
        enriched_region = enrich_geojson_with_predictions(
            stripped_geojson, {}, color_mode="region_bdl"
        )
        self.assertEqual(len(enriched_pred["features"]), 30)
        self.assertEqual(len(enriched_region["features"]), 30)

        p0 = enriched_region["features"][0]["properties"]
        self.assertEqual(p0["pine_share_pct_label"], "NULL (brak BDL)")
        self.assertEqual(p0["dominant_moisture_label"], "NULL (brak BDL)")
        self.assertEqual(len(p0["fill_color"]), 4)
        self.assertIsNone(enrich_bdl_stands_geojson(None))

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

    def test_data_audit_and_contract_approval_grz06(self) -> None:
        """
        Weryfikuje warunki ukończenia [GRZ-06]:
        - istnienie i kompletność docs/data_audit.md,
        - zatwierdzenie docs/data_contract.md i docs/mvp.md (usunięcie [DO WERYFIKACJI]).
        """
        docs_dir = ROOT_DIR / "docs"
        audit_path = docs_dir / "data_audit.md"
        contract_path = docs_dir / "data_contract.md"
        mvp_path = docs_dir / "mvp.md"

        self.assertTrue(audit_path.exists(), "Brak pliku docs/data_audit.md")
        audit_text = audit_path.read_text(encoding="utf-8")

        # Kluczowe elementy wymagane w [GRZ-06]
        self.assertIn("predicted_yield_pcs_per_h", audit_text)
        self.assertIn("area_id", audit_text)
        self.assertIn("Temporal Split", audit_text)
        self.assertIn("Macro-F1", audit_text)
        self.assertIn("Brak wpisu", audit_text)
        self.assertIn("Point-in-Time Correctness", audit_text)
        self.assertIn("Jawny Wskaźnik Warunków", audit_text)
        self.assertIn("Osoba A", audit_text)
        self.assertIn("Osoba B", audit_text)

        contract_text = contract_path.read_text(encoding="utf-8")
        mvp_text = mvp_path.read_text(encoding="utf-8")
        self.assertNotIn("[DO WERYFIKACJI]", contract_text)
        self.assertNotIn("[DO WERYFIKACJI]", mvp_text)
        self.assertIn("Wersja 2.0", contract_text)
        self.assertIn("Wersja 2.0", mvp_text)

    def test_reports_dataset_parquet_and_csv_grz07(self) -> None:
        """
        Weryfikuje zbiór obserwacji [GRZ-07]:
        - obecność i spójność `data/processed/reports.parquet`, `data/processed/reports.csv`
          oraz `data/processed/reports_quality_report.json`,
        - stabilne identyfikatory `DG...`, ślad pochodzenia (`source_url`, `place_id`),
        - brak ujemnych lub zerowych czasów zbioru (`harvest_duration_h > 0`),
        - oznaczenie i odfiltrowanie wpisów wykluczonych (`518` ogółem -> `513` VALID w CSV).
        """
        import json
        import pandas as pd

        parquet_path = PROCESSED_DIR / "reports.parquet"
        csv_path = PROCESSED_DIR / "reports.csv"
        quality_path = PROCESSED_DIR / "reports_quality_report.json"

        self.assertTrue(parquet_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertTrue(quality_path.exists())

        df_parquet = pd.read_parquet(parquet_path)
        self.assertEqual(len(df_parquet), 518)
        self.assertEqual(df_parquet["report_id"].nunique(), 518)
        self.assertTrue(df_parquet["report_id"].str.startswith("DG").all())
        self.assertTrue(df_parquet["source_url"].str.startswith("https://www.grzyby.pl/").all())
        self.assertFalse(df_parquet["is_demo"].any())

        # Czas zbioru (gdy podany) musi być dodatni (> 0.0)
        durations = df_parquet["harvest_duration_h"].dropna()
        self.assertGreater(len(durations), 100)
        self.assertTrue((durations > 0.0).all())

        # Weryfikacja wykluczonych i poprawnych rekordów
        valid_df = df_parquet[df_parquet["is_valid_for_training"]]
        self.assertEqual(len(valid_df), 513)

        csv_rows = load_reports_csv(csv_path)
        self.assertEqual(len(csv_rows), 513)
        self.assertTrue(all(not r["is_demo"] for r in csv_rows))
        self.assertTrue(all(r["area_id"].startswith("02") and len(r["area_id"]) == 4 for r in csv_rows))
        self.assertTrue(all(r["yield_pcs_per_h"] is not None and r["yield_pcs_per_h"] >= 0.0 for r in csv_rows))

        q_report = json.loads(quality_path.read_text(encoding="utf-8"))
        self.assertEqual(q_report["funnel_stages"]["stage_1_raw_extracted_from_9_weeks"], 518)
        self.assertEqual(q_report["funnel_stages"]["stage_5_valid_for_training_in_reports_csv"], 513)
        self.assertEqual(q_report["regional_coverage"]["covered_powiats_count"], 28)


if __name__ == "__main__":
    unittest.main()
