"""
Testy dymne (smoke tests) szkieletu aplikacji, kontraktu danych ([GRZ-02]),
granic regionu i wektorowej próbki leśnej BDL ([GRZ-05]), audytu ([GRZ-06]),
potoku raportów ([GRZ-07]), łączenia przestrzennego ([GRZ-08]), pogody ([GRZ-09]),
zbioru cech Point-in-Time ([GRZ-10]), EDA ([GRZ-11]), eksploracji w Streamlit ([GRZ-12])
oraz protokołu ewaluacji ([GRZ-13]).

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
    OBSERVATION_SPECIES_FILTER_LABELS,
    PILOT_REGION,
    PROCESSED_DIR,
    REFERENCE_CITIES,
    SPECIES_LABELS,
)
from src.data_loader import (
    filter_real_observations_and_aggregate_by_county,
    load_areas_geojson,
    load_dataset_bundle,
    load_real_observations_bundle,
    load_reports_csv,
    load_sprint1_audit_samples,
    load_weather_csv,
    verify_record_against_source_tables,
)
from src.geo_utils import (
    build_precise_report_points_layer_data,
    enrich_bdl_stands_geojson,
    enrich_geojson_with_predictions,
    enrich_geojson_with_real_observations,
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

    def test_spatial_features_and_no_false_precision_grz08(self) -> None:
        """
        Weryfikuje przetwarzanie przestrzymy [GRZ-08]:
        - obecność i schemat `data/processed/forest_features.parquet` i `.csv` z kluczem region_id,
        - obecność klucza region_id w `data/processed/areas.geojson`,
        - jawną klasyfikację precyzji w `reports_spatial.parquet` (point_precise vs county_aggregate),
        - brak pozornej dokładności dla raportów powiatowych (point_represents_specific_forest=False),
        - brak multiplikacji wierszy (współczynnik = 1.0; 518 wejściowych -> 518 wyjściowych),
        - brak nieprzypisanych lokalizacji (unassigned == 0).
        """
        import json
        import pandas as pd

        forest_parquet = PROCESSED_DIR / "forest_features.parquet"
        forest_csv = PROCESSED_DIR / "forest_features.csv"
        reports_spatial_parquet = PROCESSED_DIR / "reports_spatial.parquet"
        reports_spatial_csv = PROCESSED_DIR / "reports_spatial.csv"
        spatial_quality_path = PROCESSED_DIR / "spatial_quality_report.json"
        areas_path = PROCESSED_DIR / "areas.geojson"

        self.assertTrue(forest_parquet.exists())
        self.assertTrue(forest_csv.exists())
        self.assertTrue(reports_spatial_parquet.exists())
        self.assertTrue(reports_spatial_csv.exists())
        self.assertTrue(spatial_quality_path.exists())

        # 1. Sprawdzenie tabeli forest_features (30 powiatów Dolnego Śląska)
        df_forest = pd.read_parquet(forest_parquet)
        self.assertEqual(len(df_forest), 30)
        self.assertIn("region_id", df_forest.columns)
        self.assertIn("area_id", df_forest.columns)
        self.assertTrue(df_forest["region_id"].str.startswith("02").all())
        self.assertEqual(df_forest["region_id"].nunique(), 30)

        # 2. Sprawdzenie obecności region_id w areas.geojson
        areas_data, areas_by_id = load_areas_geojson(areas_path)
        self.assertEqual(len(areas_by_id), 30)
        for feat in areas_data["features"]:
            self.assertIn("region_id", feat["properties"])
            self.assertEqual(feat["properties"]["region_id"], feat["properties"]["area_id"])

        # 3. Sprawdzenie reports_spatial i niezmiennika braku multiplikacji
        df_spatial = pd.read_parquet(reports_spatial_parquet)
        self.assertEqual(len(df_spatial), 518)  # Dokładnie 1:1 z reports.parquet
        self.assertEqual(df_spatial["report_id"].nunique(), 518)

        # 4. Sprawdzenie eliminacji pozornej dokładności (false precision)
        self.assertIn("spatial_precision", df_spatial.columns)
        self.assertIn("point_represents_specific_forest", df_spatial.columns)

        county_agg = df_spatial[df_spatial["spatial_precision"] == "county_aggregate"]
        self.assertEqual(len(county_agg), 34)
        self.assertTrue((county_agg["point_represents_specific_forest"] == False).all())
        self.assertTrue(county_agg["is_county_aggregate"].all())

        point_precise = df_spatial[df_spatial["spatial_precision"] == "point_precise"]
        self.assertEqual(len(point_precise), 484)
        self.assertTrue((point_precise["point_represents_specific_forest"] == True).all())
        self.assertFalse(point_precise["is_county_aggregate"].any())

        # 5. Sprawdzenie braku nieprzypisanych lokalizacji i poprawności odległości metrycznych
        self.assertTrue(df_spatial["region_id"].notna().all())
        self.assertTrue((df_spatial["region_id"] != "").all())
        self.assertTrue(df_spatial["distance_to_county_centroid_km"].notna().all())
        self.assertTrue((df_spatial["distance_to_county_centroid_km"] >= 0.0).all())

        # 6. Sprawdzenie pliku CSV dla zbioru treningowego (dokładnie 513 rekordów)
        df_spatial_csv = pd.read_csv(reports_spatial_csv)
        self.assertEqual(len(df_spatial_csv), 513)

        # 7. Weryfikacja raportu jakości
        sq_report = json.loads(spatial_quality_path.read_text(encoding="utf-8"))
        self.assertEqual(sq_report["multiplication_invariants"]["multiplication_factor"], 1.0)
        self.assertTrue(sq_report["multiplication_invariants"]["has_1_to_1_integrity"])
        self.assertEqual(sq_report["precision_breakdown"]["unassigned_count"], 0)

    def test_pilot_weather_complete_and_no_missing_days_grz09(self) -> None:
        """
        Weryfikuje pełny zbiór pogodowy [GRZ-09]:
        - obecność i schemat `weather.parquet`, `weather.csv`, `weather_daily.parquet`
          oraz `weather_quality_report.json`,
        - unikalność klucza (area_id, date) dla wszystkich 2910 rekordów,
        - pokrycie 30 powiatów Dolnego Śląska przez 97 kolejnych dni (2025-08-01 .. 2025-11-05),
        - 0 brakujących dni w szeregach czasowych,
        - 31 dni zapasu historii przed 2025-09-01 (w 100% wyliczone okno precip_lag_21d_mm),
        - poprawność wczytywania przez loader kontraktu danych `load_weather_csv`.
        """
        import json
        import pandas as pd

        weather_parquet = PROCESSED_DIR / "weather.parquet"
        weather_csv = PROCESSED_DIR / "weather.csv"
        weather_daily_parquet = PROCESSED_DIR / "weather_daily.parquet"
        weather_quality_path = PROCESSED_DIR / "weather_quality_report.json"

        self.assertTrue(weather_parquet.exists())
        self.assertTrue(weather_csv.exists())
        self.assertTrue(weather_daily_parquet.exists())
        self.assertTrue(weather_quality_path.exists())

        # 1. Weryfikacja liczby wierszy i unikalności klucza głównego (30 powiatów x 97 dni)
        df_weather = pd.read_parquet(weather_parquet)
        self.assertEqual(len(df_weather), 2910)
        self.assertEqual(df_weather["area_id"].nunique(), 30)
        self.assertTrue(df_weather["area_id"].str.startswith("02").all())

        unique_keys = df_weather[["area_id", "date"]].drop_duplicates()
        self.assertEqual(len(unique_keys), 2910)

        # 2. Zakres dat i zapas historii
        self.assertEqual(df_weather["date"].min(), "2025-08-01")
        self.assertEqual(df_weather["date"].max(), "2025-11-05")
        self.assertEqual(df_weather["date"].nunique(), 97)

        # Na dzień pierwszych raportów (2025-09-01) precip_lag_21d_mm musi być w pełni wyliczony
        sep01_records = df_weather[df_weather["date"] == "2025-09-01"]
        self.assertEqual(len(sep01_records), 30)
        self.assertTrue(sep01_records["precip_lag_21d_mm"].notna().all())
        self.assertTrue((sep01_records["precip_lag_21d_mm"] >= 0.0).all())

        # 3. Weryfikacja wczytywania przez moduł produkcyjny src/data_loader.py
        loaded_csv_rows = load_weather_csv(weather_csv)
        self.assertEqual(len(loaded_csv_rows), 2910)
        self.assertTrue(all(not r["is_demo"] for r in loaded_csv_rows))
        self.assertTrue(all(r["area_id"].startswith("02") for r in loaded_csv_rows))

        # 4. Sprawdzenie surowego zbioru weather_daily z reprezentacją punktową
        df_daily = pd.read_parquet(weather_daily_parquet)
        self.assertEqual(len(df_daily), 2910)
        self.assertIn("spatial_representation", df_daily.columns)
        self.assertTrue((df_daily["spatial_representation"] == "point_approximation_centroid").all())
        self.assertTrue(df_daily["soil_moisture_0_7cm_m3m3"].notna().all())

        # 5. Sprawdzenie raportu jakości
        q_report = json.loads(weather_quality_path.read_text(encoding="utf-8"))
        self.assertEqual(q_report["integrity_and_missing_data"]["total_missing_days_count"], 0)
        self.assertTrue(q_report["integrity_and_missing_data"]["has_unique_area_date_keys"])
        self.assertEqual(q_report["date_range"]["history_buffer_days_before_first_report"], 31)

    def test_features_dataset_and_point_in_time_cutoff_grz10(self) -> None:
        """
        Weryfikuje zbiór cech analitycznych Point-in-Time [GRZ-10]:
        1. Skrypt odtwarza `data/processed/features.parquet` z uzgodnionym słownikiem kolumn (`FEATURE_DICTIONARY`).
        2. Każda etykieta ma identyfikator źródłowego raportu (`report_id` / `observation_id`);
           łączenie nie tworzy przypadkowych kopii (`513` -> `513`, współczynnik = 1.0).
        3. Okna pogodowe 3/7/14/21 dni nie obejmują dni po przyjętym odcięciu informacji (`feature_cutoff_date < target_date`).
        4. Sprawdzono ręcznie obliczenie jednego okna 7-dniowego (`DG174806`), jeden brak danych (`DG177228` + brak dnia pogody)
           i jeden join 3-tabelowy (`DG174714`).
        5. Braki cech (`NULL`) nie są automatycznie zamieniane w zero opadu (`0.0 mm`) ani zero zbioru (`0.0 szt./h`).
        6. Zapisano liczebność i zakres danych po połączeniu w `features_quality_report.json` i `docs/features_engineering.md`.
        """
        import json
        import pandas as pd
        from src.features import (
            FEATURE_DICTIONARY,
            compute_point_in_time_weather_features,
            index_weather_by_region_and_date,
        )

        features_parquet = PROCESSED_DIR / "features.parquet"
        features_csv = PROCESSED_DIR / "features.csv"
        quality_path = PROCESSED_DIR / "features_quality_report.json"
        doc_path = ROOT_DIR / "docs" / "features_engineering.md"
        weather_daily_parquet = PROCESSED_DIR / "weather_daily.parquet"

        self.assertTrue(features_parquet.exists(), "Brak data/processed/features.parquet")
        self.assertTrue(features_csv.exists(), "Brak data/processed/features.csv")
        self.assertTrue(quality_path.exists(), "Brak data/processed/features_quality_report.json")
        self.assertTrue(doc_path.exists(), "Brak docs/features_engineering.md")

        df_feat = pd.read_parquet(features_parquet)
        df_csv = pd.read_csv(features_csv)

        # 1. Uzgodniony słownik kolumn
        expected_cols = list(FEATURE_DICTIONARY.keys())
        self.assertEqual(list(df_feat.columns), expected_cols)
        self.assertEqual(list(df_csv.columns), expected_cols)
        self.assertEqual(len(expected_cols), 70)

        # 2. Identyfikator źródłowego raportu i brak multiplikacji (1:1 = 513 rekordów)
        self.assertEqual(len(df_feat), 513)
        self.assertEqual(len(df_csv), 513)
        self.assertEqual(df_feat["report_id"].nunique(), 513)
        self.assertEqual(df_feat["observation_id"].nunique(), 513)
        self.assertTrue((df_feat["report_id"] == df_feat["observation_id"]).all())
        self.assertTrue(df_feat["report_id"].str.startswith("DG").all())
        self.assertFalse(df_feat["is_demo"].any())

        # 3. Odcięcie informacji Point-in-Time (brak wycieku z dnia prognozy t)
        self.assertTrue((df_feat["observation_date"] == df_feat["harvest_date"]).all())
        self.assertTrue((df_feat["target_date"] == df_feat["harvest_date"]).all())
        self.assertTrue((df_feat["issued_at"] == df_feat["feature_cutoff_date"]).all())
        self.assertTrue((df_feat["feature_cutoff_date"] < df_feat["target_date"]).all())
        self.assertTrue((df_feat["window_end_date"] == df_feat["feature_cutoff_date"]).all())
        self.assertTrue((df_feat["window_7d_start_date"] < df_feat["window_end_date"]).all())
        self.assertTrue((df_feat["window_14d_start_date"] < df_feat["window_7d_start_date"]).all())
        self.assertTrue((df_feat["window_21d_start_date"] < df_feat["window_14d_start_date"]).all())
        self.assertTrue((df_feat["horizon_days"] == 1).all())

        # 4a. Ręczna weryfikacja okna 7-dniowego dla DG174806 (pow. bolesławiecki 0201, target=2025-10-05, cutoff=2025-10-04)
        row_174806 = df_feat[df_feat["report_id"] == "DG174806"].iloc[0]
        self.assertEqual(row_174806["region_id"], "0201")
        self.assertEqual(row_174806["target_date"], "2025-10-05")
        self.assertEqual(row_174806["feature_cutoff_date"], "2025-10-04")
        self.assertEqual(row_174806["window_7d_start_date"], "2025-09-28")
        self.assertEqual(row_174806["window_end_date"], "2025-10-04")
        self.assertAlmostEqual(float(row_174806["precip_lag_7d_mm"]), 13.0, places=2)
        self.assertAlmostEqual(float(row_174806["temp_mean_7d_c"]), 8.3, places=1)
        self.assertEqual(int(row_174806["rainy_days_lag_7d_count"]), 3)

        # 4b & 5. Ręczna weryfikacja braków danych (NULL != 0.0)
        row_177228 = df_feat[df_feat["report_id"] == "DG177228"].iloc[0]
        self.assertEqual(row_177228["region_id"], "0218")
        self.assertFalse(bool(row_177228["has_bdl_stand_data"]))
        self.assertTrue(pd.isna(row_177228["pine_share_pct"]))
        self.assertTrue(pd.isna(row_177228["oak_beech_share_pct"]))
        self.assertTrue(pd.isna(row_177228["spruce_share_pct"]))
        self.assertTrue(pd.isna(row_177228["mean_stand_age_years"]))
        self.assertTrue(pd.isna(row_177228["dominant_moisture_type"]))
        self.assertTrue(pd.isna(row_177228["harvest_duration_h"]))
        self.assertTrue(pd.isna(row_177228["total_pieces"]))
        self.assertAlmostEqual(float(row_177228["yield_pcs_per_h"]), 10.0, places=2)

        # Wszystkie 80 rekordów z 11 powiatów bez profilu taksacyjnego BDL ma zachowany NULL (a nie 0.0)
        no_bdl_rows = df_feat[df_feat["pine_share_pct"].isna()]
        self.assertEqual(len(no_bdl_rows), 80)
        self.assertTrue(no_bdl_rows["oak_beech_share_pct"].isna().all())
        self.assertTrue(no_bdl_rows["spruce_share_pct"].isna().all())
        self.assertTrue(no_bdl_rows["mean_stand_age_years"].isna().all())
        self.assertTrue(no_bdl_rows["dominant_moisture_type"].isna().all())

        # Raport zerowego zbioru DG174777 zachowuje yield_pcs_per_h == 0.0 oraz dominant_species == NULL
        row_174777 = df_feat[df_feat["report_id"] == "DG174777"].iloc[0]
        self.assertEqual(float(row_174777["yield_pcs_per_h"]), 0.0)
        self.assertEqual(row_174777["abundance_class"], "BRAK")
        self.assertTrue(pd.isna(row_174777["dominant_species"]))

        # Symulacja braku pojedynczego dnia w szeregu pogodowym -> okno zwraca None, a nie 0.0 mm
        df_weather_daily = pd.read_parquet(weather_daily_parquet)
        weather_lookup = index_weather_by_region_and_date(df_weather_daily)
        del weather_lookup[("0201", "2025-10-02")]
        missing_window_res = compute_point_in_time_weather_features("0201", "2025-10-05", weather_lookup)
        self.assertIsNone(missing_window_res["precip_lag_7d_mm"])
        self.assertIsNone(missing_window_res["precip_lag_14d_mm"])
        self.assertIsNone(missing_window_res["precip_lag_21d_mm"])
        self.assertAlmostEqual(float(missing_window_res["precip_cutoff_1d_mm"]), 7.9, places=2)

        # 4c. Ręczna weryfikacja łączenia 3-tabelowego (DG174714, pow. kłodzki 0208, 2025-10-05)
        row_174714 = df_feat[df_feat["report_id"] == "DG174714"].iloc[0]
        self.assertEqual(row_174714["region_id"], "0208")
        self.assertEqual(row_174714["dominant_species"], "podgrzybek_brunatny")
        self.assertAlmostEqual(float(row_174714["forest_cover_pct"]), 48.9, places=2)
        self.assertAlmostEqual(float(row_174714["spruce_share_pct"]), 58.0, places=2)
        self.assertEqual(row_174714["dominant_moisture_type"], "wilgotne")
        self.assertAlmostEqual(float(row_174714["precip_lag_7d_mm"]), 9.8, places=2)
        self.assertAlmostEqual(float(row_174714["precip_lag_14d_mm"]), 24.8, places=2)
        self.assertAlmostEqual(float(row_174714["precip_lag_21d_mm"]), 33.5, places=2)
        self.assertAlmostEqual(float(row_174714["temp_min_night_c"]), -2.1, places=2)
        self.assertAlmostEqual(float(row_174714["soil_moisture_0_7cm_m3m3"]), 0.3248, places=4)

        # 6. Sprawdzenie raportu jakości i liczebności po połączeniu
        q_report = json.loads(quality_path.read_text(encoding="utf-8"))
        self.assertEqual(q_report["dataset_summary"]["valid_training_observations_count"], 513)
        self.assertEqual(q_report["dataset_summary"]["multiplication_factor"], 1.0)
        self.assertTrue(q_report["dataset_summary"]["has_1_to_1_label_integrity"])
        self.assertTrue(q_report["temporal_ranges_and_cutoff_rules"]["no_weather_after_cutoff_verified"])
        self.assertEqual(q_report["splits_distribution"]["split_2way"]["train"], 386)
        self.assertEqual(q_report["splits_distribution"]["split_2way"]["test"], 127)

    def test_evaluation_protocol_and_baseline_grz13(self) -> None:
        """
        Weryfikuje protokół ewaluacji i modele odniesienia [GRZ-13]:
        - obecność i spójność `split_metadata.json`, `baseline_metrics.json` oraz `docs/evaluation_protocol.md`,
        - deterministyczny podział czasowy (386 Train / 127 Test = 513 łącznie),
        - brak wycieku danych (0 wspólnych ID, max_train_date < min_test_date),
        - dopasowanie predyktorów wyłącznie na zbiorze treningowym,
        - metryki regresji (MAE, RMSE) i klasyfikacji (Macro-F1, Macierz pomyłek),
        - poprawność klasyfikatora obfitości zbioru (`classify_abundance`).
        """
        import json
        import numpy as np
        import pandas as pd
        from src.evaluation import classify_abundance
        from src.baseline import GlobalMeanRegressor, MajorityClassClassifier

        split_meta_path = PROCESSED_DIR / "split_metadata.json"
        metrics_path = PROCESSED_DIR / "baseline_metrics.json"
        doc_path = ROOT_DIR / "docs" / "evaluation_protocol.md"

        self.assertTrue(split_meta_path.exists(), "Brak data/processed/split_metadata.json")
        self.assertTrue(metrics_path.exists(), "Brak data/processed/baseline_metrics.json")
        self.assertTrue(doc_path.exists(), "Brak docs/evaluation_protocol.md")

        # 1. Weryfikacja metadanych podziału (split_metadata.json)
        split_meta = json.loads(split_meta_path.read_text(encoding="utf-8"))
        self.assertEqual(split_meta["total_valid_records"], 513)

        p_train = split_meta["primary_split_train_test"]["train"]
        p_test = split_meta["primary_split_train_test"]["test"]
        self.assertEqual(p_train["count"], 386)
        self.assertEqual(p_test["count"], 127)
        self.assertEqual(p_train["count"] + p_test["count"], 513)

        # Sprawdzenie braku wycieków
        self.assertEqual(split_meta["anti_leakage_checks"]["report_id_overlap_train_test"], 0)
        self.assertTrue(split_meta["anti_leakage_checks"]["temporal_gap_maintained"])
        self.assertTrue(p_train["date_max"] < p_test["date_min"])

        train_ids = set(p_train["report_ids"])
        test_ids = set(p_test["report_ids"])
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)

        # 2. Weryfikacja metryk bazowych (baseline_metrics.json)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertIn("regression_benchmarks", metrics)
        self.assertIn("classification_benchmarks", metrics)
        self.assertIn("target_evaluation_criteria_for_ml_models", metrics)

        # Regresja: MAE i RMSE
        reg_mean = metrics["regression_benchmarks"]["global_mean"]["test_metrics"]
        self.assertGreater(reg_mean["mae_pcs_per_h"], 10.0)
        self.assertGreater(reg_mean["rmse_pcs_per_h"], 15.0)
        self.assertFalse(reg_mean["r2_interpretable"])

        # Klasyfikacja: Macro-F1 i Confusion Matrix
        cls_maj = metrics["classification_benchmarks"]["majority_class"]["test_metrics"]
        self.assertGreater(cls_maj["macro_f1"], 0.15)
        self.assertEqual(len(cls_maj["confusion_matrix"]), 3)
        self.assertEqual(len(cls_maj["confusion_matrix"][0]), 3)

        # 3. Test jednostkowy funkcji pomocniczych
        self.assertEqual(classify_abundance(2.5), "BRAK")
        self.assertEqual(classify_abundance(5.0), "SREDNIO")
        self.assertEqual(classify_abundance(20.0), "SREDNIO")
        self.assertEqual(classify_abundance(20.5), "WYSYP")

        # Test predyktorów bazowych
        g_reg = GlobalMeanRegressor().fit(None, pd.Series([10.0, 20.0, 30.0]))
        preds = g_reg.predict(3)
        np.testing.assert_array_almost_equal(preds, [20.0, 20.0, 20.0])

        m_cls = MajorityClassClassifier().fit(None, pd.Series(["BRAK", "SREDNIO", "SREDNIO"]))
        cls_preds = m_cls.predict(2)
        self.assertEqual(list(cls_preds), ["SREDNIO", "SREDNIO"])

    def test_eda_and_lag_analysis_grz11(self) -> None:
        """
        Weryfikuje eksploracyjną analizę danych i badanie opóźnienia opad -> zbiory [GRZ-11]:
        1. Co najmniej 3 czytelne wykresy PNG z jednostkami, okresem i liczebnością próby (w katalogu docs/figures/).
        2. Sprawdzenie wrażliwości wyniku na dominację pojedynczego regionu (0201, Top-3, Top-5) oraz sezonu (3 fazy jesieni).
        3. Porównanie dostępnych okien opadów (1d, 7d, 14d, 21d, przedziały tygodniowe, krzywa k=1..21d) z odnotowaniem także słabych zależności.
        4. Oddzielenie obfitości raportów (wolumen dzienny, 2.61x wyższy w weekendy) od obfitości grzybów (yield_pcs_per_h, mediana 15.0 vs 15.0 szt./h) oraz jawne zastrzeżenie korelacja != przyczynowość.
        5. Zapisane rekomendacje do modelu (REC-01..REC-05) oraz ograniczenia filtrowania gatunków (55.17% mieszane/NULL).
        """
        import json

        eda_json_path = PROCESSED_DIR / "eda_summary.json"
        eda_doc_path = ROOT_DIR / "docs" / "eda_and_lag_analysis.md"
        notebook_path = ROOT_DIR / "notebooks" / "01_eda_lag_effect.ipynb"
        figures_dir = ROOT_DIR / "docs" / "figures"

        self.assertTrue(eda_json_path.exists(), "Brak data/processed/eda_summary.json")
        self.assertTrue(eda_doc_path.exists(), "Brak docs/eda_and_lag_analysis.md")
        self.assertTrue(notebook_path.exists(), "Brak notebooks/01_eda_lag_effect.ipynb")

        # 1. Weryfikacja co najmniej 3 wykresów PNG (wygenerowano 5)
        expected_figs = [
            "eda_01_yield_and_reports_distribution.png",
            "eda_02_seasonality_and_volume_vs_yield.png",
            "eda_03_precipitation_lag_comparison.png",
            "eda_04_regional_and_seasonal_sensitivity.png",
            "eda_05_missingness_and_species_constraints.png",
        ]
        self.assertGreaterEqual(len(expected_figs), 3)
        for fname in expected_figs:
            fpath = figures_dir / fname
            self.assertTrue(fpath.exists(), f"Brak wykresu {fpath}")
            self.assertGreater(fpath.stat().st_size, 25_000, f"Wykres {fname} jest zbyt mały/pusty")

        summary = json.loads(eda_json_path.read_text(encoding="utf-8"))

        # 2. Sprawdzenie wrażliwości na dominację regionu i sezonu
        self.assertEqual(summary["yield_and_reports_distribution"]["n_valid_observations"], 513)
        self.assertEqual(summary["regional_concentration"]["top_1_region_id"], "0201")
        self.assertAlmostEqual(summary["regional_concentration"]["top_1_region_share_pct"], 19.69, places=2)

        reg_sens = summary["regional_dominance_sensitivity"]
        self.assertEqual(len(reg_sens), 4)
        for s in reg_sens:
            self.assertGreater(s["corr_rainy_days_14d"]["pearson_r"], 0.20)
            self.assertGreater(s["corr_precip_slice_15_21d"]["spearman_rho"], 0.14)

        seas_sens = summary["seasonal_dominance_sensitivity"]
        self.assertEqual(len(seas_sens), 3)
        self.assertEqual(sum(s["n"] for s in seas_sens), 513)

        # 3. Porównanie okien opadów i krzywej k=1..21 dni
        corr_map = {c["feature"]: c for c in summary["weather_and_habitat_correlations"]}
        self.assertIn("precip_cutoff_1d_mm", corr_map)
        self.assertIn("precip_lag_7d_mm", corr_map)
        self.assertIn("precip_lag_14d_mm", corr_map)
        self.assertIn("precip_lag_21d_mm", corr_map)
        self.assertIn("precip_slice_15_21d_mm", corr_map)
        self.assertIn("rainy_days_lag_14d_count", corr_map)

        self.assertLess(corr_map["precip_lag_7d_mm"]["pearson_r"], 0.0)
        self.assertGreater(corr_map["precip_lag_21d_mm"]["pearson_r"], 0.0)
        self.assertGreater(corr_map["precip_slice_15_21d_mm"]["spearman_rho"], 0.15)
        self.assertGreater(corr_map["rainy_days_lag_14d_count"]["pearson_r"], 0.25)

        lag_curve = summary["rolling_lag_curve_1_to_21_days"]
        self.assertEqual(len(lag_curve), 21)

        # 4. Oddzielenie obfitości raportów od obfitości grzybów oraz korelacja != przyczynowość
        va = summary["report_volume_vs_mushroom_abundance"]
        self.assertGreater(va["weekend_to_weekday_volume_ratio"], 2.5)
        self.assertLess(va["report_volume_mannwhitney_p"], 0.001)
        self.assertEqual(
            va["mushroom_yield_weekday_median_pcs_per_h"],
            va["mushroom_yield_weekend_median_pcs_per_h"],
        )
        self.assertGreater(va["mushroom_yield_mannwhitney_p"], 0.5)

        doc_text = eda_doc_path.read_text(encoding="utf-8")
        self.assertIn("przyczynowo", doc_text.lower())
        self.assertIn("Obfitość raportów", doc_text)
        self.assertIn("Obfitość grzybów", doc_text)

        # 5. Rekomendacje do modelu i ograniczenia filtrowania gatunków
        sp = summary["species_filtering_constraints"]
        self.assertEqual(sp["mixed_or_null_species_count"], 283)
        self.assertGreater(sp["mixed_or_null_species_pct"], 55.0)
        self.assertGreaterEqual(len(summary["modeling_recommendations"]), 5)

        nb_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(nb_data["nbformat"], 4)
        self.assertGreaterEqual(len(nb_data["cells"]), 6)

    def test_real_reports_and_coverage_in_streamlit_grz12(self) -> None:
        """
        Weryfikuje wszystkie 5 warunków ukończenia zadania [GRZ-12]:
        1. Mapa wczytuje rzeczywiste dane z `data/processed/` (`is_demo = False`, `N = 513` raportów, 30 powiatów),
           a nie tylko próbkę demonstracyjną (`data/demo/`).
        2. Widoczne są okres (`2025-09-01` .. `2025-11-02`), legenda i jednostka przestrzenna (`powiat`, TERYT 4-znaki).
        3. Filtry (zakres dat / pojedyncza data, gatunek obserwacji, powiat, promień) zmieniają mapę
           oraz liczebność dostępnych raportów.
        4. Brak raportów w powiecie (`reports_count == 0`) jest pokazany jako brak danych (`BRAK_DANYCH` / `NULL`),
           a nigdy jako `0.0 szt./h`.
        5. Można obejrzeć próbkę rekordów i sprawdzić zgodność 1:1 z tabelą źródłową (`reports.parquet`,
           `reports_spatial.parquet`, `forest_features.parquet`, `features.parquet`).
        """
        real_bundle = load_real_observations_bundle()

        # 1. Rzeczywiste dane, a nie próbka demonstracyjna
        self.assertEqual(len(real_bundle.areas_by_id), 30)
        self.assertFalse(any(p.get("is_demo", True) for p in real_bundle.areas_by_id.values()))
        self.assertEqual(len(real_bundle.reports_df), 518)
        self.assertEqual(len(real_bundle.features_df), 513)
        self.assertEqual(real_bundle.min_date, "2025-09-01")
        self.assertEqual(real_bundle.max_date, "2025-11-02")

        # 2 & 4. Pełny sezon: 28 powiatów z danymi, 2 powiaty (0217, 0262) jako BRAK_DANYCH (mean_yield = None, NIE 0.0)
        res_all = filter_real_observations_and_aggregate_by_county(
            real_bundle=real_bundle,
            date_from="2025-09-01",
            date_to="2025-11-02",
            species_filter="all",
        )
        self.assertEqual(res_all["filtered_reports_count"], 513)
        self.assertEqual(res_all["covered_counties_count"], 28)
        self.assertEqual(res_all["no_data_counties_count"], 2)
        self.assertEqual(sorted(res_all["no_data_counties_ids"]), ["0217", "0262"])

        for no_data_id in ("0217", "0262"):
            rec_empty = res_all["county_by_id"][no_data_id]
            self.assertEqual(rec_empty["reports_count"], 0)
            self.assertFalse(rec_empty["has_reports"])
            self.assertEqual(rec_empty["status_code"], "BRAK_DANYCH")
            self.assertIsNone(rec_empty["mean_yield_pcs_per_h"])

        # Sprawdzenie wzbogacenia GeoJSON na potrzeby mapy w obu trybach kolorowania
        geo_yield = enrich_geojson_with_real_observations(
            real_bundle.areas_geojson, res_all["county_by_id"], color_mode="observed_yield"
        )
        geo_cov = enrich_geojson_with_real_observations(
            real_bundle.areas_geojson, res_all["county_by_id"], color_mode="report_coverage"
        )
        self.assertEqual(len(geo_yield["features"]), 30)
        self.assertEqual(len(geo_cov["features"]), 30)

        props_0217 = [
            f["properties"] for f in geo_yield["features"] if f["properties"]["area_id"] == "0217"
        ][0]
        self.assertEqual(props_0217["status_code"], "BRAK_DANYCH")
        self.assertIn("BRAK DANYCH", props_0217["observed_yield_label"])
        self.assertEqual(props_0217["fill_color"], [180, 180, 180, 150])

        props_0201_cov = [
            f["properties"] for f in geo_cov["features"] if f["properties"]["area_id"] == "0201"
        ][0]
        self.assertEqual(props_0201_cov["reports_count"], 101)
        self.assertEqual(props_0201_cov["fill_color"], [29, 78, 137, 215])

        # Punkty precyzyjnych zgłoszeń nie obejmują raportów county_aggregate (479 z 513; 34 to county_aggregate)
        pts_all = build_precise_report_points_layer_data(res_all["filtered_reports_df"])
        self.assertEqual(len(pts_all), 479)

        # 3. Zmiana filtrów (okres dat, gatunek obserwacji, pojedynczy powiat) zmienia mapę i liczebność raportów
        res_podgrzybek = filter_real_observations_and_aggregate_by_county(
            real_bundle=real_bundle,
            date_from="2025-09-01",
            date_to="2025-11-02",
            species_filter="podgrzybek_brunatny",
        )
        self.assertEqual(res_podgrzybek["filtered_reports_count"], 114)
        self.assertLess(res_podgrzybek["covered_counties_count"], 28)
        self.assertGreater(res_podgrzybek["no_data_counties_count"], 2)

        res_borowik = filter_real_observations_and_aggregate_by_county(
            real_bundle=real_bundle,
            date_from="2025-09-01",
            date_to="2025-11-02",
            species_filter="borowik_szlachetny",
        )
        self.assertEqual(res_borowik["filtered_reports_count"], 78)

        res_single_day = filter_real_observations_and_aggregate_by_county(
            real_bundle=real_bundle,
            date_from="2025-10-05",
            date_to="2025-10-05",
            species_filter="all",
        )
        self.assertGreater(res_single_day["filtered_reports_count"], 0)
        self.assertLess(res_single_day["filtered_reports_count"], 50)
        self.assertGreater(res_single_day["no_data_counties_count"], 15)

        res_county_0208 = filter_real_observations_and_aggregate_by_county(
            real_bundle=real_bundle,
            date_from="2025-09-01",
            date_to="2025-11-02",
            species_filter="all",
            region_filter="0208",
        )
        self.assertEqual(res_county_0208["in_radius_counties_count"], 1)
        self.assertEqual(res_county_0208["covered_counties_ids"], ["0208"])
        self.assertEqual(res_county_0208["filtered_reports_count"], 24)

        # 5. Weryfikacja zgodności próbki rekordów z tabelami źródłowymi (1:1)
        for sample_rid in ("DG174806", "DG174714", "DG177228", "DG174777"):
            ver = verify_record_against_source_tables(sample_rid, real_bundle)
            self.assertTrue(
                ver["all_fields_match"],
                f"Rekord {sample_rid} ma niezgodność ze źródłem: {ver['comparison_rows']}",
            )
            self.assertEqual(len(ver["comparison_rows"]), 13)
            self.assertTrue(ver["source_url"].startswith("https://www.grzyby.pl/"))

        # Sprawdzenie słownika etykiet filtra gatunku obserwacji
        self.assertIn("mieszane", OBSERVATION_SPECIES_FILTER_LABELS)
        self.assertIn("__NULL__", OBSERVATION_SPECIES_FILTER_LABELS)

    def test_prediction_input_builder_grz15(self) -> None:
        """
        Weryfikuje budowanie wejścia do prognoz [GRZ-15]:
        1. Nazwy, jednostki i typy cech zgadzają się z tabelą treningową (`features.parquet`).
        2. Rekordy zawierają jawne `issued_at` i `target_date` dla horyzontów 1–3 dni.
        3. Przyszła rzeczywista pogoda (`date > issued_at`) nie jest używana w historycznej symulacji.
        4. Niepełna historia zwraca `NULL` zamiast `0.0` i oznacza `PARTIAL_HISTORY`.
        5. Obszar spoza pokrycia (`9999`) jest poprawnie blokowany lub oznaczany jako `OUT_OF_COVERAGE`.
        """
        import json
        import pandas as pd
        from src.features import FEATURE_DICTIONARY
        from src.inference_features import (
            MODEL_INPUT_FEATURE_COLUMNS,
            RegionOutOfCoverageError,
            build_prediction_input_row,
            build_prediction_inputs_batch,
            load_inference_reference_data,
        )

        sample_parquet = PROCESSED_DIR / "prediction_inputs_sample.parquet"
        report_json = PROCESSED_DIR / "prediction_inputs_quality_report.json"
        doc_path = ROOT_DIR / "docs" / "prediction_input_contract.md"

        self.assertTrue(sample_parquet.exists(), "Brak prediction_inputs_sample.parquet")
        self.assertTrue(report_json.exists(), "Brak prediction_inputs_quality_report.json")
        self.assertTrue(doc_path.exists(), "Brak docs/prediction_input_contract.md")

        df_train = pd.read_parquet(PROCESSED_DIR / "features.parquet")
        df_sample = pd.read_parquet(sample_parquet)

        # 1. Nazwy i typy cech zgodne z pipeline'em treningowym
        self.assertEqual(len(df_sample), 90)  # 30 powiatów * 3 horyzonty
        for col in MODEL_INPUT_FEATURE_COLUMNS:
            self.assertIn(col, df_sample.columns)
            self.assertIn(col, df_train.columns)
            self.assertIn(col, FEATURE_DICTIONARY)
            if FEATURE_DICTIONARY[col]["type"] in ("float64", "float64 (nullable)"):
                self.assertEqual(str(df_sample[col].dtype), "float64")

        # 2. Sprawdzenie dat wystawienia i docelowych dla horyzontów 1, 2, 3
        self.assertTrue((df_sample["issued_at"] == "2025-10-18").all())
        self.assertTrue((df_sample["feature_cutoff_date"] == "2025-10-18").all())
        self.assertEqual(
            sorted(df_sample["target_date"].unique().tolist()),
            ["2025-10-19", "2025-10-20", "2025-10-21"],
        )

        # 3. Brak wycieku przyszłej rzeczywistej pogody
        forest_lookup, weather_lookup = load_inference_reference_data()
        truncated_weather = {k: v for k, v in weather_lookup.items() if k[1] <= "2025-10-18"}

        df_full = build_prediction_inputs_batch(
            issued_at="2025-10-18",
            horizons=(1, 2, 3),
            region_ids=("0201", "0208", "0218"),
            forest_lookup=forest_lookup,
            weather_lookup=weather_lookup,
        )
        df_trunc = build_prediction_inputs_batch(
            issued_at="2025-10-18",
            horizons=(1, 2, 3),
            region_ids=("0201", "0208", "0218"),
            forest_lookup=forest_lookup,
            weather_lookup=truncated_weather,
        )
        pd.testing.assert_frame_equal(
            df_full[MODEL_INPUT_FEATURE_COLUMNS],
            df_trunc[MODEL_INPUT_FEATURE_COLUMNS],
        )

        # 4. Niepełna historia (2025-08-10: archiwum od 2025-08-01 -> okno 14d i 21d muszą być NULL)
        row_partial = build_prediction_input_row(
            region_id="0201",
            issued_at="2025-08-10",
            horizon_days=2,
            forest_lookup=forest_lookup,
            weather_lookup=weather_lookup,
        )
        self.assertEqual(row_partial["history_completeness"], "PARTIAL_HISTORY")
        self.assertEqual(row_partial["missing_weather_days_21d"], 11)
        self.assertIsNotNone(row_partial["precip_lag_7d_mm"])
        self.assertIsNone(row_partial["precip_lag_14d_mm"])
        self.assertIsNone(row_partial["precip_lag_21d_mm"])

        # 5. Obszar spoza pokrycia
        with self.assertRaises(RegionOutOfCoverageError):
            build_prediction_input_row(
                region_id="9999",
                issued_at="2025-10-18",
                horizon_days=1,
                forest_lookup=forest_lookup,
                weather_lookup=weather_lookup,
            )

        row_ooc = build_prediction_input_row(
            region_id="9999",
            issued_at="2025-10-18",
            horizon_days=1,
            forest_lookup=forest_lookup,
            weather_lookup=weather_lookup,
            allow_out_of_coverage=True,
        )
        self.assertFalse(row_ooc["is_region_covered"])
        self.assertEqual(row_ooc["history_completeness"], "OUT_OF_COVERAGE")

        # Raport jakości z CLI
        rep = json.loads(report_json.read_text(encoding="utf-8"))
        self.assertEqual(rep["rows_generated"], 90)
        self.assertTrue(rep["edge_case_checks"]["future_weather_leakage_test"]["identical_features_without_future_weather"])


if __name__ == "__main__":
    unittest.main()
