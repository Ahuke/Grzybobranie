#!/usr/bin/env python3
"""
Skrypt dla zadania [GRZ-08]:
Połączenie obserwacji ze strukturą przestrzenną regionu bez nadawania pozornej dokładności
oraz przygotowanie tabeli cech leśnych (forest_features) i warstwy obszarów (areas) z kluczem region_id.

Warunki ukończenia [GRZ-08]:
1. Raport o powiecie nie jest sztucznie zamieniany w punkt wskazujący konkretny las
   (jawna klasyfikacja: 'point_precise' vs 'county_aggregate' z flagą point_represents_specific_forest=False).
2. Niejednoznaczne i nieprzypisane lokalizacje są policzone i opisane w raporcie jakości.
3. Cechy leśne mają jawny sposób agregacji oraz jednostki ([%], [lata], [km²]).
4. Geometrie mają zgodny CRS (EPSG:2180 dla metrycznych odległości/powierzchni, EPSG:4326 dla GeoJSON/PyDeck).
5. Przygotowano areas i forest_features z kluczem region_id, bez mnożenia raportów przez liczbę poligonów
   (niezmiennik: dokładnie 1 wiersz wyjściowy na 1 raport, współczynnik multiplikacji = 1.0).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
SAMPLE_DIR = ROOT_DIR / "data" / "sample"
DOCS_DIR = ROOT_DIR / "docs"

AREAS_GEOJSON_PATH = PROCESSED_DIR / "areas.geojson"
REPORTS_PARQUET_PATH = PROCESSED_DIR / "reports.parquet"
REPORTS_CSV_PATH = PROCESSED_DIR / "reports.csv"

FOREST_FEATURES_PARQUET_PATH = PROCESSED_DIR / "forest_features.parquet"
FOREST_FEATURES_CSV_PATH = PROCESSED_DIR / "forest_features.csv"
REPORTS_SPATIAL_PARQUET_PATH = PROCESSED_DIR / "reports_spatial.parquet"
REPORTS_SPATIAL_CSV_PATH = PROCESSED_DIR / "reports_spatial.csv"
SPATIAL_QUALITY_REPORT_JSON_PATH = PROCESSED_DIR / "spatial_quality_report.json"

# Wzorzec identyfikujący raporty o charakterze ogólnopowiatowym / uogólnionym
COUNTY_LEVEL_REGEX = re.compile(
    r"^(?:pow\.\s*|woj\.\s*)[a-ząćęłńóśźż]+$", re.IGNORECASE
)


def _is_county_level_report(location_raw: str | None) -> bool:
    """Sprawdza, czy raport jest zgłoszeniem ogólnopowiatowym (bez nazwy miejscowości/lasu)."""
    if not location_raw:
        return False
    loc_clean = str(location_raw).strip()
    return bool(COUNTY_LEVEL_REGEX.match(loc_clean))


def build_forest_features(areas_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Przygotowuje tabelę cech leśnych zagregowanych do poziomu jednostki przestrzennej (region_id).
    Wymusza jawną obecność kluczy region_id oraz area_id, jednostek i metod agregacji.
    """
    records: list[dict[str, Any]] = []
    for _, row in areas_gdf.iterrows():
        reg_id = str(row["area_id"]).zfill(4)
        reg_name = str(row["area_name"])
        total_km2 = float(row["total_area_km2"])
        forest_pct = float(row["forest_cover_pct"]) if pd.notna(row["forest_cover_pct"]) else None
        forest_area_km2 = round(total_km2 * (forest_pct / 100.0), 2) if forest_pct is not None else None

        pine_pct = float(row["pine_share_pct"]) if pd.notna(row["pine_share_pct"]) else None
        oak_beech_pct = (
            float(row["oak_beech_share_pct"]) if pd.notna(row["oak_beech_share_pct"]) else None
        )
        spruce_pct = float(row["spruce_share_pct"]) if pd.notna(row["spruce_share_pct"]) else None
        mean_age = (
            float(row["mean_stand_age_years"]) if pd.notna(row["mean_stand_age_years"]) else None
        )
        moisture = (
            str(row["dominant_moisture_type"]).strip()
            if pd.notna(row["dominant_moisture_type"]) and row["dominant_moisture_type"] is not None
            else None
        )
        has_bdl = bool(row["has_bdl_stand_data"])
        stands_count = int(row["bdl_sample_stands_count"]) if pd.notna(row["bdl_sample_stands_count"]) else 0

        agg_method = (
            "bdl_stand_weighted_mean_epsg2180"
            if has_bdl
            else "gus_bdl_official_forestry_prg"
        )

        records.append(
            {
                "region_id": reg_id,
                "area_id": reg_id,
                "region_name": reg_name,
                "voivodeship_id": "02",
                "total_area_km2": total_km2,
                "forest_cover_pct": forest_pct,
                "forest_area_km2": forest_area_km2,
                "pine_share_pct": pine_pct,
                "oak_beech_share_pct": oak_beech_pct,
                "spruce_share_pct": spruce_pct,
                "mean_stand_age_years": mean_age,
                "dominant_moisture_type": moisture,
                "has_bdl_stand_data": has_bdl,
                "bdl_sample_stands_count": stands_count,
                "aggregation_method": agg_method,
                "crs_metric": "EPSG:2180",
                "crs_display": "EPSG:4326",
            }
        )

    records.sort(key=lambda r: r["region_id"])
    df = pd.DataFrame(records)
    return df, records


def build_spatial_observations(
    reports_df: pd.DataFrame,
    areas_gdf: gpd.GeoDataFrame,
    forest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Łączy raporty ze strukturą przestrzenną bez nadawania pozornej dokładności
    i bez mnożenia raportów przez liczbę poligonów (1 raport -> dokładnie 1 rekord).
    """
    areas_lookup = {
        str(r["area_id"]).zfill(4): {
            "region_name": str(r["area_name"]),
            "centroid_lon": float(r["centroid_lon"]),
            "centroid_lat": float(r["centroid_lat"]),
            "forest_cover_pct": float(r["forest_cover_pct"]) if pd.notna(r["forest_cover_pct"]) else None,
            "pine_share_pct": float(r["pine_share_pct"]) if pd.notna(r["pine_share_pct"]) else None,
            "oak_beech_share_pct": (
                float(r["oak_beech_share_pct"]) if pd.notna(r["oak_beech_share_pct"]) else None
            ),
            "spruce_share_pct": float(r["spruce_share_pct"]) if pd.notna(r["spruce_share_pct"]) else None,
            "mean_stand_age_years": (
                float(r["mean_stand_age_years"]) if pd.notna(r["mean_stand_age_years"]) else None
            ),
            "dominant_moisture_type": (
                str(r["dominant_moisture_type"]) if pd.notna(r["dominant_moisture_type"]) else None
            ),
            "has_bdl_stand_data": bool(r["has_bdl_stand_data"]),
        }
        for _, r in areas_gdf.iterrows()
    }

    # Transformacja powiatów i punktów do układu metrycznego EPSG:2180 w celu wyliczenia odległości
    areas_2180 = areas_gdf.set_index("area_id").to_crs("EPSG:2180")
    centroids_2180 = areas_2180.geometry.centroid

    points_gdf = gpd.GeoDataFrame(
        reports_df,
        geometry=[Point(float(r["lon"]), float(r["lat"])) for _, r in reports_df.iterrows()],
        crs="EPSG:4326",
    ).to_crs("EPSG:2180")

    spatial_records: list[dict[str, Any]] = []
    county_agg_count = 0
    point_precise_count = 0
    border_resolved_count = 0
    unassigned_count = 0

    for idx, row in reports_df.iterrows():
        rid = str(row["report_id"])
        reg_id = str(row["area_id"]).zfill(4)
        loc_raw = str(row.get("location_raw") or "").strip()
        quality_status = str(row.get("quality_status") or "VALID")
        quality_reason = str(row.get("quality_reason") or "ok")
        is_valid = bool(row.get("is_valid_for_training", True))

        if quality_status == "VALID_BORDER_LOCALITY_RESOLVED":
            border_resolved_count += 1

        if not reg_id or reg_id not in areas_lookup:
            unassigned_count += 1
            area_meta = {}
            dist_km = None
        else:
            area_meta = areas_lookup[reg_id]
            # Odległość metryczna punktu od geometrycznego środka powiatu (w kilometrach)
            pt_geom = points_gdf.geometry.iloc[idx]
            cnt_geom = centroids_2180.loc[reg_id]
            dist_km = round(float(pt_geom.distance(cnt_geom)) / 1000.0, 2)

        is_county_agg = _is_county_level_report(loc_raw)
        if is_county_agg:
            county_agg_count += 1
            spatial_precision = "county_aggregate"
            point_represents_specific_forest = False
            precision_note = (
                "Raport podany wyłącznie na poziomie powiatu; współrzędne reprezentują "
                "umowny punkt i nie wskazują konkretnego wydzielenia leśnego. "
                "Cechy siedliskowe przypisano z agregacji powiatowej (forest_features)."
            )
        else:
            point_precise_count += 1
            spatial_precision = "point_precise"
            point_represents_specific_forest = True
            precision_note = (
                "Raport ze zidentyfikowaną miejscowością lub leśnictwem; współrzędne "
                "wskazują konkretny punkt terenu przypisany przestrzennie w EPSG:2180."
            )

        spatial_records.append(
            {
                "report_id": rid,
                "date": str(row["date"]),
                "harvest_date": str(row["harvest_date"]),
                "publication_date": str(row["publication_date"]),
                "region_id": reg_id,
                "area_id": reg_id,
                "region_name": area_meta.get("region_name", ""),
                "location_raw": loc_raw,
                "place_id": str(row.get("place_id", "")),
                "spatial_precision": spatial_precision,
                "is_county_aggregate": is_county_agg,
                "point_represents_specific_forest": point_represents_specific_forest,
                "spatial_precision_note": precision_note,
                "yield_pcs_per_h": (
                    float(row["yield_pcs_per_h"]) if pd.notna(row["yield_pcs_per_h"]) else None
                ),
                "dominant_species": (
                    str(row["dominant_species"])
                    if pd.notna(row["dominant_species"]) and row["dominant_species"] is not None
                    else None
                ),
                "harvest_duration_h": (
                    float(row["harvest_duration_h"]) if pd.notna(row["harvest_duration_h"]) else None
                ),
                "total_pieces": (
                    int(row["total_pieces"]) if pd.notna(row["total_pieces"]) else None
                ),
                "worminess_pct": (
                    float(row["worminess_pct"]) if pd.notna(row["worminess_pct"]) else None
                ),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "centroid_lat": area_meta.get("centroid_lat"),
                "centroid_lon": area_meta.get("centroid_lon"),
                "distance_to_county_centroid_km": dist_km,
                "forest_cover_pct": area_meta.get("forest_cover_pct"),
                "pine_share_pct": area_meta.get("pine_share_pct"),
                "oak_beech_share_pct": area_meta.get("oak_beech_share_pct"),
                "spruce_share_pct": area_meta.get("spruce_share_pct"),
                "mean_stand_age_years": area_meta.get("mean_stand_age_years"),
                "dominant_moisture_type": area_meta.get("dominant_moisture_type"),
                "has_bdl_stand_data": area_meta.get("has_bdl_stand_data", False),
                "quality_status": quality_status,
                "is_valid_for_training": is_valid,
                "quality_reason": quality_reason,
                "source_url": str(row.get("source_url", "")),
                "is_demo": False,
            }
        )

    out_df = pd.DataFrame(spatial_records)
    # Weryfikacja niezmiennika: dokładnie tyle samo wierszy co na wejściu (współczynnik multiplikacji = 1.0)
    assert len(out_df) == len(reports_df), (
        f"Błąd multiplikacji: wejście={len(reports_df)}, wyjście={len(out_df)}"
    )

    metrics = {
        "total_reports_processed": len(out_df),
        "point_precise_reports_count": point_precise_count,
        "county_aggregate_reports_count": county_agg_count,
        "border_locality_resolved_count": border_resolved_count,
        "unassigned_reports_count": unassigned_count,
        "multiplication_factor": round(float(len(out_df)) / float(len(reports_df)), 4),
        "valid_for_training_count": int(out_df["is_valid_for_training"].sum()),
    }
    return out_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Łączenie obserwacji z geometrią i przygotowanie forest_features ([GRZ-08])"
    )
    parser.add_argument(
        "--areas-geojson",
        type=Path,
        default=AREAS_GEOJSON_PATH,
        help="Ścieżka do data/processed/areas.geojson",
    )
    parser.add_argument(
        "--reports-parquet",
        type=Path,
        default=REPORTS_PARQUET_PATH,
        help="Ścieżka do data/processed/reports.parquet",
    )
    args = parser.parse_args()

    print("=== Rozpoczęcie potoku przetwarzania przestrzennego [GRZ-08] ===")
    areas_gdf = gpd.read_file(args.areas_geojson)
    print(f"Wczytano {len(areas_gdf)} jednostek powiatowych z: {args.areas_geojson}")

    # 1. Zapewnij jawną obecność 'region_id' w properties pliku areas.geojson
    areas_geojson_raw = json.loads(args.areas_geojson.read_text(encoding="utf-8"))
    for feature in areas_geojson_raw.get("features", []):
        props = feature.setdefault("properties", {})
        aid = str(props.get("area_id") or "").zfill(4)
        props["region_id"] = aid
        props["area_id"] = aid
    args.areas_geojson.write_text(
        json.dumps(areas_geojson_raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Zaktualizowano {args.areas_geojson} o klucz 'region_id'")

    # 2. Przygotuj tabelę forest_features (Parquet + CSV)
    forest_df, forest_records = build_forest_features(areas_gdf)
    FOREST_FEATURES_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    forest_df.to_parquet(FOREST_FEATURES_PARQUET_PATH, index=False, engine="pyarrow")

    csv_forest_cols = [
        "region_id",
        "area_id",
        "region_name",
        "voivodeship_id",
        "total_area_km2",
        "forest_cover_pct",
        "forest_area_km2",
        "pine_share_pct",
        "oak_beech_share_pct",
        "spruce_share_pct",
        "mean_stand_age_years",
        "dominant_moisture_type",
        "has_bdl_stand_data",
        "bdl_sample_stands_count",
        "aggregation_method",
        "crs_metric",
    ]
    with FOREST_FEATURES_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_forest_cols)
        writer.writeheader()
        for r in forest_records:
            writer.writerow({k: (r[k] if r[k] is not None else "") for k in csv_forest_cols})
    print(f"Zapisano tabelę cech leśnych ({len(forest_df)} wierszy):")
    print(f" - {FOREST_FEATURES_PARQUET_PATH}")
    print(f" - {FOREST_FEATURES_CSV_PATH}")

    # 3. Wczytaj raporty z [GRZ-07] i połącz bez nadawania pozornej dokładności
    reports_df = pd.read_parquet(args.reports_parquet)
    print(f"Wczytano {len(reports_df)} obserwacji z: {args.reports_parquet}")

    spatial_df, metrics = build_spatial_observations(
        reports_df=reports_df,
        areas_gdf=areas_gdf,
        forest_df=forest_df,
    )
    spatial_df.to_parquet(REPORTS_SPATIAL_PARQUET_PATH, index=False, engine="pyarrow")

    # Zapisz wersję CSV dla rekordów ważnych treningowo (513 wierszy)
    valid_spatial = spatial_df[spatial_df["is_valid_for_training"]].copy()
    valid_spatial.to_csv(REPORTS_SPATIAL_CSV_PATH, index=False, encoding="utf-8")
    print(f"Zapisano tabelę przestrzenną obserwacji ({len(spatial_df)} wierszy, {len(valid_spatial)} VALID):")
    print(f" - {REPORTS_SPATIAL_PARQUET_PATH}")
    print(f" - {REPORTS_SPATIAL_CSV_PATH}")

    # 4. Wygeneruj raport jakości i zgodności przestrzennej
    quality_payload = {
        "task": "[GRZ-08] Połączenie obserwacji z geometrią bez pozornej dokładności",
        "spatial_key": "region_id (TERYT 4 znaki, np. 0201)",
        "spatial_key_alias": "area_id",
        "crs_rules": {
            "metric_calculations": "EPSG:2180 (PUWG 1992 - odległości w metrach, pole powierzchni)",
            "storage_and_pydeck": "EPSG:4326 (WGS84 - stopnie dziesiętne lon/lat)",
        },
        "multiplication_invariants": {
            "input_reports_count": len(reports_df),
            "output_spatial_reports_count": len(spatial_df),
            "multiplication_factor": metrics["multiplication_factor"],
            "has_1_to_1_integrity": bool(len(reports_df) == len(spatial_df)),
        },
        "precision_breakdown": {
            "point_precise_count": metrics["point_precise_reports_count"],
            "county_aggregate_count": metrics["county_aggregate_reports_count"],
            "county_aggregate_pct": round(
                100.0 * metrics["county_aggregate_reports_count"] / len(reports_df), 2
            ),
            "border_locality_resolved_count": metrics["border_locality_resolved_count"],
            "unassigned_count": metrics["unassigned_reports_count"],
        },
        "forest_features_units": {
            "total_area_km2": "km² (powierzchnia powiatu z geometrii PRG)",
            "forest_cover_pct": "% (lesistość powiatu z GUS BDL)",
            "forest_area_km2": "km² (powierzchnia lasów w powiecie)",
            "pine_share_pct": "% (udział drzewostanów z panującą sosną SO w BDL)",
            "oak_beech_share_pct": "% (udział drzewostanów z dębem DB lub bukiem BK w BDL)",
            "spruce_share_pct": "% (udział drzewostanów ze świerkiem ŚW w BDL)",
            "mean_stand_age_years": "lata (średni ważony wiek drzewostanów D-STAN)",
            "dominant_moisture_type": "enum: suche, swieze, wilgotne, bagienne",
        },
        "regional_coverage": {
            "regions_count": len(forest_df),
            "regions_with_bdl_profile": int(forest_df["has_bdl_stand_data"].sum()),
            "regions_without_bdl_profile": int((~forest_df["has_bdl_stand_data"]).sum()),
        },
    }

    SPATIAL_QUALITY_REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPATIAL_QUALITY_REPORT_JSON_PATH.write_text(
        json.dumps(quality_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Zapisano raport jakości: {SPATIAL_QUALITY_REPORT_JSON_PATH}")

    print("\n=== Podsumowanie [GRZ-08] ===")
    print(f"- Obserwacje ogółem:                      {metrics['total_reports_processed']}")
    print(f"- Raporty punktowe (point_precise):       {metrics['point_precise_reports_count']}")
    print(f"- Raporty powiatowe (county_aggregate):   {metrics['county_aggregate_reports_count']} (brak pozornej dokładności)")
    print(f"- Rozwiązane graniczne:                   {metrics['border_locality_resolved_count']}")
    print(f"- Nieprzypisane lokalizacje:              {metrics['unassigned_reports_count']}")
    print(f"- Współczynnik multiplikacji:             {metrics['multiplication_factor']} (brak duplikacji wierszy)")


if __name__ == "__main__":
    main()
