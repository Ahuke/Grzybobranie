"""
Punkt wejścia aplikacji Streamlit: "Gdzie na grzyby?" ([GRZ-02], [GRZ-05], [GRZ-12]).

Uruchomienie:
    streamlit run app.py
"""

from __future__ import annotations

import datetime
import os
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.config import (
    ABUNDANCE_CLASSES,
    ACADEMIC_ATTRIBUTION,
    OBSERVATION_SPECIES_FILTER_LABELS,
    PILOT_REGION,
    REFERENCE_CITIES,
    SPECIES_HABITAT_HINTS,
    SPECIES_LABELS,
)
from src.data_loader import (
    DatasetBundle,
    RealObservationsBundle,
    filter_real_observations_and_aggregate_by_county,
    load_dataset_bundle,
    load_real_observations_bundle,
    load_sprint1_audit_samples,
    verify_record_against_source_tables,
)
from src.geo_utils import (
    build_precise_report_points_layer_data,
    enrich_bdl_stands_geojson,
    enrich_geojson_with_predictions,
    enrich_geojson_with_real_observations,
    haversine_km,
)


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1380px;
    }
    .app-header {
        border-bottom: 2px solid #2C5530;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1F2421;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #4A524D;
        margin-top: 0.25rem;
    }
    .status-banner {
        padding: 0.65rem 1rem;
        border-left: 4px solid #2C5530;
        background-color: #EFECE6;
        color: #1F2421;
        font-size: 0.86rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.75rem;
    }
    .status-banner-warn {
        border-left-color: #C98E2A;
        background-color: #F7EFE0;
    }
    .context-strip {
        padding: 0.6rem 1rem;
        background-color: #FFFFFF;
        border: 1px solid #DCD9D0;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #1F2421;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #DCD9D0;
        border-radius: 4px;
        padding: 0.75rem 1rem;
    }
    .legend-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 0.75rem;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 12px;
        background: #FFFFFF;
        border: 1px solid #DCD9D0;
        border-radius: 4px;
        font-size: 0.83rem;
        font-weight: 500;
        color: #1F2421;
    }
    .legend-swatch {
        width: 13px;
        height: 13px;
        border-radius: 2px;
        display: inline-block;
    }
    .footer-note {
        font-size: 0.78rem;
        color: #5A625D;
        border-top: 1px solid #DCD9D0;
        padding-top: 0.85rem;
        margin-top: 2rem;
    }
</style>
"""


def build_ranking_records(
    bundle: DatasetBundle,
    target_date: str,
    species_filter: str,
    origin_lat: float,
    origin_lon: float,
    radius_km: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Łączy predykcje dla wybranej daty i gatunku z cechami siedliskowymi powiatów (areas)
    oraz najnowszym dostępnym rekordem pogodowym (weather) i wylicza odległość od punktu startowego.
    Zachowane dla kompatybilności wstecznej z trybem demonstracyjnym ([GRZ-02], [GRZ-05]).
    """
    preds_filtered = [
        p
        for p in bundle.predictions
        if p["target_date"] == target_date and p["species_filter"] == species_filter
    ]
    preds_by_area = {p["area_id"]: p for p in preds_filtered}

    weather_by_area: dict[str, dict[str, Any]] = {}
    for w in sorted(bundle.weather, key=lambda x: x["date"]):
        if w["date"] <= target_date:
            weather_by_area[w["area_id"]] = w

    records: list[dict[str, Any]] = []
    records_by_area: dict[str, dict[str, Any]] = {}

    for area_id, area_props in bundle.areas_by_id.items():
        c_lat = float(area_props["centroid_lat"])
        c_lon = float(area_props["centroid_lon"])
        dist_km = haversine_km(origin_lat, origin_lon, c_lat, c_lon)
        in_radius = dist_km <= radius_km

        pred = preds_by_area.get(area_id, {})
        w_rec = weather_by_area.get(area_id, {})

        record = {
            "area_id": area_id,
            "area_name": area_props.get("area_name", area_id),
            "total_area_km2": area_props.get("total_area_km2"),
            "distance_km": dist_km,
            "in_radius": in_radius,
            "predicted_yield_pcs_per_h": pred.get("predicted_yield_pcs_per_h"),
            "abundance_class": pred.get("abundance_class", "BRAK_DANYCH"),
            "confidence_score": pred.get("confidence_score"),
            "habitat_match_note": pred.get("habitat_match_note"),
            "generated_at": pred.get("generated_at", "—"),
            "model_version": pred.get("model_version", "—"),
            "forest_cover_pct": area_props.get("forest_cover_pct"),
            "pine_share_pct": area_props.get("pine_share_pct"),
            "oak_beech_share_pct": area_props.get("oak_beech_share_pct"),
            "spruce_share_pct": area_props.get("spruce_share_pct"),
            "mean_stand_age_years": area_props.get("mean_stand_age_years"),
            "dominant_moisture_type": area_props.get("dominant_moisture_type"),
            "has_bdl_stand_data": bool(
                area_props.get("has_bdl_stand_data")
                or (
                    area_props.get("pine_share_pct") is not None
                    and area_props.get("mean_stand_age_years") is not None
                )
            ),
            "bdl_sample_stands_count": int(area_props.get("bdl_sample_stands_count") or 0),
            "precip_lag_7d_mm": w_rec.get("precip_lag_7d_mm"),
            "precip_lag_14d_mm": w_rec.get("precip_lag_14d_mm"),
            "temp_min_night_c": w_rec.get("temp_min_night_c"),
            "soil_moisture_0_7cm_m3m3": w_rec.get("soil_moisture_0_7cm_m3m3"),
            "is_demo": bool(pred.get("is_demo", area_props.get("is_demo", True))),
        }
        records.append(record)
        records_by_area[area_id] = record

    records.sort(
        key=lambda r: (
            r["in_radius"],
            r["predicted_yield_pcs_per_h"] if r["predicted_yield_pcs_per_h"] is not None else -1.0,
            r["forest_cover_pct"] if r["forest_cover_pct"] is not None else -1.0,
            -r["distance_km"],
        ),
        reverse=True,
    )
    return records, records_by_area


def render_legend(
    color_mode: str = "observed_yield",
    show_bdl_stands: bool = False,
    show_report_points: bool = False,
) -> None:
    """Wyświetla legendę warstwy powiatów, punktów zgłoszeń oraz wydzieleń BDL."""
    items_html = []
    if color_mode == "region_bdl":
        forest_items = [
            ("#225C34", "Lesistość ≥ 45% (wysoka)"),
            ("#448452", "Lesistość 30 – 45% (umiarkowana)"),
            ("#7AA874", "Lesistość 18 – 30% (średnia)"),
            ("#B2C6A5", "Lesistość < 18% lub brak paczki BDL (NULL)"),
        ]
        for hex_col, label in forest_items:
            items_html.append(
                f"<div class='legend-item'>"
                f"<span class='legend-swatch' style='background:{hex_col};'></span>"
                f"<span>{label}</span>"
                f"</div>"
            )
    elif color_mode == "report_coverage":
        coverage_items = [
            ("#1D4E89", "Wysokie pokrycie (≥ 20 raportów w powiecie)"),
            ("#3A7CA5", "Średnie pokrycie (5 – 19 raportów w powiecie)"),
            ("#81B2CD", "Niskie pokrycie (1 – 4 raporty w powiecie)"),
            ("#B4B4B4", "Brak danych (0 raportów — NULL, nie 0.0 szt./h)"),
        ]
        for hex_col, label in coverage_items:
            items_html.append(
                f"<div class='legend-item'>"
                f"<span class='legend-swatch' style='background:{hex_col};'></span>"
                f"<span>{label}</span>"
                f"</div>"
            )
    else:
        for key in ["WYSYP", "SREDNIO", "BRAK", "BRAK_DANYCH"]:
            info = ABUNDANCE_CLASSES[key]
            items_html.append(
                f"<div class='legend-item'>"
                f"<span class='legend-swatch' style='background:{info['hex']};'></span>"
                f"<span>{info['label']}</span>"
                f"</div>"
            )

    if show_report_points:
        items_html.append(
            "<div class='legend-item'>"
            "<span class='legend-swatch' style='background:#1F2421;border-radius:50%;'></span>"
            "<span>Punkt zgłoszenia z potwierdzoną miejscowością (point_precise)</span>"
            "</div>"
        )

    if show_bdl_stands:
        bdl_items = [
            ("#2E8B57", "BDL: Sosna (SO)"),
            ("#1B4F72", "BDL: Świerk (ŚW)"),
            ("#B76E22", "BDL: Buk / Dąb (BK/DB)"),
            ("#8C8C87", "BDL: Użytek nieleśny"),
        ]
        for hex_col, label in bdl_items:
            items_html.append(
                f"<div class='legend-item'>"
                f"<span class='legend-swatch' style='background:{hex_col};'></span>"
                f"<span>{label}</span>"
                f"</div>"
            )

    st.markdown(
        f"<div class='legend-bar'>{''.join(items_html)}</div>",
        unsafe_allow_html=True,
    )


def render_real_observations_view(
    real_bundle: RealObservationsBundle,
    audit_samples: dict[str, Any],
) -> None:
    """
    Główny widok eksploracji rzeczywistych raportów i pokrycia danych przed modelem ([GRZ-12]).
    """
    min_d = datetime.date.fromisoformat(real_bundle.min_date)
    max_d = datetime.date.fromisoformat(real_bundle.max_date)

    # --- PANEL BOCZNY (FILTRY OBSERWACJI [GRZ-12]) ---
    with st.sidebar:
        st.divider()
        st.subheader("1. Wybór okresu / daty obserwacji")

        date_filter_mode = st.radio(
            "Tryb filtrowania daty",
            options=["range", "single_day"],
            format_func=lambda m: (
                "Zakres dat / podokres sezonu" if m == "range" else "Pojedynczy dzień zbioru"
            ),
            horizontal=False,
        )

        if date_filter_mode == "range":
            preset_options = {
                "all_season": f"Cały okres danych ({real_bundle.min_date} .. {real_bundle.max_date})",
                "sep_2025": "Wrzesień 2025 (2025-09-01 .. 2025-09-30)",
                "oct_early": "Wczesny październik (2025-10-01 .. 2025-10-15)",
                "oct_late": "Późny październik i listopad (2025-10-16 .. 2025-11-02)",
                "custom": "Własny zakres z kalendarza",
            }
            selected_preset = st.selectbox(
                "Predefiniowany podokres",
                options=list(preset_options.keys()),
                format_func=lambda k: preset_options[k],
                index=0,
            )
            preset_bounds = {
                "all_season": (min_d, max_d),
                "sep_2025": (datetime.date(2025, 9, 1), datetime.date(2025, 9, 30)),
                "oct_early": (datetime.date(2025, 10, 1), datetime.date(2025, 10, 15)),
                "oct_late": (datetime.date(2025, 10, 16), max_d),
            }
            if selected_preset in preset_bounds:
                default_start, default_end = preset_bounds[selected_preset]
            else:
                default_start, default_end = (min_d, max_d)

            picked_dates = st.date_input(
                "Zakres dat zbioru (`observation_date`)",
                value=(default_start, default_end),
                min_value=min_d,
                max_value=max_d,
            )
            if isinstance(picked_dates, (tuple, list)) and len(picked_dates) == 2:
                date_from = picked_dates[0].isoformat()
                date_to = picked_dates[1].isoformat()
            elif isinstance(picked_dates, (tuple, list)) and len(picked_dates) == 1:
                date_from = picked_dates[0].isoformat()
                date_to = picked_dates[0].isoformat()
            else:
                date_from = picked_dates.isoformat()  # type: ignore[union-attr]
                date_to = date_from
        else:
            avail_dates = sorted(
                real_bundle.features_df["observation_date"].astype(str).unique().tolist(),
                reverse=True,
            )
            selected_single = st.selectbox(
                "Wybierz dzień z raportami (`observation_date`)",
                options=avail_dates,
                index=0,
                help="Lista dni w sezonie jesiennym 2025 posiadających co najmniej jeden raport.",
            )
            date_from = selected_single
            date_to = selected_single

        st.divider()
        st.subheader("2. Filtr gatunku (obserwacje)")
        selected_species = st.selectbox(
            "Deklarowany gatunek dominujący (`dominant_species`)",
            options=list(OBSERVATION_SPECIES_FILTER_LABELS.keys()),
            format_func=lambda k: OBSERVATION_SPECIES_FILTER_LABELS.get(k, k),
            index=0,
            help=(
                "Na etapie [GRZ-12] filtr gatunku filtruje rzeczywiste raporty terenowe z grzyby.pl "
                "(nie uruchamia osobnego modelu ML dla gatunku)."
            ),
        )
        st.caption(
            f"Kontekst siedliskowy BDL: {SPECIES_HABITAT_HINTS.get(selected_species, '—')}"
        )

        st.divider()
        st.subheader("3. Jednostka przestrzenna i odległość")
        county_options = {"ALL": "Wszystkie powiaty (30 jednostek TERYT 02)"}
        for aid, aprops in sorted(
            real_bundle.areas_by_id.items(), key=lambda kv: kv[1].get("area_name", kv[0])
        ):
            county_options[aid] = f"{aprops.get('area_name', aid)} (TERYT {aid})"

        selected_county = st.selectbox(
            "Filtr jednostki przestrzennej (powiat)",
            options=list(county_options.keys()),
            format_func=lambda k: county_options[k],
            index=0,
        )

        selected_city = st.selectbox(
            "Punkt odniesienia odległości",
            options=list(REFERENCE_CITIES.keys()),
            index=0,
        )
        origin_lat, origin_lon = REFERENCE_CITIES[selected_city]

        radius_km = st.slider(
            "Promień od punktu [km]",
            min_value=20,
            max_value=160,
            value=160,
            step=10,
            help="Odległość w linii prostej (Haversine) do centroidu powiatu.",
        )

        st.divider()
        st.caption(
            f"Konfiguracja regionu:\n"
            f"TERYT: {PILOT_REGION.voivodeship_id} ({PILOT_REGION.voivodeship_name})\n"
            f"Jednostka: {PILOT_REGION.spatial_unit_name} ({PILOT_REGION.spatial_key_name})\n"
            f"Układ mapy: {PILOT_REGION.crs_display} | Metryczny: {PILOT_REGION.crs_metric}\n"
            f"Strefa: {PILOT_REGION.timezone}"
        )

    # --- FILTROWANIE I AGREGACJA RZECZYWISTYCH DANYCH ---
    filter_res = filter_real_observations_and_aggregate_by_county(
        real_bundle=real_bundle,
        date_from=date_from,
        date_to=date_to,
        species_filter=selected_species,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        radius_km=float(radius_km),
        region_filter=selected_county,
    )
    filtered_df: pd.DataFrame = filter_res["filtered_reports_df"]
    county_records: list[dict[str, Any]] = filter_res["county_records"]
    county_by_id: dict[str, dict[str, Any]] = filter_res["county_by_id"]

    in_filter_counties = [c for c in county_records if c["in_radius"]]
    counties_with_data = [c for c in in_filter_counties if c["has_reports"]]
    counties_no_data = [c for c in in_filter_counties if not c["has_reports"]]

    # --- PASEK STATUSU I WIDOCZNE METADANE (OKRES, JEDNOSTKA PRZESTRZENNA, FILTRY) ---
    st.markdown(
        f"<div class='status-banner'><b>Źródło danych ([GRZ-12]):</b> {real_bundle.status_message}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='context-strip'>"
        f"<b>Pełny okres zbioru danych:</b> <code>{real_bundle.min_date} – {real_bundle.max_date}</code> (63 dni sezonu jesiennego 2025) "
        f"&nbsp;|&nbsp; <b>Aktywny filtr okresu:</b> <code>{date_from} – {date_to}</code> "
        f"&nbsp;|&nbsp; <b>Jednostka przestrzenna:</b> <code>powiat (4-znakowy kod TERYT 0201–0265, N=30)</code> "
        f"&nbsp;|&nbsp; <b>Filtr gatunku (obserwacje):</b> <code>{OBSERVATION_SPECIES_FILTER_LABELS.get(selected_species, selected_species)}</code>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- KARTY PODSUMOWANIA (DYNAMICZNE POD WPŁYWEM FILTRÓW) ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Dostępne raporty (po filtracji)",
        f"{len(filtered_df)} z {len(real_bundle.features_df)}",
        delta=f"Okres: {date_from} .. {date_to}",
        delta_color="off",
    )
    kpi2.metric(
        "Pokrycie powiatów danymi",
        f"{len(counties_with_data)} z {len(in_filter_counties)} w filtrze",
        delta=f"Brak danych (NULL): {len(counties_no_data)} pow.",
        delta_color="off",
    )
    if not filtered_df.empty and counties_with_data:
        mean_y = float(filtered_df["yield_pcs_per_h"].mean())
        med_y = float(filtered_df["yield_pcs_per_h"].median())
        zero_cnt = int((filtered_df["yield_pcs_per_h"] == 0.0).sum())
        kpi3.metric(
            "Średni zbiór z raportów",
            f"{mean_y:.1f} szt./h",
            delta=f"Mediana: {med_y:.1f} szt./h | Puste koszyki (0/h): {zero_cnt}",
            delta_color="off",
        )
        top_county = max(counties_with_data, key=lambda c: int(c["reports_count"]))
        kpi4.metric(
            "Najwięcej raportów w filtrze",
            f"{top_county['reports_count']} rap.",
            delta=f"{top_county['area_name']} ({top_county['area_id']})",
            delta_color="off",
        )
    else:
        kpi3.metric(
            "Średni zbiór z raportów",
            "BRAK DANYCH (NULL)",
            delta="0 raportów spełnia kryteria filtru",
            delta_color="off",
        )
        kpi4.metric(
            "Najwięcej raportów w filtrze",
            "0 raportów",
            delta="Brak obserwacji w zadanym filtrze",
            delta_color="off",
        )

    # --- ZAKŁADKI GŁÓWNE ([GRZ-12]) ---
    tab_map, tab_records, tab_methodology = st.tabs(
        [
            "Mapa pokrycia danych i obserwowanych zbiorów",
            "Próbka rekordów i weryfikacja zgodności z tabelami źródłowymi",
            "Metodologia, kontrakt danych i wyniki EDA",
        ]
    )

    with tab_map:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([4, 2, 2])
        with ctrl_col1:
            color_mode = st.radio(
                "Tryb kolorowania powiatów na mapie",
                options=["observed_yield", "report_coverage", "region_bdl"],
                format_func=lambda m: {
                    "observed_yield": "Obserwowana wydajność zbiorów [szt./h] vs Brak danych",
                    "report_coverage": "Pokrycie danymi — liczba raportów w powiecie",
                    "region_bdl": "Granice regionu i lesistość BDL ([GRZ-05])",
                }[m],
                horizontal=True,
            )
        with ctrl_col2:
            show_report_points = st.checkbox(
                "Pokaż punkty zgłoszeń (`point_precise`)",
                value=True,
                help=(
                    "Wyświetla punkty wyłącznie dla raportów ze zweryfikowaną miejscowością/lasem "
                    "(`point_represents_specific_forest == True`), bez pozornej dokładności dla raportów ogólnopowiatowych."
                ),
            )
        with ctrl_col3:
            bdl_available = bool(audit_samples.get("bdl_stands_geojson"))
            show_bdl_stands = st.checkbox(
                "Nałóż wydzielenia BDL (N=105)",
                value=False,
                help="Warstwa wektorowa 105 wydzieleń leśnych BDL w 7 klastrach na Dolnym Śląsku.",
            )

        st.markdown(
            f"#### Mapa jednostek przestrzennych (`powiat`, TERYT `0201–0265`) — "
            f"okres `{date_from}` do `{date_to}` | "
            f"filtr gatunku: {OBSERVATION_SPECIES_FILTER_LABELS.get(selected_species, selected_species)}"
        )

        render_legend(
            color_mode=color_mode,
            show_bdl_stands=(show_bdl_stands and bdl_available),
            show_report_points=show_report_points,
        )

        enriched_geojson = enrich_geojson_with_real_observations(
            real_bundle.areas_geojson,
            county_by_id,
            color_mode=color_mode,
        )

        polygon_layer = pdk.Layer(
            "GeoJsonLayer",
            data=enriched_geojson,
            opacity=0.72,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=True,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            get_line_width=140,
            line_width_min_pixels=1.5,
            pickable=True,
            auto_highlight=True,
        )
        deck_layers = [polygon_layer]

        if show_bdl_stands and bdl_available:
            enriched_bdl = enrich_bdl_stands_geojson(audit_samples.get("bdl_stands_geojson"))
            if enriched_bdl:
                bdl_layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=enriched_bdl,
                    opacity=0.88,
                    stroked=True,
                    filled=True,
                    extruded=False,
                    get_fill_color="properties.fill_color",
                    get_line_color="properties.line_color",
                    get_line_width=35,
                    line_width_min_pixels=1.2,
                    pickable=True,
                    auto_highlight=True,
                )
                deck_layers.append(bdl_layer)

        precise_points = build_precise_report_points_layer_data(filtered_df)
        if show_report_points and precise_points:
            points_layer = pdk.Layer(
                "ScatterplotLayer",
                data=precise_points,
                get_position="[lon, lat]",
                get_fill_color="fill_color",
                get_line_color=[20, 25, 20, 240],
                line_width_min_pixels=1.5,
                get_radius=2400,
                pickable=True,
                auto_highlight=True,
            )
            deck_layers.append(points_layer)

        origin_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[
                {
                    "tooltip_header": f"Punkt odniesienia odległości: {selected_city}",
                    "tooltip_sub": f"Promień filtra: <b>{radius_km} km</b>",
                    "tooltip_bdl": f"Współrzędne WGS84: <code>{origin_lat:.4f}, {origin_lon:.4f}</code>",
                    "lat": origin_lat,
                    "lon": origin_lon,
                }
            ],
            get_position="[lon, lat]",
            get_fill_color=[35, 78, 112, 235],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            get_radius=4000,
            pickable=True,
        )
        deck_layers.append(origin_layer)

        view_state = pdk.ViewState(
            latitude=PILOT_REGION.center_lat,
            longitude=PILOT_REGION.center_lon,
            zoom=PILOT_REGION.default_zoom,
            pitch=0,
        )

        tooltip = {
            "html": (
                "<b>{tooltip_header}</b><br/>"
                "{tooltip_sub}<hr style='margin:4px 0;'/>"
                "{tooltip_bdl}"
            ),
            "style": {
                "backgroundColor": "#1F2421",
                "color": "#F7F6F2",
                "fontSize": "12px",
                "padding": "10px",
                "borderRadius": "4px",
            },
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=deck_layers,
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light",
            ),
            use_container_width=True,
        )

        county_agg_cnt = (
            int((filtered_df["point_represents_specific_forest"] == False).sum())  # noqa: E712
            if not filtered_df.empty
            else 0
        )
        st.caption(
            f"Jednostka przestrzenna: powiat (GUGiK PRG, 30 powiatów woj. dolnośląskiego TERYT 02, CRS: EPSG:4326) | "
            f"Okres zbioru danych: {real_bundle.min_date} – {real_bundle.max_date} (aktywny filtr: {date_from} – {date_to}) | "
            f"Raporty punktowe na mapie (`point_precise`): {len(precise_points)} | "
            f"Raporty ogólnopowiatowe przypisane wyłącznie do poligonu powiatu (`county_aggregate`): {county_agg_cnt}."
        )

        st.markdown(
            f"#### Pokrycie danych i statystyki obserwacji w powiatach "
            f"({len(in_filter_counties)} pow. w filtrze: {len(counties_with_data)} z raportami, "
            f"{len(counties_no_data)} ze statusem `BRAK_DANYCH`)"
        )
        st.caption(
            "Zgodnie z kontraktem danych (`[GRZ-06]`, `[GRZ-10]`, `[GRZ-12]`) powiaty bez wpisów w wybranym okresie i filtrze "
            "mają jawny status `BRAK_DANYCH (0 raportów — NULL)` oraz puste pola średniej wydajności (`None`), "
            "co odróżnia brak obserwacji od jawnego zerowego zbioru (`0.0 szt./h`)."
        )

        if not in_filter_counties:
            st.warning(
                "Żaden powiat nie spełnia warunków filtru przestrzennego. Zwiększ promień na pasku bocznym."
            )
        else:
            df_counties = pd.DataFrame(in_filter_counties)
            county_cols = {
                "area_id": "TERYT",
                "area_name": "Powiat",
                "distance_km": "Odległość [km]",
                "reports_count": "Liczba raportów",
                "status_code": "Status pokrycia / obfitości",
                "mean_yield_pcs_per_h": "Śr. zbiór [szt./h] (NULL=brak danych)",
                "median_yield_pcs_per_h": "Mediana [szt./h]",
                "max_yield_pcs_per_h": "Maks. [szt./h]",
                "zero_harvest_reports_count": "Puste koszyki (0/h)",
                "point_precise_count": "Punkty miejsc. (precise)",
                "county_aggregate_count": "Ogólnopowiatowe (agg)",
                "last_report_date": "Ostatni raport",
                "forest_cover_pct": "Lesistość BDL [%]",
                "pine_share_pct": "Sosna [%]",
                "oak_beech_share_pct": "Dąb/Buk [%]",
                "spruce_share_pct": "Świerk [%]",
                "precip_lag_14d_mm": "Opad 14d [mm]",
            }
            df_counties_show = df_counties[list(county_cols.keys())].rename(columns=county_cols)
            st.dataframe(df_counties_show, use_container_width=True, hide_index=True)

    with tab_records:
        st.markdown(
            f"#### 1. Próbka przefiltrowanych rekordów obserwacyjnych (`N = {len(filtered_df)}` z `{len(real_bundle.features_df)}`)"
        )
        st.caption(
            "Tabela prezentuje rzeczywiste zgłoszenia z `grzyby.pl` spełniające aktywne filtry okresu, gatunku, "
            "jednostki przestrzennej i promienia wraz z dołączonymi cechami Point-in-Time (`t - 1`)."
        )

        if filtered_df.empty:
            st.info(
                "Brak raportów spełniających wybrane kryteria filtru (status: BRAK DANYCH). "
                "Poszerz zakres dat lub wybierz 'Wszystkie raporty' w filtrze gatunku."
            )
        else:
            preview_cols = {
                "report_id": "ID raportu",
                "observation_date": "Data zbioru (t)",
                "feature_cutoff_date": "Odcięcie cech (t-1)",
                "region_id": "TERYT",
                "region_name": "Powiat",
                "location_raw": "Miejscowość / las (grzyby.pl)",
                "yield_pcs_per_h": "Wydajność [szt./h]",
                "abundance_class": "Klasa obfitości",
                "dominant_species": "Gatunek dominujący",
                "spatial_precision": "Precyzja geo",
                "precip_lag_7d_mm": "Opad 7d [mm]",
                "precip_lag_14d_mm": "Opad 14d [mm]",
                "precip_lag_21d_mm": "Opad 21d [mm]",
                "temp_mean_7d_c": "Śr. temp. 7d [°C]",
                "soil_moisture_0_7cm_m3m3": "Wilg. gleby 0–7 cm",
                "forest_cover_pct": "Lesistość [%]",
                "source_url": "URL źródłowy",
            }
            available_preview_cols = [c for c in preview_cols if c in filtered_df.columns]
            df_preview = (
                filtered_df[available_preview_cols]
                .sort_values(["observation_date", "report_id"], ascending=[False, True])
                .rename(columns=preview_cols)
            )
            st.dataframe(df_preview, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown(
            "#### 2. Weryfikacja zgodności rekordu z tabelami źródłowymi (`reports.parquet` vs `reports_spatial.parquet` vs `forest_features.parquet` vs `features.parquet`)"
        )
        st.caption(
            "Wybierz dowolny raport z aktywnej próbki (lub z pełnego zbioru `N = 513`), aby sprawdzić zgodność 1:1 "
            "pomiędzy surową tabelą zgłoszeń (`reports.parquet`), złączeniem przestrzennym (`reports_spatial.parquet`), "
            "profilem leśnym BDL (`forest_features.parquet`) oraz tabelą analityczną (`features.parquet`)."
        )

        candidate_df = filtered_df if not filtered_df.empty else real_bundle.features_df
        candidate_df = candidate_df.sort_values(
            ["observation_date", "report_id"], ascending=[False, True]
        )
        candidate_options: dict[str, str] = {}
        for _, r_row in candidate_df.head(200).iterrows():
            rid = str(r_row["report_id"])
            sp_txt = (
                "NULL (brak deklaracji)"
                if r_row["dominant_species"] is None or pd.isna(r_row["dominant_species"])
                else str(r_row["dominant_species"])
            )
            candidate_options[rid] = (
                f"{rid} | {r_row['observation_date']} | {r_row['region_name']} ({r_row['region_id']}) | "
                f"{float(r_row['yield_pcs_per_h']):.1f} szt./h | gatunek: {sp_txt}"
            )

        selected_report_id = st.selectbox(
            "Wybierz rekord do inspekcji zgodności ze źródłem",
            options=list(candidate_options.keys()),
            format_func=lambda k: candidate_options[k],
            index=0,
        )

        verification = verify_record_against_source_tables(selected_report_id, real_bundle)
        v_col1, v_col2 = st.columns([2, 3])
        with v_col1:
            status_txt = (
                "ZGODNY 1:1 WE WSZYSTKICH TABELACH ŹRÓDŁOWYCH"
                if verification["all_fields_match"]
                else "WYKRYTO ROZBIEŻNOŚĆ"
            )
            st.markdown(
                f"**Status audytu rekordu `{selected_report_id}`:** `{status_txt}`\n\n"
                f"- **Data zbioru (`observation_date`):** `{verification['observation_date']}`\n"
                f"- **Odcięcie cech (`feature_cutoff_date`):** `{verification['feature_cutoff_date']}`\n"
                f"- **Powiat (`region_id`):** `{verification['region_name']} (TERYT {verification['region_id']})`\n"
                f"- **Lokalizacja w zgłoszeniu (`location_raw`):** `{verification['location_raw']}`\n"
                f"- **Link do wpisu źródłowego:** [{verification['source_url']}]({verification['source_url']})"
            )
        with v_col2:
            df_checks = pd.DataFrame(verification["comparison_rows"])
            st.dataframe(df_checks, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### 3. Podsumowanie spójności tabel w `data/processed/`")
        summary_rows = [
            {
                "Plik w data/processed/": "areas.geojson",
                "Liczba rekordów": len(real_bundle.areas_by_id),
                "Jednostka / Klucz": "powiat (area_id, TERYT 0201–0265)",
                "Opis zawartości": "30 poligonów powiatów woj. dolnośląskiego (GUGiK PRG, EPSG:4326 + BDL)",
            },
            {
                "Plik w data/processed/": "reports.parquet",
                "Liczba rekordów": len(real_bundle.reports_df),
                "Jednostka / Klucz": "report_id (DG...)",
                "Opis zawartości": "518 pobranych zgłoszeń z grzyby.pl (513 walidowanych w oknie 01.09–02.11 + 5 odrzuconych przez walidator temporalny)",
            },
            {
                "Plik w data/processed/": "reports_spatial.parquet",
                "Liczba rekordów": len(real_bundle.reports_spatial_df),
                "Jednostka / Klucz": "report_id -> region_id",
                "Opis zawartości": "Złączenie przestrzenne zgłoszeń z poligonami powiatów (484 point_precise, 34 county_aggregate)",
            },
            {
                "Plik w data/processed/": "forest_features.parquet",
                "Liczba rekordów": len(real_bundle.forest_features_df),
                "Jednostka / Klucz": "region_id (TERYT 0201–0265)",
                "Opis zawartości": "Statyczne cechy siedliskowe i udział gatunków drzewostanowych BDL dla 30 powiatów",
            },
            {
                "Plik w data/processed/": "weather.csv",
                "Liczba rekordów": len(real_bundle.weather_rows),
                "Jednostka / Klucz": "(region_id, date)",
                "Opis zawartości": "2910 rekordów dobowych ERA5-Land (30 powiatów × 97 dni: 2025-08-01 .. 2025-11-05, 0 braków)",
            },
            {
                "Plik w data/processed/": "features.parquet",
                "Liczba rekordów": len(real_bundle.features_df),
                "Jednostka / Klucz": "report_id (1:1 z walidowanymi raportami)",
                "Opis zawartości": "513 rekordów analitycznych Point-in-Time (okna opadów 7/14/21d liczone do t-1 włącznie)",
            },
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with tab_methodology:
        st.markdown("#### Założenia metodologiczne, kontrakt danych (`[GRZ-06]`) i wnioski z EDA (`[GRZ-11]`)")
        st.markdown(
            """
            1. **Zakres etapu `[GRZ-12]` (Eksploracja rzeczywistych obserwacji przed modelem):**
               - Aplikacja wczytuje rzeczywisty zbiór obserwacyjny z `data/processed/` (`N = 513` zweryfikowanych raportów w okresie `2025-09-01 – 2025-11-02`, obejmujących `28 z 30` powiatów woj. dolnośląskiego).
               - **Filtr gatunku dotyczy na tym etapie rzeczywistych obserwacji (`dominant_species`), a nie osobnego modelu ML dla gatunku.** Ze względu na fakt, że `37,8%` raportów ma etykietę `mieszane`, a `17,3%` nie deklaruje gatunku (`NULL`), osobne modele nadzorowane per gatunek dysponowałyby zbyt małą próbą (`19–79` rekordów).
            2. **Semantyka braków danych (`NULL` vs `0.0 szt./h`):**
               - **Brak raportów w powiecie (`reports_count == 0`) jest prezentowany na mapie, w kartach KPI i w tabeli jako `BRAK DANYCH (0 raportów — NULL)`**, a nie jako zerowy zbiór (`0.0 szt./h`).
               - W całym 63-dniowym sezonie `0` raportów mają 2 powiaty (`0217 powiat strzeliński` oraz `0262 powiat m. Legnica`), a na poziomie pojedynczych powiat-dni brak raportów występuje w `81,2%` komórek siatki (`1534 / 1890`).
               - Jawne raporty zerowego zbioru (`yield_pcs_per_h == 0.0`, tj. potwierdzone „puste koszyki”, `N = 45`) są zachowane jako twarde obserwacje negatywne w klasie `BRAK (< 5 szt./h)`.
            3. **Jednostka przestrzenna i uczciwość wizualizacji (`powiat`, TERYT `0201–0265`):**
               - Główną jednostką przestrzenną na mapie jest **powiat (`area_id`)**.
               - Punkty zgłoszeń (`ScatterplotLayer`) są wyświetlane wyłącznie dla wpisów ze zweryfikowaną miejscowością (`point_represents_specific_forest == True`, `479 / 513` wpisów). Raporty ogólnopowiatowe (`county_aggregate`, `34 / 513`) reprezentują wyłącznie poligon powiatu.
            4. **Odcięcie czasowe cech (`Point-in-Time Cutoff` z `[GRZ-10]`):**
               - Dla każdego raportu z datą zbioru `observation_date = t` okna opadów (`precip_lag_7d_mm`, `precip_lag_14d_mm`, `precip_lag_21d_mm`), temperatury i wilgotności gleby ERA5-Land są wyliczone wyłącznie z pełnych dób poprzedzających zbiór (`[t - k, t - 1]`, gdzie `feature_cutoff_date = t - 1`).
            """
        )


def render_demo_predictions_view(
    bundle: DatasetBundle,
    audit_samples: dict[str, Any],
) -> None:
    """
    Widok próbki demonstracyjnej (`data/demo/`, `is_demo=True`) ze Sprintu 1 ([GRZ-02], [GRZ-05]).
    """
    with st.sidebar:
        st.divider()
        available_dates = sorted({p["target_date"] for p in bundle.predictions}, reverse=True)
        selected_date = st.selectbox(
            "Data prognozy (demo)",
            options=available_dates if available_dates else ["2026-10-31"],
            help="Data docelowa w próbce demonstracyjnej data/demo/.",
        )

        available_species = [
            sp for sp in SPECIES_LABELS if any(p["species_filter"] == sp for p in bundle.predictions)
        ] or list(SPECIES_LABELS.keys())

        selected_species = st.selectbox(
            "Filtr gatunku (demo)",
            options=available_species,
            format_func=lambda sp: SPECIES_LABELS.get(sp, sp),
        )
        st.caption(f"Siedlisko BDL: {SPECIES_HABITAT_HINTS.get(selected_species, '—')}")

        st.divider()
        st.subheader("Filtr odległości")
        selected_city = st.selectbox(
            "Punkt odniesienia",
            options=list(REFERENCE_CITIES.keys()),
            index=0,
        )
        origin_lat, origin_lon = REFERENCE_CITIES[selected_city]

        radius_km = st.slider(
            "Promień od punktu [km]",
            min_value=20,
            max_value=160,
            value=120,
            step=10,
            help="Odległość w linii prostej (Haversine) do centroidu powiatu.",
        )

    banner_cls = "status-banner status-banner-warn" if bundle.is_fallback_to_demo else "status-banner"
    st.markdown(
        f"<div class='{banner_cls}'><b>Status danych:</b> {bundle.status_message}</div>",
        unsafe_allow_html=True,
    )

    records, records_by_area = build_ranking_records(
        bundle=bundle,
        target_date=selected_date,
        species_filter=selected_species,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        radius_km=float(radius_km),
    )
    in_radius_records = [r for r in records if r["in_radius"]]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Powiaty w promieniu", f"{len(in_radius_records)} z {len(records)}")
    valid_yields = [r for r in in_radius_records if r["predicted_yield_pcs_per_h"] is not None]
    if valid_yields:
        best_rec = max(valid_yields, key=lambda r: float(r["predicted_yield_pcs_per_h"]))
        kpi2.metric(
            "Maksymalna prognoza (demo)",
            f"{best_rec['predicted_yield_pcs_per_h']:.1f} szt./h",
            delta=best_rec["area_name"],
            delta_color="off",
        )
    else:
        kpi2.metric("Maksymalna prognoza (demo)", "brak danych")

    bdl_covered_count = sum(1 for r in records if r["has_bdl_stand_data"])
    kpi3.metric("Powiaty z profilem BDL", f"{bdl_covered_count} z {len(records)}")
    valid_precip = [
        float(r["precip_lag_14d_mm"])
        for r in in_radius_records
        if r["precip_lag_14d_mm"] is not None
    ]
    kpi4.metric(
        "Średni opad skumulowany (14d)",
        f"{(sum(valid_precip) / len(valid_precip)):.1f} mm" if valid_precip else "brak danych",
    )

    render_legend(color_mode="predictions", show_bdl_stands=False)
    enriched_geojson = enrich_geojson_with_predictions(
        bundle.areas_geojson,
        records_by_area,
        color_mode="predictions",
    )
    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        data=enriched_geojson,
        opacity=0.72,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        get_line_width=140,
        line_width_min_pixels=1.5,
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=PILOT_REGION.center_lat,
        longitude=PILOT_REGION.center_lon,
        zoom=PILOT_REGION.default_zoom,
        pitch=0,
    )
    tooltip = {
        "html": (
            "<b>{tooltip_header}</b><br/>"
            "{tooltip_sub}<hr style='margin:4px 0;'/>"
            "{tooltip_bdl}"
        ),
        "style": {
            "backgroundColor": "#1F2421",
            "color": "#F7F6F2",
            "fontSize": "12px",
            "padding": "10px",
            "borderRadius": "4px",
        },
    }
    st.pydeck_chart(
        pdk.Deck(
            layers=[polygon_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="light",
        ),
        use_container_width=True,
    )
    if in_radius_records:
        st.dataframe(pd.DataFrame(in_radius_records), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Gdzie na grzyby — Dolny Śląsk",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        f"<div class='app-header'>"
        f"<p class='app-title'>Gdzie na grzyby — przestrzenna eksploracja raportów i analiza wysypu</p>"
        f"<p class='app-subtitle'>"
        f"Obszar pilotażowy: <b>{PILOT_REGION.voivodeship_name} (TERYT {PILOT_REGION.voivodeship_id}, 30 powiatów)</b> "
        f"&nbsp;|&nbsp; Jednostka przestrzenna: <b>{PILOT_REGION.spatial_unit_name}</b> "
        f"&nbsp;|&nbsp; Jednostka wyniku: <b>szt./h</b>"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Źródło danych aplikacji")
        default_mode = os.getenv("GRZYBY_DATA_ENV", "real").lower()
        mode_options = {
            "real": "Rzeczywiste raporty i pokrycie (data/processed/, [GRZ-12])",
            "demo": "Próbka demonstracyjna UI (data/demo/, [GRZ-02])",
        }
        selected_mode = st.selectbox(
            "Tryb danych",
            options=list(mode_options.keys()),
            index=0 if default_mode != "demo" else 1,
            format_func=lambda k: mode_options[k],
            help=(
                "Domyślny tryb [GRZ-12] wczytuje rzeczywiste dane obserwacyjne z data/processed/ "
                "(reports.parquet, features.parquet, forest_features.parquet, areas.geojson)."
            ),
        )

    audit_samples = load_sprint1_audit_samples()

    if selected_mode == "real":
        real_bundle = load_real_observations_bundle()
        render_real_observations_view(real_bundle=real_bundle, audit_samples=audit_samples)
    else:
        demo_bundle = load_dataset_bundle("demo")
        render_demo_predictions_view(bundle=demo_bundle, audit_samples=audit_samples)

    st.markdown(
        f"<div class='footer-note'>{ACADEMIC_ATTRIBUTION}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
