"""
Punkt wejścia aplikacji Streamlit: "Gdzie na grzyby?" ([GRZ-02]).

Uruchomienie:
    streamlit run app.py
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.config import (
    ABUNDANCE_CLASSES,
    ACADEMIC_ATTRIBUTION,
    PILOT_REGION,
    REFERENCE_CITIES,
    SPECIES_HABITAT_HINTS,
    SPECIES_LABELS,
)
from src.data_loader import (
    DatasetBundle,
    load_dataset_bundle,
    load_sprint1_audit_samples,
)
from src.geo_utils import enrich_geojson_with_predictions, haversine_km


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
        margin-bottom: 1.1rem;
    }
    .status-banner-warn {
        border-left-color: #C98E2A;
        background-color: #F7EFE0;
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
            "dominant_moisture_type": area_props.get("dominant_moisture_type"),
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
            -r["distance_km"],
        ),
        reverse=True,
    )
    return records, records_by_area


def render_legend() -> None:
    """Wyświetla legendę klas obfitości wysypu oraz jednostkę wyniku."""
    items_html = []
    for key in ["WYSYP", "SREDNIO", "BRAK", "BRAK_DANYCH"]:
        info = ABUNDANCE_CLASSES[key]
        items_html.append(
            f"<div class='legend-item'>"
            f"<span class='legend-swatch' style='background:{info['hex']};'></span>"
            f"<span>{info['label']}</span>"
            f"</div>"
        )
    st.markdown(
        f"<div class='legend-bar'>{''.join(items_html)}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Gdzie na grzyby — Dolny Śląsk",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        f"<div class='app-header'>"
        f"<p class='app-title'>Gdzie na grzyby — przestrzenna analiza i predykcja wysypu</p>"
        f"<p class='app-subtitle'>"
        f"Obszar pilotażowy: <b>{PILOT_REGION.voivodeship_name} (TERYT {PILOT_REGION.voivodeship_id})</b> "
        f"&nbsp;|&nbsp; Jednostka agregacji: <b>{PILOT_REGION.spatial_unit_name}</b> "
        f"&nbsp;|&nbsp; Jednostka wyniku: <b>szt./h</b>"
        f"</p></div>",
        unsafe_allow_html=True,
    )

    # --- PANEL BOCZNY (SIDEBAR) ---
    with st.sidebar:
        st.subheader("Parametry widoku")

        default_env = os.getenv("GRZYBY_DATA_ENV", "demo").lower()
        env_options = {
            "demo": "Próbka demonstracyjna (data/demo/)",
            "prod": "Dane przetworzone (data/processed/)",
        }
        selected_env = st.selectbox(
            "Zbiór danych",
            options=list(env_options.keys()),
            index=0 if default_env != "prod" else 1,
            format_func=lambda k: env_options[k],
            help="Próbka data/demo/ (is_demo=True) służy do integracji interfejsu i nie trafia do treningu ML.",
        )

        bundle = load_dataset_bundle(selected_env)

        available_dates = sorted({p["target_date"] for p in bundle.predictions}, reverse=True)
        selected_date = st.selectbox(
            "Data prognozy",
            options=available_dates if available_dates else ["2026-10-31"],
            help="Data docelowa w formacie ISO (YYYY-MM-DD).",
        )

        available_species = [
            sp for sp in SPECIES_LABELS if any(p["species_filter"] == sp for p in bundle.predictions)
        ] or list(SPECIES_LABELS.keys())

        selected_species = st.selectbox(
            "Filtr gatunku",
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
            help="Odległość w linii prostej (Haversine) do centroidu leśnego powiatu.",
        )

        st.divider()
        st.caption(
            f"Konfiguracja regionu:\n"
            f"TERYT: {PILOT_REGION.voivodeship_id} ({PILOT_REGION.voivodeship_name})\n"
            f"Klucz: {PILOT_REGION.spatial_key_name} | Układ: {PILOT_REGION.crs_display}\n"
            f"Strefa: {PILOT_REGION.timezone}"
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

    # --- KARTY PODSUMOWANIA ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Powiaty w promieniu",
        f"{len(in_radius_records)} z {len(records)}",
    )

    valid_yields = [
        r for r in in_radius_records if r["predicted_yield_pcs_per_h"] is not None
    ]
    if valid_yields:
        best_rec = max(valid_yields, key=lambda r: float(r["predicted_yield_pcs_per_h"]))
        kpi2.metric(
            "Maksymalna prognoza",
            f"{best_rec['predicted_yield_pcs_per_h']:.1f} szt./h",
            delta=best_rec["area_name"],
            delta_color="off",
        )
    else:
        kpi2.metric("Maksymalna prognoza", "brak danych")

    wysyp_count = sum(1 for r in in_radius_records if r["abundance_class"] == "WYSYP")
    kpi3.metric(
        "Powiaty z klasą WYSYP",
        f"{wysyp_count}",
    )

    valid_precip = [
        float(r["precip_lag_14d_mm"])
        for r in in_radius_records
        if r["precip_lag_14d_mm"] is not None
    ]
    kpi4.metric(
        "Średni opad skumulowany (14d)",
        f"{(sum(valid_precip) / len(valid_precip)):.1f} mm" if valid_precip else "brak danych",
    )

    # --- ZAKŁADKI GŁÓWNE ---
    tab_map, tab_samples, tab_info = st.tabs(
        [
            "Mapa i zestawienie powiatów",
            "Przegląd danych wejściowych i próbek",
            "Metodologia i kontrakt danych",
        ]
    )

    with tab_map:
        st.markdown(
            f"#### Prognoza wydajności zbiorów na dzień `{selected_date}` — "
            f"{SPECIES_LABELS.get(selected_species, selected_species)}"
        )
        render_legend()

        enriched_geojson = enrich_geojson_with_predictions(
            bundle.areas_geojson, records_by_area
        )

        polygon_layer = pdk.Layer(
            "GeoJsonLayer",
            data=enriched_geojson,
            opacity=0.78,
            stroked=True,
            filled=True,
            extruded=False,
            wireframe=True,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            get_line_width=110,
            line_width_min_pixels=1.5,
            pickable=True,
            auto_highlight=True,
        )

        origin_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[
                {
                    "name": f"Punkt odniesienia: {selected_city}",
                    "lat": origin_lat,
                    "lon": origin_lon,
                }
            ],
            get_position="[lon, lat]",
            get_fill_color=[35, 78, 112, 235],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            get_radius=4200,
            pickable=False,
        )

        view_state = pdk.ViewState(
            latitude=PILOT_REGION.center_lat,
            longitude=PILOT_REGION.center_lon,
            zoom=PILOT_REGION.default_zoom,
            pitch=0,
        )

        tooltip = {
            "html": (
                "<b>{area_name}</b> (TERYT: <code>{area_id}</code>)<br/>"
                "Prognoza: <b>{predicted_yield_label}</b> ({abundance_label})<br/>"
                "Wskaźnik dopasowania: <b>{confidence_label}</b><br/>"
                "Odległość w linii prostej: <b>{distance_label}</b><hr style='margin:4px 0;'/>"
                "Lesistość: <b>{forest_cover_pct}%</b> | "
                "Sosna: <b>{pine_share_pct_label}</b> | "
                "Dąb/Buk: <b>{oak_beech_share_pct_label}</b> | "
                "Świerk: <b>{spruce_share_pct_label}</b><br/>"
                "Siedlisko BDL: <b>{dominant_moisture_label}</b><br/>"
                "<i>{habitat_match_note}</i>"
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
                layers=[polygon_layer, origin_layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light",
            ),
            use_container_width=True,
        )

        sample_meta = bundle.predictions[0] if bundle.predictions else {}
        st.caption(
            f"Data docelowa: {selected_date} | "
            f"Data wygenerowania: {sample_meta.get('generated_at', '—')} | "
            f"Wersja modelu: {sample_meta.get('model_version', '—')} | "
            f"Jednostka wyniku: szt./h (liczba owocników na osobo-godzinę) | "
            f"Punkt odniesienia (granatowy znacznik): {selected_city}."
        )

        st.markdown(f"#### Zestawienie powiatów w promieniu {radius_km} km ({selected_city})")
        if not in_radius_records:
            st.warning(
                "Brak powiatów w zadanym promieniu. Zwiększ promień na pasku bocznym."
            )
        else:
            df_rank = pd.DataFrame(in_radius_records)
            display_cols = {
                "area_id": "TERYT",
                "area_name": "Powiat",
                "distance_km": "Odległość [km]",
                "predicted_yield_pcs_per_h": "Prognoza [szt./h]",
                "abundance_class": "Klasa",
                "confidence_score": "Dopasowanie [0-1]",
                "forest_cover_pct": "Lesistość [%]",
                "pine_share_pct": "Sosna [%]",
                "oak_beech_share_pct": "Dąb/Buk [%]",
                "spruce_share_pct": "Świerk [%]",
                "precip_lag_14d_mm": "Opad 14d [mm]",
                "temp_min_night_c": "Temp. min [°C]",
                "soil_moisture_0_7cm_m3m3": "Wilg. gleby 0-7cm [m³/m³]",
                "habitat_match_note": "Uwagi siedliskowo-pogodowe",
            }
            df_show = df_rank[list(display_cols.keys())].rename(columns=display_cols)
            st.dataframe(df_show, use_container_width=True, hide_index=True)

    with tab_samples:
        st.markdown("#### Podgląd plików kontraktu danych oraz próbek audytowych ze Sprintu 1")
        st.caption(
            "Sekcja diagnostyczna do weryfikacji formatu plików wymienianych pomiędzy modułem danych a interfejsem."
        )

        audit_samples = load_sprint1_audit_samples()

        col_rep, col_wea = st.columns(2)
        with col_rep:
            st.markdown("**1. Raporty zbieraczy (`reports`)**")
            st.dataframe(pd.DataFrame(bundle.reports), use_container_width=True, hide_index=True)

        with col_wea:
            st.markdown("**2. Szeregi pogodowe (`weather`)**")
            st.dataframe(pd.DataFrame(bundle.weather), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**3. Próbka audytowa raportów z grzyby.pl (`data/sample/reports_audit_52_ds.csv`, N = 52)**")
        if audit_samples["reports_audit"]:
            df_audit_rep = pd.DataFrame(audit_samples["reports_audit"])
            st.caption(
                f"Liczba rekordów: {len(df_audit_rep)} | Pokrycie: 18 powiatów woj. dolnośląskiego | "
                f"Okres zbiorów: 2025-09-29 .. 2025-10-05."
            )
            st.dataframe(df_audit_rep.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Plik `data/sample/reports_audit_52_ds.csv` nie został znaleziony.")

        st.markdown("**4. Próbka historii pogody Open-Meteo ERA5-Land (`data/sample/weather_daily_sample_ds.csv`, N = 210)**")
        if audit_samples["weather_sample"]:
            df_audit_wea = pd.DataFrame(audit_samples["weather_sample"])
            st.caption(
                f"Liczba rekordów dobowych: {len(df_audit_wea)} (6 lokalizacji × 35 dni, 2025-09-01 .. 2025-10-05). "
                f"Zmienne glebowe (warstwy 0–7 cm oraz 7–28 cm) pochodzą z modelu reanalizy ERA5-Land (siatka ~9 km), "
                f"a nie z pomiarów terenowych in-situ."
            )
            st.dataframe(df_audit_wea.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Plik `data/sample/weather_daily_sample_ds.csv` nie został znaleziony.")

    with tab_info:
        st.markdown("#### Założenia metodologiczne i ograniczenia")
        st.markdown(
            """
            1. **Definicja zmiennej wyjściowej (`predicted_yield_pcs_per_h` `[szt./h]`):**
               - Szacowana wydajność grzybobrania w sztukach na godzinę na jedną osobę, odpowiadająca raportom z portalu `grzyby.pl`.
               - Agregacja przestrzenna odbywa się na poziomie powiatu (`area_id`, 4-znakowy kod TERYT). Wynik nie wskazuje pojedynczego oddziału leśnego.
            2. **Semantyka braków danych (`NULL` vs `0.0`):**
               - `0.0` oznacza potwierdzony pomiar zerowy (np. brak opadu `0.0 mm` lub jawny raport zerowego zbioru `(0/h)`).
               - `NULL` (`NaN` / `None`) oznacza brak informacji (np. niepodany czas zbierania lub brak atrybutu BDL).
               - Brak raportu w danym powiecie i dniu nie jest traktowany jako zerowy zbiór.
            3. **Charakter danych glebowych:**
               - Temperatura i objętościowa wilgotność gleby (`soil_moisture_0_7cm_m3m3`) pochodzą z numerycznego modelu powierzchni lądu **ECMWF ERA5-Land** (oczko siatki ok. 9 × 9 km) udostępnianego przez Open-Meteo API, a nie z czujników terenowych.
            4. **Izolacja danych demonstracyjnych (`data/demo/`):**
               - Pliki w `data/demo/` posiadają flagę `is_demo = True` oraz oznaczenie `DEMO_MOCK_v0` i są wykluczone z potoku treningowego ML.
            """
        )

    st.markdown(
        f"<div class='footer-note'>{ACADEMIC_ATTRIBUTION}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
