"""
Konfiguracja regionu pilotażowego, ścieżek danych oraz słowników domenowych
zgodna z ustaleniami w docs/mvp.md oraz docs/data_contract.md ([GRZ-01], [GRZ-02]).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Katalogi projektu
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
DEMO_DIR: Path = DATA_DIR / "demo"
SAMPLE_DIR: Path = DATA_DIR / "sample"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
DOCS_DIR: Path = ROOT_DIR / "docs"
MODELS_DIR: Path = ROOT_DIR / "models"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"


@dataclass(frozen=True)
class RegionConfig:
    """Konfiguracja regionu pilotażowego (MVP)."""

    voivodeship_id: str
    voivodeship_name: str
    spatial_unit_name: str
    spatial_key_name: str
    crs_display: str
    crs_metric: str
    timezone: str
    center_lat: float
    center_lon: float
    default_zoom: float
    default_radius_km: int


PILOT_REGION = RegionConfig(
    voivodeship_id=os.getenv("GRZYBY_REGION_TERYT", "02"),
    voivodeship_name="Województwo dolnośląskie",
    spatial_unit_name="powiat (4-znakowy kod TERYT)",
    spatial_key_name="area_id",
    crs_display="EPSG:4326 (WGS84)",
    crs_metric="EPSG:2180 (PUWG 1992)",
    timezone="Europe/Warsaw",
    center_lat=51.08,
    center_lon=16.35,
    default_zoom=7.4,
    default_radius_km=60,
)

# Punkty referencyjne (miasta startowe w regionie pilotażowym) do filtra odległości
REFERENCE_CITIES: dict[str, tuple[float, float]] = {
    "Wrocław (centrum regionu)": (51.1079, 17.0385),
    "Jelenia Góra (Karkonosze)": (50.9044, 15.7194),
    "Wałbrzych (Sudety Środkowe)": (50.7714, 16.2843),
    "Bolesławiec (Bory Dolnośląskie)": (51.2630, 15.5651),
    "Kłodzko (Kotlina Kłodzka)": (50.4347, 16.6548),
    "Legnica": (51.2070, 16.1553),
    "Milicz (Dolina Baryczy)": (51.5297, 17.2714),
    "Oleśnica (Wzgórza Twardogórskie)": (51.2089, 17.3828),
}

# Słownik gatunków grzybów zgodny z docs/data_contract.md (Sekcja 3.1)
SPECIES_LABELS: dict[str, str] = {
    "all": "Ogółem (wszystkie gatunki)",
    "borowik_szlachetny": "Borowik szlachetny",
    "podgrzybek_brunatny": "Podgrzybek brunatny",
    "kurka": "Pieprznik jadalny (kurka)",
    "maslak_zwyczajny": "Maślak zwyczajny",
    "kania": "Czubajka kania",
}

SPECIES_HABITAT_HINTS: dict[str, str] = {
    "all": "Wszystkie typy siedlisk leśnych (BDL)",
    "borowik_szlachetny": "Lasy dębowe, bukowe, mieszane i górskie bory świerkowe",
    "podgrzybek_brunatny": "Bory sosnowe i świerkowe na siedliskach świeżych i wilgotnych",
    "kurka": "Bory sosnowe i mieszane na siedliskach świeżych (z mchem)",
    "maslak_zwyczajny": "Młodniki sosnowe i obrzeża borów sosnowych",
    "kania": "Przecinki, obrzeża lasów mieszanych i liściastych",
}

# Klasy obfitości wysypu zgodne z docs/mvp.md (Sekcja 3) — stonowana paleta kartograficzna
ABUNDANCE_CLASSES: dict[str, dict[str, object]] = {
    "WYSYP": {
        "label": "WYSYP (> 20 szt./h)",
        "description": "Masowy wysyp grzybów (> 20 szt./h)",
        "rgba": [44, 107, 63, 190],
        "hex": "#2C6B3F",
    },
    "SREDNIO": {
        "label": "ŚREDNIO (5 – 20 szt./h)",
        "description": "Umiarkowany zbiór (5 – 20 szt./h)",
        "rgba": [201, 142, 42, 185],
        "hex": "#C98E2A",
    },
    "BRAK": {
        "label": "BRAK (< 5 szt./h)",
        "description": "Pojedyncze sztuki lub pusto (< 5 szt./h)",
        "rgba": [168, 66, 50, 175],
        "hex": "#A84232",
    },
    "BRAK_DANYCH": {
        "label": "Poza promieniem / brak danych",
        "description": "Brak predykcji dla wybranego filtra lub obszar poza promieniem",
        "rgba": [150, 150, 144, 115],
        "hex": "#969690",
    },
}

# Wymagana klauzula atrybucji i charakteru akademickiego ([GRZ-03] / zgoda autora grzyby.pl)
ACADEMIC_ATTRIBUTION: str = (
    "Dane o zbiorach pochodzą z portalu grzyby.pl – © Marek Snowarski, "
    "Grzyby Polski, atlas grzybów (www.grzyby.pl). "
    "Dane meteorologiczne i modelowana wilgotność gleby (ERA5-Land, siatka ~9 km): Open-Meteo.com. "
    "Dane o siedliskach leśnych: Bank Danych o Lasach (BDL). "
    "Projekt wyłącznie o charakterze dydaktycznym (na zaliczenie przedmiotu), niekomercyjny."
)
