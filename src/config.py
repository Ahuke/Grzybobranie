"""
Konfiguracja regionu pilotażowego, ścieżek danych oraz słowników domenowych
zgodna z ustaleniami w docs/mvp.md oraz docs/data_contract.md ([GRZ-01], [GRZ-02], [GRZ-12]).
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

# Rozszerzony słownik filtra obserwacji dla rzeczywistych raportów ([GRZ-12])
# Filtr gatunku dotyczy na tym etapie obserwacji, a nie osobnego modelu dla gatunku.
OBSERVATION_SPECIES_FILTER_LABELS: dict[str, str] = {
    "all": "Wszystkie raporty ogółem",
    "mieszane": "Zbiory mieszane (wielogatunkowe)",
    "podgrzybek_brunatny": "Podgrzybek brunatny",
    "borowik_szlachetny": "Borowik szlachetny",
    "kania": "Czubajka kania",
    "maslak_zwyczajny": "Maślak zwyczajny",
    "kurka": "Pieprznik jadalny (kurka)",
    "__NULL__": "Brak deklaracji gatunku (NULL, w tym zbiory 0.0 szt./h)",
}

SPECIES_HABITAT_HINTS: dict[str, str] = {
    "all": "Wszystkie typy siedlisk leśnych (BDL)",
    "mieszane": "Różne typy siedlisk leśnych (kosz wielogatunkowy)",
    "borowik_szlachetny": "Lasy dębowe, bukowe, mieszane i górskie bory świerkowe",
    "podgrzybek_brunatny": "Bory sosnowe i świerkowe na siedliskach świeżych i wilgotnych",
    "kurka": "Bory sosnowe i mieszane na siedliskach świeżych (z mchem)",
    "maslak_zwyczajny": "Młodniki sosnowe i obrzeża borów sosnowych",
    "kania": "Przecinki, obrzeża lasów mieszanych i liściastych",
    "__NULL__": "Raporty ogólne bez wskazania pojedynczego gatunku lub jawne puste wyjścia (0.0 szt./h)",
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
        "label": "Brak danych (0 raportów — NULL)",
        "description": "Brak raportów w wybranym okresie/filtrze lub obszar poza promieniem (NULL != 0.0 szt./h)",
        "rgba": [150, 150, 144, 115],
        "hex": "#969690",
    },
    "NIEDOSTEPNY": {
        "label": "Niedostępny (brak wystarczających danych)",
        "description": "Obszar bez pełnej 21-dniowej historii pogodowej lub spoza obszaru pokrycia",
        "rgba": [180, 180, 176, 145],
        "hex": "#B4B4B0",
    },
}

# Ciągło-przedziałowa skala kolorów zgodna z jednostką wyniku modelu: szt./h ([GRZ-16])
# Progi 5.0 i 20.0 szt./h są w 100% zgodne z granicami klas BRAK / SREDNIO / WYSYP.
YIELD_COLOR_SCALE_BINS: list[dict[str, object]] = [
    {
        "key": "VERY_HIGH",
        "min_val": 30.0,
        "max_val": None,
        "label": "> 30 szt./h (silny WYSYP)",
        "rgba": [20, 83, 45, 220],
        "hex": "#14532D",
    },
    {
        "key": "HIGH",
        "min_val": 20.0,
        "max_val": 30.0,
        "label": "20 – 30 szt./h (WYSYP)",
        "rgba": [46, 125, 50, 205],
        "hex": "#2E7D32",
    },
    {
        "key": "MEDIUM_HIGH",
        "min_val": 12.0,
        "max_val": 20.0,
        "label": "12 – 20 szt./h (ŚREDNIO+)",
        "rgba": [138, 184, 78, 200],
        "hex": "#8AB84E",
    },
    {
        "key": "MEDIUM_LOW",
        "min_val": 5.0,
        "max_val": 12.0,
        "label": "5 – 12 szt./h (ŚREDNIO)",
        "rgba": [217, 155, 42, 195],
        "hex": "#D99B2A",
    },
    {
        "key": "LOW",
        "min_val": 0.0,
        "max_val": 5.0,
        "label": "< 5 szt./h (BRAK)",
        "rgba": [176, 70, 54, 190],
        "hex": "#B04636",
    },
]

# Etykiety źródeł wyników — wyraźne rozróżnienie ML vs regułowy vs obserwacje vs demo ([GRZ-16])
PREDICTION_SOURCE_METADATA: dict[str, dict[str, str]] = {
    "ML_MODEL": {
        "badge": "WYNIK ML",
        "label": "Model uczenia maszynowego (Random Forest, rf_pipeline_v1.0)",
        "color": "#1E5631",
        "bg": "#E9F3EC",
    },
    "RULE_BASED_INDEX": {
        "badge": "WYNIK REGUŁOWY",
        "label": "Jawny Wskaźnik Warunków Grzybobrania (model regułowy, RULE_SUITABILITY_INDEX_v1.0)",
        "color": "#1D4E89",
        "bg": "#EAF1F8",
    },
    "REAL_OBSERVATIONS": {
        "badge": "OBSERWACJE TERENOWE",
        "label": "Rzeczywiste raporty grzybiarzy (grzyby.pl, data/processed/)",
        "color": "#5C472A",
        "bg": "#F4EFE6",
    },
    "DEMO_MOCK": {
        "badge": "DANE DEMONSTRACYJNE (MOCK)",
        "label": "Syntetyczna próbka demonstracyjna interfejsu (data/demo/, is_demo=True)",
        "color": "#9A4E0F",
        "bg": "#FBF0E4",
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
