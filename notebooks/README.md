# Analizy i Eksploracja Danych (`notebooks/`)

Katalog przeznaczony na notatniki Jupyter (`.ipynb`) oraz raporty analityczne:
- [`01_eda_lag_effect.ipynb`](01_eda_lag_effect.ipynb) — kompletny notatnik eksploracyjnej analizy danych (EDA) i badania opóźnienia opad $\rightarrow$ raportowane zbiory (`[GRZ-11]`),
- pełny raport analityczny z osadzonymi wykresami: [`docs/eda_and_lag_analysis.md`](../docs/eda_and_lag_analysis.md),
- skrypt odtwarzający obliczenia, wykresy PNG (`docs/figures/`) i notatnik: `python scripts/run_eda_and_lag_analysis.py`.

> **Zasada higieny repozytorium:** Przed zatwierdzeniem zmian do Git wyczyść ciężkie wyjścia komórek lub zapisuj gotowe wykresy podsumowujące w `docs/figures/`. Katalog `.ipynb_checkpoints/` jest ignorowany w `.gitignore`.
