# Artefakty Modeli ML (`models/`)

Katalog na wytrenowane potoki predykcyjne (`Pipeline` scikit-learn / Random Forest) oraz pliki metadanych JSON (`[GRZ-14]`):
- Duże pliki binarne (`*.pkl`, `*.joblib`) są ignorowane w `.gitignore`, aby nie obciążać repozytorium Git.
- Konfiguracje cech, metryki ewaluacji i lekkie metadane modeli mogą być wersjonowane jako pliki `.json`.
- **Zasada bezpieczeństwa:** Żaden model nie może być trenowany na danych z katalogu `data/demo/` (`is_demo == True`).
