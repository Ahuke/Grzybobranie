# UWAGA: Dane Demonstracyjne (Mock dla Interfejsu Streamlit)

> **OSTRZEŻENIE:** Pliki w katalogu `data/demo/` zawierają **syntetyczną próbkę demonstracyjną** stworzoną wyłącznie w celu umożliwienia równoległej budowy i testowania interfejsu aplikacji Streamlit zgodnie z kontraktem `docs/data_contract.md`.

## Zasady bezpieczeństwa:
1. **ZAKAZ TRENINGU ML:** Żaden plik z katalogu `data/demo/` **nie może** trafiać do zbioru treningowego, walidacyjnego ani testowego modelu Machine Learning.
2. **Jawne oznaczenie:**
   - Każdy rekord we wszystkich czterech plikach posiada flagę `is_demo = True` (`true` w GeoJSON).
   - Identyfikatory raportów zaczynają się od prefiksu `DEMO_REP_`.
   - Wersja modelu w predykcjach jest oznaczona jako `DEMO_MOCK_v0`.
3. **Testowanie warunków brzegowych i braków danych (`NULL` vs `0.0`):**
   - Próbka celowo zawiera pojedyncze puste wartości (`NULL` / `null`) w polach opcjonalnych oraz zerowe pomiary (`0.0`), aby interfejs użytkownika mógł zweryfikować poprawne wyświetlanie braków danych.
