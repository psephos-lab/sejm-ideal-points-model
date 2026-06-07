# Strona: Punkty idealne posłów Sejmu

Statyczna strona (GitHub Pages) wizualizująca 1-wymiarowe punkty idealne posłów
Sejmu X kadencji. Czysty HTML/JS — bez backendu, bez build-stepu.

## Pliki
- `index.html` — struktura strony
- `style.css` — style
- `app.js` — wizualizacja D3 (beeswarm + wyszukiwarka + legenda klubów + panel średnich)
- `ideal_points.json` — dane (generowane z `../make_site_data.py` na podstawie wyników MCMC)

## Podgląd lokalny
```bash
cd docs
python3 -m http.server 8000
# otwórz http://localhost:8000
```
(Trzeba serwera HTTP — `fetch()` JSON-a nie zadziała z `file://`.)

## Aktualizacja danych
Po ponownym przeliczeniu modelu:
```bash
python make_site_data.py     # z katalogu głównego repo -> nadpisuje docs/ideal_points.json
```

## Publikacja na GitHub Pages
1. Wypchnij repo na GitHub.
2. Settings → Pages → Build and deployment → Source: **Deploy from a branch**.
3. Branch: `main`, folder: **/docs**. Zapisz.
4. Strona pojawi się pod `https://<user>.github.io/<repo>/` (zwykle w ~1 min).

### (Opcjonalnie) auto-aktualizacja przez GitHub Actions
Cron co miesiąc może odpalić `fetch_data.py` + model + `make_site_data.py` i zacommitować
świeży `docs/ideal_points.json` — strona zaktualizuje się sama. (Bieg MCMC ~17 min mieści
się w limicie joba.)
