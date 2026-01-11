# Croatian Property Price Estimator 🏠

## Opis projekta

Ovaj projekt predstavlja prvi alat za procjenu cijene nekretnina u Hrvatskoj, inspiriran poznatim Zillow Zestimate modelom. Cilj je stvoriti pouzdan sustav za predviđanje tržišne vrijednosti nekretnina baziran na machine learning algoritmima i opsežnoj analizi dostupnih podataka s hrvatskog tržišta nekretnina.

## Motivacija

Na hrvatskom tržištu ne postoji javno dostupan alat koji omogućava brzu i pouzdanu procjenu vrijednosti nekretnina. Ovaj projekt nastoji popuniti tu prazninu pružajući:

- **Transparentnost**: Objektivnu procjenu cijene bazirano na tržišnim podacima
- **Informiranost**: Kupci i prodavači mogu donositi bolje odluke
- **Pristupačnost**: Besplatan alat dostupan svima

## Ključne značajke

### 1. Prikupljanje podataka (Web Scraping)
- Automatsko prikupljanje oglasa s internetskih portala
- Ekstrakcija strukturiranih podataka (površina, broj soba, lokacija, cijena...)
- **LLM-powered analiza**: Korištenje velikih jezičnih modela za ekstrakciju dodatnih značajki iz tekstualnih opisa oglasa

### 2. Geolokacijske značajke
- Izračun blizine ključnih sadržaja (škole, trgovine, javni prijevoz)
- Walk Score i druge metrike dostupnosti
- Rješavanje problema neprecizne lokacije kroz analizu patterna u oglasima

### 3. Machine Learning model
- Predviđanje cijene na osnovu:
  - Osnovnih karakteristika nekretnine
  - Lokacijskih faktora
  - Dodatnih ekstrahiranih značajki iz opisa
  - Tržišnih trendova

### 4. Napredne funkcionalnosti (Budućnost)
- Ekonomski pokazatelji: kamatne stope, tržišni trendovi
- Savjeti o optimalnom timingu kupnje/prodaje
- Analiza isplativosti investicije

## Tehnička arhitektura

### Faza 1: Prikupljanje i priprema podataka
1. Web scraping oglasa s multiple platformi
2. LLM-based ekstrakcija kategorija iz nestrukturiranih opisa
3. Geolokacijska obogaćivanje podataka
4. Čišćenje i validacija dataseta

### Faza 2: Razvoj modela
1. Eksplorativna analiza podataka (EDA)
2. Feature engineering
3. Treniranje različitih ML modela
4. Model evaluacija i selekcija

### Faza 3: Deployment
1. Backend API za procjene
2. Web sučelje za korisnike
3. Kontinuirano ažuriranje modela s novim podacima

## Tehnologije

TBD

## Izazovi i rješenja

### Problem: Neprecizna lokacija
- **Izazov**: Oglasi prikazuju krug unutar kojeg se nalazi nekretnina, ne točnu adresu
- **Pristup**: Analiza konzistentnosti pozicija u krugovima, korištenje centroida kao aproksimacije
- **Alternativa**: Dodavanje "uncertainty radius" kao dodatne značajke

### Problem: Nedostatak javne baze prodaja
- **Izazov**: Ne postoji javna evidencija stvarnih prodajnih cijena
- **Pristup**: Korištenje oglašenih cijena kao proxy, filtriranje outliera i nerealno postavljenih oglasa

## Roadmap

### Faza 1: MVP ✨
- [ ] Implementacija web scrapera
- [ ] LLM ekstrakcija dodatnih kategorija
- [ ] Kreiranje inicijalnog dataseta
- [ ] Baseline ML model
- [ ] Osnovni web interface

### Faza 2: Poboljšanja 🚀
- [ ] Geolokacijske značajke i external API integracije
- [ ] Model refinement i ensemble pristup
- [ ] Vizualizacija tržišnih trendova
- [ ] Dodavanje novih izvora podataka

### Faza 3: Advanced Features 💡
- [ ] Ekonomski indikatori i timing savjeti
- [ ] Komparativna analiza s drugim nekretninama
- [ ] Investicijski kalkulator (ROI)