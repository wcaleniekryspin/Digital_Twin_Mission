# Satellite Launch System | Digital Twin 3D

## O projekcie
Projekt przedstawia zaawansowany prototyp Cyfrowego Bliźniaka (Digital Twin) systemu nośnego rakiety orbitalnej, stworzony w środowisku Python z interaktywnym kokpitem menedżerskim Dash (Plotly). Aplikacja służy do optymalizacji i symulacji trajektorii lotów rakietowych w czasie rzeczywistym, oferując potężne narzędzia telemetryczne dedykowane inżynierii kosmicznej.

## Kluczowe funkcjonalności
**Fotorealistyczna wizualizacja 3D (Mapbox Satellite):** Dynamiczne mapowanie płaskiej trajektorii lotu 2D na sferyczne współrzędne geograficzne globu ziemskiego z wykorzystaniem trójwymiarowych map satelitarnych.
**Autonomiczny Optymalizator Trajektorii (AI Powell Optimizer):** Algorytm oparty na metodzie Powella, który dynamicznie dostosowuje czas rozpoczęcia manewru *Gravity Turn* oraz końcowe pochylenie rakiety w celu maksymalizacji dystansu kątowego (jak najdalszego okrążenia Ziemi) oraz budowania prędkości orbitalnej.
**Zaawansowana Telemetria 2X2:** Podział paneli telemetrycznych na dedykowane, czytelne wykresy w jednostkach metrycznych (km): profil wysokości, prędkości, dynamicznego zużycia paliwa dla obu stopni oraz analizy stabilności aerodynamicznej (Kąt Natarcia - AoA).
**Fizyczny Silnik Integracji Numerycznej:** Wykorzystanie solvera `solve_ivp` (RK45) zaimplementowanego wraz z barometrycznym modelem gęstości atmosfery oraz precyzyjnym wykrywaniem zdarzeń krytycznych (*ground impact event*).

## Interface aplikacji
<img width="1392" height="855" alt="obraz" src="https://github.com/user-attachments/assets/25235131-c221-4f70-a555-c6926577850d" />
<img width="1392" height="855" alt="obraz" src="https://github.com/user-attachments/assets/34eb849d-a89a-4ca6-91c7-61ccecb7fb37" />
<img width="1231" height="781" alt="obraz" src="https://github.com/user-attachments/assets/88068a09-f91e-4b7d-879d-565f541576ad" />
