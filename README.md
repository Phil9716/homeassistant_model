# Hausmodell – selbstlernende Wärmepumpen-Steuerung für Home Assistant

AppDaemon-App für eine Vaillant aroTHERM plus mit Fußbodenheizung, PV und Batterie.

## Aufbau

| Pfad | Inhalt |
|---|---|
| `apps/hausmodell/waermemodell.py` | Thermisches 2R2C-Hausmodell, Kalman-Filter, nächtliches Lernen, Prognosen |
| `apps/hausmodell/hausmodell.yaml` | Konfiguration (Entitäten, Komfortgrenzen) |
| `tests/test_synthetic.py` | Offline-Test mit synthetischen Daten: `python3 tests/test_synthetic.py` |
| `update.sh` | Holt die neueste Version auf den Home-Assistant-Rechner |

## Modell

- **Zustände:** Raumtemperatur der Leiträume (gemessen), Estrichtemperatur (geschätzt)
- **Eingänge:** Außentemperatur, Heizleistung (Durchfluss × Spreizung je Heizkreis), PV-Leistung als Sonnenersatz
- **Lernen:** jede Nacht 02:10 Least-Squares-Anpassung auf den letzten 21 Tagen; neue Parameter
  werden nur übernommen, wenn sie die letzten 48 h (nicht zum Lernen benutzt) mindestens gleich gut vorhersagen
- **Laufend:** alle 15 min Kalman-Filter (Rückläufe dienen als Messung der Estrichtemperatur)
- **Manuell neu lernen:** Event `hausmodell_lernen` in Home Assistant auslösen

## Sensoren in Home Assistant

`sensor.hausmodell_waermeverlust`, `_guete`, `_estrich_temperatur`, `_heizleistung`,
`_auskuehlzeit`, `_waermebedarf_24h`

## Installation auf Home Assistant (einmalig, im Terminal-App)

```sh
cd /addon_configs/a0d7b954_appdaemon
git clone https://github.com/Phil9716/homeassistant_model.git
rm -rf apps/hausmodell && ln -s ../homeassistant_model/apps/hausmodell apps/hausmodell
```

Danach genügt für Updates: `sh /addon_configs/a0d7b954_appdaemon/homeassistant_model/update.sh`
