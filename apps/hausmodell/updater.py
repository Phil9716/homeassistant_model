"""
Updater – holt neue Versionen der Hausmodell-Dateien direkt von GitHub.

Auslösen: Event `hausmodell_update` in Home Assistant (z. B. per Dashboard-Knopf).
AppDaemon erkennt geänderte Dateien und lädt die Apps danach automatisch neu.
"""

import hashlib
import os
import urllib.request

import appdaemon.plugins.hass.hassapi as hass


class Updater(hass.Hass):

    def initialize(self):
        self.base = self.args.get(
            "quelle", "https://raw.githubusercontent.com/Phil9716/homeassistant_model/main/apps/hausmodell/")
        self.files = self.args.get("dateien", ["waermemodell.py", "wpmodell.py", "updater.py", "hausmodell.yaml"])
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.listen_event(self.update, "hausmodell_update")
        self._publish("bereit", {})

    def _publish(self, state, attrs):
        self.set_state("sensor.hausmodell_update", state=state,
                       attributes={"friendly_name": "Hausmodell Update", "icon": "mdi:github", **attrs})

    def update(self, event, data, kwargs):
        changed, errors = [], []
        # Erst alles laden, dann schreiben – so bleibt bei Netzfehlern der alte Stand vollständig
        downloads = {}
        for name in self.files:
            try:
                with urllib.request.urlopen(self.base + name + "?nocache=" + os.urandom(4).hex(), timeout=20) as r:
                    downloads[name] = r.read()
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {e}")
        if errors:
            self.log(f"Update abgebrochen: {errors}", level="WARNING")
            self._publish("fehler", {"fehler": errors})
            return
        # .py zuerst, .yaml zuletzt, damit die Konfiguration nie auf fehlenden Code zeigt
        for name in sorted(downloads, key=lambda n: n.endswith(".yaml")):
            path = os.path.join(self.dir, name)
            new = downloads[name]
            try:
                with open(path, "rb") as f:
                    old = f.read()
            except FileNotFoundError:
                old = b""
            if hashlib.sha256(old).digest() != hashlib.sha256(new).digest():
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(new)
                os.replace(tmp, path)
                changed.append(name)
        self.log(f"Update fertig, geändert: {changed or 'nichts'}")
        self._publish("aktuell", {"geaendert": changed})
