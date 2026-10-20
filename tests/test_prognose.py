"""Offline-Test Prognoseabgleich.  python3 tests/test_prognose.py"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_synthetic  # noqa: F401,E402  (registriert den Hass-Ersatz)
sys.path.insert(0, os.path.join(ROOT, "apps", "hausmodell"))
import prognose as pg  # noqa: E402

# 1) Lernen: morgens (8–10 Uhr) ist die Prognose systematisch doppelt so hoch wie die Wirklichkeit
rows = []
for day in range(14):
    for h in range(7, 17):
        for _ in range(4):
            fc = 3.0
            act = 1.5 if h in (8, 9) else 3.0
            rows.append({"t": 0, "h": h, "g": "pv:ost", "l": 1, "p": fc, "i": act})
            rows.append({"t": 0, "h": h, "g": "pv:west", "l": 1, "p": fc, "i": fc})
f = pg.hourly_factors(rows, "pv:ost")
fw = pg.hourly_factors(rows, "pv:west")
assert fw[8] == 1.0, "West darf nicht von Ost-Fehlern beeinflusst werden"
print("Faktoren:", {h: f[h] for h in range(7, 12)})
assert 0.5 < f[8] < 0.6 and 0.5 < f[9] < 0.6, "Morgen-Überschätzung nicht gelernt"
assert 0.97 < f[12] < 1.03, "Mittag fälschlich korrigiert"
assert f[3] == 1.0, "Nachtstunden ohne Daten müssen 1 bleiben"

# 2) Live-Korrektur: gemessen 25 % der Prognose -> erste Stunde stark, nach 6 h kaum noch korrigiert
raw = np.full(25, 4.0)
leads = np.arange(25) * 0.25
out = pg.correct_pv(raw, [12] * 25, leads, {12: 1.0}, nowcast=0.25)
print("korrigiert 0 h / 2 h / 6 h:", out[0].round(2), out[8].round(2), out[24].round(2))
assert out[0] < 1.2 and 2.6 < out[8] < 3.1 and out[24] > 3.6

# 3) Fehlerstatistik
st = pg.error_stats(rows, "pv:ost")
assert st["1"]["bias"] > 0, "Überschätzung muss als positiver Bias erscheinen"


# 4) Durchlauf der App: Schnappschuss -> Abgleich -> Archiv
class FakeApp(pg.Prognose):
    def __init__(self, args, states, attrs):
        self.args, self._s, self._a, self.out = args, states, attrs, {}

    def get_state(self, e, attribute=None):
        if attribute == "all":
            return {"attributes": self._a.get(e, {})}
        if attribute:
            return self._a.get(e, {}).get(attribute)
        return self._s.get(e)

    def set_state(self, e, state=None, attributes=None):
        self.out[e] = (state, attributes)

    def log(self, *a, **k): pass
    def run_every(self, *a): pass
    def run_daily(self, *a): pass


now = pd.Timestamp.now(tz="UTC").floor("15min")
ms = [int((now + pd.Timedelta(minutes=15 * k)).timestamp() * 1000) for k in range(1, 121)]
plan = {"raum_plan": [[m, 21.0] for m in ms], "grundlast_kw": [[m, 0.4] for m in ms]}
watts = {p: {(now + pd.Timedelta(minutes=15 * k)).isoformat(): v for k in range(0, 121)}
         for p, v in {"ost": 3200.0, "sued": 2000.0, "west": 500.0}.items()}
flaechen = {p: {"ist": f"s.{p}", "prognose": [f"fc.{p}"]} for p in watts}
d = tempfile.mkdtemp()
app = FakeApp({"ordner": d, "pv_flaechen": flaechen, "ist": {"raum": "s.ti"}, "verbrauch": "s.v", "wp_leistung": "s.wp"},
              {"s.ost": "800", "s.sued": "2000", "s.west": "500", "s.ti": "20.5", "s.v": "500", "s.wp": "0"},
              {"sensor.optimierer_plan": plan, **{f"fc.{p}": {"watts": w} for p, w in watts.items()}})
app.initialize()
for _ in range(15):
    app.sample({})
app.tick({})
assert any(app.pending.values()), "kein Schnappschuss gespeichert"
key = min(app.pending, key=int)
app.pending = {str(int(now.timestamp() * 1000)): app.pending[key]}
app.tick({})
got = {r["g"]: r["i"] for r in app.rows}
assert got["pv:ost"] == 0.8 and got["pv_roh"] == 3.3, got
print("Nowcast je Fläche:", app.nowcast, "| Abgleiche:", len(app.rows))
assert app.nowcast["ost"] < 0.4 and 0.9 < app.nowcast["sued"] < 1.1
raw, cor = app.pv_forecast()
print("Summe roh/korr nächste 15 min [W]:", raw[0].round(), cor[0].round())
assert cor[0] < raw[0] - 1500, "Ost-Überschätzung nicht korrigiert"
a = app.out["sensor.pv_prognose"][1]
assert "ost_korr" in a and a["nowcast"]["ost"] < 0.4
print("OK")
