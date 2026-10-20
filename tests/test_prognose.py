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
            rows.append({"t": 0, "h": h, "g": "pv_roh", "l": 1, "p": fc, "i": act})
f = pg.hourly_factors(rows)
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
st = pg.error_stats(rows, "pv_roh")
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
plan = {"pv_kw_roh": [[m, 3.0] for m in ms], "pv_kw": [[m, 3.0] for m in ms],
        "raum_plan": [[m, 21.0] for m in ms], "grundlast_kw": [[m, 0.4] for m in ms]}
d = tempfile.mkdtemp()
app = FakeApp({"ordner": d, "ist": {"pv_roh": "s.pv", "raum": "s.ti"}, "verbrauch": "s.v", "wp_leistung": "s.wp"},
              {"s.pv": "800", "s.ti": "20.5", "s.v": "500", "s.wp": "0"}, {"sensor.optimierer_plan": plan})
app.initialize()
app.tick({})
assert any(app.pending.values()), "kein Schnappschuss gespeichert"
# 15 min später: Ziel der 15-min-Prognose ist erreicht
key = min(app.pending, key=int)
app.pending = {str(int(now.timestamp() * 1000)): app.pending[key]}
app.tick({})
assert app.rows and app.rows[0]["i"] == 0.8, "Abgleich mit Messwert fehlt"
print("Nowcast-Faktor:", app.nowcast, "| Abgleiche:", len(app.rows))
assert app.nowcast is not None and app.nowcast < 0.4
print("OK")
