"""Offline-Test: erzeugt synthetische Hausdaten mit bekannten Parametern und prüft,
ob das Modell sie wiederfindet.   Aufruf:  python3 tests/test_synthetic.py"""
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- minimaler Ersatz für appdaemon.plugins.hass.hassapi -------------------
class _Hass:
    def __init__(self, args, hist, states):
        self.args, self._h, self._s, self.out = args, hist, states, {}

    def log(self, m, level="INFO"):
        print(level, m)

    def get_state(self, e):
        return self._s.get(e)

    def set_state(self, e, state=None, attributes=None):
        self.out[e] = state

    def get_history(self, entity_id, **kw):
        return [self._h.get(entity_id, [])]

    def run_every(self, *a): pass
    def run_daily(self, *a): pass
    def run_in(self, *a): pass
    def listen_event(self, *a): pass

    def call_service(self, *a, **k):
        raise RuntimeError("offline")


for name in ("appdaemon", "appdaemon.plugins", "appdaemon.plugins.hass"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["appdaemon.plugins.hass.hassapi"] = types.SimpleNamespace(Hass=_Hass)
sys.path.insert(0, os.path.join(ROOT, "apps", "hausmodell"))
import waermemodell as hm  # noqa: E402

TRUE = {"R_is": 0.0015, "R_ia": 0.005, "C_i": 6e6, "C_s": 6e7, "a_sol": 0.1}


def main():
    rng = np.random.default_rng(1)
    n = 96 * 21
    t = pd.date_range(pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(days=21), periods=n, freq="15min")
    h = np.arange(n) / 4 % 24
    Ta = 5 + 5 * np.sin((h - 9) / 24 * 2 * np.pi)
    Ppv = np.clip(4000 * np.sin((h - 7) / 12 * np.pi), 0, None)
    Q = np.where((h > 2) & (h < 8), 6000, np.where((h > 11) & (h < 15), 4000, 0.0))
    Ti, Ts = hm.simulate(TRUE, 21, 24, Ta, Q, Ppv)
    Ti = Ti + rng.normal(0, 0.05, n)
    rl = Ts + 0.5
    vl = rl + Q / (1900 * 1.163)

    def mk(v):
        return [{"last_changed": str(t[i]), "state": str(v[i])} for i in range(n)]

    hist = {"s.ti": mk(Ti), "s.ta": mk(Ta), "s.pv": mk(Ppv), "s.vl": mk(vl), "s.rl": mk(rl)}
    states = {"s.ti": "21", "s.ta": "5", "s.pv": "0", "s.vl": "30", "s.rl": "27", "i.fl": "1900"}
    store = os.path.join(tempfile.mkdtemp(), "state.json")
    args = {"raumtemperatur": ["s.ti"], "aussentemperatur": "s.ta", "pv_leistung": "s.pv", "vorlauf": "s.vl",
            "heizkreise": [{"ruecklauf": "s.rl", "durchfluss": "i.fl"}], "speicher": store, "wetter": "w.x"}
    app = hm.Hausmodell(args, hist, states)
    app.initialize()
    app.learn({})
    s = app._summary()
    print(s, app.fit_info)
    assert app.fit_info["rmse_48h"] < 0.1, "Vorhersagefehler zu groß"
    assert abs(s["H_W_per_K"] - 200) < 20, "Wärmeverlust nicht wiedergefunden"
    app.step({})
    print("OK", app.out)


if __name__ == "__main__":
    main()
