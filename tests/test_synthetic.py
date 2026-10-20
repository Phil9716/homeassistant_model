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

    # Variante mit elektrischer WP-Leistung × COP als Wärmequelle (wie in der echten Anlage)
    cop = app._cop(vl, Ta)
    pel = np.where(Q > 0, Q / cop, 8.0)
    hist["s.pel"] = mk(pel)
    rl_still = np.where(Q > 0, rl, rl - 3)      # stehende Pumpe: Rücklauffühler kühlen ab
    hist["s.rl"] = mk(rl_still)
    args2 = dict(args, wp_leistung="s.pel", speicher=os.path.join(tempfile.mkdtemp(), "s2.json"))
    app2 = hm.Hausmodell(args2, hist, dict(states, **{"s.pel": "2000"}))
    app2.initialize()
    app2.learn({})
    s2 = app2._summary()
    print(s2, app2.fit_info)
    assert app2.fit_info["rmse_48h"] < 0.15, "Vorhersagefehler (P_el-Variante) zu groß"
    assert abs(s2["H_W_per_K"] - 200) < 30, "Wärmeverlust (P_el-Variante) nicht wiedergefunden"

    # Variante 3: gemessene Wärme (eBUS, kW) erst seit 7 Tagen, davor nichts; P_el absichtlich falsch (×1,5)
    pth = np.where(np.arange(n) >= n - 96 * 7, Q / 1000.0, np.nan)
    hist["s.pth"] = [{"last_changed": str(t[i]), "state": str(pth[i])} for i in range(n) if not np.isnan(pth[i])]
    hist["s.pel"] = mk(pel)
    args3 = dict(args2, wp_waerme="s.pth", speicher=os.path.join(tempfile.mkdtemp(), "s3.json"))
    app3 = hm.Hausmodell(args3, hist, dict(states, **{"s.pel": "2000", "s.pth": "6.0"}))
    app3.initialize()
    df = app3._dataset(21)
    last = df.index >= t[n - 96 * 7]
    ref = np.where(Q > 300, Q, 0)[-last.sum():]
    assert abs(df["Q"][last].sum() / ref.sum() - 1) < 0.03, "Messwert nicht übernommen"
    assert app3.q_gemessen > 0 and app3.q_geschaetzt > 0
    assert app3._heat_now(20.0, 5.0) == 6000.0
    app3.learn({})
    s3 = app3._summary()
    print("gemessene Wärme:", s3, app3.fit_info)
    assert abs(s3["H_W_per_K"] - 200) < 30


if __name__ == "__main__":
    main()
