"""Offline-Test des WP-Modells mit synthetischen Daten.  python3 tests/test_wpmodell.py"""
import os, sys, tempfile, types
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_synthetic  # noqa: F401,E402  (registriert den Hass-Ersatz)
sys.path.insert(0, os.path.join(ROOT, "apps", "hausmodell"))
import wpmodell as wm  # noqa: E402

rng = np.random.default_rng(2)
n = 96 * 20
t = pd.date_range(pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(days=20), periods=n, freq="15min")
ta = 5 + 6 * np.sin(np.arange(n) / 96 * 2 * np.pi) + rng.normal(0, 1, n)
vl = 28 + 0.3 * (15 - ta)
on = rng.random(n) < 0.5
pth = np.where(on, 4 + rng.random(n) * 3, 0)
cop = wm.cop_model(0.5, 0.15, vl, ta) * (1 + rng.normal(0, 0.05, n))
pel = np.where(on, pth / cop, 0)
mk = lambda v: [{"last_changed": str(t[i]), "state": str(v[i])} for i in range(n)]
hist = {"pel": mk(pel), "pth": mk(pth), "vl": mk(vl), "ta": mk(ta)}
args = {"leistung_elektrisch": "pel", "leistung_thermisch": "pth", "vorlauf": "vl", "aussentemperatur": "ta",
        "speicher": os.path.join(tempfile.mkdtemp(), "wp.json")}
app = wm.WPModell(args, hist, {"vl": "30", "ta": "5", "pel": "1.5", "pth": "5"})
app.initialize(); app.learn({}); app.live({})
print(app.p, app.info, app.out)
assert abs(app.p["eta"] - 0.5) < 0.03 and abs(app.p["d"] - 0.15) < 0.05
print("OK")
