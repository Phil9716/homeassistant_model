"""Offline-Test des Optimierers (nur die LP-Funktion).  python3 tests/test_optimierer.py"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_synthetic  # noqa: F401,E402  (registriert den Hass-Ersatz)
sys.path.insert(0, os.path.join(ROOT, "apps", "hausmodell"))
import optimierer as op  # noqa: E402
import waermemodell as hm  # noqa: E402

CFG = {"preis": 0.30, "strafe_kalt": 0.6, "strafe_warm": 0.2, "restwert_akku": 0.15, "akku_kwh": 7.6,
       "akku_kw": 5.0, "eta_laden": 0.95, "eta_entladen": 0.85, "q_max_kw": 8.0}


def scenario(start_hour=8, pv_peak=8.0):
    n = op.STEPS
    hour = (start_hour + np.arange(1, n + 1) * op.DT_H) % 24
    ta = 8 + 4 * np.sin((hour - 9) / 24 * 2 * np.pi)
    pv = np.clip(pv_peak * 1000 * np.sin((hour - 8) / 9 * np.pi), 0, None)
    base = np.full(n, 0.35)
    cop = np.full(n, 4.0)
    t_min = np.where((hour >= 6) & (hour < 22), 21.0, 20.5)
    t_max = np.full(n, 23.5)
    return hour, ta, pv, base, cop, t_min, t_max


def run(pv_peak, x0=(21.5, 23.5)):
    hour, ta, pv, base, cop, t_min, t_max = scenario(pv_peak=pv_peak)
    res = op.plan(dict(hm.DEFAULT_PARAMS), list(x0), ta, pv, base, cop, t_min, t_max, 0.0, CFG)
    assert res is not None, "LP nicht lösbar"
    sunny = (hour >= 8) & (hour < 17)
    print(f"PV-Spitze {pv_peak} kW: Wärme 24h {res['q'][:96].sum() * op.DT_H:.1f} kWh, "
          f"davon Sonnenstunden {res['q'][:96][sunny[:96]].sum() * op.DT_H:.1f} kWh, "
          f"Netz {res['g'][:96].sum() * op.DT_H:.1f} kWh, Raum min/max {res['ti'].min():.1f}/{res['ti'].max():.1f}")
    return res, sunny


res, sunny = run(8.0)
assert res["q"][:96][sunny[:96]].sum() > 0.6 * res["q"][:96].sum(), "bei Sonne sollte überwiegend mittags geheizt werden"
assert res["ti"][40:].min() > 20.2, "Komfort nachts verletzt"
res2, _ = run(0.0)
res3, _ = run(8.0, x0=(19.4, 23.0))
assert res3["q"][:8].mean() > 4, "kaltes Haus muss sofort geheizt werden"
assert res2["g"].sum() > res["g"].sum(), "ohne Sonne muss mehr Netzstrom anfallen"
print("OK")
