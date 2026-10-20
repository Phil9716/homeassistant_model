"""Offline-Test Überschussregler: Regelkreis mit einfacher WP-Strecke.  python3 tests/test_regler.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_synthetic  # noqa: F401,E402
sys.path.insert(0, os.path.join(ROOT, "apps", "hausmodell"))
import regler as rg  # noqa: E402
from wpmodell import cop_model  # noqa: E402


def cop(vl, ta):
    return float(cop_model(0.45, 0.1, vl, ta))


G_TRUE, TS, TA = 1.4, 23.0, 6.0          # wahre Wärmeabgabe weicht vom Startwert 1,0 ab
cfg = dict(rg.DEFAULT_CFG)
st = {"g": 1.0, "korr": 0.0}
vl_soll, vl_ist, log = 32.0, 31.0, []
for minute in range(240):
    surplus = 2500.0 if minute < 80 else (900.0 if minute < 140 else 3500.0)   # Sonne, Wolke, volle Sonne
    raum_max = 22.0 if minute < 200 else 23.6                                  # zum Schluss: T_max erreicht
    vl_ist += (vl_soll - vl_ist) * 0.25                                         # WP folgt träge
    q = G_TRUE * (vl_ist - TS)
    p_el = q / cop(vl_ist, TA) * 1000.0
    inp = {"ueberschuss_w": surplus, "p_el_w": p_el, "ts": TS, "ta": TA, "q_kw": q, "vl_mess": vl_ist,
           "vl_soll": vl_soll, "verdichter": True, "sperre": None, "raum_min": 21.5, "raum_max": raum_max,
           "t_min": 21.0, "t_max": 23.5, "q_plan_kw": 5.0, "jetzt_min": float(minute)}
    new, info = rg.regler_schritt(st, inp, cfg, cop)
    if new is not None:
        vl_soll = float(new)
    log.append((minute, surplus, round(p_el), vl_soll, info["modus"]))

for m in (20, 79, 100, 139, 170, 199, 239):
    print(log[m])
print("gelerntes G:", round(st["g"], 2))


def err(m):
    return abs(log[m][2] - (log[m][1] - cfg["reserve_w"]))


assert err(79) < 350, "Sonne: Überschuss nicht getroffen"
assert err(139) < 350, "Wolke: nicht heruntergeregelt"
assert err(199) < 400, "volle Sonne: nicht hochgeregelt"
assert log[239][3] <= TS + 3, "T_max: Vorlauf nicht minimal"
assert 1.2 < st["g"] < 1.6, "Wärmeabgabe der FBH nicht gelernt"
# Schreibhäufigkeit: höchstens alle 3 min
writes = [i for i in range(1, len(log)) if log[i][3] != log[i - 1][3]]
assert all(b - a >= 3 for a, b in zip(writes, writes[1:])), "zu häufig geschrieben"
print("Schreibvorgänge in 4 h:", len(writes))
print("OK")
