"""
Überschussregler – führt die Vorlauftemperatur so nach, dass die WP genau den PV-Überschuss aufnimmt.

Arbeitsteilung: Der Optimierer entscheidet, OB geheizt wird (Ein/Aus). Läuft die WP in Betriebsart „Manuell“,
stellt der Regler jede Minute ein, WIE STARK – über den Vorlauf-Sollwert (Min = Max = Sollwert).

  Ziel-Strom   P = gemessener Überschuss − Reserve   (Überschuss inkl. Akku-Ladeleistung, ohne WP)
               bei Untertemperatur mindestens die geplante Leistung des Optimierers, bei T_max nur Minimum
  Vorsteuerung Q = P · COP(VL, Luft)   →   VL = T_Estrich + Q / G          (G = Wärmeabgabe FBH je K, gelernt)
  Nachregelung I-Anteil auf (P_Ziel − P_gemessen), begrenzt, eingefroren bei Abtauen/Warmwasser/Begrenzung
  Schreiben    nur ganze Grad, höchstens alle 3 min
"""

import json
from collections import deque
from datetime import datetime

import appdaemon.plugins.hass.hassapi as hass
import numpy as np

DEFAULT_CFG = {"reserve_w": 200.0, "p_max_w": 4000.0, "vl_min": 25.0, "vl_max": 40.0, "ki": 0.15,
               "korr_max": 6.0, "schreib_min": 3.0, "g_start": 1.0, "g_lo": 0.4, "g_hi": 3.0, "g_alpha": 0.05}


def feedforward_vl(p_w, ts, ta, g, cop_fn, vl_lo, vl_hi):
    """Vorlauf, bei dem die WP etwa p_w Watt Strom aufnimmt (Fixpunkt, COP hängt vom Vorlauf ab)."""
    vl = ts + 5.0
    for _ in range(6):
        q_kw = max(p_w, 0.0) / 1000.0 * cop_fn(vl, ta)
        vl = float(np.clip(ts + q_kw / g, vl_lo, vl_hi))
    return vl


def regler_schritt(st, inp, cfg, cop_fn):
    """Ein Reglerschritt (1 min). st wird verändert. Rückgabe: (neuer Sollwert oder None, Info)."""
    info = {}
    if inp.get("ueberschuss_w") is None or inp.get("ts") is None or inp.get("p_el_w") is None:
        return None, {"modus": "warte auf Messwerte"}
    ts, ta = inp["ts"], inp["ta"]
    vl_lo = max(cfg["vl_min"], ts + 2.0)
    vl_hi = cfg["vl_max"]

    # 1) Ziel-Leistung
    p_ziel, modus = inp["ueberschuss_w"] - cfg["reserve_w"], "Überschuss"
    cop_now = cop_fn(inp.get("vl_mess") or ts + 8.0, ta)
    p_plan = (inp.get("q_plan_kw") or 0.0) * 1000.0 / max(cop_now, 1.5)
    if inp.get("raum_min") is not None and inp.get("t_min") is not None and inp["raum_min"] < inp["t_min"] \
            and p_plan > p_ziel:
        p_ziel, modus = p_plan, "Komfort (Plan des Optimierers)"
    if inp.get("raum_max") is not None and inp["raum_max"] >= inp["t_max"]:
        p_ziel, modus = 0.0, "Komfortgrenze erreicht – minimal"
    p_ziel = float(np.clip(p_ziel, 0.0, cfg["p_max_w"]))

    # 2) Wärmeabgabe der FBH lernen: G = Q / (VL − T_Estrich), nur im ruhigen Heizbetrieb
    q, vl_m = inp.get("q_kw"), inp.get("vl_mess")
    if inp.get("verdichter") and q and vl_m and q > 2.0 and vl_m - ts > 2.0 \
            and inp.get("vl_soll") is not None and abs(vl_m - inp["vl_soll"]) < 1.5:
        g_new = q / (vl_m - ts)
        st["g"] = float(np.clip((1 - cfg["g_alpha"]) * st["g"] + cfg["g_alpha"] * g_new, cfg["g_lo"], cfg["g_hi"]))

    # 3) Vorsteuerung + I-Anteil
    vl_ff = feedforward_vl(p_ziel, ts, ta, st["g"], cop_fn, vl_lo, vl_hi)
    fehler_w = p_ziel - inp["p_el_w"]
    if inp.get("verdichter") and not inp.get("sperre"):
        step = cfg["ki"] * fehler_w / 1000.0
        neu = float(np.clip(st["korr"] + step, -cfg["korr_max"], cfg["korr_max"]))
        # Anti-Windup: nicht weiter in die Richtung integrieren, in der der Sollwert schon anliegt
        if not ((vl_ff + neu >= vl_hi and step > 0) or (vl_ff + neu <= vl_lo and step < 0)):
            st["korr"] = neu
    vl = float(np.clip(vl_ff + st["korr"], vl_lo, vl_hi))
    vl_int = int(round(vl))
    info.update({"modus": modus, "p_ziel_w": round(p_ziel), "p_mess_w": round(inp["p_el_w"]),
                 "fehler_w": round(fehler_w), "vl_vorsteuerung": round(vl_ff, 1), "korrektur_k": round(st["korr"], 2),
                 "vl_rechnerisch": round(vl, 1), "g_kw_pro_k": round(st["g"], 2), "estrich": round(ts, 1)})

    # 4) Schreiben?
    if inp.get("sperre"):
        info["modus"] = f"{modus} – pausiert ({inp['sperre']})"
        return None, info
    cur = inp.get("vl_soll")
    since = inp["jetzt_min"] - st.get("last_write", -1e9)
    if cur is None or (abs(vl_int - cur) >= 1 and since >= cfg["schreib_min"]):
        st["last_write"] = inp["jetzt_min"]
        return vl_int, info
    return None, info


class Regler(hass.Hass):

    def initialize(self):
        a = self.args
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update({k: float(v) for k, v in a.get("parameter", {}).items()})
        self.e = a["entitaeten"]
        self.store = a.get("speicher", "/config/regler_state.json")
        self.st = {"g": self.cfg["g_start"], "korr": 0.0}
        try:
            with open(self.store) as f:
                self.st["g"] = float(json.load(f).get("g", self.cfg["g_start"]))
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            pass
        self.surplus = deque(maxlen=5)
        self.active = False
        self.run_every(self.step, "now+20", 60)

    def _f(self, e, d=None):
        if not e:
            return d
        try:
            return float(self.get_state(e))
        except (TypeError, ValueError):
            return d

    def aktiv(self):
        return self.active

    def _cop(self, vl, ta):
        try:
            return max(self.get_app("wpmodell").cop(vl, ta), 1.5)
        except Exception:  # noqa: BLE001
            return 3.5

    def step(self, kwargs):
        e = self.e
        s = self._f(e["ueberschuss"])
        if s is not None:
            self.surplus.append(s)
        betrieb = self.get_state(e["betriebsart"])
        enabled = (self.get_state(e["automatik"]) == "on" and self.get_state(e["regler_an"]) != "off"
                   and self.get_state(e["ebus_ok"]) == "on" and betrieb == "Manuell")
        self.active = enabled
        if not enabled:
            self.st["korr"] = 0.0
            self._publish(None, {"modus": f"inaktiv (Betriebsart {betrieb})"})
            return
        try:
            haus = self.get_app("hausmodell")
            ts = float(haus.x[1]) if getattr(haus, "x", None) is not None else None
        except Exception:  # noqa: BLE001
            ts = None
        try:
            opt = self.get_app("optimierer")
            q_plan, t_min = getattr(opt, "q_next", None), getattr(opt, "t_min_now", None)
            t_max = opt.t_max
        except Exception:  # noqa: BLE001
            q_plan, t_min, t_max = None, None, 23.5
        status = str(self.get_state(e["status"]) or "")
        sperre = None
        if "defrost" in status or "hwc" in status:
            sperre = "Abtauen/Warmwasser"
        vl_soll = self._f(e["min_vorlauf"])
        inp = {"ueberschuss_w": float(np.mean(self.surplus)) if self.surplus else None,
               "p_el_w": self._f(e["wp_strom"]), "ts": ts,
               "ta": self._f(e["luft"], self._f(e["aussen"], 5.0)),
               "q_kw": self._f(e["waerme"]), "vl_mess": self._f(e["vorlauf"]), "vl_soll": vl_soll,
               "verdichter": "compressor_active" in status, "sperre": sperre,
               "raum_min": self._f(e["raum_min"]), "raum_max": self._f(e["raum_max"]),
               "t_min": t_min, "t_max": t_max, "q_plan_kw": q_plan,
               "jetzt_min": datetime.now().timestamp() / 60.0}
        g_old = self.st["g"]
        vl, info = regler_schritt(self.st, inp, self.cfg, self._cop)
        if vl is not None:
            self.call_service("number/set_value", entity_id=e["min_vorlauf"], value=vl)
            self.call_service("number/set_value", entity_id=e["max_vorlauf"], value=max(vl, 30))
            info["geschrieben"] = f"{vl} °C um {datetime.now().strftime('%H:%M')}"
            self.log(f"Regler: Vorlauf → {vl} °C ({info})")
        if abs(self.st["g"] - g_old) > 0.01:
            try:
                with open(self.store, "w") as f:
                    json.dump({"g": self.st["g"]}, f)
            except OSError:
                pass
        self._publish(vl if vl is not None else vl_soll, info)

    def _publish(self, vl, info):
        p = info.get("p_ziel_w")
        self.set_state("sensor.ueberschussregler", state=str(p if p else 0.0001), attributes={
            "friendly_name": "Überschussregler Ziel-Leistung", "unit_of_measurement": "W", "device_class": "power",
            "state_class": "measurement", "icon": "mdi:sine-wave", "vorlauf_soll": vl,
            "ueberschuss_w": round(float(np.mean(self.surplus))) if self.surplus else None, **info})
