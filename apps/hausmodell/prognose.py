"""
Prognoseabgleich – vergleicht jede Prognose mit dem, was tatsächlich eingetreten ist, und lernt daraus.

Alle 15 min:
  1. Schnappschuss: Aus dem aktuellen Fahrplan werden die Prognosewerte für feste Vorlaufzeiten
     (15 min, 1, 3, 6, 12, 24 h) gemerkt – je Größe (PV roh, PV korrigiert, Leiträume, Grundlast, Räume).
  2. Abgleich: Prognosen, deren Zielzeitpunkt jetzt erreicht ist, werden mit dem Messwert verglichen
     und ins Fehlerarchiv geschrieben.
  3. Nowcast: Verhältnis gemessene / prognostizierte PV der letzten 30 min.

Nachts: Korrekturfaktoren für die PV-Prognose je Tagesstunde aus den letzten 14 Tagen (zum Wert 1
geschrumpft, solange wenig Daten vorliegen) und Gütekennzahlen je Größe und Vorlaufzeit.

Andere Apps nutzen pv_korrektur() – der Optimierer plant damit mit der korrigierten PV-Prognose.
"""

import json
import os
from collections import deque
from datetime import datetime

import appdaemon.plugins.hass.hassapi as hass
import numpy as np
import pandas as pd

LEADS_H = [0.25, 1, 3, 6, 12, 24]
STEP_MIN = 15


def _nz(v, nd=2):
    r = round(float(v), nd)
    return r if r != 0 else 0.0001


def hourly_factors(rows, n0_kwh=2.0, min_fc_kw=0.2):
    """Korrekturfaktor je Tagesstunde: (Σ Ist + n0) / (Σ Prognose + n0) – schrumpft bei wenig Daten zu 1."""
    act = {h: 0.0 for h in range(24)}
    fc = {h: 0.0 for h in range(24)}
    for r in rows:
        if r["g"] != "pv_roh" or r["l"] < 1 or r["l"] > 6 or r["p"] < min_fc_kw:
            continue
        h = r["h"]
        act[h] += r["i"] * STEP_MIN / 60
        fc[h] += r["p"] * STEP_MIN / 60
    return {h: round((act[h] + n0_kwh) / (fc[h] + n0_kwh), 3) for h in range(24)}


def correct_pv(raw_kw, hours, leads_h, factors, nowcast):
    """Korrigierte PV: Stundenfaktor, in den ersten Stunden zum aktuellen Ist/Prognose-Verhältnis hin gezogen."""
    raw_kw = np.asarray(raw_kw, float)
    k = np.array([factors.get(int(h), 1.0) for h in hours])
    if nowcast is None:
        return raw_kw * k
    w = np.exp(-np.asarray(leads_h, float) / 2.0)          # Einfluss: 2 h noch ~37 %, 6 h ~5 %
    return raw_kw * ((1 - w) * k + w * nowcast)


def error_stats(rows, group):
    out = {}
    for lead in LEADS_H:
        e = [r["p"] - r["i"] for r in rows if r["g"] == group and r["l"] == lead]
        if len(e) >= 4:
            e = np.array(e)
            out[str(lead)] = {"mae": round(float(np.mean(np.abs(e))), 2),
                              "bias": round(float(np.mean(e)), 2), "n": int(len(e))}
    return out


class Prognose(hass.Hass):

    def initialize(self):
        a = self.args
        self.dir = a.get("ordner", "/config/prognose")
        os.makedirs(self.dir, exist_ok=True)
        self.f_pending = os.path.join(self.dir, "offen.json")
        self.f_errors = os.path.join(self.dir, "fehler.jsonl")
        self.plan = a.get("plan", "sensor.optimierer_plan")
        self.ist = a["ist"]                     # {"pv_roh": sensor, "raum": sensor, ...}
        self.raeume = a.get("raeume", {})       # {"Bad": "sensor.prognose_bad", ...}
        self.verbrauch = a.get("verbrauch")
        self.wp = a.get("wp_leistung")
        self.days = int(a.get("lern_tage", 14))
        self.pending = {}
        self.rows = []
        self.factors = {h: 1.0 for h in range(24)}
        self.nowcast = None
        self.pv_window = deque(maxlen=2)
        self._load()
        self.relearn({})
        self.run_every(self.tick, "now+150", STEP_MIN * 60)
        self.run_daily(self.relearn, "03:30:00")

    # ------------------------------------------------------------ storage
    def _load(self):
        try:
            with open(self.f_pending) as f:
                self.pending = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.pending = {}
        cutoff = datetime.now().timestamp() - self.days * 86400
        try:
            with open(self.f_errors) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("t", 0) >= cutoff:
                        self.rows.append(r)
        except FileNotFoundError:
            pass

    def _save_pending(self):
        tmp = self.f_pending + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.pending, f)
        os.replace(tmp, self.f_pending)

    # ------------------------------------------------------------- values
    def _f(self, e, d=None):
        try:
            return float(self.get_state(e))
        except (TypeError, ValueError):
            return d

    def _actuals(self):
        act = {}
        pv = self._f(self.ist.get("pv_roh"))
        if pv is not None:
            act["pv_roh"] = act["pv_korr"] = pv / 1000.0
        ti = self._f(self.ist.get("raum"))
        if ti is not None:
            act["raum"] = ti
        v, w = self._f(self.verbrauch), self._f(self.wp, 0.0)
        if v is not None:
            act["grundlast"] = max(v - (w or 0.0), 0.0) / 1000.0
        for name, ent in self.raeume.items():
            m = self._f(self.get_state(ent, attribute="quelle") or "")
            if m is not None:
                act[f"raum:{name}"] = m
        return act

    def _series(self):
        """Prognosereihen aus dem aktuellen Plan: {gruppe: {ms: wert}}."""
        attrs = self.get_state(self.plan, attribute="all") or {}
        attrs = attrs.get("attributes", {})
        src = {"pv_roh": "pv_kw_roh", "pv_korr": "pv_kw", "raum": "raum_plan", "grundlast": "grundlast_kw"}
        out = {}
        for g, key in src.items():
            lst = attrs.get(key) or []
            out[g] = {int(p[0]): float(p[1]) for p in lst if len(p) == 2}
        for name, ent in self.raeume.items():
            lst = self.get_state(ent, attribute="prognose") or []
            out[f"raum:{name}"] = {int(p[0]): float(p[1]) for p in lst if len(p) == 2}
        return out

    # --------------------------------------------------------------- tick
    def tick(self, kwargs):
        now = pd.Timestamp.now(tz="UTC").floor(f"{STEP_MIN}min")
        now_ms = int(now.timestamp() * 1000)
        act = self._actuals()
        hour = int(now.tz_convert("Europe/Berlin").hour)

        # 1) fällige Prognosen auswerten
        due = [k for k in self.pending if int(k) <= now_ms]
        new_rows = []
        for k in due:
            for g, lead, val in self.pending.pop(k):
                if g in act and int(k) > now_ms - 2 * STEP_MIN * 60000:   # nur frische Ziele
                    new_rows.append({"t": int(k) / 1000, "h": hour, "g": g, "l": lead,
                                     "p": round(val, 3), "i": round(act[g], 3)})
        if new_rows:
            with open(self.f_errors, "a") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")
            self.rows.extend(new_rows)

        # 2) neuen Schnappschuss merken
        series = self._series()
        for g, vals in series.items():
            for lead in LEADS_H:
                tgt = now_ms + int(lead * 3600 * 1000)
                if tgt in vals:
                    self.pending.setdefault(str(tgt), []).append((g, lead, vals[tgt]))
        self._save_pending()

        # 3) Nowcast: Ist/Prognose der letzten 30 min (nur wenn nennenswert PV erwartet)
        fc_now = series.get("pv_roh", {}).get(now_ms + STEP_MIN * 60000)
        if fc_now is not None and "pv_roh" in act:
            self.pv_window.append((act["pv_roh"], fc_now))
        fc_sum = sum(f for _, f in self.pv_window)
        if len(self.pv_window) >= 1 and fc_sum >= 0.6:
            self.nowcast = float(np.clip(sum(i for i, _ in self.pv_window) / fc_sum, 0.1, 1.6))
        else:
            self.nowcast = None
        self._publish()

    # ------------------------------------------------------------ learning
    def relearn(self, kwargs):
        cutoff = datetime.now().timestamp() - self.days * 86400
        self.rows = [r for r in self.rows if r["t"] >= cutoff]
        self.factors = hourly_factors(self.rows)
        self.log(f"PV-Korrekturfaktoren gelernt ({len(self.rows)} Abgleiche): "
                 f"{ {h: f for h, f in self.factors.items() if f != 1.0} }")
        self._publish()

    def pv_korrektur(self, idx, raw_w):
        """Für den Optimierer: korrigierte PV-Prognose in W für die Zeitachse idx (UTC)."""
        now = pd.Timestamp.now(tz="UTC")
        leads = np.array([(t - now).total_seconds() / 3600 for t in idx])
        hours = idx.tz_convert("Europe/Berlin").hour
        return correct_pv(np.asarray(raw_w) / 1000.0, hours, leads, self.factors, self.nowcast) * 1000.0

    def _publish(self):
        pv = error_stats(self.rows, "pv_roh")
        pvk = error_stats(self.rows, "pv_korr")
        mae1 = pv.get("1", {}).get("mae")
        self.set_state("sensor.prognose_guete_pv", state=str(mae1 if mae1 is not None else "unknown"), attributes={
            "friendly_name": "Prognosegüte PV (Fehler 1 h vorher)", "unit_of_measurement": "kW",
            "icon": "mdi:solar-power-variant", "state_class": "measurement",
            "roh": pv, "korrigiert": pvk, "nowcast_faktor": _nz(self.nowcast) if self.nowcast else "–",
            "faktor_je_stunde": {str(h): _nz(f, 2) for h, f in self.factors.items()},
            "abgleiche": len(self.rows)})
        raum = error_stats(self.rows, "raum")
        mae6 = raum.get("6", {}).get("mae")
        rooms = {name: error_stats(self.rows, f"raum:{name}") for name in self.raeume}
        self.set_state("sensor.prognose_guete_raum", state=str(mae6 if mae6 is not None else "unknown"), attributes={
            "friendly_name": "Prognosegüte Leiträume (Fehler 6 h vorher)", "unit_of_measurement": "K",
            "icon": "mdi:home-thermometer-outline", "state_class": "measurement",
            "leitraeume": raum, "raeume": rooms, "grundlast": error_stats(self.rows, "grundlast")})
