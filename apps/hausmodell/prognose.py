"""
Prognoseabgleich v2 – vergleicht Prognosen mit der Wirklichkeit und lernt daraus, PV getrennt je Ausrichtung.

PV je Fläche (Ost / Süd / West):
  * Rohprognose aus Open-Meteo Solar Forecast (15-min-Werte, heute/morgen/übermorgen)
  * Messung: String-Leistung, jede Minute abgetastet, über 15 min gemittelt
  * Lernend: Korrekturfaktor je Fläche und Tagesstunde aus den letzten 14 Tagen (zu 1 geschrumpft)
  * Live: Ist/Prognose-Verhältnis der letzten 30 min je Fläche, klingt mit der Vorlaufzeit ab
Weitere Größen (Leiträume, Räume, Grundlast) werden aus dem Fahrplan des Optimierers abgeglichen.

Alle 15 min: Prognose neu rechnen → fällige Prognosen mit Messwert vergleichen (Archiv) → Schnappschuss.
Nachts 03:30: Faktoren neu lernen. Der Optimierer holt sich pv_forecast(idx).
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
STEPS = 4 * 30


def _nz(v, nd=2):
    r = round(float(v), nd)
    return r if r != 0 else 0.0001


def hourly_factors(rows, group, n0_kwh=1.0, min_fc_kw=0.1):
    """Korrekturfaktor je Tagesstunde: (Σ Ist + n0) / (Σ Prognose + n0) – bei wenig Daten nahe 1."""
    act = {h: 0.0 for h in range(24)}
    fc = {h: 0.0 for h in range(24)}
    for r in rows:
        if r["g"] != group or r["l"] < 1 or r["l"] > 6 or r["p"] < min_fc_kw:
            continue
        act[r["h"]] += r["i"] * STEP_MIN / 60
        fc[r["h"]] += r["p"] * STEP_MIN / 60
    return {h: round(float(np.clip((act[h] + n0_kwh) / (fc[h] + n0_kwh), 0.2, 1.5)), 3) for h in range(24)}


def correct_pv(raw_kw, hours, leads_h, factors, nowcast):
    """Stundenfaktor; in den ersten Stunden zusätzlich zum aktuellen Ist/Prognose-Verhältnis hin gezogen."""
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
        self.flaechen = a["pv_flaechen"]        # {ost: {ist: sensor, prognose: [..]}, ...}
        self.ist = a.get("ist", {})             # {raum: sensor}
        self.raeume = a.get("raeume", {})
        self.verbrauch = a.get("verbrauch")
        self.wp = a.get("wp_leistung")
        self.days = int(a.get("lern_tage", 14))
        self.pending, self.rows = {}, []
        self.factors = {p: {h: 1.0 for h in range(24)} for p in self.flaechen}
        self.nowcast = {p: None for p in self.flaechen}
        self.samples = {p: deque(maxlen=STEP_MIN) for p in self.flaechen}
        self.nc_window = {p: deque(maxlen=2) for p in self.flaechen}
        self.fc = None
        self._load()
        self.relearn({})
        self.run_every(self.sample, "now+3", 60)
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

    def sample(self, kwargs):
        for p, cfg in self.flaechen.items():
            v = self._f(cfg["ist"])
            if v is not None:
                self.samples[p].append(max(v, 0.0) / 1000.0)

    def _timeline(self):
        t0 = pd.Timestamp.now(tz="UTC").floor(f"{STEP_MIN}min")
        return pd.date_range(t0 + pd.Timedelta(minutes=STEP_MIN), periods=STEPS, freq=f"{STEP_MIN}min")

    def _raw_plane(self, plane, idx):
        """Rohprognose einer Fläche in kW auf der Zeitachse idx."""
        total = pd.Series(0.0, index=idx)
        for ent in self.flaechen[plane]["prognose"]:
            w = self.get_state(ent, attribute="watts") or {}
            if w:
                s = pd.Series({pd.Timestamp(k): float(v) for k, v in w.items()}).sort_index()
                s.index = pd.to_datetime(s.index, utc=True)
                total = total.add(s.reindex(idx).fillna(0.0), fill_value=0.0)
        return total.values / 1000.0

    def pv_forecast(self, idx=None):
        """Für den Optimierer: (roh, korrigiert) Gesamt-PV in W auf der Zeitachse idx."""
        idx = idx if idx is not None else self._timeline()
        now = pd.Timestamp.now(tz="UTC")
        leads = np.array([(t - now).total_seconds() / 3600 for t in idx])
        hours = idx.tz_convert("Europe/Berlin").hour
        raw_tot, cor_tot, per = np.zeros(len(idx)), np.zeros(len(idx)), {}
        for p in self.flaechen:
            raw = self._raw_plane(p, idx)
            cor = correct_pv(raw, hours, leads, self.factors[p], self.nowcast[p])
            per[p] = (raw, cor)
            raw_tot += raw
            cor_tot += cor
        self.fc = (idx, per, raw_tot, cor_tot)
        return raw_tot * 1000.0, cor_tot * 1000.0

    def _actuals(self):
        act = {}
        tot = 0.0
        for p in self.flaechen:
            if self.samples[p]:
                act[f"pv:{p}"] = float(np.mean(self.samples[p]))
                tot += act[f"pv:{p}"]
        if act:
            act["pv_roh"] = act["pv_korr"] = tot
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

    def _plan_series(self):
        attrs = (self.get_state(self.plan, attribute="all") or {}).get("attributes", {})
        out = {}
        for g, key in {"raum": "raum_plan", "grundlast": "grundlast_kw"}.items():
            out[g] = {int(p[0]): float(p[1]) for p in attrs.get(key) or [] if len(p) == 2}
        for name, ent in self.raeume.items():
            lst = self.get_state(ent, attribute="prognose") or []
            out[f"raum:{name}"] = {int(p[0]): float(p[1]) for p in lst if len(p) == 2}
        return out

    # --------------------------------------------------------------- tick
    def tick(self, kwargs):
        now = pd.Timestamp.now(tz="UTC").floor(f"{STEP_MIN}min")
        now_ms = int(now.timestamp() * 1000)
        hour = int(now.tz_convert("Europe/Berlin").hour)
        act = self._actuals()

        # 1) Live-Korrektur je Fläche: Ist der letzten 15 min gegen Rohprognose dieses Zeitpunkts
        idx_now = pd.DatetimeIndex([now])
        for p in self.flaechen:
            fc_now = float(self._raw_plane(p, idx_now)[0])
            if f"pv:{p}" in act:
                self.nc_window[p].append((act[f"pv:{p}"], fc_now))
            fsum = sum(f for _, f in self.nc_window[p])
            self.nowcast[p] = (float(np.clip(sum(i for i, _ in self.nc_window[p]) / fsum, 0.1, 1.6))
                               if fsum >= 0.3 else None)

        # 2) fällige Prognosen auswerten
        new_rows = []
        for k in [k for k in self.pending if int(k) <= now_ms]:
            for g, lead, val in self.pending.pop(k):
                if g in act and int(k) > now_ms - 2 * STEP_MIN * 60000:
                    new_rows.append({"t": int(k) / 1000, "h": hour, "g": g, "l": lead,
                                     "p": round(val, 3), "i": round(act[g], 3)})
        if new_rows:
            with open(self.f_errors, "a") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")
            self.rows.extend(new_rows)

        # 3) aktuelle Prognose rechnen und Schnappschuss merken
        idx = self._timeline()
        self.pv_forecast(idx)
        _, per, raw_tot, cor_tot = self.fc
        ms = [int(t.timestamp() * 1000) for t in idx]
        series = {"pv_roh": dict(zip(ms, raw_tot)), "pv_korr": dict(zip(ms, cor_tot))}
        for p, (raw, _) in per.items():
            series[f"pv:{p}"] = dict(zip(ms, raw))
        series.update(self._plan_series())
        for g, vals in series.items():
            for lead in LEADS_H:
                tgt = now_ms + int(lead * 3600 * 1000)
                if tgt in vals:
                    self.pending.setdefault(str(tgt), []).append((g, lead, float(vals[tgt])))
        self._save_pending()
        self._publish()

    # ------------------------------------------------------------ learning
    def relearn(self, kwargs):
        cutoff = datetime.now().timestamp() - self.days * 86400
        self.rows = [r for r in self.rows if r["t"] >= cutoff]
        for p in self.flaechen:
            self.factors[p] = hourly_factors(self.rows, f"pv:{p}")
        self.log(f"PV-Faktoren gelernt ({len(self.rows)} Abgleiche): "
                 + "; ".join(f"{p}: { {h: f for h, f in fa.items() if f != 1.0} }" for p, fa in self.factors.items()))
        self._publish()

    # ------------------------------------------------------------ publish
    def _publish(self):
        attrs = {"friendly_name": "PV-Prognose (korrigiert, nächste 15 min)", "unit_of_measurement": "kW",
                 "icon": "mdi:solar-power-variant", "state_class": "measurement"}
        state = "unknown"
        if self.fc is not None:
            idx, per, raw_tot, cor_tot = self.fc
            ms = [int(t.timestamp() * 1000) for t in idx]

            def ser(a):
                return [[m, _nz(v)] for m, v in zip(ms, a)]
            for p, (raw, cor) in per.items():
                attrs[f"{p}_roh"], attrs[f"{p}_korr"] = ser(raw), ser(cor)
                attrs[f"{p}_heute_kwh_roh"] = _nz(raw[:48].sum() * STEP_MIN / 60, 1)
                attrs[f"{p}_heute_kwh_korr"] = _nz(cor[:48].sum() * STEP_MIN / 60, 1)
            attrs["gesamt_roh"], attrs["gesamt_korr"] = ser(raw_tot), ser(cor_tot)
            state = str(_nz(cor_tot[0]))
        attrs["nowcast"] = {p: (_nz(v) if v is not None else "–") for p, v in self.nowcast.items()}
        self.set_state("sensor.pv_prognose", state=state, attributes=attrs)

        pv = {p: error_stats(self.rows, f"pv:{p}") for p in self.flaechen}
        tot = error_stats(self.rows, "pv_roh")
        mae1 = tot.get("1", {}).get("mae")
        self.set_state("sensor.prognose_guete_pv", state=str(mae1 if mae1 is not None else "unknown"), attributes={
            "friendly_name": "Prognosegüte PV (Fehler 1 h vorher)", "unit_of_measurement": "kW",
            "icon": "mdi:solar-power-variant", "state_class": "measurement",
            "roh": tot, "korrigiert": error_stats(self.rows, "pv_korr"), "flaechen": pv,
            "nowcast_faktor": {p: (_nz(v) if v is not None else "–") for p, v in self.nowcast.items()},
            "faktor_je_stunde": {p: {str(h): _nz(f) for h, f in fa.items() if f != 1.0}
                                 for p, fa in self.factors.items()},
            "abgleiche": len(self.rows)})
        raum = error_stats(self.rows, "raum")
        mae6 = raum.get("6", {}).get("mae")
        self.set_state("sensor.prognose_guete_raum", state=str(mae6 if mae6 is not None else "unknown"), attributes={
            "friendly_name": "Prognosegüte Leiträume (Fehler 6 h vorher)", "unit_of_measurement": "K",
            "icon": "mdi:home-thermometer-outline", "state_class": "measurement",
            "leitraeume": raum, "raeume": {n: error_stats(self.rows, f"raum:{n}") for n in self.raeume},
            "grundlast": error_stats(self.rows, "grundlast")})
