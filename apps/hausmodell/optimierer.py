"""
Optimierer v1 – plant die Wärmepumpe für die nächsten 30 h (15-min-Raster) und setzt den Plan um.

Lineares Programm (scipy/HiGHS), je Zeitschritt k:
  q_k  thermische Heizleistung in den Estrich [kW]      (Entscheidung)
  c_k / d_k  Akku laden / entladen [kW]   g_k Netzbezug [kW]   e_k Einspeisung [kW]
  lo_k / hi_k  Komfort-Unter-/Überschreitung [K]   s_k  Akku-Energie [kWh]

  Bilanz:   PV_k + d_k + g_k = Grundlast_k + q_k / COP_k + c_k + e_k
  Akku:     s_{k+1} = s_k + dt·(η_c·c_k − d_k/η_d)
  Haus:     T_raum = T_frei + G·q      (lineares 2R2C-Hausmodell, exakt diskretisiert)
  Komfort:  T_min − lo ≤ T_raum ≤ T_max + hi
  Ziel:     min Σ 0,30 €·g·dt + Komfortstrafe − Restwert Akku       (Einspeisung = 0 €)

Umsetzung (nur mit Automatik-Schalter und frischen eBUS-Daten):
  geplante Leistung der nächsten Stunde ≥ Schwelle → Betriebsart „Manuell“ + Max-Vorlauf passend zur Leistung,
  sonst „Aus“. Mindestlaufzeit / Mindestpause gegen Takten.
"""

from datetime import datetime

import appdaemon.plugins.hass.hassapi as hass
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import linprog

DT_MIN = 15
DT_H = DT_MIN / 60.0
STEPS = 4 * 30


def _s(v):
    return str(v)


def _find_forecast(obj):
    """Sucht in einer beliebig verschachtelten Service-Antwort die Liste 'forecast'."""
    if isinstance(obj, dict):
        if isinstance(obj.get("forecast"), list):
            return obj["forecast"]
        for v in obj.values():
            r = _find_forecast(v)
            if r is not None:
                return r
    return None


def discretize(p):
    """Exakte Diskretisierung (Halteglied) des 2R2C-Modells. Eingänge u = [T_a, P_pv, Q]."""
    R_is, R_ia, C_i, C_s, a_sol = p["R_is"], p["R_ia"], p["C_i"], p["C_s"], p["a_sol"]
    Ac = np.array([[-(1 / R_is + 1 / R_ia) / C_i, 1 / (C_i * R_is)],
                   [1 / (C_s * R_is), -1 / (C_s * R_is)]])
    Bc = np.array([[1 / (C_i * R_ia), a_sol / C_i, 0.0],
                   [0.0, 0.0, 1 / C_s]])
    M = np.zeros((5, 5))
    M[:2, :2], M[:2, 2:] = Ac, Bc
    E = expm(M * DT_MIN * 60)
    return E[:2, :2], E[:2, 2:]


def plan(params, x0, ta, ppv_w, base_kw, cop, t_min, t_max, soc0_kwh, cfg):
    """Löst das LP. Gibt dict mit Plan-Arrays zurück (oder None)."""
    n = len(ta)
    Ad, Bd = discretize(params)
    # freie Trajektorie (ohne Heizen) und Impulsantwort der Heizung auf die Raumtemperatur
    x = np.array(x0, float)
    ti_free = np.empty(n)
    for k in range(n):
        x = Ad @ x + Bd[:, 0] * ta[k] + Bd[:, 1] * ppv_w[k]
        ti_free[k] = x[0]
    h = np.empty(n)
    v = Bd[:, 2] * 1000.0          # Wirkung von 1 kW
    for m in range(n):
        h[m] = v[0]
        v = Ad @ v
    G = np.zeros((n, n))
    for k in range(n):
        G[k, :k + 1] = h[k::-1]

    nq, nc, nd, ng, ne, nlo, nhi, ns = n, n, n, n, n, n, n, n + 1
    off = np.cumsum([0, nq, nc, nd, ng, ne, nlo, nhi])
    iq, ic, idd, ig, ie, ilo, ihi, is_ = (slice(off[i], off[i] + l) for i, l in
                                         enumerate([nq, nc, nd, ng, ne, nlo, nhi, ns]))
    nv = off[-1] + ns

    cost = np.zeros(nv)
    cost[ig] = cfg["preis"] * DT_H
    cost[iq] = 1e-4 * DT_H
    cost[ilo] = cfg["strafe_kalt"] * DT_H
    cost[ihi] = cfg["strafe_warm"] * DT_H
    cost[is_.stop - 1] = -cfg["restwert_akku"]

    A_eq, b_eq = [], []
    eta_c, eta_d = cfg["eta_laden"], cfg["eta_entladen"]
    for k in range(n):
        row = np.zeros(nv)                       # Leistungsbilanz
        row[iq.start + k] = 1.0 / cop[k]
        row[ic.start + k] = 1.0
        row[ie.start + k] = 1.0
        row[idd.start + k] = -1.0
        row[ig.start + k] = -1.0
        A_eq.append(row)
        b_eq.append(ppv_w[k] / 1000.0 - base_kw[k])
        row = np.zeros(nv)                       # Akku
        row[is_.start + k + 1] = 1.0
        row[is_.start + k] = -1.0
        row[ic.start + k] = -DT_H * eta_c
        row[idd.start + k] = DT_H / eta_d
        A_eq.append(row)
        b_eq.append(0.0)
    A_ub = np.zeros((2 * n, nv))
    b_ub = np.zeros(2 * n)
    A_ub[:n, iq] = -G
    A_ub[:n, ilo] = -np.eye(n)
    b_ub[:n] = ti_free - t_min
    A_ub[n:, iq] = G
    A_ub[n:, ihi] = -np.eye(n)
    b_ub[n:] = t_max - ti_free

    cap, pmax = cfg["akku_kwh"], cfg["akku_kw"]
    bounds = ([(0, cfg["q_max_kw"])] * n + [(0, pmax)] * n + [(0, pmax)] * n + [(0, None)] * n
              + [(0, None)] * n + [(0, None)] * n + [(0, None)] * n
              + [(min(max(soc0_kwh, 0), cap), min(max(soc0_kwh, 0), cap))] + [(0, cap)] * n)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if res.status != 0:
        return None
    z = res.x
    q = z[iq]
    return {"q": q, "ti": ti_free + G @ q, "ti_frei": ti_free, "p_el": q / cop, "g": z[ig], "e": z[ie],
            "soc": z[is_][1:], "lo": z[ilo], "kosten": float(cfg["preis"] * DT_H * z[ig].sum())}


class Optimierer(hass.Hass):

    def initialize(self):
        a = self.args
        self.cfg = {
            "preis": float(a.get("strompreis", 0.30)),
            "strafe_kalt": float(a.get("strafe_kalt_eur_kh", 0.6)),
            "strafe_warm": float(a.get("strafe_warm_eur_kh", 0.2)),
            "restwert_akku": float(a.get("restwert_akku_eur_kwh", 0.15)),
            "akku_kwh": float(a.get("akku_nutzbar_kwh", 7.6)),
            "akku_kw": float(a.get("akku_leistung_kw", 5.0)),
            "eta_laden": float(a.get("akku_eta_laden", 0.95)),
            "eta_entladen": float(a.get("akku_eta_entladen", 0.85)),
            "q_max_kw": float(a.get("heizleistung_max_kw", 8.0)),
        }
        self.t_min_tag = float(a.get("t_min_tag", 21.0))
        self.t_min_nacht = float(a.get("t_min_nacht", 20.5))
        self.t_max = float(a.get("t_max", 23.5))
        self.tag_von, self.tag_bis = int(a.get("tag_von", 6)), int(a.get("tag_bis", 22))
        self.vl_plan = float(a.get("vorlauf_plan", 33))
        self.schwelle_kw = float(a.get("einschalt_schwelle_kw", 1.5))
        self.min_lauf = int(a.get("mindestlaufzeit_min", 60))
        self.min_pause = int(a.get("mindestpause_min", 30))
        self.ua_fbh = float(a.get("fbh_kw_pro_k", 0.6))
        self.e = a["entitaeten"]
        self.raeume = a.get("raeume", {})
        self.pv_entities = a.get("pv_prognose", [])
        self.room_fit = {}
        self.base_profile = None
        self.last_switch = None
        self.last_cmd = None
        self.run_every(self.optimize, "now+90", DT_MIN * 60)
        self.run_daily(self.fit_rooms, "03:15:00")
        self.listen_event(lambda *x: self.optimize({}), "optimierer_jetzt")
        self.listen_state(self.on_automatik, self.e["automatik"])
        self.run_in(self.fit_rooms, 60)

    # ------------------------------------------------------------- helpers
    def _f(self, e, d=None):
        try:
            return float(self.get_state(e))
        except (TypeError, ValueError):
            return d

    def _haus(self):
        try:
            return self.get_app("hausmodell")
        except Exception:  # noqa: BLE001
            return None

    def _wp(self):
        try:
            return self.get_app("wpmodell")
        except Exception:  # noqa: BLE001
            return None

    def on_automatik(self, entity, attribute, old, new, kwargs):
        if new == "on":
            # Die einfache PV-Boost-Regel würde sonst dagegen arbeiten
            for ent in (self.e["pv_boost_automatik"], self.e["pv_boost_aktiv"]):
                if self.get_state(ent) == "on":
                    self.call_service("input_boolean/turn_off", entity_id=ent)
            self.optimize({})

    # -------------------------------------------------------------- inputs
    def _timeline(self):
        t0 = pd.Timestamp.now(tz="UTC").floor(f"{DT_MIN}min")
        return pd.date_range(t0 + pd.Timedelta(minutes=DT_MIN), periods=STEPS, freq=f"{DT_MIN}min")

    def _pv_forecast(self, idx):
        total = pd.Series(0.0, index=idx)
        found = False
        for ent in self.pv_entities:
            w = self.get_state(ent, attribute="watts") or {}
            if not w:
                continue
            s = pd.Series({pd.Timestamp(k): float(v) for k, v in w.items()}).sort_index()
            s.index = pd.to_datetime(s.index, utc=True)
            total = total.add(s.reindex(idx).fillna(0.0), fill_value=0.0)
            found = True
        return total.values, found

    def _ta_forecast(self, idx, ta_now):
        try:
            r = self.call_service("weather/get_forecasts", entity_id=self.e["wetter"], type="hourly")
            fc = _find_forecast(r) or []
            s = pd.Series({pd.Timestamp(f["datetime"]): float(f["temperature"]) for f in fc
                           if "temperature" in f and "datetime" in f}).sort_index()
            if len(s):
                s.index = pd.to_datetime(s.index, utc=True)
                s = s.reindex(s.index.union(idx)).interpolate(method="time").ffill().bfill()
                return s.reindex(idx).values, True
        except Exception as ex:  # noqa: BLE001
            self.log(f"Wetterprognose fehlgeschlagen: {ex}", level="WARNING")
        return np.full(len(idx), ta_now), False

    def _base_load(self, idx, haus):
        """Grundlast je Stunde: 30-%-Perzentil des Hausverbrauchs (blendet Heizbetrieb weitgehend aus)."""
        if self.base_profile is None and haus is not None:
            try:
                end = pd.Timestamp.now(tz="UTC").floor("15min")
                gidx = pd.date_range(end - pd.Timedelta(days=10), end, freq="15min")
                s = haus._grid(haus._history(self.e["verbrauch"], gidx[0].to_pydatetime()), gidx).dropna()
                if len(s) > 96:
                    loc = s.tz_convert("Europe/Berlin")
                    prof = loc.groupby(loc.index.hour).quantile(0.3).clip(150, 1500)
                    self.base_profile = {int(h): float(v) / 1000.0 for h, v in prof.items()}
            except Exception as ex:  # noqa: BLE001
                self.log(f"Grundlast-Profil fehlgeschlagen: {ex}", level="WARNING")
        prof = self.base_profile or {}
        hours = idx.tz_convert("Europe/Berlin").hour
        return np.array([prof.get(int(h), 0.4) for h in hours])

    def _t_min(self, idx):
        hours = idx.tz_convert("Europe/Berlin").hour
        return np.where((hours >= self.tag_von) & (hours < self.tag_bis), self.t_min_tag, self.t_min_nacht)

    # ----------------------------------------------------------- room model
    def fit_rooms(self, kwargs):
        haus = self._haus()
        if haus is None:
            return
        end = pd.Timestamp.now(tz="UTC").floor("15min")
        idx = pd.date_range(end - pd.Timedelta(days=10), end, freq="60min")
        st = idx[0].to_pydatetime()
        try:
            ref = pd.concat([haus._grid(haus._history(e, st), idx) for e in haus.e_ti], axis=1).mean(axis=1)
        except Exception as ex:  # noqa: BLE001
            self.log(f"Raumabgleich fehlgeschlagen: {ex}", level="WARNING")
            return
        fit = {}
        for name, ent in self.raeume.items():
            r = haus._grid(haus._history(ent, st), idx)
            ok = r.notna() & ref.notna()
            if ok.sum() < 48:
                fit[name] = {"a": 0.0, "g": 1.0, "n": int(ok.sum())}
                continue
            g, a = np.polyfit(ref[ok].values, r[ok].values, 1)
            if not 0.3 <= g <= 1.5:      # unplausibel -> nur Versatz
                g, a = 1.0, float(np.median(r[ok] - ref[ok]))
            res = r[ok] - (a + g * ref[ok])
            fit[name] = {"a": float(a), "g": float(g), "n": int(ok.sum()),
                         "rmse": round(float(np.sqrt(np.mean(res ** 2))), 2)}
        self.room_fit = fit
        self.log(f"Raumabgleich: {fit}")

    # ------------------------------------------------------------ optimize
    def optimize(self, kwargs):
        haus, wp = self._haus(), self._wp()
        if haus is None or getattr(haus, "x", None) is None:
            self.log("Hausmodell noch nicht bereit", level="INFO")
            return
        idx = self._timeline()
        ta_now = self._f(self.e["aussen"], 10.0)
        ta, ta_ok = self._ta_forecast(idx, ta_now)
        pv, pv_ok = self._pv_forecast(idx)
        base = self._base_load(idx, haus)
        if wp is not None:
            cop = np.array([max(wp.cop(self.vl_plan, t), 1.5) for t in ta])
        else:
            cop = np.full(len(ta), 3.5)
        soc = self._f(self.e["akku_soc"], 0.0) / 100.0 * self.cfg["akku_kwh"]
        t_min = self._t_min(idx)
        x0 = np.array(haus.x, float)
        res = plan(dict(haus.params), x0, ta, pv, base, cop, t_min, np.full(len(ta), self.t_max), soc, self.cfg)
        if res is None:
            self.log("Optimierung nicht lösbar", level="WARNING")
            self.set_state("sensor.optimierer_plan", state="Fehler",
                           attributes={"friendly_name": "Optimierer Plan", "icon": "mdi:alert"})
            return
        self._publish(idx, res, pv, base, ta, cop, soc, pv_ok, ta_ok, haus)
        self._dispatch(res, x0)

    # ------------------------------------------------------------- dispatch
    def _dispatch(self, res, x0):
        q_next = float(np.mean(res["q"][:4]))
        want_on = q_next >= self.schwelle_kw
        ts = float(x0[1])
        vl_min = max(30.0, self._f(self.e["min_vorlauf"], 30.0))   # Max-Vorlauf nie unter Min-Vorlauf
        vl = int(round(min(max(ts + q_next / self.ua_fbh, vl_min), 40)))
        reason = f"geplant {q_next:.1f} kW in der nächsten Stunde"
        auto = self.get_state(self.e["automatik"]) == "on"
        ebus_ok = self.get_state(self.e["ebus_ok"]) == "on"
        state_now = self.get_state(self.e["betriebsart"])
        is_on = state_now not in (None, "Aus", "unavailable", "unknown")
        now = datetime.now()
        mins = (now - self.last_switch).total_seconds() / 60 if self.last_switch else 1e9
        action = "keine"
        if not auto:
            action = "Automatik aus – nur Planung"
        elif not ebus_ok:
            action = "eBUS ohne frische Daten – nichts geschaltet"
        elif want_on and not is_on:
            if mins >= self.min_pause:
                self._set_vl(vl)
                self.call_service("select/select_option", entity_id=self.e["betriebsart"], option="Manuell")
                self.last_switch, action = now, f"EIN (Max-Vorlauf {vl} °C)"
            else:
                action = f"EIN geplant, Mindestpause noch {self.min_pause - mins:.0f} min"
        elif not want_on and is_on:
            if mins >= self.min_lauf:
                self.call_service("select/select_option", entity_id=self.e["betriebsart"], option="Aus")
                self.last_switch, action = now, "AUS"
            else:
                action = f"AUS geplant, Mindestlaufzeit noch {self.min_lauf - mins:.0f} min"
        elif want_on and is_on:
            if self._set_vl(vl):
                action = f"Max-Vorlauf → {vl} °C"
        if action not in ("keine",) and not action.startswith("Automatik"):
            self.log(f"Optimierer: {action} ({reason})")
        self.set_state("sensor.optimierer_entscheidung", state=_s("Heizen" if want_on else "Aus"),
                       attributes={"friendly_name": "Optimierer Entscheidung", "icon": "mdi:robot",
                                   "aktion": action, "grund": reason, "ziel_vorlauf": vl,
                                   "automatik": auto, "zeit": now.strftime("%H:%M")})

    def _set_vl(self, vl):
        cur = self._f(self.e["max_vorlauf"])
        if cur is None or abs(cur - vl) >= 1:
            self.call_service("number/set_value", entity_id=self.e["max_vorlauf"], value=vl)
            return True
        return False

    # -------------------------------------------------------------- publish
    def _publish(self, idx, res, pv, base, ta, cop, soc, pv_ok, ta_ok, haus):
        ms = [int(t.timestamp() * 1000) for t in idx]

        def series(arr, nd=2):
            return [[m, round(float(v), nd)] for m, v in zip(ms, arr)]

        day = slice(0, 96)
        self.set_state("sensor.optimierer_plan", state=_s(round(float(res["q"][:4].mean()), 2)), attributes={
            "friendly_name": "Optimierer geplante Heizleistung", "unit_of_measurement": "kW",
            "icon": "mdi:chart-timeline-variant", "state_class": "measurement",
            "heizleistung_kw": series(res["q"]), "wp_strom_kw": series(res["p_el"]),
            "netzbezug_kw": series(res["g"]), "akku_kwh": series(res["soc"]),
            "pv_kw": series(pv / 1000.0), "grundlast_kw": series(base),
            "raum_plan": series(res["ti"]), "raum_ohne_heizen": series(res["ti_frei"]),
            "aussen": series(ta, 1), "cop": series(cop),
            "waerme_24h_kwh": round(float(res["q"][day].sum() * DT_H), 1),
            "wp_strom_24h_kwh": round(float(res["p_el"][day].sum() * DT_H), 1),
            "netz_24h_kwh": round(float(res["g"][day].sum() * DT_H), 1),
            "kosten_24h_eur": round(float(res["g"][day].sum() * DT_H * self.cfg["preis"]), 2),
            "pv_prognose_ok": pv_ok, "wetter_prognose_ok": ta_ok,
            "akku_start_kwh": round(soc, 2), "zeit": datetime.now().strftime("%H:%M")})
        # Raumprognosen unter dem geplanten Fahrplan
        ti_now = float(haus.x[0])
        k = np.arange(1, len(idx) + 1)
        decay = np.exp(-k * DT_H / 8.0)
        for name, ent in self.raeume.items():
            f = self.room_fit.get(name, {"a": 0.0, "g": 1.0})
            meas = self._f(ent)
            base_now = f["a"] + f["g"] * ti_now
            err = (meas - base_now) if meas is not None else 0.0
            pred = f["a"] + f["g"] * res["ti"] + err * decay
            slug = name.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace(" ", "_")
            self.set_state(f"sensor.prognose_{slug}", state=_s(round(float(pred[23]), 1)), attributes={
                "friendly_name": f"Prognose {name} (+6 h)", "unit_of_measurement": "°C",
                "device_class": "temperature", "state_class": "measurement", "icon": "mdi:crystal-ball",
                "prognose": series(pred, 1), "gemessen_jetzt": meas, "quelle": ent,
                "abgleich": f, "min_24h": round(float(pred[:96].min()), 1),
                "max_24h": round(float(pred[:96].max()), 1)})
