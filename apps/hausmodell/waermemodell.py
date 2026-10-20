"""
Hausmodell v1 – selbstlernendes thermisches 2R2C-Modell für Fußbodenheizung.

Zustände:  T_i  = Raumtemperatur (Leiträume, gemessen)
           T_s  = Estrich-/Speichertemperatur (nicht gemessen, geschätzt)
Eingänge:  T_a  = Außentemperatur
           Q    = Heizleistung in den Estrich [W]  (aus Durchflüssen × Spreizung)
           P_pv = PV-Leistung [W] als Ersatzgröße für Sonneneinstrahlung

    C_i dT_i/dt = (T_s - T_i)/R_is + (T_a - T_i)/R_ia + a_sol * P_pv
    C_s dT_s/dt = (T_i - T_s)/R_is + Q

Lernen:
  * nachts: Parameter per Least-Squares (Simulationsfehler) auf rollierendem Fenster
  * alle 15 min: Kalman-Filter schätzt T_s und korrigiert den Zustand
  * Güte: Simulationsfehler auf den letzten 48 h (nicht zum Fitten benutzt)
"""

import json
import math
import os
from datetime import datetime, timedelta

import appdaemon.plugins.hass.hassapi as hass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DT_MIN = 15                     # Zeitschritt in Minuten
DT = DT_MIN * 60.0              # in Sekunden
CP_WATER = 1.163                # Wh/(l·K)

# Startwerte (typisches EFH mit FBH) – werden durch das Lernen ersetzt
DEFAULT_PARAMS = {
    "R_is": 0.0020,   # K/W   Estrich -> Raum
    "R_ia": 0.0060,   # K/W   Raum -> außen  (H = 1/R ≈ 170 W/K)
    "C_i": 8.0e6,     # J/K   Raumluft + Möbel + Wände (schnell)
    "C_s": 4.0e7,     # J/K   Estrich (langsam)
    "a_sol": 0.15,    # W/W   Anteil PV-Leistung als Solargewinn
}
PARAM_KEYS = list(DEFAULT_PARAMS.keys())
# Physikalisch plausible Bereiche (EFH mit FBH): H 60–400 W/K, Raum 0,5–5 kWh/K, Estrich 2–25 kWh/K
BOUNDS_LO = np.log([5e-4, 2.5e-3, 1.8e6, 7.2e6, 0.0001])
BOUNDS_HI = np.log([5e-3, 1.7e-2, 1.8e7, 9.0e7, 1.0])


def _s(value):
    """AppDaemon verwirft "falsy" Zustände wie 0.0 – daher immer als Text übergeben."""
    return str(value)


def simulate(params, T_i0, T_s0, T_a, Q, P_pv):
    """Explizite Euler-Simulation über alle Zeitschritte."""
    R_is, R_ia, C_i, C_s, a_sol = (params[k] for k in PARAM_KEYS)
    n = len(T_a)
    Ti = np.empty(n)
    Ts = np.empty(n)
    ti, ts = T_i0, T_s0
    for k in range(n):
        Ti[k], Ts[k] = ti, ts
        dti = ((ts - ti) / R_is + (T_a[k] - ti) / R_ia + a_sol * P_pv[k]) / C_i
        dts = ((ti - ts) / R_is + Q[k]) / C_s
        ti += DT * dti
        ts += DT * dts
    return Ti, Ts


class Hausmodell(hass.Hass):

    # ------------------------------------------------------------------ setup
    def initialize(self):
        a = self.args
        ti = a["raumtemperatur"]
        self.e_ti = ti if isinstance(ti, list) else [ti]   # mehrere Räume -> Mittelwert
        self.e_ta = a["aussentemperatur"]
        self.e_pv = a["pv_leistung"]
        self.e_vl = a["vorlauf"]
        # Heizwärme = elektrische WP-Leistung × COP, nur wenn die WP wirklich in die FBH heizt.
        # Ohne diese Angabe: alter Weg über Durchfluss × Spreizung (unsicher bei stehender Pumpe).
        self.e_pel = a.get("wp_leistung")
        self.pel_min = float(a.get("wp_laeuft_ab_w", 200))
        self.circuits = a["heizkreise"]          # Liste {ruecklauf, durchfluss}
        self.e_bad = a.get("bad_temperatur")
        self.e_weather = a.get("wetter")
        self.t_min = float(a.get("komfort_min", 20.5))
        self.t_soll = float(a.get("komfort_soll", 21.5))
        self.fit_days = int(a.get("lern_tage", 21))
        self.store = a.get("speicher", "/config/hausmodell_state.json")

        self.params = dict(DEFAULT_PARAMS)
        self.fit_info = {"status": "noch nicht gelernt"}
        self.x = None                 # Kalman-Zustand [T_i, T_s]
        self.P = np.diag([0.25, 4.0])  # Kovarianz
        self._load()

        # Kalman + Prognose alle 15 Minuten, Lernen jede Nacht 02:10
        self.run_every(self.step, "now+30", DT_MIN * 60)
        self.run_daily(self.learn, "02:10:00")
        self.listen_event(self.on_learn_event, "hausmodell_lernen")
        # Beim Start sofort einmal lernen, falls noch nie geschehen
        if self.fit_info.get("status") != "ok":
            self.run_in(self.learn, 60)

    # ----------------------------------------------------------- persistence
    def _load(self):
        try:
            with open(self.store) as f:
                s = json.load(f)
            self.params.update(s.get("params", {}))
            # ältere Lernstände in die aktuellen Grenzen holen
            lo, hi = np.exp(BOUNDS_LO), np.exp(BOUNDS_HI)
            for i, k in enumerate(PARAM_KEYS):
                self.params[k] = float(min(max(self.params[k], lo[i]), hi[i]))
            self.fit_info = s.get("fit_info", self.fit_info)
            self.log("Gespeicherte Modellparameter geladen")
        except FileNotFoundError:
            self.log("Keine gespeicherten Parameter – starte mit Standardwerten")
        except Exception as e:  # noqa: BLE001
            self.log(f"Laden fehlgeschlagen: {e}", level="WARNING")

    def _save(self):
        tmp = self.store + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"params": self.params, "fit_info": self.fit_info}, f, indent=2)
        os.replace(tmp, self.store)

    # ------------------------------------------------------------- helpers
    def _f(self, entity, default=None):
        try:
            return float(self.get_state(entity))
        except (TypeError, ValueError):
            return default

    def _cop(self, vl, ta):
        """COP aus dem WP-Modell (falls geladen), sonst Carnot-Schätzung mit Gütegrad 0,45."""
        try:
            wp = self.get_app("wpmodell")
            eta, d = wp.p["eta"], wp.p["d"]
        except Exception:  # noqa: BLE001
            eta, d = 0.45, 0.10
        vl, ta = np.asarray(vl, float), np.asarray(ta, float)
        th = vl + 273.15 + 3.0
        tc = ta + 273.15 - 5.0
        cop = eta * th / np.maximum(th - tc, 5.0) * (1 - d * np.exp(-((ta - 1.0) / 3.5) ** 2))
        return np.clip(cop, 1.5, 7.0)

    def _running(self, pel, vl, ti):
        """WP heizt in die FBH: Strom fließt und der Verteiler-Vorlauf liegt deutlich über Raum."""
        return pel is not None and vl is not None and ti is not None and pel > self.pel_min and vl - ti > 3.0

    def _heat_now(self, ti=None, ta=None):
        """Aktuelle Heizleistung in W."""
        vl = self._f(self.e_vl)
        if vl is None:
            return None
        if self.e_pel:
            pel = self._f(self.e_pel, 0.0)
            if not self._running(pel, vl, ti):
                return 0.0
            return float(pel * self._cop(vl, ta if ta is not None else 10.0))
        q = 0.0
        for c in self.circuits:
            rl = self._f(c["ruecklauf"])
            fl = self._f(c["durchfluss"], 0.0)
            if rl is None:
                continue
            q += fl * CP_WATER * max(vl - rl, 0.0)
        return q

    def _slab_measurement(self, ti=None):
        """Mittlerer Rücklauf als Messwert für die Estrichtemperatur – nur bei Durchströmung."""
        vl = self._f(self.e_vl)
        rls = [self._f(c["ruecklauf"]) for c in self.circuits]
        rls = [r for r in rls if r is not None]
        if vl is None or not rls:
            return None
        rl = float(np.mean(rls))
        if self.e_pel and not self._running(self._f(self.e_pel, 0.0), vl, ti):
            return None
        return rl if (vl - rl) > 0.5 else None

    # ------------------------------------------------------------ history
    def _history(self, entity, start):
        raw = self.get_history(entity_id=entity, days=self.fit_days + 1)
        rows = []
        items = raw[0] if raw and isinstance(raw, list) and raw and isinstance(raw[0], list) else (raw or [])
        if items:
            for item in items:
                try:
                    rows.append((pd.Timestamp(item["last_changed"]), float(item["state"])))
                except (TypeError, ValueError, KeyError):
                    pass
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series(dict(rows)).sort_index()
        s.index = pd.to_datetime(s.index, utc=True)
        return s[~s.index.duplicated(keep="last")]

    def _grid(self, series, idx):
        """Auf 15-min-Raster bringen: letzten Wert halten, zeitlich mitteln."""
        if series.empty:
            return pd.Series(np.nan, index=idx)
        s = series.reindex(series.index.union(idx)).sort_index().ffill()
        return s.resample(f"{DT_MIN}min").mean().reindex(idx).interpolate(limit=4)

    def _dataset(self, days):
        end = pd.Timestamp.now(tz="UTC").floor(f"{DT_MIN}min")
        start = end - pd.Timedelta(days=days)
        idx = pd.date_range(start, end, freq=f"{DT_MIN}min")
        st = start.to_pydatetime()
        df = pd.DataFrame(index=idx)
        rooms = [self._grid(self._history(e, st), idx) for e in self.e_ti]
        df["Ti"] = pd.concat(rooms, axis=1).mean(axis=1)
        df["Ta"] = self._grid(self._history(self.e_ta, st), idx)
        df["Ppv"] = self._grid(self._history(self.e_pv, st), idx).fillna(0).clip(lower=0)
        vl = self._grid(self._history(self.e_vl, st), idx)
        rl_all = [self._grid(self._history(c["ruecklauf"], st), idx) for c in self.circuits]
        rl_mean = pd.concat(rl_all, axis=1).mean(axis=1)
        if self.e_pel:
            pel = self._grid(self._history(self.e_pel, st), idx).fillna(0)
            running = (pel > self.pel_min) & ((vl - df["Ti"]) > 3.0)
            cop = pd.Series(self._cop(vl.fillna(30).values, df["Ta"].fillna(10).values), index=idx)
            df["Q"] = (pel * cop).where(running, 0.0).fillna(0)
            df["Ts_meas"] = rl_mean.where(running & ((vl - rl_mean) > 0.5))
        else:
            q = pd.Series(0.0, index=idx)
            for c, rl in zip(self.circuits, rl_all):
                fl = self._f(c["durchfluss"], 0.0)
                q += (fl * CP_WATER * (vl - rl).clip(lower=0)).fillna(0)
            df["Q"] = q
            df["Ts_meas"] = rl_mean.where((vl - rl_mean) > 1.0)
        self.log(f"Datensatz: {len(df)} Zeilen, gültig Ti={df['Ti'].notna().sum()}, "
                 f"Ta={df['Ta'].notna().sum()}, Estrich-Messung={df['Ts_meas'].notna().sum()}")
        return df.dropna(subset=["Ti", "Ta"])

    # ------------------------------------------------------------- learning
    def on_learn_event(self, event, data, kwargs):
        self.learn({})

    def learn(self, kwargs):
        try:
            df = self._dataset(self.fit_days)
        except Exception as e:  # noqa: BLE001
            self.log(f"Daten laden fehlgeschlagen: {e}", level="ERROR")
            return
        if len(df) < 4 * 24 * 4:   # < 4 Tage
            self.fit_info = {"status": f"zu wenig Daten ({len(df) // 96} Tage)"}
            self._publish_model()
            return

        win = 4 * 24                          # 24-h-Simulationsfenster
        n_val = 4 * 48                        # letzte 48 h nur zur Prüfung
        ts_m = df["Ts_meas"].values
        ti_v, ta_v, q_v, pv_v = (df[c].values for c in ("Ti", "Ta", "Q", "Ppv"))

        # Fenster starten nur dort, wo die Estrichtemperatur über den Rücklauf bekannt ist
        def pick_starts(lo, hi, spacing):
            out, last = [], -10 ** 9
            for i in np.where(~np.isnan(ts_m[lo:hi]))[0] + lo:
                if i - last >= spacing and i + win <= hi:
                    out.append(int(i))
                    last = i
            return out

        split = len(df) - n_val
        starts = pick_starts(0, split, win // 2)
        val_starts = pick_starts(split, len(df), win // 4)
        if len(starts) < 4:
            self.fit_info = {"status": f"zu wenig Heizphasen in den Daten ({len(starts)})"}
            self._publish_model()
            return

        def residuals(logp):
            p = dict(zip(PARAM_KEYS, np.exp(logp)))
            res = []
            for i in starts:
                sl = slice(i, i + win)
                Ti, Ts = simulate(p, ti_v[i], ts_m[i], ta_v[sl], q_v[sl], pv_v[sl])
                res.append(Ti - ti_v[sl])
                m = ts_m[sl]
                ok = ~np.isnan(m)
                if ok.any():   # Rückläufe stützen die Estrich-Schätzung (geringer gewichtet)
                    res.append(0.3 * (Ts[ok] - m[ok]))
            r = np.concatenate(res)
            return np.nan_to_num(r, nan=0.0)

        def fit_from(x0):
            x0 = np.clip(x0, BOUNDS_LO + 1e-6, BOUNDS_HI - 1e-6)
            return least_squares(residuals, x0, bounds=(BOUNDS_LO, BOUNDS_HI),
                                 loss="soft_l1", f_scale=0.3, max_nfev=600)

        try:
            cands = [np.log([self.params[k] for k in PARAM_KEYS]),
                     np.log([DEFAULT_PARAMS[k] for k in PARAM_KEYS])]
            sol = min((fit_from(x) for x in cands), key=lambda r: r.cost)
        except Exception as e:  # noqa: BLE001
            self.log(f"Lernen fehlgeschlagen: {e}", level="ERROR")
            return
        new = {k: float(v) for k, v in zip(PARAM_KEYS, np.exp(sol.x))}

        # Prüfung auf den letzten 48 h (nicht zum Lernen benutzt)
        def val_rmse(p):
            if not val_starts:
                return float("nan")
            errs = []
            for i in val_starts:
                end = min(i + win, len(df))
                sl = slice(i, end)
                Ti, _ = simulate(p, ti_v[i], ts_m[i], ta_v[sl], q_v[sl], pv_v[sl])
                errs.append(Ti - ti_v[sl])
            e = np.concatenate(errs)
            return float(np.sqrt(np.nanmean(e ** 2)))

        rmse_new, rmse_old = val_rmse(new), val_rmse(self.params)
        if math.isnan(rmse_new):
            accepted = True
        else:
            accepted = (math.isnan(rmse_old) or rmse_new <= rmse_old * 1.05
                        or self.fit_info.get("status") != "ok")
        if accepted:
            self.params = new
        self.fit_info = {
            "status": "ok",
            "zeit": datetime.now().isoformat(timespec="minutes"),
            "tage": round(len(df) / 96, 1),
            "rmse_48h": None if math.isnan(rmse_new) else round(rmse_new if accepted else rmse_old, 3),
            "rmse_neu": None if math.isnan(rmse_new) else round(rmse_new, 3),
            "rmse_alt": None if math.isnan(rmse_old) else round(rmse_old, 3),
            "lernfenster": len(starts),
            "uebernommen": accepted,
        }
        self._save()
        self.log(f"Gelernt: {self.fit_info} -> {self._summary()}")
        self._publish_model()

    def _summary(self):
        p = self.params
        H = 1.0 / p["R_ia"]
        return {
            "H_W_per_K": round(float(H), 1),
            "tau_raum_h": round(float(p["C_i"] * p["R_ia"] / 3600), 1),
            "tau_estrich_h": round(float(p["C_s"] * (p["R_is"] + p["R_ia"]) / 3600), 1),
            "C_estrich_kWh_per_K": round(float(p["C_s"] / 3.6e6), 2),
            "solar_faktor": round(float(p["a_sol"]), 3),
        }

    def _publish_model(self):
        s = self._summary()
        self.set_state("sensor.hausmodell_waermeverlust", state=_s(s["H_W_per_K"]),
                       attributes={"unit_of_measurement": "W/K", "friendly_name": "Hausmodell Wärmeverlust H",
                                   "icon": "mdi:home-export-outline", "state_class": "measurement", **s})
        rmse = self.fit_info.get("rmse_48h")
        self.set_state("sensor.hausmodell_guete", state=_s(rmse if rmse is not None else "unknown"),
                       attributes={"unit_of_measurement": "K", "friendly_name": "Hausmodell Vorhersagefehler (48 h)",
                                   "icon": "mdi:target", "state_class": "measurement", **self.fit_info})

    # ---------------------------------------------------- kalman + forecast
    def step(self, kwargs):
        tis = [v for v in (self._f(e) for e in self.e_ti) if v is not None]
        ti = float(np.mean(tis)) if tis else None
        ta = self._f(self.e_ta)
        if ti is None or ta is None:
            return
        q = self._heat_now(ti, ta) or 0.0
        ppv = max(self._f(self.e_pv, 0.0), 0.0)
        p = self.params
        R_is, R_ia, C_i, C_s, a_sol = (p[k] for k in PARAM_KEYS)

        if self.x is None:
            ts_m = self._slab_measurement(ti)
            self.x = np.array([ti, ts_m if ts_m is not None else ti + 1.0])

        # Vorhersage-Schritt (lineares System)
        A = np.array([[1 - DT / C_i * (1 / R_is + 1 / R_ia), DT / (C_i * R_is)],
                      [DT / (C_s * R_is), 1 - DT / (C_s * R_is)]])
        B = np.array([DT / C_i * (ta / R_ia + a_sol * ppv), DT / C_s * q])
        self.x = A @ self.x + B
        Qn = np.diag([0.01, 0.02])
        self.P = A @ self.P @ A.T + Qn

        # Mess-Schritt: Raumtemperatur immer, Estrich über Rücklauf wenn durchströmt
        ts_m = self._slab_measurement(ti)
        if ts_m is not None:
            Hm = np.eye(2)
            z = np.array([ti, ts_m])
            Rm = np.diag([0.05, 0.5])
        else:
            Hm = np.array([[1.0, 0.0]])
            z = np.array([ti])
            Rm = np.array([[0.05]])
        S = Hm @ self.P @ Hm.T + Rm
        K = self.P @ Hm.T @ np.linalg.inv(S)
        innov = z - Hm @ self.x
        self.x = self.x + K @ innov
        self.P = (np.eye(2) - K @ Hm) @ self.P

        self.set_state("sensor.hausmodell_estrich_temperatur", state=_s(round(float(self.x[1]), 2)),
                       attributes={"unit_of_measurement": "°C", "device_class": "temperature",
                                   "state_class": "measurement", "friendly_name": "Hausmodell Estrichtemperatur (geschätzt)",
                                   "icon": "mdi:floor-plan", "innovation_raum_K": round(float(innov[0]), 3)})
        self.set_state("sensor.hausmodell_heizleistung", state=_s(round(q)),
                       attributes={"unit_of_measurement": "W", "device_class": "power", "state_class": "measurement",
                                   "friendly_name": "Hausmodell Heizleistung (Durchfluss × Spreizung)", "icon": "mdi:radiator"})
        self._forecast(ta)

    def _outdoor_forecast(self, ta_now, steps):
        """Stündliche Außentemperatur-Prognose auf 15-min-Raster; Fallback: konstant."""
        temps = None
        if self.e_weather:
            try:
                r = self.call_service("weather/get_forecasts", entity_id=self.e_weather, type="hourly")
                fc = (r or {}).get("result", r or {}).get("response", r or {})
                fc = fc.get(self.e_weather, {}).get("forecast", []) if isinstance(fc, dict) else []
                hourly = [float(f["temperature"]) for f in fc if "temperature" in f]
                if hourly:
                    temps = np.repeat(hourly, 60 // DT_MIN)
            except Exception as e:  # noqa: BLE001
                self.log(f"Wetterprognose nicht verfügbar: {e}", level="DEBUG")
        if temps is None or len(temps) < steps:
            pad = np.full(steps, ta_now)
            if temps is not None:
                pad[:len(temps)] = temps
            temps = pad
        return temps[:steps]

    def _forecast(self, ta_now):
        """Auskühlprognose ohne Heizen + Wärmebedarf der nächsten 24 h."""
        steps = 4 * 24
        ta = self._outdoor_forecast(ta_now, steps)
        Ti, Ts = simulate(self.params, float(self.x[0]), float(self.x[1]), ta,
                          np.zeros(steps), np.zeros(steps))
        below = np.where(Ti < self.t_min)[0]
        hours = round(float(below[0]) * DT_MIN / 60, 1) if len(below) else 24.0
        H = 1.0 / self.params["R_ia"]
        demand_kwh = max(float(H) * (self.t_soll - float(np.mean(ta))) * 24 / 1000, 0.0)
        now = datetime.now()
        series = [{"t": (now + timedelta(minutes=DT_MIN * k)).strftime("%H:%M"), "Ti": round(float(Ti[k]), 2)}
                  for k in range(0, steps, 4)]
        self.set_state("sensor.hausmodell_auskuehlzeit", state=_s(hours),
                       attributes={"unit_of_measurement": "h", "friendly_name": "Hausmodell Auskühlzeit ohne Heizen", "grenze_temperatur": self.t_min,
                                   "icon": "mdi:timer-sand", "prognose_ohne_heizen": series,
                                   "aussentemp_mittel_24h": round(float(np.mean(ta)), 1)})
        self.set_state("sensor.hausmodell_waermebedarf_24h", state=_s(round(demand_kwh, 1)),
                       attributes={"unit_of_measurement": "kWh", "friendly_name": "Hausmodell Wärmebedarf nächste 24 h",
                                   "icon": "mdi:fire", "soll_temperatur": self.t_soll})
