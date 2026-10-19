"""
Wärmepumpen-Modell v1 – lernt Arbeitszahl (COP) und Leistungsgrenzen der aroTHERM aus eBUS-Daten.

COP-Modell (Carnot-Anteil mit Abtau-Delle):
    COP = eta * T_h / (T_h - T_c) * (1 - d * exp(-((T_a - 1) / 3.5)^2))
    T_h = Vorlauf + dT_h   [K]   (Kondensator etwas wärmer als Vorlauf)
    T_c = Außen   - dT_c   [K]   (Verdampfer kälter als Außenluft)
Gelernt werden eta (Gütegrad) und d (Abtauverlust um ~1 °C).
Leistungsgrenzen: kleinste/größte thermische Leistung je Außentemperatur-Band (Perzentile).
"""

import json
import os
from datetime import datetime

import appdaemon.plugins.hass.hassapi as hass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DT_H = 3.0
DT_C = 5.0
DEFAULT = {"eta": 0.45, "d": 0.10}
BANDS = list(range(-15, 21, 3))      # Außentemperatur-Bänder [°C]


def cop_model(eta, d, t_flow, t_out):
    th = np.asarray(t_flow) + 273.15 + DT_H
    tc = np.asarray(t_out) + 273.15 - DT_C
    carnot = th / np.maximum(th - tc, 5.0)
    defrost = 1.0 - d * np.exp(-((np.asarray(t_out) - 1.0) / 3.5) ** 2)
    return eta * carnot * defrost


def _s(v):
    return str(v)


class WPModell(hass.Hass):

    def initialize(self):
        a = self.args
        self.e_pel = a["leistung_elektrisch"]      # kW
        self.e_pth = a["leistung_thermisch"]       # kW
        self.e_vl = a["vorlauf"]
        self.e_ta = a["aussentemperatur"]
        self.max_vl_heizen = float(a.get("max_vorlauf_heizen", 45))   # darüber: Warmwasser
        self.days = int(a.get("lern_tage", 30))
        self.store = a.get("speicher", "/config/wpmodell_state.json")
        self.p = dict(DEFAULT)
        self.info = {"status": "noch nicht gelernt"}
        self.limits = {}
        try:
            with open(self.store) as f:
                s = json.load(f)
            self.p.update(s.get("params", {}))
            self.info = s.get("info", self.info)
            self.limits = s.get("limits", {})
        except FileNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001
            self.log(f"Laden fehlgeschlagen: {e}", level="WARNING")
        self.run_daily(self.learn, "02:30:00")
        self.run_every(self.live, "now+20", 300)
        self.listen_event(lambda *a: self.learn({}), "wpmodell_lernen")
        self.run_in(self.learn, 45)

    # ------------------------------------------------------------ helpers
    def _f(self, e):
        try:
            return float(self.get_state(e))
        except (TypeError, ValueError):
            return None

    def _hist(self, e, idx):
        raw = self.get_history(entity_id=e, days=self.days + 1)
        items = raw[0] if raw and isinstance(raw[0], list) else (raw or [])
        rows = []
        for it in items:
            try:
                rows.append((pd.Timestamp(it["last_changed"]), float(it["state"])))
            except (TypeError, ValueError, KeyError):
                pass
        if not rows:
            return pd.Series(np.nan, index=idx)
        s = pd.Series(dict(rows)).sort_index()
        s.index = pd.to_datetime(s.index, utc=True)
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(s.index.union(idx)).sort_index().ffill()
        return s.resample("15min").mean().reindex(idx)

    # ------------------------------------------------------------- learn
    def learn(self, kwargs):
        end = pd.Timestamp.now(tz="UTC").floor("15min")
        idx = pd.date_range(end - pd.Timedelta(days=self.days), end, freq="15min")
        df = pd.DataFrame({"pel": self._hist(self.e_pel, idx), "pth": self._hist(self.e_pth, idx),
                           "vl": self._hist(self.e_vl, idx), "ta": self._hist(self.e_ta, idx)}).dropna()
        run = df[(df.pel > 0.25) & (df.pth > 0.5) & (df.vl < self.max_vl_heizen) & (df.vl > 18)]
        run = run.assign(cop=run.pth / run.pel)
        run = run[(run.cop > 1.0) & (run.cop < 9.0)]
        self.log(f"WP-Daten: {len(df)} Zeilen, davon Heizbetrieb {len(run)}")

        # Leistungsgrenzen je Außentemperatur-Band (braucht nur wenige Punkte)
        limits = {}
        for lo in BANDS:
            band = run[(run.ta >= lo) & (run.ta < lo + 3)]
            if len(band) >= 8:
                limits[str(lo)] = {"min_kw": round(float(band.pth.quantile(0.1)), 2),
                                   "max_kw": round(float(band.pth.quantile(0.95)), 2),
                                   "cop_mittel": round(float(band.cop.median()), 2), "n": int(len(band))}
        if limits:
            self.limits = limits

        if len(run) < 40:
            self.info = {"status": f"zu wenig Heizbetrieb ({len(run)} Viertelstunden, mind. 40)",
                         "zeit": datetime.now().isoformat(timespec="minutes")}
            self._save()
            self._publish()
            return

        w = np.sqrt(run.pth.values)   # stärker gewichten, wenn viel Wärme fließt (genauer gemessen)

        def res(x):
            return w * (cop_model(x[0], x[1], run.vl.values, run.ta.values) - run.cop.values)

        sol = least_squares(res, [self.p["eta"], self.p["d"]], bounds=([0.2, 0.0], [0.75, 0.4]),
                            loss="soft_l1", f_scale=0.3)
        self.p = {"eta": float(sol.x[0]), "d": float(sol.x[1])}
        pred = cop_model(self.p["eta"], self.p["d"], run.vl.values, run.ta.values)
        rmse = float(np.sqrt(np.mean((pred - run.cop.values) ** 2)))
        self.info = {"status": "ok", "zeit": datetime.now().isoformat(timespec="minutes"),
                     "viertelstunden": int(len(run)), "rmse_cop": round(rmse, 2),
                     "aussentemp_bereich": [round(float(run.ta.min()), 1), round(float(run.ta.max()), 1)]}
        self.log(f"WP gelernt: {self.p} {self.info}")
        self._save()
        self._publish()

    def _save(self):
        tmp = self.store + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"params": self.p, "info": self.info, "limits": self.limits}, f, indent=2)
        os.replace(tmp, self.store)

    # ------------------------------------------------------------ publish
    def cop(self, t_flow, t_out):
        return float(cop_model(self.p["eta"], self.p["d"], t_flow, t_out))

    def _publish(self):
        curve = [{"aussen": t, "cop_30": round(self.cop(30, t), 2), "cop_35": round(self.cop(35, t), 2),
                  "cop_40": round(self.cop(40, t), 2)} for t in range(-10, 21, 5)]
        self.set_state("sensor.wpmodell_guetegrad", state=_s(round(self.p["eta"], 3)),
                       attributes={"friendly_name": "WP-Modell Gütegrad (Anteil Carnot)", "icon": "mdi:heat-pump",
                                   "state_class": "measurement", "abtauverlust": round(self.p["d"], 3),
                                   "cop_kurve": curve, "leistungsgrenzen": self.limits, **self.info})

    def live(self, kwargs):
        vl, ta, pel, pth = (self._f(e) for e in (self.e_vl, self.e_ta, self.e_pel, self.e_pth))
        if vl is None or ta is None:
            return
        pred = round(self.cop(min(vl, self.max_vl_heizen), ta), 2)
        meas = round(pth / pel, 2) if (pel and pth and pel > 0.25 and pth > 0.5) else None
        self.set_state("sensor.wpmodell_cop", state=_s(pred),
                       attributes={"friendly_name": "WP-Modell COP (aktuelle Bedingungen)", "icon": "mdi:gauge",
                                   "state_class": "measurement", "cop_gemessen": meas,
                                   "vorlauf": vl, "aussen": ta})
