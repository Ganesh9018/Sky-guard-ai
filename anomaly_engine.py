"""
============================================================================
SkyGuard — MoES AI/ML Anomaly Detection Engine for AWS (Python reference)
============================================================================
Implements the same 5-layer detection stack as the browser dashboard:

  L1  Physical plausibility      — hard range limits per parameter
  L2  Temporal learning          — EWMA baseline + seasonal hour-of-day mean
  L3  Spatial consistency        — neighbor-station cross validation
  L4  Multivariate physics       — T/RH dewpoint coherence, pressure coupling,
                                   frozen-value detection
  L5  Explainable AI             — per-layer contributions, human-readable
                                   evidence (SHAP-style additive attribution)

Outputs per reading:
  verdict (ANOMALY / WARNING / NORMAL), confidence %, root-cause class,
  corrected (imputed) values, sensor health score + maintenance ETA.

Usage:
  python anomaly_engine.py            # runs the built-in demo
  python anomaly_engine.py --demo     # same
  python anomaly_engine.py --csv data.csv   # CSV with columns:
      station_id,timestamp,temperature,humidity,pressure
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PHYSICAL_LIMITS = {
    "temperature": (-10.0, 60.0),   # deg C
    "humidity":    (0.0, 100.0),    # %
    "pressure":    (870.0, 1084.0), # hPa
}

LAYER_WEIGHTS = {"L1": 0.30, "L2": 0.30, "L3": 0.25, "L4": 0.15}
ANOMALY_THRESHOLD = 0.50
WARNING_THRESHOLD = 0.30
Z_THRESHOLD = 2.5
NEIGHBOR_RADIUS_DEG = 0.35
MIN_NEIGHBORS = 2

PARAM_LABEL = {
    "temperature": "Temperature",
    "humidity":    "Humidity",
    "pressure":    "Pressure",
}
PARAM_UNIT = {"temperature": "°C", "humidity": "%", "pressure": " hPa"}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Reading:
    station_id: str
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float


@dataclass
class StationState:
    station_id: str
    lat: float = 0.0
    lng: float = 0.0
    name: str = ""
    state: str = ""
    region: str = ""
    baseline: dict = field(default_factory=dict)     # param -> dict
    health: dict = field(default_factory=lambda: {
        "drift": 0.0, "faults": 0, "samples": 0, "uptime": 100.0
    })
    last_good: dict | None = None    # last trusted reading {temperature,humidity,pressure}
    recent: list = field(default_factory=list)       # recent readings for L4 analysis


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class AnomalyEngine:
    def __init__(self, z_threshold: float = Z_THRESHOLD):
        self.z_threshold = z_threshold
        self.stations: dict[str, StationState] = {}

    # ---------------- station registry ----------------
    def register_station(self, station_id: str, lat: float, lng: float, name: str = "", state: str = "", region: str = ""):
        st = StationState(station_id, lat, lng, name=name, state=state, region=region)
        for p in PHYSICAL_LIMITS:
            st.baseline[p] = {"ewma": None, "sd": None, "n": 0, "hourly": {}}
        self.stations[station_id] = st
        return st

    def search_stations(self, query: str):
        q = (query or "").strip().lower()
        if not q:
            return [
                {
                    "station_id": s.station_id,
                    "name": s.name or s.station_id,
                    "state": s.state,
                    "region": s.region,
                    "lat": s.lat,
                    "lng": s.lng,
                }
                for s in self.stations.values()
            ]
        matches = []
        for s in self.stations.values():
            haystack = " ".join([
                s.station_id,
                s.name or "",
                s.state or "",
                s.region or "",
            ]).lower()
            if q in haystack:
                matches.append({
                    "station_id": s.station_id,
                    "name": s.name or s.station_id,
                    "state": s.state,
                    "region": s.region,
                    "lat": s.lat,
                    "lng": s.lng,
                })
        return matches

    # ---------------- learning ----------------
    def _learn(self, st: StationState, param: str, value: float, hour: int):
        b = st.baseline[param]
        alpha = 0.08
        if b["ewma"] is None:
            b["ewma"], b["sd"], b["n"] = value, 0.8, 1
        else:
            b["ewma"] = alpha * value + (1 - alpha) * b["ewma"]
            dev = abs(value - b["ewma"])
            b["sd"] = 0.9 * (b["sd"] or 0.8) + 0.1 * dev
            b["n"] += 1
        slot = b["hourly"].setdefault(hour, [])
        slot.append(value)
        if len(slot) > 60:          # bounded seasonal memory
            slot.pop(0)

    def _seasonal_mean(self, st: StationState, param: str, hour: int):
        slot = st.baseline[param]["hourly"].get(hour)
        if not slot or len(slot) < 3:
            return None
        return statistics.mean(slot)

    # ---------------- L1: physical plausibility ----------------
    def _l1_physical(self, r: Reading):
        issues = []
        vals = {"temperature": r.temperature, "humidity": r.humidity, "pressure": r.pressure}
        for p, v in vals.items():
            lo, hi = PHYSICAL_LIMITS[p]
            if v < lo or v > hi:
                issues.append((p, f"{PARAM_LABEL[p]} {v:.1f}{PARAM_UNIT[p]} outside physical range [{lo},{hi}]{PARAM_UNIT[p]}"))
        return issues

    # ---------------- L2: temporal z-scores ----------------
    def _l2_temporal(self, st: StationState, r: Reading):
        issues = []
        hour = r.timestamp.hour
        vals = {"temperature": r.temperature, "humidity": r.humidity, "pressure": r.pressure}
        for p, v in vals.items():
            b = st.baseline[p]
            if b["n"] < 6:
                continue
            sd = max(b["sd"] or 0.001, 0.25)
            seas = self._seasonal_mean(st, p, hour)
            reference = seas if seas is not None else b["ewma"]
            if reference is None:
                continue
            z = abs((v - reference) / sd)
            if z >= self.z_threshold:
                if seas is not None:
                    issues.append((p, f"{PARAM_LABEL[p]} deviates {z:.1f}σ from seasonal expectation ({seas:.1f}{PARAM_UNIT[p]})"))
                else:
                    issues.append((p, f"{PARAM_LABEL[p]} deviates {z:.1f}σ from learned baseline ({b['ewma']:.1f}{PARAM_UNIT[p]})"))
        return issues

    # ---------------- L3: spatial consistency ----------------
    def _l3_spatial(self, r: Reading):
        issues = []
        me = self.stations.get(r.station_id)
        if me is None:
            return issues
        neighbors = [
            s for s in self.stations.values()
            if s.station_id != r.station_id
            and math.hypot(s.lat - me.lat, s.lng - me.lng) < NEIGHBOR_RADIUS_DEG
            and s.last_good is not None
        ]
        if len(neighbors) < MIN_NEIGHBORS:
            return issues
        vals = {"temperature": r.temperature, "humidity": r.humidity, "pressure": r.pressure}
        for p, v in vals.items():
            nv = [s.last_good[p] for s in neighbors if s.last_good is not None]
            if len(nv) < MIN_NEIGHBORS:
                continue
            mean = statistics.mean(nv)
            spread = statistics.pstdev(nv) or 0.55
            z = abs((v - mean) / max(spread * 1.2, 0.5))
            if z >= self.z_threshold + 0.4:
                issues.append((p, f"{PARAM_LABEL[p]} {v:.1f}{PARAM_UNIT[p]} disagrees with {len(neighbors)} neighbors (avg {mean:.1f}{PARAM_UNIT[p]}, {z:.1f}σ)"))
        return issues

    # ---------------- L4: multivariate physics ----------------
    def _l4_multivariate(self, st: StationState, r: Reading, recent: list[Reading]):
        issues = []
        t, h, p = r.temperature, r.humidity, r.pressure
        dp = t - (100 - h) / 5
        if t >= 45 and h >= 85:
            issues.append(("temperature", f"Physics conflict: {t:.0f}°C with {h:.0f}% RH implausible (dewpoint {dp:.0f}°C ≥ T)"))
        if recent:
            drop = recent[-1].pressure - p
            if drop >= 5 and abs(r.temperature - recent[-1].temperature) < 3:
                issues.append(("pressure", f"Pressure fell {drop:.1f} hPa in one interval — verify against wind/rain response"))
        if len(recent) >= 8:
            tail = recent[-8:]
            if all(abs(x.temperature - tail[-1].temperature) < 1e-6 for x in tail):
                issues.append(("temperature", "Frozen value: temperature unchanged for 8+ consecutive samples → stuck sensor"))
        if abs(t - (recent[-1].temperature if recent else t)) > 18 and h > 85 and p < 1010:
            issues.append(("temperature", "Rapid heat rise with high humidity and falling pressure suggests sensor spike or storm front"))
        return issues

    def _extract_z(self, msg: str) -> float:
        """Pull the σ figure out of an evidence message (e.g. '6.8σ')."""
        import re
        m = re.search(r"([0-9.]+)σ", msg)
        return float(m.group(1)) if m else self.z_threshold + 1.0

    # ---------------- root-cause classifier ----------------
    def _classify(self, layers, frozen):
        keys = {p for iss in layers.values() for p, _ in iss}
        spat, temp = layers["L3"], layers["L2"]
        if frozen:
            return {"code": "STUCK_SENSOR", "label": "Stuck/Frozen Sensor",
                    "action": "Restart datalogger; check sensor cable & ADC"}
        if {"temperature", "humidity"} <= keys and layers["L4"]:
            return {"code": "SENSOR_FAULT", "label": "Multi-Sensor Fault",
                    "action": "Recalibrate station; verify RH + T probes"}
        if spat and not temp:
            return {"code": "SPATIAL_OUTLIER", "label": "Spatial Outlier (likely sensor)",
                    "action": "Cross-check with neighbor stations; inspect sensor"}
        if temp and spat:
            return {"code": "SENSOR_FAULT", "label": "Sensor Fault (temporal+spatial)",
                    "action": "Schedule calibration; flag data as suspect"}
        if spat:
            return {"code": "SPATIAL_OUTLIER", "label": "Spatial Outlier",
                    "action": "Verify against neighbors before use"}
        if temp and not spat:
            return {"code": "REAL_EVENT", "label": "Possible Real Weather Event",
                    "action": "Monitor — no spatial disagreement; may be genuine"}
        if layers["L1"]:
            return {"code": "RANGE_VIOLATION", "label": "Range Violation",
                    "action": "Check sensor wiring / datalogger config"}
        return {"code": "OK", "label": "Nominal", "action": "None"}

    # ---------------- health ----------------
    def _update_health(self, st: StationState, is_anomaly: bool):
        H = st.health
        H["samples"] += 1
        if is_anomaly:
            H["faults"] += 1
        b = st.baseline["temperature"]
        if b["n"] > 20 and b["ewma"] is not None:
            bias = abs(st.last_good["temperature"] - b["ewma"]) if st.last_good else 0
            H["drift"] = 0.95 * H["drift"] + 0.05 * bias
        H["uptime"] = max(0.0, 100 - (H["faults"] / max(H["samples"], 1)) * 100)

    def health_score(self, station_id: str) -> int:
        st = self.stations.get(station_id)
        if not st:
            return 100
        H = st.health
        return int(max(0, min(100, round(100 - H["drift"] * 6 -
                 (H["faults"] / max(H["samples"], 1)) * 60))))

    def maintenance_days(self, station_id: str) -> int:
        return max(1, round(self.health_score(station_id) / 100 * 90))

    # ---------------- master inference ----------------
    def process(self, r: Reading, learn: bool = True) -> dict:
        if r.station_id not in self.stations:
            self.register_station(r.station_id, 0.0, 0.0)
        st = self.stations[r.station_id]
        recent = st.recent

        layers = {
            "L1": self._l1_physical(r),
            "L2": self._l2_temporal(st, r),
            "L3": self._l3_spatial(r),
            "L4": self._l4_multivariate(st, r, recent),
        }
        frozen = any("Frozen" in m for _, m in layers["L4"])

        # weighted fusion score — stronger influence from confirmed real spikes,
        # while keeping spatial and temporal disagreement sensitive and stable.
        score = 0.0
        for name, issues in layers.items():
            if not issues:
                continue
            w = LAYER_WEIGHTS[name]
            if name in ("L1", "L4"):
                score += w * min(1.0, len(issues) / 2)
            else:
                zmax = max(self._extract_z(m) for _, m in issues)
                strength = min(1.0, zmax / (self.z_threshold + 1.5))
                score += w * strength

        if any("deviates" in msg or "disagrees" in msg for _, msg in layers["L2"] + layers["L3"]):
            score += 0.06

        layers_hit = [k for k, v in layers.items() if v]

        # Cross-layer agreement bonus: independent layers confirming each
        # other is strong evidence (SHAP-style interaction term).
        if len(layers_hit) >= 2:
            score += 0.15 * (len(layers_hit) - 1)
        # A frozen sensor is unambiguous — hard anomaly signal.
        if frozen:
            score = max(score, ANOMALY_THRESHOLD + 0.05)

        is_anomaly = score >= ANOMALY_THRESHOLD
        is_warning = not is_anomaly and score >= WARNING_THRESHOLD

        agreement = len(layers_hit) / 4
        confidence = int(max(40, min(99, round((0.55 + 0.45 * agreement) * (100 if is_anomaly else 70)))))

        root_cause = self._classify(layers, frozen)

        # corrected (imputed) values for flagged params
        corrected = {}
        flagged = {p for iss in layers.values() for p, _ in iss}
        hour = r.timestamp.hour
        for p in ("temperature", "humidity", "pressure"):
            if p not in flagged:
                corrected[p] = None
                continue
            candidates = []
            seas = self._seasonal_mean(st, p, hour)
            if seas is not None:
                candidates.append(seas)
            neigh = [
                s.last_good[p] for s in self.stations.values()
                if s.station_id != r.station_id and s.last_good
                and math.hypot(s.lat - st.lat, s.lng - st.lng) < NEIGHBOR_RADIUS_DEG
            ]
            if neigh:
                candidates.append(statistics.median(neigh))
            if not candidates and st.baseline[p]["ewma"] is not None:
                candidates.append(st.baseline[p]["ewma"])
            corrected[p] = statistics.mean(candidates) if candidates else None

        # update state
        if learn and not is_anomaly:
            self._learn(st, "temperature", r.temperature, hour)
            self._learn(st, "humidity", r.humidity, hour)
            self._learn(st, "pressure", r.pressure, hour)
        if not is_anomaly:
            st.last_good = {"temperature": r.temperature, "humidity": r.humidity, "pressure": r.pressure}
        self._update_health(st, is_anomaly)
        recent.append(r)
        if len(recent) > 30:
            recent.pop(0)

        verdict = "ANOMALY" if is_anomaly else ("WARNING" if is_warning else "NORMAL")
        return {
            "station_id": r.station_id,
            "timestamp": r.timestamp.isoformat(),
            "verdict": verdict,
            "confidence": confidence,
            "score": round(score, 3),
            "layers_triggered": layers_hit,
            "evidence": [m for iss in layers.values() for _, m in iss],
            "root_cause": root_cause,
            "corrected": {k: (round(v, 2) if v is not None else None) for k, v in corrected.items()},
            "health_score": self.health_score(r.station_id),
            "maintenance_due_days": self.maintenance_days(r.station_id),
        }


# --------------------------------------------------------------------------
# Fault injection (for evaluation on anomaly-injected data)
# --------------------------------------------------------------------------
def inject_fault(reading: Reading, fault: str, rng: random.Random) -> Reading:
    """Return a copy of the reading with a synthetic sensor fault applied."""
    r = Reading(reading.station_id, reading.timestamp,
                reading.temperature, reading.humidity, reading.pressure)
    if fault == "spike":
        r.temperature += rng.uniform(14, 22)
        r.humidity = min(100.0, r.humidity + rng.uniform(20, 30))
    elif fault == "stuck":
        pass  # caller keeps previous values
    elif fault == "drift":
        r.temperature += rng.uniform(0.5, 1.2)
    elif fault == "frozen":
        pass  # caller repeats previous value
    elif fault == "comm_loss":
        r.temperature = r.humidity = r.pressure = float("nan")
    elif fault == "pressure_drop":
        r.pressure -= rng.uniform(8, 14)
    return r


# --------------------------------------------------------------------------
# Demo: simulated network with injected faults + evaluation metrics
# --------------------------------------------------------------------------
STATIONS = [
    # id, lat, lng, base T, base RH, base P
    ("MUM-01", 18.906, 72.815, 29, 78, 1008),
    ("MUM-02", 19.088, 72.852, 30, 74, 1007),
    ("MUM-03", 19.055, 72.840, 30, 76, 1008),
    ("MUM-04", 19.136, 72.855, 31, 70, 1007),
    ("MUM-05", 19.180, 72.960, 29, 82, 1009),
    ("MUM-06", 19.033, 73.020, 31, 71, 1007),
]

def run_demo():
    rng = random.Random(42)
    engine = AnomalyEngine()
    for sid, lat, lng, *_ in STATIONS:
        engine.register_station(sid, lat, lng)

    t0 = datetime(2026, 8, 30, 0, 0)
    prev = {sid: {"t": float(bt), "h": float(bh), "p": float(bp)}
            for sid, _, _, bt, bh, bp in STATIONS}

    # fault schedule: (tick, station, fault)
    schedule = {40: ("MUM-03", "spike"), 70: ("MUM-05", "frozen"),
                100: ("MUM-02", "pressure_drop"), 130: ("MUM-04", "drift")}
    active = {}   # station -> (fault, ticks_left)
    tp = fp = fn = 0
    injected = 0
    anomalies_log = []

    print("=" * 78)
    print("SkyGuard MoES demo — 200 ticks, 6 stations, 4 injected faults")
    print("=" * 78)

    for tick in range(200):
        ts = t0 + timedelta(minutes=10 * tick)
        hour = ts.hour
        for sid, lat, lng, bt, bh, bp in STATIONS:
            # normal evolution: random walk around base + fresh diurnal cycle
            # (diurnal term is recomputed each tick so it never accumulates)
            walk_t = prev[sid]["t"] + rng.uniform(-0.4, 0.4)
            walk_t = max(bt - 4, min(bt + 4, walk_t))          # bounded walk
            t = walk_t + 2.5 * math.sin((hour - 6) / 24 * 2 * math.pi)
            h = min(99, max(20, prev[sid]["h"] + rng.uniform(-1, 1)))
            h = max(bh - 6, min(bh + 6, h))                    # bounded walk
            p = prev[sid]["p"] + rng.uniform(-0.5, 0.5)
            p = max(bp - 3, min(bp + 3, p))                    # bounded walk

            # fault injection
            if tick in schedule and schedule[tick][0] == sid:
                fault = schedule[tick][1]
                active[sid] = (fault, 10)
                injected += 1
            if sid in active:
                fault, left = active[sid]
                r = Reading(sid, ts, t, h, p)
                r = inject_fault(r, fault, rng)
                if fault == "frozen":
                    r.temperature = prev[sid].get("frozen_t", t)
                    prev[sid]["frozen_t"] = r.temperature
                t, h, p = r.temperature, r.humidity, r.pressure
                left -= 1
                if left <= 0:
                    active.pop(sid)
                    prev[sid].pop("frozen_t", None)
                else:
                    active[sid] = (fault, left)

            prev[sid]["t"] = t
            prev[sid]["h"] = h
            prev[sid]["p"] = p
            reading = Reading(sid, ts, t, h, p)
            out = engine.process(reading)

            expected_anomaly = sid in active
            detected = out["verdict"] == "ANOMALY"
            if expected_anomaly and detected:
                tp += 1
            elif expected_anomaly and not detected:
                fn += 1
            elif not expected_anomaly and detected:
                fp += 1

            if detected:
                anomalies_log.append(out)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"\nInjected faults : {injected}")
    print(f"TP / FP / FN    : {tp} / {fp} / {fn}")
    print(f"Precision       : {precision*100:.1f}%")
    print(f"Recall          : {recall*100:.1f}%")
    print(f"F1 score        : {f1:.2f}")

    print("\n--- Sample detected anomalies (explainable output) ---")
    for a in anomalies_log[:5]:
        print(f"\n[{a['timestamp']}] {a['station_id']} → {a['verdict']} "
              f"(confidence {a['confidence']}%)")
        print(f"  Root cause : {a['root_cause']['label']}")
        print(f"  Action     : {a['root_cause']['action']}")
        for e in a["evidence"]:
            print(f"  • {e}")
        corr = {k: v for k, v in a["corrected"].items() if v is not None}
        if corr:
            print(f"  Corrected  : {corr}")
        print(f"  Health     : {a['health_score']}/100, maintenance in {a['maintenance_due_days']} days")

    print("\n--- Sensor health summary ---")
    for sid in engine.stations:
        print(f"  {sid}: health {engine.health_score(sid)}/100, "
              f"maintenance due in {engine.maintenance_days(sid)} days")


def run_csv(path: str):
    engine = AnomalyEngine()
    # register stations lazily with 0,0 coords (spatial layer disabled)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["station_id"]
            if sid not in engine.stations:
                engine.register_station(sid, float(row.get("lat", 0) or 0),
                                        float(row.get("lng", 0) or 0))
            ts = datetime.fromisoformat(row["timestamp"])
            r = Reading(sid, ts, float(row["temperature"]),
                        float(row["humidity"]), float(row["pressure"]))
            out = engine.process(r)
            flag = "🚨" if out["verdict"] == "ANOMALY" else ("⚠️" if out["verdict"] == "WARNING" else "  ")
            print(f"{flag} {out['timestamp']} {sid:8s} {out['verdict']:8s} "
                  f"conf={out['confidence']:2d}% {out['root_cause']['label']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SkyGuard MoES AWS anomaly engine")
    ap.add_argument("--csv", help="CSV file with station_id,timestamp,temperature,humidity,pressure")
    ap.add_argument("--demo", action="store_true", help="run built-in simulation demo")
    args = ap.parse_args()
    if args.csv:
        run_csv(args.csv)
    else:
        run_demo()