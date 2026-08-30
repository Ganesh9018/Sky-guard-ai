# SkyWatch — MoES AI/ML Intelligent Anomaly Detection for Automatic Weather Stations (AWS)

A self-aware, self-healing weather observation network: real-time anomaly detection over AWS data streams with **explainable AI**, **confidence scores**, **root-cause classification**, **sensor health prediction**, and **corrected-data estimation** — delivered through an animated visualization dashboard with a live station map.

---

## 1. What This Solves

| Challenge Requirement | Where It Is Implemented |
|---|---|
| Detect anomalies in real-time AWS streams | `dashboard.js` — 2-second inference loop over all stations |
| Sensor faults, spikes, frozen values, comm errors | Fault-Injection Lab + L1/L4 layers (spike, stuck, frozen, drift, comm-loss, pressure-drop) |
| Learn normal temporal & seasonal patterns | L2 — EWMA baseline + hour-of-day seasonal memory per station/parameter |
| Multivariate consistency analysis | L4 — T/RH dewpoint coherence, pressure-vs-wind/rain coupling |
| Confidence scores + explainable reasoning | L5 — per-layer contribution bars + human-readable evidence list |
| Predict sensor degradation & maintenance | Health engine — drift tracking, fault-rate uptime, maintenance ETA in days |
| Corrected/imputed values (optional) | Imputation — blend of seasonal mean + neighbor median + EWMA fallback |
| Real-time alerts with severity & confidence | Alert feed + toast notifications, severity from confidence |
| Root-cause classification | 7 classes: STUCK_SENSOR, SENSOR_FAULT, SPATIAL_OUTLIER, REAL_EVENT, RANGE_VIOLATION, … |
| Visualization dashboard | Leaflet map, KPIs, charts, tables, health panel |
| Evaluation on anomaly-injected data | Built-in Fault-Injection Lab with live Precision / Recall / F1 |
| Edge deployability (ESP32) | Python reference engine is stdlib-only; L1+L2 run in O(1) memory — portable to MicroPython |

---

## 2. Architecture — The 5-Layer Detection Stack

```
AWS stream ──► L1 Physical plausibility      (hard range limits)
            ──► L2 Temporal learning         (EWMA + seasonal hour-of-day z-scores)
            ──► L3 Spatial consistency       (neighbor cross-validation, <0.35°)
            ──► L4 Multivariate physics      (dewpoint, pressure coupling, frozen)
            ──► L5 Explainable fusion        (weighted vote → verdict + confidence)
                     │
                     ├─► Root-cause classifier (7 classes + suggested action)
                     ├─► Corrected-value imputation (seasonal mean ⊕ neighbor median)
                     ├─► Sensor health & maintenance prediction
                     └─► Alerts (severity + confidence) → dashboard feed
```

**Fusion weights:** L1 = 0.30, L2 = 0.30, L3 = 0.25, L4 = 0.15.
**Verdict:** ANOMALY if fused score ≥ 0.50; WARNING if ≥ 0.30.
**Confidence:** scales with cross-layer agreement (independent layers confirming each other).

**Self-healing loop:** the engine learns baselines *only from trusted (non-anomalous) readings*, so a faulty sensor cannot poison its own model.

---

## 3. Files

| File | Purpose |
|---|---|
| `index.html` | Landing page (rise-in animations, hover effects, custom cursor, animated borders) |
| `login.html` | Signup / Login (localStorage auth; dashboard opens only after sign-in) |
| `dashboard.html` | Main visualization dashboard |
| `styles.css`, `dashboard.css` | All styling and animations |
| `app.js` | Cursor, background-shift, reveals, auth flow |
| `dashboard.js` | **Browser AI engine** + map + charts + fault-injection lab |
| `anomaly_engine.py` | **Python reference engine** (stdlib-only) with demo + CSV mode |

---

## 4. How to Run

### A. Web dashboard (no build step)
Open `index.html` in any browser → Sign up → Dashboard launches.
- Click any station marker → full XAI breakdown (confidence, layer contributions, evidence, root cause, corrected values, health).
- Use **🧪 Fault-Injection Lab** buttons to inject spike/stuck/drift/frozen/comm-loss/pressure-drop faults and watch Precision/Recall/F1 update live.
- Adjust the **Z-Score Threshold** slider to change sensitivity.

> Note: the map needs internet for OpenStreetMap tiles; everything else works offline.

### B. Python engine (example usage)
```bash
python anomaly_engine.py          # built-in demo: 200 ticks, 6 stations, 4 injected faults
python anomaly_engine.py --csv data.csv
```
CSV columns: `station_id,timestamp,temperature,humidity,pressure` (optional `lat,lng` enable the spatial layer).

Example output:
```
Injected faults : 4
TP / FP / FN    : 33 / 0 / 7
Precision       : 100.0%
Recall          : 82.5%
F1 score        : 0.90

[2026-08-30T06:40:00] MUM-03 → ANOMALY (confidence 92%)
  Root cause : Multi-Sensor Fault
  Action     : Recalibrate station; verify RH + T probes
  • Temperature deviates 6.8σ from learned baseline (29.9°C)
  • Physics conflict: 48°C with 96% RH implausible (dewpoint 46°C ≥ T)
  Corrected  : {'temperature': 30.1, 'humidity': 76.4}
  Health     : 84/100, maintenance in 76 days
```

---

## 5. Use Cases (Documented Scenarios)

### UC-1 — Sudden temperature spike with high humidity (the challenge example)
**Input:** AWS reports 55 °C with 90 % RH; neighbors read 30 °C.
**Pipeline:** L1 passes (55 < 60) → L2 flags +8σ vs baseline → L3 flags disagreement with 4 neighbors → L4 flags dewpoint impossibility.
**Output:** ANOMALY, confidence ~95 %, root cause `SENSOR_FAULT`, corrected value ≈ 30 °C, alert raised, health score drops, maintenance ETA shortens.

### UC-2 — Frozen/stuck sensor
**Input:** identical temperature for 8+ consecutive samples.
**Output:** L4 frozen-value rule fires → `STUCK_SENSOR` → action: restart datalogger, check cable/ADC.

### UC-3 — Slow sensor drift (degradation prediction)
**Input:** +0.9 °C per tick bias over 25 ticks.
**Output:** L2 EWMA tracks slowly; health engine accumulates drift → health score decays → maintenance ETA counts down *before* hard failure.

### UC-4 — Communication error
**Input:** station stops reporting.
**Output:** marker turns grey (offline), excluded from spatial consensus so it cannot corrupt neighbors, uptime metric falls.

### UC-5 — Real weather event (no false alarm)
**Input:** genuine 42 °C heat spike at one station; neighbors 39–41 °C.
**Output:** L2 flags but L3 does **not** disagree strongly → root cause `REAL_EVENT` → "Monitor, may be genuine" — the system distinguishes weather from sensor faults.

### UC-6 — Pressure drop without storm response
**Input:** −10 hPa in 10 min but wind 5 km/h, rain 0 mm.
**Output:** L4 coupling rule fires → sensor-suspect verdict instead of a false storm alert.

### UC-7 — Fleet maintenance planning
**Input:** weeks of operation.
**Output:** health panel ranks all stations; low-health stations get short maintenance ETAs → crew dispatch prioritization.

---

## 6. Evaluation Criteria Mapping

| Criterion (weight) | How this project scores it |
|---|---|
| Innovation & Novelty (25 %) | 5-layer explainable fusion + self-healing learning loop + spatial consensus + imputation in a zero-dependency browser engine |
| Detection Accuracy (20 %) | Multi-layer voting; live Precision/Recall/F1 on injected faults (demo: F1 ≈ 0.9) |
| Real-Time Capability (15 %) | O(1) per-reading inference; 2 s tick in browser; streaming-ready Python engine |
| Explainability (10 %) | Per-layer contribution bars, evidence list, root-cause + action (SHAP-style additive attribution) |
| Scalability (10 %) | Stateless per-station models; trivially parallelizable; region switcher demonstrates multi-region scale |
| Practical Deployability (10 %) | Stdlib-only Python (edge/ESP32/MicroPython-portable), no backend needed for web demo |
| Visualization/UI (5 %) | Animated dark dashboard, live map, charts, health bars, alert feed |
| Energy Efficiency (5 %) | O(1) memory per station, no matrix ops; L1+L2 alone run on microcontrollers |

---

## 7. Edge AI / ESP32 Path

The Python engine's L1 (range checks) and L2 (EWMA + z-score) require only ~10 floats of state per station — directly portable to MicroPython on ESP32 for on-sensor pre-filtering, with L3/L4 running at the gateway. This matches the "Edge AI for low-power deployment" suggestion.

---

## 8. Notes

- Auth is client-side (localStorage) for demo purposes; production would use a real identity provider.
- Station telemetry is simulated in-browser so the demo is fully self-contained; the same engine consumes real streams via the Python `--csv` mode.