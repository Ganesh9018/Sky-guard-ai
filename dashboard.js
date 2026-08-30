/* ============================================================
   SkyWatch — MoES AI/ML Anomaly Detection Engine (Browser)
   ------------------------------------------------------------
   Layers:
     L1 Physical plausibility      (range/step/rate limits)
     L2 Temporal learning          (EWMA baseline + seasonal hour-of-day)
     L3 Spatial consistency        (neighbor-station cross-validation)
     L4 Multivariate physics       (T/dewpoint, RH, pressure coupling)
     L5 Explainable AI             (per-layer contribution + human-readable reasons)
   Outputs: anomaly verdict, confidence %, root-cause class,
            corrected/imputed values, sensor health & maintenance ETA
   ============================================================ */

/* ---------------- Region data (5 regions, 31 stations) ---------------- */
const REGIONS = {
  mumbai: {
    name: "Mumbai Region",
    center: [19.05, 72.95],
    zoom: 10,
    stations: [
      { id: "MUM-01", name: "Colaba Observatory", lat: 18.906, lng: 72.815, base: { t: 29, h: 78, p: 1008, w: 12, r: 2 } },
      { id: "MUM-02", name: "Santacruz Airport",   lat: 19.088, lng: 72.852, base: { t: 30, h: 74, p: 1007, w: 10, r: 1 } },
      { id: "MUM-03", name: "Bandra Station",      lat: 19.055, lng: 72.840, base: { t: 30, h: 76, p: 1008, w: 14, r: 0 } },
      { id: "MUM-04", name: "Andheri Tech Park",   lat: 19.136, lng: 72.855, base: { t: 31, h: 70, p: 1007, w: 9,  r: 0 } },
      { id: "MUM-05", name: "Thane Creek",         lat: 19.180, lng: 72.960, base: { t: 29, h: 82, p: 1009, w: 16, r: 3 } },
      { id: "MUM-06", name: "Navi Mumbai SEZ",     lat: 19.033, lng: 73.020, base: { t: 31, h: 71, p: 1007, w: 11, r: 1 } },
      { id: "MUM-07", name: "Powai Lake",          lat: 19.117, lng: 72.905, base: { t: 29, h: 79, p: 1008, w: 8,  r: 2 } },
      { id: "MUM-08", name: "Vasai Road",          lat: 19.391, lng: 72.839, base: { t: 30, h: 75, p: 1008, w: 13, r: 1 } }
    ]
  },
  delhi: {
    name: "Delhi NCR Region",
    center: [28.60, 77.20],
    zoom: 10,
    stations: [
      { id: "DEL-01", name: "Safdarjung",        lat: 28.586, lng: 77.190, base: { t: 27, h: 55, p: 1010, w: 10, r: 0 } },
      { id: "DEL-02", name: "Palam Airport",     lat: 28.566, lng: 77.100, base: { t: 28, h: 52, p: 1009, w: 12, r: 0 } },
      { id: "DEL-03", name: "Ridge Road",        lat: 28.635, lng: 77.170, base: { t: 26, h: 58, p: 1011, w: 8,  r: 1 } },
      { id: "DEL-04", name: "Noida Sector 62",   lat: 28.627, lng: 77.370, base: { t: 28, h: 54, p: 1010, w: 9,  r: 0 } },
      { id: "DEL-05", name: "Gurugram Cyber City", lat: 28.495, lng: 77.086, base: { t: 29, h: 50, p: 1009, w: 11, r: 0 } },
      { id: "DEL-06", name: "Dwarka West",       lat: 28.592, lng: 77.046, base: { t: 27, h: 56, p: 1010, w: 14, r: 0 } },
      { id: "DEL-07", name: "Ghaziabad Indirapuram", lat: 28.645, lng: 77.370, base: { t: 28, h: 57, p: 1010, w: 10, r: 1 } }
    ]
  },
  bangalore: {
    name: "Bengaluru Region",
    center: [12.97, 77.60],
    zoom: 10.5,
    stations: [
      { id: "BLR-01", name: "Kempegowda Airport", lat: 13.197, lng: 77.706, base: { t: 24, h: 62, p: 1013, w: 12, r: 1 } },
      { id: "BLR-02", name: "HAL Airport",        lat: 12.959, lng: 77.648, base: { t: 24, h: 64, p: 1013, w: 10, r: 0 } },
      { id: "BLR-03", name: "Whitefield",         lat: 12.969, lng: 77.749, base: { t: 25, h: 60, p: 1012, w: 9,  r: 0 } },
      { id: "BLR-04", name: "Jayanagar",          lat: 12.925, lng: 77.593, base: { t: 24, h: 66, p: 1013, w: 8,  r: 1 } },
      { id: "BLR-05", name: "Hebbal Lake",        lat: 13.035, lng: 77.597, base: { t: 23, h: 68, p: 1014, w: 11, r: 2 } },
      { id: "BLR-06", name: "Electronic City",    lat: 12.845, lng: 77.660, base: { t: 25, h: 61, p: 1012, w: 10, r: 0 } }
    ]
  },
  chennai: {
    name: "Chennai Region",
    center: [13.05, 80.25],
    zoom: 10.5,
    stations: [
      { id: "CHE-01", name: "Nungambakkam",     lat: 13.067, lng: 80.243, base: { t: 31, h: 70, p: 1009, w: 14, r: 1 } },
      { id: "CHE-02", name: "Meenambakkam Airport", lat: 12.990, lng: 80.170, base: { t: 32, h: 68, p: 1008, w: 12, r: 0 } },
      { id: "CHE-03", name: "Marina Beach",     lat: 13.050, lng: 80.282, base: { t: 30, h: 76, p: 1010, w: 18, r: 2 } },
      { id: "CHE-04", name: "Anna Nagar",       lat: 13.085, lng: 80.210, base: { t: 31, h: 71, p: 1009, w: 11, r: 1 } },
      { id: "CHE-05", name: "Ennore Port",      lat: 13.208, lng: 80.320, base: { t: 30, h: 78, p: 1010, w: 20, r: 3 } }
    ]
  },
  kolkata: {
    name: "Kolkata Region",
    center: [22.58, 88.36],
    zoom: 10.5,
    stations: [
      { id: "KOL-01", name: "Alipore",           lat: 22.530, lng: 88.320, base: { t: 30, h: 74, p: 1008, w: 10, r: 2 } },
      { id: "KOL-02", name: "Dum Dum Airport",   lat: 22.645, lng: 88.428, base: { t: 31, h: 72, p: 1007, w: 12, r: 1 } },
      { id: "KOL-03", name: "Salt Lake Sector V", lat: 22.575, lng: 88.432, base: { t: 30, h: 75, p: 1008, w: 9,  r: 2 } },
      { id: "KOL-04", name: "Howrah Maidan",     lat: 22.588, lng: 88.340, base: { t: 31, h: 73, p: 1007, w: 11, r: 1 } },
      { id: "KOL-05", name: "Diamond Harbour",   lat: 22.190, lng: 88.190, base: { t: 30, h: 80, p: 1009, w: 16, r: 4 } }
    ]
  }
};

/* ---------------- State ---------------- */
let map, markerLayer;
let stations = [];
let markers = {};
let selectedId = null;
let zThreshold = 2.5;
let showHeat = true;
let showLabels = true;
let currentRegion = "mumbai";
const history = {};            // id -> [{t,h,p,w,r,ts}]
const MAX_HISTORY = 90;

/* Evaluation metrics (fault-injection lab) */
const evalState = { injected: 0, detected: 0, tp: 0, fp: 0, fn: 0, tn: 0, log: [] };

const $ = (id) => document.getElementById(id);
const rand = (a, b) => Math.random() * (b - a) + a;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const dist = (a, b) => Math.hypot(a.lat - b.lat, a.lng - b.lng);

/* ============================================================
   AI ENGINE
   ============================================================ */
const AI = {
  /* --- Learned baselines per station/param --- */
  baseline: {},   // id -> {t:{ewma,sd,hourly:{h0:{mean,sd}...}}, ...}

  initBaseline(st) {
    this.baseline[st.id] = {};
    ["t", "h", "p"].forEach((k) => {
      this.baseline[st.id][k] = {
        ewma: st.base[k === "t" ? "t" : k],
        sd: k === "t" ? 0.8 : k === "h" ? 3 : 0.8,
        n: 1,
        hourly: {}   // seasonal memory: hour -> {sum, sumSq, n}
      };
    });
  },

  /* Update learned model with a trusted (non-anomalous) reading */
  learn(id, key, value, hour) {
    const b = this.baseline[id]?.[key];
    if (!b) return;
    const alpha = 0.08;
    b.ewma = alpha * value + (1 - alpha) * b.ewma;
    b.n++;
    // running sd via Welford-ish approximation
    const dev = Math.abs(value - b.ewma);
    b.sd = 0.9 * b.sd + 0.1 * dev;

    // seasonal memory (hour-of-day)
    const hslot = b.hourly[hour] || (b.hourly[hour] = { sum: 0, sumSq: 0, n: 0 });
    hslot.sum += value; hslot.sumSq += value * value; hslot.n++;
  },

  seasonalMean(id, key, hour) {
    const slot = this.baseline[id]?.[key]?.hourly[hour];
    if (!slot || slot.n < 3) return null;
    return slot.sum / slot.n;
  },

  /* --- L1: physical plausibility --- */
  physical(s) {
    const r = s.reading;
    const issues = [];
    if (r.t < -10 || r.t > 60) issues.push({ k: "t", msg: `Temperature ${r.t.toFixed(1)}°C outside physical range [-10,60]°C` });
    if (r.h < 0 || r.h > 100) issues.push({ k: "h", msg: `Humidity ${r.h.toFixed(0)}% outside [0,100]%` });
    if (r.p < 870 || r.p > 1084) issues.push({ k: "p", msg: `Pressure ${r.p.toFixed(0)} hPa outside [870,1084] hPa` });
    if (r.w < 0 || r.w > 120) issues.push({ k: "w", msg: `Wind ${r.w.toFixed(0)} km/h outside [0,120] km/h` });
    return issues;
  },

  /* --- L2: temporal z-score vs EWMA + seasonal expectation --- */
  temporal(s, hour) {
    const out = [];
    ["t", "h", "p"].forEach((k) => {
      const b = this.baseline[s.id]?.[k];
      if (!b || b.n < 8) return;
      const zEwma = Math.abs((s.reading[k] - b.ewma) / (b.sd || 0.001));
      const seas = this.seasonalMean(s.id, k, hour);
      const zSeas = seas !== null ? Math.abs((s.reading[k] - seas) / (b.sd || 0.001)) : 0;
      const z = Math.max(zEwma, zSeas);
      if (z >= zThreshold) {
        out.push({
          k, z,
          msg: seas !== null && zSeas >= zEwma
            ? `${PARAM[k].label} deviates ${z.toFixed(1)}σ from seasonal expectation (${seas.toFixed(1)}${PARAM[k].unit})`
            : `${PARAM[k].label} deviates ${z.toFixed(1)}σ from learned baseline (${b.ewma.toFixed(1)}${PARAM[k].unit})`
        });
      }
    });
    return out;
  },

  /* --- L3: spatial consistency vs neighbors (<0.35°) --- */
  spatial(s, all) {
    const out = [];
    const neigh = all.filter((o) => o.id !== s.id && !o.offline && !o.anomaly && dist(s, o) < 0.35);
    if (neigh.length < 2) return out;
    ["t", "h", "p"].forEach((k) => {
      const vals = neigh.map((o) => o.reading[k]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const spread = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length) || 0.001;
      const z = Math.abs((s.reading[k] - mean) / (spread * 1.6));
      if (z >= zThreshold + 0.5) {
        out.push({
          k, z,
          msg: `${PARAM[k].label} ${s.reading[k].toFixed(1)}${PARAM[k].unit} disagrees with ${neigh.length} neighbors (avg ${mean.toFixed(1)}${PARAM[k].unit}, ${z.toFixed(1)}σ)`
        });
      }
    });
    return out;
  },

  /* --- L4: multivariate physics consistency --- */
  multivariate(s) {
    const out = [];
    const { t, h, p } = s.reading;
    // Dewpoint gap: T and RH must cohere (dewpoint spread = 100-RH approx at low RH)
    const dp = t - (100 - h) / 5;
    if (t >= 45 && h >= 85) {
      out.push({ k: "t", msg: `Physics conflict: ${t.toFixed(0)}°C with ${h.toFixed(0)}% RH is meteorologically implausible (dewpoint ${dp.toFixed(0)}°C > T)` });
    }
    // Pressure drop with no wind/rain response
    const h10 = (history[s.id] || []).slice(-10);
    if (h10.length >= 6) {
      const pDrop = h10[0].p - p;
      if (pDrop >= 5 && s.reading.w < 20 && s.reading.r < 2) {
        out.push({ k: "p", msg: `Pressure fell ${pDrop.toFixed(1)} hPa rapidly but wind (${s.reading.w.toFixed(0)} km/h) and rain (${s.reading.r.toFixed(1)} mm) show no storm response → sensor suspect` });
      }
    }
    // Frozen sensor: identical values many ticks
    if (h10.length >= 8) {
      const frozen = h10.slice(-8).every((x) => Math.abs(x.t - h10[h10.length - 1].t) < 0.001);
      if (frozen) out.push({ k: "t", msg: "Frozen value: temperature unchanged for 8+ consecutive samples → stuck sensor" });
    }
    return out;
  },

  /* --- Master inference: fuse layers, explain, impute, classify --- */
  infer(s, all) {
    const hour = new Date().getHours();
    const phys = this.physical(s);
    const temp = this.temporal(s, hour);
    const spat = this.spatial(s, all);
    const mult = this.multivariate(s);

    const allIssues = [...phys, ...temp, ...spat, ...mult];
    const layersHit = new Set([
      ...(phys.length ? ["L1"] : []),
      ...(temp.length ? ["L2"] : []),
      ...(spat.length ? ["L3"] : []),
      ...(mult.length ? ["L4"] : [])
    ]);

    // Weighted vote: L1=0.30, L2=0.30, L3=0.25, L4=0.15
    let score = 0;
    if (phys.length) score += 0.30 * Math.min(1, phys.length / 2);
    if (temp.length) score += 0.30 * Math.min(1, Math.max(...temp.map(i => i.z / (zThreshold + 2))));
    if (spat.length) score += 0.25 * Math.min(1, Math.max(...spat.map(i => i.z / (zThreshold + 2))));
    if (mult.length) score += 0.15 * Math.min(1, mult.length / 2);

    const isAnomaly = score >= 0.5;
    const isWarning = !isAnomaly && score >= 0.3;

    // Confidence: agreement between independent layers boosts it
    const agreement = layersHit.size / 4;
    const confidence = clamp(Math.round((0.55 + 0.45 * agreement) * (isAnomaly ? 100 : 70)), 40, 99);

    // Root-cause classification
    const rootCause = this.classify(s, { phys, temp, spat, mult, frozen: mult.some(m => m.msg.includes("Frozen")) });

    // Corrected value imputation: blend seasonal mean + neighbor median
    const corrected = {};
    const flagged = new Set(allIssues.map(i => i.k));
    ["t", "h", "p"].forEach((k) => {
      if (!flagged.has(k)) { corrected[k] = null; return; }
      const seas = this.seasonalMean(s.id, k, hour);
      const neigh = all.filter((o) => o.id !== s.id && !o.offline && dist(s, o) < 0.35).map((o) => o.reading[k]).sort((a, b) => a - b);
      const med = neigh.length ? neigh[Math.floor(neigh.length / 2)] : null;
      const parts = [];
      if (seas !== null) parts.push(seas);
      if (med !== null) parts.push(med);
      corrected[k] = parts.length ? parts.reduce((a, b) => a + b, 0) / parts.length : this.baseline[s.id]?.[k]?.ewma ?? s.reading[k];
    });

    return { isAnomaly, isWarning, score, confidence, layersHit: [...layersHit], issues: allIssues, rootCause, corrected };
  },

  /* Root-cause classifier */
  classify(s, f) {
    const keys = new Set([...f.phys, ...f.temp, ...f.spat, ...f.mult].map(i => i.k));
    if (f.frozen) return { code: "STUCK_SENSOR", label: "Stuck/Frozen Sensor", action: "Restart datalogger; check sensor cable & ADC" };
    if (keys.has("t") && keys.has("h") && f.mult.length) return { code: "SENSOR_FAULT", label: "Multi-Sensor Fault", action: "Recalibrate station; verify RH + T probes" };
    if (f.spat.length && !f.temp.length) return { code: "SPATIAL_OUTLIER", label: "Spatial Outlier (likely sensor)", action: "Cross-check with neighbor stations; inspect sensor" };
    if (f.temp.length && f.spat.length) return { code: "SENSOR_FAULT", label: "Sensor Fault (temporal+spatial)", action: "Schedule calibration; flag data as suspect" };
    if (f.spat.length) return { code: "SPATIAL_OUTLIER", label: "Spatial Outlier", action: "Verify against neighbors before use" };
    if (f.temp.length && !f.spat.length) return { code: "REAL_EVENT", label: "Possible Real Weather Event", action: "Monitor — no spatial disagreement; may be genuine microburst/heat spike" };
    if (f.phys.length) return { code: "RANGE_VIOLATION", label: "Range Violation", action: "Check sensor wiring / datalogger config" };
    return { code: "OK", label: "Nominal", action: "None" };
  },

  /* --- Sensor health: degradation tracking + maintenance ETA --- */
  health: {},  // id -> {drift, noise, faults, uptime, samples}
  initHealth(st) {
    this.health[st.id] = { drift: 0, noise: 0.8, faults: 0, uptime: 100, samples: 0 };
  },
  updateHealth(s, verdict) {
    const H = this.health[s.id];
    if (!H) return;
    H.samples++;
    if (verdict.isAnomaly) H.faults++;
    // drift = slow bias of reading vs baseline
    const b = this.baseline[s.id]?.t;
    if (b && b.n > 20) {
      const bias = Math.abs(s.reading.t - b.ewma);
      H.drift = 0.95 * H.drift + 0.05 * bias;
    }
    H.uptime = clamp(100 - (H.faults / Math.max(H.samples, 1)) * 100, 0, 100);
  },
  healthScore(id) {
    const H = this.health[id];
    if (!H) return 100;
    return clamp(Math.round(100 - H.drift * 6 - (H.faults / Math.max(H.samples, 1)) * 60), 0, 100);
  },
  maintenanceDays(id) {
    const hs = this.healthScore(id);
    return Math.max(1, Math.round((hs / 100) * 90));
  }
};

const PARAM = {
  t: { label: "Temperature", unit: "°C" },
  h: { label: "Humidity", unit: "%" },
  p: { label: "Pressure", unit: " hPa" }
};

/* ============================================================
   FAULT INJECTION LAB (evaluation on anomaly-injected data)
   ============================================================ */
const FaultLab = {
  active: null,   // {type, stationId, ticksLeft}
  types: ["spike", "stuck", "drift", "frozen", "comm_loss", "random_drop"],

  inject(type) {
    const online = stations.filter((s) => !s.offline);
    if (!online.length) return;
    const s = online[Math.floor(Math.random() * online.length)];
    this.active = { type, stationId: s.id, ticksLeft: type === "drift" ? 25 : 10, origin: { ...s.reading } };
    evalState.injected++;
    evalState.log.push({ t: Date.now(), type, station: s.id });
    toast(`🧪 Injected ${type.toUpperCase()} fault at ${s.name} — watch the AI catch it`, "info");
  },

  tickApply(s) {
    const f = this.active;
    if (!f || f.stationId !== s.id) return false;
    const r = s.reading;
    switch (f.type) {
      case "spike": r.t = clamp(r.t + rand(14, 22), -20, 65); r.h = clamp(r.h + 25, 0, 100); break;
      case "stuck": r.t = f.origin.t; r.h = f.origin.h; r.p = f.origin.p; break;
      case "drift": r.t += 0.9; break;
      case "frozen": r.t = f.origin.t; break;
      case "comm_loss": s.offline = true; break;
      case "random_drop": r.p = clamp(r.p - rand(8, 14), 870, 1084); break;
    }
    f.ticksLeft--;
    if (f.ticksLeft <= 0) { this.active = null; if (f.type === "comm_loss") setTimeout(() => (s.offline = false), 6000); }
    return true;
  },

  /* Score detection: if station flagged while fault active → TP */
  score(s, verdict) {
    const f = this.active;
    if (!f || f.stationId !== s.id) {
      if (verdict.isAnomaly) evalState.fp++;
      return;
    }
    if (verdict.isAnomaly) { evalState.tp++; evalState.detected++; }
    else evalState.fn++;
  },

  metrics() {
    const { tp, fp, fn } = evalState;
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;
    return { precision, recall, f1, injected: evalState.injected, detected: evalState.detected };
  }
};

/* ============================================================
   SIMULATION TICK
   ============================================================ */
function tick() {
  stations.forEach((s) => {
    if (s.offline) { AI.updateHealth(s, { isAnomaly: false }); return; }

    FaultLab.tickApply(s);

    const drift = rand(-0.5, 0.5);
    s.reading.t = clamp(s.reading.t + drift, s.base.t - 8, s.base.t + 8);
    s.reading.h = clamp(s.reading.h + rand(-1.2, 1.2), 15, 99);
    s.reading.p = clamp(s.reading.p + rand(-0.6, 0.6), 980, 1035);
    s.reading.w = clamp(s.reading.w + rand(-1.5, 1.5), 0, 60);
    s.reading.r = clamp(s.reading.r + (Math.random() < 0.15 ? rand(0, 1.2) : -0.1), 0, 60);

    const verdict = AI.infer(s, stations);
    s.verdict = verdict;
    s.status = verdict.isAnomaly ? "danger" : verdict.isWarning ? "warn" : "ok";

    FaultLab.score(s, verdict);

    // Learn only from trusted readings (self-healing loop)
    if (!verdict.isAnomaly) {
      const hour = new Date().getHours();
      ["t", "h", "p"].forEach((k) => AI.learn(s.id, k, s.reading[k], hour));
    }
    AI.updateHealth(s, verdict);

    if (verdict.isAnomaly && !s.wasAnomaly) raiseAlert(s, verdict);
    s.wasAnomaly = verdict.isAnomaly;

    if (!history[s.id]) history[s.id] = [];
    history[s.id].push({ ...s.reading, ts: Date.now() });
    if (history[s.id].length > MAX_HISTORY) history[s.id].shift();
  });

  renderMarkers();
  renderTable();
  updateKPIs();
  updateHealthPanel();
  updateEvalPanel();
  if (selectedId) renderDetail(selectedId);
  drawChart();
}

/* ---------------- Alerts ---------------- */
function raiseAlert(s, v) {
  const item = document.createElement("div");
  item.className = "alert-item " + (v.confidence >= 80 ? "critical" : "warning");
  item.innerHTML = `
    <span class="alert-icon">${v.confidence >= 80 ? "🚨" : "⚠️"}</span>
    <div class="alert-body">
      <b>${s.name} (${s.id})</b>
      <span>${v.rootCause.label} · conf ${v.confidence}% · ${v.issues[0]?.msg || "multivariate conflict"}</span>
    </div>
    <span class="alert-time">${new Date().toLocaleTimeString()}</span>`;
  const list = $("alertsList");
  list.querySelector(".empty-state")?.remove();
  list.prepend(item);
  while (list.children.length > 40) list.lastChild.remove();
  toast(`🚨 ${v.rootCause.label} @ ${s.name} (confidence ${v.confidence}%)`);
}

function toast(text, kind = "alert") {
  let holder = document.querySelector(".toast");
  if (!holder) { holder = document.createElement("div"); holder.className = "toast"; document.body.appendChild(holder); }
  const t = document.createElement("div");
  t.className = "toast-item" + (kind === "info" ? " info" : "");
  t.textContent = text;
  holder.appendChild(t);
  setTimeout(() => { t.style.opacity = "0"; t.style.transition = "opacity .5s"; }, 4200);
  setTimeout(() => t.remove(), 4800);
}

/* ============================================================
   RENDERING
   ============================================================ */
function initMap() {
  map = L.map("map").setView(REGIONS.mumbai.center, REGIONS.mumbai.zoom);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18, attribution: "© OpenStreetMap contributors"
  }).addTo(map);
  markerLayer = L.layerGroup().addTo(map);
}

function markerIcon(s) {
  const label = showLabels ? `<div class="marker-label">${s.name}</div>` : "";
  return L.divIcon({
    className: "station-marker",
    html: `<div class="marker-pin ${s.status}"></div>${label}`,
    iconSize: [18, 18], iconAnchor: [9, 9]
  });
}

function renderMarkers() {
  markerLayer.clearLayers();
  if (showHeat) {
    stations.filter((s) => s.status !== "ok" && s.status !== "offline").forEach((s) => {
      const color = s.status === "danger" ? "#ff4d5e" : "#ffb020";
      L.circle([s.lat, s.lng], { radius: s.status === "danger" ? 2600 : 1500, color, weight: 1.5, fillColor: color, fillOpacity: 0.12, interactive: false }).addTo(markerLayer);
    });
  }
  stations.forEach((s) => {
    const m = L.marker([s.lat, s.lng], { icon: markerIcon(s) }).addTo(markerLayer);
    m.bindPopup(`<b>${s.name}</b><br><span style="font-size:.8rem">${s.id} · ${s.status.toUpperCase()}${s.verdict ? ` · conf ${s.verdict.confidence}%` : ""}</span><br>🌡️ ${s.reading.t.toFixed(1)}°C · 💧 ${s.reading.h.toFixed(0)}%<br>🌬️ ${s.reading.p.toFixed(0)} hPa · 💨 ${s.reading.w.toFixed(0)} km/h`);
    m.on("click", () => selectStation(s.id));
    markers[s.id] = m;
  });
}

function selectStation(id) {
  selectedId = id;
  renderDetail(id);
  const s = stations.find((x) => x.id === id);
  if (s) {
    map.setView([s.lat, s.lng], Math.max(map.getZoom(), 12), { animate: true });
    markers[id]?.openPopup();
  }
  drawChart();
}

function renderDetail(id) {
  const s = stations.find((x) => x.id === id);
  if (!s) return;
  const v = s.verdict || { confidence: 0, issues: [], rootCause: { label: "Learning…", action: "Collecting baseline" }, corrected: {}, layersHit: [] };
  const pill = $("detailStatus");
  pill.className = "status-pill " + s.status;
  pill.textContent = s.status === "ok" ? "NORMAL" : s.status.toUpperCase();

  const corrRows = Object.entries(v.corrected || {})
    .filter(([k, val]) => val !== null)
    .map(([k, val]) => `<div class="reading"><div class="r-label">✔ Corrected ${PARAM[k].label}</div><div class="r-value" style="color:var(--ok)">${val.toFixed(1)}${PARAM[k].unit}</div></div>`)
    .join("");

  const issueRows = v.issues.length
    ? v.issues.map((i) => `<li>${i.msg}</li>`).join("")
    : "<li>No rule violations — all layers agree reading is nominal.</li>";

  const contrib = ["L1 Physical", "L2 Temporal", "L3 Spatial", "L4 Physics"].map((name, idx) => {
    const hit = v.layersHit.includes("L" + (idx + 1));
    return `<div class="contrib-row"><span>${name}</span><div class="contrib-bar"><i style="width:${hit ? 100 : 8}%;background:${hit ? "var(--danger)" : "rgba(90,140,255,.3)"}"></i></div><b>${hit ? "TRIGGERED" : "pass"}</b></div>`;
  }).join("");

  const hs = AI.healthScore(s.id);
  const days = AI.maintenanceDays(s.id);

  $("detailBody").innerHTML = `
    <h4 style="font-size:1.05rem">${s.name} <span style="font-size:.72rem;color:var(--text-dim)">${s.id}</span></h4>
    <div class="conf-row">
      <span>Confidence</span>
      <div class="conf-bar"><i style="width:${v.confidence}%;background:${v.confidence >= 80 ? "var(--danger)" : "var(--warn)"}"></i></div>
      <b>${v.confidence}%</b>
    </div>
    <div class="reading-grid">
      <div class="reading"><div class="r-label">🌡️ Temperature</div><div class="r-value">${s.reading.t.toFixed(1)}°C</div></div>
      <div class="reading"><div class="r-label">💧 Humidity</div><div class="r-value">${s.reading.h.toFixed(0)}%</div></div>
      <div class="reading"><div class="r-label">🌬️ Pressure</div><div class="r-value">${s.reading.p.toFixed(1)} hPa</div></div>
      <div class="reading"><div class="r-label">💨 Wind</div><div class="r-value">${s.reading.w.toFixed(0)} km/h</div></div>
    </div>
    ${corrRows ? `<h5 class="mini-h">🩹 Corrected (imputed) values</h5><div class="reading-grid">${corrRows}</div>` : ""}
    <h5 class="mini-h">🧠 Explainable AI — layer contributions</h5>
    <div class="contrib">${contrib}</div>
    <h5 class="mini-h">📋 Evidence</h5>
    <ul class="evidence">${issueRows}</ul>
    <div class="anomaly-note ${s.status === "ok" ? "ok" : ""}">
      <b>Root cause: ${v.rootCause.label}</b><br>Suggested action: ${v.rootCause.action}
    </div>
    <h5 class="mini-h">🩺 Sensor Health</h5>
    <div class="conf-row"><span>Health score</span><div class="conf-bar"><i style="width:${hs}%;background:${hs > 70 ? "var(--ok)" : hs > 40 ? "var(--warn)" : "var(--danger)"}"></i></div><b>${hs}/100</b></div>
    <p class="hint">Predicted maintenance due in <b>${days} days</b> · uptime ${AI.health[s.id]?.uptime.toFixed(0) ?? 100}%</p>`;
}

function renderTable() {
  const tbody = $("stationTable");
  tbody.innerHTML = stations.map((s) => {
    const v = s.verdict || {};
    const scoreColor = s.status === "danger" ? "var(--danger)" : s.status === "warn" ? "var(--warn)" : "var(--ok)";
    return `<tr data-id="${s.id}">
      <td class="td-name">${s.name}<br><span style="font-size:.7rem;color:var(--text-dim)">${s.id}</span></td>
      <td><span class="badge ${s.status}">${s.status}</span></td>
      <td>${s.offline ? "--" : s.reading.t.toFixed(1)}</td>
      <td>${s.offline ? "--" : s.reading.h.toFixed(0)}</td>
      <td>${s.offline ? "--" : s.reading.p.toFixed(0)}</td>
      <td>${s.offline ? "--" : s.reading.w.toFixed(0)}</td>
      <td>${s.offline ? "--" : s.reading.r.toFixed(1)}</td>
      <td><span class="score-bar"><i style="width:${Math.round((v.score || 0) * 100)}%;background:${scoreColor}"></i></span>${v.confidence ?? 0}%</td>
    </tr>`;
  }).join("");
  $("tableCount").textContent = `${stations.length} stations · ${REGIONS[currentRegion].name}`;
  tbody.querySelectorAll("tr").forEach((tr) => tr.addEventListener("click", () => selectStation(tr.dataset.id)));
}

function updateKPIs() {
  const active = stations.filter((s) => !s.offline);
  const anomalies = stations.filter((s) => s.status === "danger");
  const avgT = active.length ? active.reduce((a, s) => a + s.reading.t, 0) / active.length : 0;
  const avgH = active.length ? active.reduce((a, s) => a + s.reading.h, 0) / active.length : 0;
  $("kpiStations").textContent = stations.length;
  $("kpiAnomalies").textContent = anomalies.length;
  $("kpiTemp").textContent = avgT.toFixed(1) + "°C";
  $("kpiHum").textContent = avgH.toFixed(0) + "%";
}

function updateHealthPanel() {
  const el = $("healthList");
  if (!el) return;
  el.innerHTML = stations.map((s) => {
    const hs = AI.healthScore(s.id);
    const col = hs > 70 ? "var(--ok)" : hs > 40 ? "var(--warn)" : "var(--danger)";
    return `<div class="health-row"><span class="td-name">${s.name}</span><div class="conf-bar"><i style="width:${hs}%;background:${col}"></i></div><b style="color:${col}">${hs}</b></div>`;
  }).join("");
}

function updateEvalPanel() {
  const m = FaultLab.metrics();
  $("evalInjected").textContent = m.injected;
  $("evalDetected").textContent = m.detected;
  $("evalPrecision").textContent = (m.precision * 100).toFixed(0) + "%";
  $("evalRecall").textContent = (m.recall * 100).toFixed(0) + "%";
  $("evalF1").textContent = m.f1.toFixed(2);
}

/* ---------------- Chart ---------------- */
function drawChart() {
  const canvas = $("tempChart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.offsetWidth || 600;
  const cssH = 150;
  if (canvas.width !== cssW * dpr) canvas.width = cssW * dpr;
  if (canvas.height !== cssH * dpr) canvas.height = cssH * dpr;
  canvas.style.height = cssH + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = cssW, H = cssH;

  let series, series2, label;
  if (selectedId && history[selectedId]) {
    const h = history[selectedId];
    series = h.map((r) => r.t);
    series2 = h.map((r) => r.p / 10);   // pressure /10 for overlay
    label = stations.find((s) => s.id === selectedId)?.name || selectedId;
  } else {
    const hs = Object.values(history);
    const len = Math.max(0, ...hs.map((h) => h.length));
    series = []; series2 = [];
    for (let i = 0; i < len; i++) {
      let st = 0, sp = 0, n = 0;
      hs.forEach((h) => { if (h[i]) { st += h[i].t; sp += h[i].p / 10; n++; } });
      if (n) { series.push(st / n); series2.push(sp / n); }
    }
    label = "Network average";
  }
  $("chartStation").textContent = label;

  ctx.clearRect(0, 0, W, H);
  if (series.length < 2) {
    ctx.fillStyle = "#8fa3c8"; ctx.font = "13px sans-serif";
    ctx.fillText("Collecting telemetry…", 20, H / 2);
    return;
  }
  const all = [...series, ...series2];
  const min = Math.min(...all) - 0.8, max = Math.max(...all) + 0.8;
  const px = (i) => (i / (series.length - 1)) * (W - 40) + 20;
  const py = (v) => H - 24 - ((v - min) / (max - min)) * (H - 44);

  ctx.strokeStyle = "rgba(90,140,255,0.12)"; ctx.lineWidth = 1;
  for (let g = 0; g <= 3; g++) { const y = 20 + (g / 3) * (H - 44); ctx.beginPath(); ctx.moveTo(20, y); ctx.lineTo(W - 20, y); ctx.stroke(); }

  const plot = (data, color, fill) => {
    ctx.beginPath();
    ctx.moveTo(px(0), py(data[0]));
    data.forEach((v, i) => ctx.lineTo(px(i), py(v)));
    if (fill) {
      ctx.lineTo(px(data.length - 1), H - 24); ctx.lineTo(px(0), H - 24); ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, fill); grad.addColorStop(1, "rgba(0,212,255,0.02)");
      ctx.fillStyle = grad; ctx.fill();
      ctx.beginPath(); ctx.moveTo(px(0), py(data[0]));
      data.forEach((v, i) => ctx.lineTo(px(i), py(v)));
    }
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.shadowColor = color; ctx.shadowBlur = 6;
    ctx.stroke(); ctx.shadowBlur = 0;
  };
  plot(series2, "rgba(124,92,255,0.55)", null);
  plot(series, "#00d4ff", "rgba(0,212,255,0.30)");

  const lx = px(series.length - 1), ly = py(series[series.length - 1]);
  ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fillStyle = "#ff6901"; ctx.fill();
}

/* ---------------- Region loading ---------------- */
function loadRegion(key) {
  currentRegion = key;
  const r = REGIONS[key];
  $("regionName").textContent = r.name;
  stations = r.stations.map((st) => ({
    ...st,
    reading: { t: st.base.t + rand(-1, 1), h: st.base.h + rand(-3, 3), p: st.base.p + rand(-1, 1), w: st.base.w, r: st.base.r },
    offline: false, status: "ok", verdict: null, wasAnomaly: false
  }));
  stations.forEach((st) => { AI.initBaseline(st); AI.initHealth(st); history[st.id] = []; });
  selectedId = null;
  $("detailStatus").className = "status-pill";
  $("detailStatus").textContent = "Select a station";
  $("detailBody").innerHTML = `<div class="empty-state"><div class="empty-icon">🛰️</div><p>Click any station marker on the map<br>to inspect live AI analysis.</p></div>`;
  map.setView(r.center, r.zoom, { animate: true });
  renderMarkers(); renderTable(); updateKPIs(); updateHealthPanel(); drawChart();
}

/* ---------------- Auth guard ---------------- */
(function initUser() {
  const session = window.SkyAuth ? SkyAuth.session() : null;
  if (!session) { window.location.href = "login.html"; return; }
  $("userName").textContent = session.name;
  $("userAvatar").textContent = session.name.trim().charAt(0).toUpperCase() || "?";
  $("logoutBtn").addEventListener("click", () => { SkyAuth.logout(); window.location.href = "login.html"; });
})();

/* ---------------- Clock ---------------- */
setInterval(() => { $("liveClock").textContent = new Date().toLocaleTimeString(); }, 1000);

/* ---------------- Controls ---------------- */
$("regionSelect").addEventListener("change", (e) => loadRegion(e.target.value));

$("zSlider").addEventListener("input", (e) => {
  zThreshold = parseFloat(e.target.value);
  $("zLabel").textContent = zThreshold.toFixed(1) + "σ";
});

$("toggleHeat").addEventListener("click", (e) => { showHeat = !showHeat; e.currentTarget.classList.toggle("active", showHeat); renderMarkers(); });
$("toggleLabels").addEventListener("click", (e) => { showLabels = !showLabels; e.currentTarget.classList.toggle("active", showLabels); renderMarkers(); });
$("fitBtn").addEventListener("click", () => { const r = REGIONS[currentRegion]; map.setView(r.center, r.zoom, { animate: true }); });
$("clearAlerts").addEventListener("click", () => {
  $("alertsList").innerHTML = `<div class="empty-state small"><p>No anomalies detected yet.<br>Monitoring…</p></div>`;
});

document.querySelectorAll(".fault-btn").forEach((b) =>
  b.addEventListener("click", () => FaultLab.inject(b.dataset.fault))
);

$("stationSearch").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  if (!q) return;
  const match = stations.find((s) => s.name.toLowerCase().includes(q) || s.id.toLowerCase().includes(q));
  if (match) selectStation(match.id);
});

document.querySelectorAll(".side-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".side-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    if (view === "alerts") $("alertsList").scrollIntoView({ behavior: "smooth", block: "center" });
    if (view === "stations") $("stationTable").scrollIntoView({ behavior: "smooth", block: "center" });
    if (view === "analytics") $("tempChart").scrollIntoView({ behavior: "smooth", block: "center" });
    if (view === "overview") window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

window.addEventListener("resize", drawChart);

/* ---------------- Boot ---------------- */
initMap();
loadRegion("mumbai");
$("toggleHeat").classList.add("active");
$("toggleLabels").classList.add("active");
setInterval(tick, 2000);
setTimeout(tick, 500);