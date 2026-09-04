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

/* ---------------- Region data (India-wide network) ---------------- */
const INDIA_NETWORK = [
  { id: "DEL-01", name: "Safdarjung", state: "Delhi", region: "North India", lat: 28.586, lng: 77.190, base: { t: 27, h: 55, p: 1010, w: 10, r: 0 } },
  { id: "DEL-02", name: "Palam Airport", state: "Delhi", region: "North India", lat: 28.566, lng: 77.100, base: { t: 28, h: 52, p: 1009, w: 12, r: 0 } },
  { id: "DEL-03", name: "Ridge Road", state: "Delhi", region: "North India", lat: 28.635, lng: 77.170, base: { t: 26, h: 58, p: 1011, w: 8, r: 1 } },
  { id: "CHA-01", name: "Chandigarh Sector 17", state: "Punjab", region: "North India", lat: 30.733, lng: 76.788, base: { t: 25, h: 57, p: 1012, w: 9, r: 0 } },
  { id: "JAI-01", name: "Jaipur Sanganer", state: "Rajasthan", region: "North India", lat: 26.810, lng: 75.800, base: { t: 28, h: 48, p: 1010, w: 11, r: 0 } },
  { id: "AMR-01", name: "Amritsar Airport", state: "Punjab", region: "North India", lat: 31.709, lng: 74.797, base: { t: 24, h: 60, p: 1013, w: 10, r: 0 } },
  { id: "LCN-01", name: "Lucknow Gomti", state: "Uttar Pradesh", region: "North India", lat: 26.847, lng: 80.947, base: { t: 27, h: 53, p: 1011, w: 8, r: 1 } },
  { id: "DUN-01", name: "Dehradun", state: "Uttarakhand", region: "North India", lat: 30.316, lng: 78.032, base: { t: 23, h: 62, p: 1012, w: 7, r: 1 } },
  { id: "SRI-01", name: "Srinagar Airport", state: "Jammu & Kashmir", region: "North India", lat: 34.083, lng: 74.797, base: { t: 18, h: 64, p: 1014, w: 6, r: 2 } },

  { id: "MUM-01", name: "Colaba Observatory", state: "Maharashtra", region: "West India", lat: 18.906, lng: 72.815, base: { t: 29, h: 78, p: 1008, w: 12, r: 2 } },
  { id: "MUM-02", name: "Santacruz Airport", state: "Maharashtra", region: "West India", lat: 19.088, lng: 72.852, base: { t: 30, h: 74, p: 1007, w: 10, r: 1 } },
  { id: "MUM-03", name: "Bandra Station", state: "Maharashtra", region: "West India", lat: 19.055, lng: 72.840, base: { t: 30, h: 76, p: 1008, w: 14, r: 0 } },
  { id: "PUN-01", name: "Pune Airport", state: "Maharashtra", region: "West India", lat: 18.579, lng: 73.908, base: { t: 26, h: 59, p: 1010, w: 9, r: 1 } },
  { id: "NAG-01", name: "Nagpur Sadar", state: "Maharashtra", region: "Central India", lat: 21.146, lng: 79.084, base: { t: 29, h: 52, p: 1010, w: 10, r: 1 } },
  { id: "AHD-01", name: "Ahmedabad Airport", state: "Gujarat", region: "West India", lat: 23.078, lng: 72.634, base: { t: 30, h: 49, p: 1009, w: 13, r: 0 } },
  { id: "SUR-01", name: "Surat City", state: "Gujarat", region: "West India", lat: 21.170, lng: 72.831, base: { t: 31, h: 55, p: 1008, w: 14, r: 1 } },
  { id: "RAJ-01", name: "Rajkot Airport", state: "Gujarat", region: "West India", lat: 22.309, lng: 70.802, base: { t: 29, h: 52, p: 1009, w: 12, r: 0 } },
  { id: "GOA-01", name: "Panaji", state: "Goa", region: "West India", lat: 15.49, lng: 73.827, base: { t: 29, h: 70, p: 1010, w: 11, r: 2 } },
  { id: "BPL-01", name: "Bhopal Airport", state: "Madhya Pradesh", region: "Central India", lat: 23.287, lng: 77.337, base: { t: 27, h: 50, p: 1011, w: 8, r: 0 } },
  { id: "IND-01", name: "Indore Airport", state: "Madhya Pradesh", region: "Central India", lat: 22.721, lng: 75.801, base: { t: 26, h: 45, p: 1012, w: 9, r: 0 } },

  { id: "BLR-01", name: "Kempegowda Airport", state: "Karnataka", region: "South India", lat: 13.197, lng: 77.706, base: { t: 24, h: 62, p: 1013, w: 12, r: 1 } },
  { id: "BLR-02", name: "HAL Airport", state: "Karnataka", region: "South India", lat: 12.959, lng: 77.648, base: { t: 24, h: 64, p: 1013, w: 10, r: 0 } },
  { id: "BLR-03", name: "Whitefield", state: "Karnataka", region: "South India", lat: 12.969, lng: 77.749, base: { t: 25, h: 60, p: 1012, w: 9, r: 0 } },
  { id: "CHE-01", name: "Nungambakkam", state: "Tamil Nadu", region: "South India", lat: 13.067, lng: 80.243, base: { t: 31, h: 70, p: 1009, w: 14, r: 1 } },
  { id: "CHE-02", name: "Meenambakkam Airport", state: "Tamil Nadu", region: "South India", lat: 12.990, lng: 80.170, base: { t: 32, h: 68, p: 1008, w: 12, r: 0 } },
  { id: "COI-01", name: "Coimbatore Airport", state: "Tamil Nadu", region: "South India", lat: 11.031, lng: 76.966, base: { t: 28, h: 64, p: 1011, w: 10, r: 0 } },
  { id: "HYD-01", name: "Begumpet Airport", state: "Telangana", region: "South India", lat: 17.453, lng: 78.467, base: { t: 28, h: 58, p: 1010, w: 11, r: 1 } },
  { id: "TVM-01", name: "Thiruvananthapuram", state: "Kerala", region: "South India", lat: 8.524, lng: 76.936, base: { t: 29, h: 72, p: 1010, w: 15, r: 3 } },
  { id: "KOC-01", name: "Kochi Aluva", state: "Kerala", region: "South India", lat: 10.030, lng: 76.331, base: { t: 30, h: 74, p: 1009, w: 14, r: 2 } },
  { id: "VJA-01", name: "Vijayawada Airport", state: "Andhra Pradesh", region: "South India", lat: 16.518, lng: 80.648, base: { t: 30, h: 61, p: 1011, w: 12, r: 1 } },
  { id: "VIZ-01", name: "Visakhapatnam Airport", state: "Andhra Pradesh", region: "South India", lat: 17.727, lng: 83.224, base: { t: 30, h: 66, p: 1011, w: 13, r: 1 } },

  { id: "KOL-01", name: "Alipore", state: "West Bengal", region: "East India", lat: 22.530, lng: 88.320, base: { t: 30, h: 74, p: 1008, w: 10, r: 2 } },
  { id: "KOL-02", name: "Dum Dum Airport", state: "West Bengal", region: "East India", lat: 22.645, lng: 88.428, base: { t: 31, h: 72, p: 1007, w: 12, r: 1 } },
  { id: "PAT-01", name: "Patna Airport", state: "Bihar", region: "East India", lat: 25.594, lng: 85.091, base: { t: 29, h: 56, p: 1010, w: 9, r: 1 } },
  { id: "BBS-01", name: "Bhubaneswar Airport", state: "Odisha", region: "East India", lat: 20.249, lng: 85.812, base: { t: 30, h: 67, p: 1009, w: 11, r: 1 } },
  { id: "RNC-01", name: "Ranchi Airport", state: "Jharkhand", region: "East India", lat: 23.315, lng: 85.321, base: { t: 27, h: 55, p: 1010, w: 10, r: 1 } },
  { id: "GAY-01", name: "Gaya Airport", state: "Bihar", region: "East India", lat: 24.744, lng: 84.951, base: { t: 29, h: 58, p: 1010, w: 8, r: 0 } },
  { id: "SIL-01", name: "Siliguri", state: "West Bengal", region: "North-East India", lat: 26.727, lng: 88.395, base: { t: 27, h: 72, p: 1010, w: 7, r: 2 } },

  { id: "GUA-01", name: "Guwahati Airport", state: "Assam", region: "North-East India", lat: 26.106, lng: 91.585, base: { t: 27, h: 74, p: 1009, w: 8, r: 3 } },
  { id: "DIB-01", name: "Dibrugarh Airport", state: "Assam", region: "North-East India", lat: 27.483, lng: 95.016, base: { t: 26, h: 77, p: 1008, w: 7, r: 4 } },
  { id: "SHL-01", name: "Shillong Airport", state: "Meghalaya", region: "North-East India", lat: 25.561, lng: 91.886, base: { t: 21, h: 76, p: 1011, w: 7, r: 5 } },
  { id: "IMP-01", name: "Imphal Airport", state: "Manipur", region: "North-East India", lat: 24.760, lng: 93.896, base: { t: 24, h: 70, p: 1011, w: 9, r: 3 } },
  { id: "AGT-01", name: "Agartala Airport", state: "Tripura", region: "North-East India", lat: 23.890, lng: 91.259, base: { t: 27, h: 74, p: 1010, w: 8, r: 3 } },
  { id: "ITN-01", name: "Itanagar Airport", state: "Arunachal Pradesh", region: "North-East India", lat: 27.55, lng: 93.867, base: { t: 24, h: 70, p: 1011, w: 7, r: 3 } },
  { id: "GAN-01", name: "Gangtok", state: "Sikkim", region: "North-East India", lat: 27.332, lng: 88.614, base: { t: 18, h: 68, p: 1012, w: 6, r: 3 } },
  { id: "KOH-01", name: "Kohima Airport", state: "Nagaland", region: "North-East India", lat: 25.666, lng: 94.107, base: { t: 23, h: 72, p: 1011, w: 8, r: 4 } }
];

const REGIONS = {
  india: { name: "All India Network", center: [22.5, 78.96], zoom: 5, stations: INDIA_NETWORK },
  north: { name: "North India", center: [28.6, 77.2], zoom: 6, stations: INDIA_NETWORK.filter((s) => s.region === "North India") },
  south: { name: "South India", center: [15.5, 78.5], zoom: 6, stations: INDIA_NETWORK.filter((s) => s.region === "South India") },
  west: { name: "West India", center: [22.0, 73.5], zoom: 6, stations: INDIA_NETWORK.filter((s) => s.region === "West India") },
  east: { name: "East India", center: [24.8, 85.0], zoom: 6, stations: INDIA_NETWORK.filter((s) => s.region === "East India") },
  northeast: { name: "North-East India", center: [26.5, 90.0], zoom: 6, stations: INDIA_NETWORK.filter((s) => s.region === "North-East India") },
  central: { name: "Central India", center: [23.0, 78.7], zoom: 6, stations: INDIA_NETWORK.filter((s) => s.region === "Central India") }
};

async function fetchLiveStations() {
  try {
    const res = await fetch("/api/imd/stations");
    if (!res.ok) return null;
    const payload = await res.json();
    if (!payload || !Array.isArray(payload.stations) || !payload.stations.length) return null;
    return payload.stations.map((s) => ({
      id: s.station_id || s.id,
      name: s.name || s.station_name || s.station_id || 'Station',
      state: s.state || 'India',
      region: s.region || 'India',
      lat: Number(s.lat ?? s.latitude ?? 0),
      lng: Number(s.lng ?? s.lon ?? s.longitude ?? 0),
      base: {
        t: Number(s.temperature ?? 26),
        h: Number(s.humidity ?? 60),
        p: Number(s.pressure ?? 1010),
        w: Number(s.wind ?? 10),
        r: Number(s.rain ?? 0)
      }
    }));
  } catch (err) {
    return null;
  }
}

/* ---------------- State ---------------- */
let map, markerLayer;
let stations = [];
let markers = {};
let selectedId = null;
let zThreshold = 2.5;
let showHeat = true;
let showLabels = true;
let currentRegion = "india";
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

  inject(type, stationId = selectedId || null) {
    const online = stations.filter((s) => !s.offline);
    if (!online.length) return;
    const s = online.find((x) => x.id === stationId) || online[Math.floor(Math.random() * online.length)];
    this.active = { type, stationId: s.id, ticksLeft: type === "drift" ? 25 : 10, origin: { ...s.reading } };
    evalState.injected++;
    evalState.log.push({ t: Date.now(), type, station: s.id });
    toast(`🧪 Injected ${type.toUpperCase()} fault at ${s.name} — AI detection is active`, "info");
    if (selectedId !== s.id) selectStation(s.id);
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
  updateWorkspaceData();
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
  map = L.map("map").setView(REGIONS.india.center, REGIONS.india.zoom);
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

function updateWorkspaceData() {
  const ranked = [...stations].sort((a, b) => AI.healthScore(a.id) - AI.healthScore(b.id));
  const maintenance = $("maintenanceList");
  if (maintenance) {
    maintenance.innerHTML = ranked.slice(0, 8).map((s) => {
      const health = AI.healthScore(s.id);
      const color = health > 70 ? "var(--ok)" : health > 40 ? "var(--warn)" : "var(--danger)";
      return `<div class="maintenance-row"><div><b>${s.name}</b><span>${s.id} · due in ${AI.maintenanceDays(s.id)} days</span></div><div class="maintenance-score"><i style="width:${health}%;background:${color}"></i><b style="color:${color}">${health}%</b></div></div>`;
    }).join("");
  }
  const ready = stations.filter((s) => AI.healthScore(s.id) > 70).length;
  $("readyCount")?.replaceChildren(document.createTextNode(ready));
  $("warningCount")?.replaceChildren(document.createTextNode(stations.length - ready));
  const active = stations.filter((s) => !s.offline);
  const anomalies = stations.filter((s) => s.status === "danger").length;
  const avgTemp = active.length ? active.reduce((sum, s) => sum + s.reading.t, 0) / active.length : 0;
  const avgHumidity = active.length ? active.reduce((sum, s) => sum + s.reading.h, 0) / active.length : 0;
  const summary = $("reportSummary");
  if (summary) summary.innerHTML = `<div><span>Reporting stations</span><b>${active.length}</b></div><div><span>Active anomalies</span><b class="report-danger">${anomalies}</b></div><div><span>Mean temperature</span><b>${avgTemp.toFixed(1)}°C</b></div><div><span>Mean humidity</span><b>${avgHumidity.toFixed(0)}%</b></div>`;
  const source = $("dataSourceLabel")?.textContent || "Live telemetry";
  if ($("reportSource")) $("reportSource").textContent = `${source} · generated ${new Date().toLocaleString()}`;
  if ($("lastSync")) $("lastSync").textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const quality = stations.length ? Math.round(stations.reduce((sum, station) => sum + AI.healthScore(station.id), 0) / stations.length) : 0;
  const reporting = stations.length ? Math.round((active.length / stations.length) * 100) : 0;
  $("qualityScore")?.replaceChildren(document.createTextNode(`${quality}%`));
  $("reportingScore")?.replaceChildren(document.createTextNode(`${reporting}%`));
  $("reviewScore")?.replaceChildren(document.createTextNode(String(stations.length - ready)));
  if ($("qualityBar")) $("qualityBar").style.width = `${quality}%`;
  if ($("reportingBar")) $("reportingBar").style.width = `${reporting}%`;
  const activity = $("overviewActivity");
  if (activity) {
    const anomalyCount = stations.filter((station) => station.status === "danger").length;
    const warningCount = stations.filter((station) => station.status === "warn").length;
    activity.innerHTML = `<div class="activity-row"><span class="activity-dot ${anomalyCount ? "danger" : "ok"}"></span><div><b>${anomalyCount ? `${anomalyCount} anomaly${anomalyCount === 1 ? "" : "ies"} detected` : "No critical anomalies"}</b><small>AI scan completed across the network</small></div></div><div class="activity-row"><span class="activity-dot ${warningCount ? "warn" : "ok"}"></span><div><b>${warningCount ? `${warningCount} station${warningCount === 1 ? "" : "s"} need review` : "All stations within range"}</b><small>Health and telemetry checks are current</small></div></div><div class="activity-row"><span class="activity-dot ok"></span><div><b>Live stream connected</b><small>Last sync ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small></div></div>`;
  }
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
 async function loadRegion(key) {
  currentRegion = key;
  const r = REGIONS[key];
  $("regionName").textContent = r.name;

  const liveStations = await fetchLiveStations();
  const isLiveSource = Array.isArray(liveStations) && liveStations.length > 0;
  const sourceStations = isLiveSource ? liveStations : r.stations;
  const regionStations = sourceStations.filter((st) => !key || key === 'india' || st.region === r.name || st.region === key || st.state === key);

  stations = regionStations.map((st) => ({
    ...st,
    reading: { t: Number(st.base?.t ?? st.temperature ?? 26) + rand(-1, 1), h: Number(st.base?.h ?? st.humidity ?? 60) + rand(-3, 3), p: Number(st.base?.p ?? st.pressure ?? 1010) + rand(-1, 1), w: Number(st.base?.w ?? st.wind ?? 10), r: Number(st.base?.r ?? st.rain ?? 0) },
    offline: false, status: "ok", verdict: null, wasAnomaly: false
  }));

  if (!stations.length) {
    stations = r.stations.map((st) => ({
      ...st,
      reading: { t: st.base.t + rand(-1, 1), h: st.base.h + rand(-3, 3), p: st.base.p + rand(-1, 1), w: st.base.w, r: st.base.r },
      offline: false, status: "ok", verdict: null, wasAnomaly: false
    }));
  }

  const sourceLabel = isLiveSource ? 'Data source: IMD live feed' : 'Data source: local India fallback';
  const ds = $('dataSourceLabel');
  if (ds) ds.textContent = sourceLabel;

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
  if (!q) {
    renderTable();
    return;
  }
  const match = stations.find((s) =>
    s.name.toLowerCase().includes(q) ||
    s.id.toLowerCase().includes(q) ||
    (s.state || "").toLowerCase().includes(q) ||
    (s.region || "").toLowerCase().includes(q)
  );
  if (match) selectStation(match.id);
});

const VIEW_META = {
  overview: ["Dashboard Overview", "A live view of station health, alerts, and weather telemetry."],
  map: ["Live Network Map", "Inspect station locations and drill into current telemetry."],
  maintenance: ["Maintenance Planning", "Prioritize field work using predicted sensor health."],
  analysis: ["Analysis Studio", "Review trends, alerts, and anomaly detection performance."],
  reports: ["Network Reports", "A concise operational summary for the current observation window."]
};

function setDashboardView(view) {
  document.querySelectorAll("[data-dashboard-view]").forEach((section) => {
    section.hidden = section.dataset.dashboardView !== view;
  });
  document.querySelectorAll(".side-btn").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  const meta = VIEW_META[view] || VIEW_META.overview;
  $("pageTitle").textContent = meta[0];
  $("pageSubtitle").textContent = meta[1];
  if (view === "map") setTimeout(() => map?.invalidateSize(), 0);
  updateWorkspaceData();
}

document.querySelectorAll(".side-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".side-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    setDashboardView(view);
  });
});

$("printReport")?.addEventListener("click", () => window.print());

window.addEventListener("resize", drawChart);

/* ---------------- Boot ---------------- */
initMap();
loadRegion("india");
setDashboardView("overview");
$("toggleHeat").classList.add("active");
$("toggleLabels").classList.add("active");
setInterval(tick, 2000);
setTimeout(tick, 500);