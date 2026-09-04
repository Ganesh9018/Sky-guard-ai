const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const PORT = process.env.PORT || 3001;
const IMD_BASE = (process.env.IMD_API_BASE_URL || 'https://api.imd.gov.in').replace(/\/$/, '');
const IMD_API_KEY = process.env.IMD_API_KEY || '';
const ROOT = __dirname;

function getJson(url) {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const mod = parsed.protocol === 'https:' ? https : http;

    const req = mod.get(parsed, { headers: { Accept: 'application/json', ...(IMD_API_KEY ? { 'x-api-key': IMD_API_KEY } : {}) } }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        if (res.statusCode >= 400) {
          try {
            const payload = JSON.parse(data);
            reject(new Error(`IMD API error ${res.statusCode}: ${JSON.stringify(payload)}`));
          } catch {
            reject(new Error(`IMD API error ${res.statusCode}: ${data}`));
          }
          return;
        }

        try {
          resolve(JSON.parse(data || '{}'));
        } catch (err) {
          resolve({ raw: data });
        }
      });
    });

    req.on('error', reject);
  });
}

function sanitizeStation(raw) {
  const id = raw.station_id || raw.stationId || raw.id || `IMD-${Math.random().toString(36).slice(2, 8)}`;
  const temperature = Number(raw.temperature ?? raw.temp ?? raw.current_temperature ?? raw.air_temp ?? 0);
  const humidity = Number(raw.humidity ?? raw.rh ?? raw.relative_humidity ?? 0);
  const pressure = Number(raw.pressure ?? raw.sea_level_pressure ?? raw.atm_pressure ?? 0);
  const wind = Number(raw.wind_speed ?? raw.ws ?? raw.wind ?? 0);
  const rain = Number(raw.rainfall ?? raw.rain ?? raw.precipitation ?? 0);
  const lat = Number(raw.lat ?? raw.latitude ?? 0);
  const lng = Number(raw.lng ?? raw.lon ?? raw.longitude ?? 0);

  return {
    station_id: String(id),
    name: raw.name || raw.station_name || raw.station || String(id),
    state: raw.state || raw.state_name || 'India',
    region: raw.region || raw.zone || 'India',
    lat: isFinite(lat) ? lat : 0,
    lng: isFinite(lng) ? lng : 0,
    temperature: isFinite(temperature) ? temperature : 0,
    humidity: isFinite(humidity) ? humidity : 0,
    pressure: isFinite(pressure) ? pressure : 0,
    wind: isFinite(wind) ? wind : 0,
    rain: isFinite(rain) ? rain : 0,
    source: 'IMD',
    timestamp: raw.timestamp || new Date().toISOString(),
  };
}

function normalizeImdPayload(payload) {
  if (!payload || typeof payload !== 'object') return [];

  const candidates = [];

  const pushObject = (obj) => {
    if (!obj || typeof obj !== 'object') return;
    candidates.push(obj);
  };

  if (Array.isArray(payload.data)) payload.data.forEach(pushObject);
  if (Array.isArray(payload.stations)) payload.stations.forEach(pushObject);
  if (Array.isArray(payload.weather)) payload.weather.forEach(pushObject);
  if (Array.isArray(payload.results)) payload.results.forEach(pushObject);

  const direct = payload.data || payload.stations || payload.weather || payload.results;
  if (Array.isArray(direct)) direct.forEach(pushObject);

  if (!candidates.length && payload && Object.keys(payload).length) {
    candidates.push(payload);
  }

  return candidates.map((item) => sanitizeStation(item));
}

async function fetchStationFeed() {
  const endpoints = [
    `${IMD_BASE}/api/v1/current_wx`,
    `${IMD_BASE}/api/v1/observations`,
    `${IMD_BASE}/` 
  ];

  let merged = [];

  for (const url of endpoints) {
    try {
      const payload = await getJson(url);
      const stations = normalizeImdPayload(payload);
      if (stations.length) merged = merged.concat(stations);
    } catch (err) {
      console.warn(`IMD fetch failed for ${url}: ${err.message}`);
    }
  }

  if (!merged.length) {
    return {
      source: 'fallback',
      stations: [],
      message: 'No live IMD data available from the current endpoint configuration.',
    };
  }

  const dedupMap = new Map();
  for (const station of merged) {
    const key = `${station.station_id}-${station.name}`.toLowerCase();
    if (!dedupMap.has(key)) dedupMap.set(key, station);
  }

  return {
    source: 'IMD',
    stations: Array.from(dedupMap.values()).slice(0, 1200),
    generated_at: new Date().toISOString(),
  };
}

function serveStaticFile(filePath, res) {
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.ico': 'image/x-icon',
    '.txt': 'text/plain; charset=utf-8'
  };

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found');
      return;
    }

    res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
    res.end(data);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === 'GET' && url.pathname === '/api/imd/stations') {
    try {
      const feed = await fetchStationFeed();
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' });
      res.end(JSON.stringify(feed, null, 2));
    } catch (error) {
      res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' });
      res.end(JSON.stringify({ error: error.message }, null, 2));
    }
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' });
    res.end(JSON.stringify({ ok: true, service: 'SkyWatch IMD Proxy', timestamp: new Date().toISOString() }));
    return;
  }

  let filePath = ROOT;
  const relativePath = url.pathname === '/' ? '/index.html' : url.pathname;
  filePath = path.join(ROOT, relativePath);

  if (filePath.startsWith(ROOT) === false) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Forbidden');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (!err && stats.isFile()) {
      serveStaticFile(filePath, res);
      return;
    }

    if (!err && stats.isDirectory()) {
      serveStaticFile(path.join(filePath, 'index.html'), res);
      return;
    }

    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Not found');
  });
});

server.listen(PORT, () => {
  console.log(`SkyWatch IMD proxy running on http://localhost:${PORT}`);
  console.log(`IMD base: ${IMD_BASE}`);
  console.log('Endpoints available:');
  console.log('  GET /api/health');
  console.log('  GET /api/imd/stations');
});
