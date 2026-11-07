// Obtención de precios y cálculo de rendimientos, media y covarianza.
// Reemplaza a yfinance + pandas del script de Python.

export const TICKERS_DEFAULT = {
  SP500: 'SPY',
  GOLD: 'GLD',
  YPF: 'YPF',
  'COCA-COLA': 'KO',
  QQQ: 'QQQ',
  'MERCADO LIBRE': 'MELI',
  NVIDIA: 'NVDA',
};

// Descarga precios mensuales ajustados de un ticker desde Yahoo Finance.
// El navegador no puede llamar a Yahoo directamente (CORS), así que se usa
// el proxy "/yahoo" definido en vite.config.js.
async function descargarTicker(simbolo, period, interval) {
  const url = `/yahoo/v8/finance/chart/${encodeURIComponent(simbolo)}?range=${period}&interval=${interval}&events=div,split`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Yahoo respondió ${res.status} para ${simbolo}`);
  const json = await res.json();
  const r = json?.chart?.result?.[0];
  if (!r) throw new Error(json?.chart?.error?.description || `Sin datos para ${simbolo}`);

  const fechas = r.timestamp.map((t) => new Date(t * 1000).toISOString().slice(0, 7)); // AAAA-MM
  const adj = r.indicators?.adjclose?.[0]?.adjclose ?? r.indicators.quote[0].close;

  const serie = new Map();
  fechas.forEach((f, i) => {
    if (adj[i] != null) serie.set(f, adj[i]); // si hay dos registros del mismo mes, queda el último
  });
  return serie;
}

/**
 * Descarga precios de todos los activos y los alinea por fecha.
 * @returns {{ fechas: string[], activos: string[], precios: number[][] }}
 *          precios[t][i] = precio del activo i en la fecha t
 */
export async function obtenerPrecios(tickers, period = '5y', interval = '1mo') {
  const nombres = Object.keys(tickers);
  const series = await Promise.all(nombres.map((k) => descargarTicker(tickers[k], period, interval)));

  // Solo fechas presentes en todos los activos (equivale a dropna).
  const fechas = [...series[0].keys()].filter((f) => series.every((s) => s.has(f))).sort();
  const precios = fechas.map((f) => series.map((s) => s.get(f)));

  return { fechas, activos: nombres, precios };
}

// Rendimientos simples: precio_t / precio_{t-1} - 1 (pct_change().dropna()).
export function calcularRendimientos({ fechas, activos, precios }) {
  const retornos = [];
  for (let t = 1; t < precios.length; t++) {
    retornos.push(precios[t].map((p, i) => p / precios[t - 1][i] - 1));
  }
  return { fechas: fechas.slice(1), activos, retornos };
}

// Media por activo y matriz de covarianzas muestral (ddof = 1, como pandas).
export function estadisticas(retornos) {
  const T = retornos.length;
  const n = retornos[0].length;
  const mu = new Array(n).fill(0);
  for (const fila of retornos) fila.forEach((r, i) => (mu[i] += r / T));

  const cov = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const fila of retornos) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) cov[i][j] += ((fila[i] - mu[i]) * (fila[j] - mu[j])) / (T - 1);
    }
  }
  return { mu, cov };
}

// ------------------------------------------------------------
// CSV: primera columna fecha, el resto precios por activo.
// ------------------------------------------------------------

export function parsearCSV(texto) {
  const lineas = texto.trim().split(/\r?\n/).filter(Boolean);
  const sep = lineas[0].includes(';') ? ';' : ',';
  const cabecera = lineas[0].split(sep).map((s) => s.trim());
  const activos = cabecera.slice(1);
  const fechas = [];
  const precios = [];
  for (const linea of lineas.slice(1)) {
    const celdas = linea.split(sep).map((s) => s.trim());
    const valores = celdas.slice(1).map(Number);
    if (valores.some((v) => !Number.isFinite(v))) continue;
    fechas.push(celdas[0]);
    precios.push(valores);
  }
  if (precios.length < 3) throw new Error('El CSV necesita al menos 3 filas de precios válidas.');
  return { fechas, activos, precios };
}

// ------------------------------------------------------------
// Datos de demostración (sintéticos, NO son precios reales).
// Sirven para probar la interfaz sin conexión a Yahoo.
// ------------------------------------------------------------

export function datosDemo(meses = 61, seed = 42) {
  // Rendimiento y volatilidad mensual aproximados por activo.
  const perfil = {
    SP500: [0.011, 0.045],
    GOLD: [0.007, 0.04],
    YPF: [0.025, 0.14],
    'COCA-COLA': [0.006, 0.04],
    QQQ: [0.015, 0.06],
    'MERCADO LIBRE': [0.018, 0.12],
    NVIDIA: [0.045, 0.14],
  };
  const activos = Object.keys(perfil);
  // Exposición de cada activo a un factor de mercado común (genera correlación).
  const beta = [0.9, 0.1, 0.5, 0.5, 1.0, 0.7, 0.8];

  let a = seed >>> 0;
  const rng = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(rng() || 1e-12)) * Math.cos(2 * Math.PI * rng());

  const fechas = [];
  const precios = [];
  let actual = activos.map(() => 100);
  const inicio = new Date(Date.UTC(2021, 10, 1));
  for (let t = 0; t < meses; t++) {
    const d = new Date(inicio);
    d.setUTCMonth(inicio.getUTCMonth() + t);
    fechas.push(d.toISOString().slice(0, 7));
    if (t > 0) {
      const mercado = gauss();
      actual = actual.map((p, i) => {
        const [m, s] = perfil[activos[i]];
        const b = beta[i];
        const shock = b * mercado + Math.sqrt(1 - b * b) * gauss();
        return p * (1 + m + s * shock);
      });
    }
    precios.push(actual.slice());
  }
  return { fechas, activos, precios };
}
