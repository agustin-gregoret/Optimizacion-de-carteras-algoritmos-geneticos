// Algoritmo genético para optimización de carteras (Markowitz).
// Traducción de TPI_EvaluarCartera.py. No depende del navegador,
// así que también se puede usar desde Node (por ejemplo, en tests).

export const DEFAULT_PARAMS = {
  tamPoblacion: 100,
  generaciones: 100,
  probMutacion: 0.05,
  desviacionMutacion: 0.05,
  probCrossover: 0.8,
  lambda: 2.0, // coeficiente de aversión al riesgo
  riskFreeRate: 0.005, // tasa libre de riesgo mensual
  minInversion: 0.05,
  maxInversion: 0.25,
  seed: null, // número para resultados reproducibles, o null para aleatorio
};

// ------------------------------------------------------------
// Números aleatorios
// ------------------------------------------------------------

// Generador con semilla (mulberry32) para poder repetir corridas.
export function crearRng(seed) {
  if (seed === null || seed === undefined || seed === '') return Math.random;
  let a = Number(seed) >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Normal(media, desvío) con Box-Muller (equivale a np.random.normal).
function normal(rng, media, desvio) {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return media + desvio * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const suma = (arr) => arr.reduce((s, x) => s + x, 0);
const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

// ------------------------------------------------------------
// Validación de restricciones
// ------------------------------------------------------------

export function validarParametros(n, params) {
  if (n * params.minInversion > 1) {
    throw new Error('La inversión mínima no permite distribuir el 100% del capital.');
  }
  if (n * params.maxInversion < 1) {
    throw new Error('La inversión máxima no permite distribuir el 100% del capital.');
  }
}

// ------------------------------------------------------------
// Fábrica: crea todas las funciones del AG para unos datos dados
// ------------------------------------------------------------

/**
 * @param {object} datos  { mu: number[], cov: number[][] }
 * @param {object} params parámetros (ver DEFAULT_PARAMS)
 */
export function crearAG(datos, params = {}) {
  const P = { ...DEFAULT_PARAMS, ...params };
  const { mu, cov } = datos;
  const n = mu.length;
  const rng = crearRng(P.seed);
  const MIN = P.minInversion;
  const MAX = P.maxInversion;

  validarParametros(n, P);

  // Ajusta una cartera para que MIN <= peso_i <= MAX y sum(peso_i) = 1,
  // redistribuyendo el capital de forma iterativa.
  function ajustarRestricciones(proporciones) {
    let pesos = proporciones.map((x) => Math.max(Number(x), 0));

    if (suma(pesos) === 0) return generarCartera();

    let total = suma(pesos);
    pesos = pesos.map((x) => clip(x / total, MIN, MAX));

    for (let iter = 0; iter < 100; iter++) {
      const diferencia = 1 - suma(pesos);
      if (Math.abs(diferencia) < 1e-12) break;

      // Si falta capital, se agrega donde hay lugar hasta MAX.
      // Si sobra, se quita donde hay margen sobre MIN.
      const capacidad = diferencia > 0 ? pesos.map((p) => MAX - p) : pesos.map((p) => p - MIN);
      const capacidadTotal = suma(capacidad);
      if (capacidadTotal <= 0) break;

      pesos = pesos.map((p, i) => clip(p + diferencia * (capacidad[i] / capacidadTotal), MIN, MAX));
    }

    total = suma(pesos);
    pesos = pesos.map((p) => p / total);

    // Corrección final por redondeo.
    const diferencia = 1 - suma(pesos);
    if (Math.abs(diferencia) > 1e-12) {
      const idx =
        diferencia > 0
          ? pesos.findIndex((p) => p < MAX - 1e-12)
          : pesos.findIndex((p) => p > MIN + 1e-12);
      if (idx >= 0) pesos[idx] += diferencia;
    }

    return pesos;
  }

  // Genera un individuo: cada gen es el porcentaje invertido en un activo.
  function generarCartera() {
    const pesos = new Array(n).fill(MIN);
    const capacidades = new Array(n).fill(MAX - MIN);
    let restante = 1 - suma(pesos);

    while (restante > 1e-12) {
      const indices = [];
      for (let i = 0; i < n; i++) if (capacidades[i] > 1e-12) indices.push(i);
      if (indices.length === 0) break;

      const valores = indices.map(() => rng());
      const totalValores = suma(valores);

      indices.forEach((i, k) => {
        const asignacion = Math.min((valores[k] / totalValores) * restante, capacidades[i]);
        pesos[i] += asignacion;
        capacidades[i] -= asignacion;
      });

      restante = 1 - suma(pesos);
    }

    return ajustarRestricciones(pesos);
  }

  function generarPoblacion() {
    return Array.from({ length: P.tamPoblacion }, generarCartera);
  }

  // --------------------------------------------------------
  // Métricas y funciones de fitness
  // --------------------------------------------------------

  function retornoCartera(w) {
    let r = 0;
    for (let i = 0; i < n; i++) r += w[i] * mu[i];
    return r;
  }

  function riesgoCartera(w) {
    let varianza = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) varianza += w[i] * cov[i][j] * w[j];
    }
    return Math.sqrt(Math.max(varianza, 0));
  }

  function fitnessLambda(w) {
    return retornoCartera(w) - P.lambda * riesgoCartera(w);
  }

  function fitnessSharpe(w) {
    const riesgo = riesgoCartera(w);
    if (riesgo === 0) return 0;
    return (retornoCartera(w) - P.riskFreeRate) / riesgo;
  }

  const FITNESS = { lambda: fitnessLambda, sharpe: fitnessSharpe };

  // --------------------------------------------------------
  // Operadores genéticos
  // --------------------------------------------------------

  // Selección por truncamiento: se queda con la mejor mitad.
  function seleccionTruncamiento(poblacion, fitnessValues) {
    const indices = poblacion.map((_, i) => i).sort((a, b) => fitnessValues[b] - fitnessValues[a]);
    return indices.slice(0, Math.floor(P.tamPoblacion / 2)).map((i) => poblacion[i]);
  }

  // Cruce aritmético: hijo = alpha * p1 + (1 - alpha) * p2.
  function crossover(p1, p2) {
    if (rng() > P.probCrossover) return p1.slice();
    const alpha = rng();
    const hijo = p1.map((x, i) => alpha * x + (1 - alpha) * p2[i]);
    return ajustarRestricciones(hijo);
  }

  // Mutación gaussiana gen a gen.
  function mutacion(individuo) {
    const nuevo = individuo.slice();
    for (let i = 0; i < n; i++) {
      if (rng() < P.probMutacion) nuevo[i] += normal(rng, 0, P.desviacionMutacion);
    }
    return ajustarRestricciones(nuevo);
  }

  function crearNuevaPoblacion(seleccionados) {
    const nueva = [];
    while (nueva.length < P.tamPoblacion) {
      const p1 = seleccionados[Math.floor(rng() * seleccionados.length)];
      const p2 = seleccionados[Math.floor(rng() * seleccionados.length)];
      nueva.push(mutacion(crossover(p1, p2)));
    }
    return nueva;
  }

  // --------------------------------------------------------
  // Algoritmo principal
  // --------------------------------------------------------

  /**
   * Ejecuta el AG.
   * @param {'lambda'|'sharpe'|Function} fitness
   * @param {number[][]} [poblacionInicial]
   * @returns {{mejor, mejorFitness, historialPoblaciones, historialMejores, historialFitness}}
   */
  function ejecutar(fitness = 'lambda', poblacionInicial = null) {
    const fn = typeof fitness === 'function' ? fitness : FITNESS[fitness];
    let poblacion = poblacionInicial ? poblacionInicial.map((w) => w.slice()) : generarPoblacion();

    let mejor = null;
    let mejorFitness = -Infinity;
    const historialPoblaciones = [];
    const historialMejores = [];
    const historialFitness = []; // { mejor, promedio } por generación

    for (let g = 0; g < P.generaciones; g++) {
      historialPoblaciones.push(poblacion.map((w) => w.slice()));

      const fitnessValues = poblacion.map(fn);

      let bestIdx = 0;
      for (let i = 1; i < fitnessValues.length; i++) {
        if (fitnessValues[i] > fitnessValues[bestIdx]) bestIdx = i;
      }

      const actual = poblacion[bestIdx].slice();
      const fitnessActual = fitnessValues[bestIdx];
      historialMejores.push(actual);
      historialFitness.push({ mejor: fitnessActual, promedio: suma(fitnessValues) / fitnessValues.length });

      if (fitnessActual > mejorFitness) {
        mejorFitness = fitnessActual;
        mejor = actual.slice();
      }

      poblacion = crearNuevaPoblacion(seleccionTruncamiento(poblacion, fitnessValues));
    }

    return { mejor, mejorFitness, historialPoblaciones, historialMejores, historialFitness };
  }

  return {
    params: P,
    n,
    ajustarRestricciones,
    generarCartera,
    generarPoblacion,
    retornoCartera,
    riesgoCartera,
    fitnessLambda,
    fitnessSharpe,
    ejecutar,
  };
}

// ------------------------------------------------------------
// Utilidades de métricas
// ------------------------------------------------------------

// Rendimiento: Tasa Efectiva Anual. Riesgo: regla de la raíz del tiempo.
export function anualizarMetricas(rendimientoMensual, riesgoMensual) {
  return {
    rendimiento: (1 + rendimientoMensual) ** 12 - 1,
    riesgo: riesgoMensual * Math.sqrt(12),
  };
}

// Resumen de una cartera (mensual y anualizado).
export function resumenCartera(ag, w) {
  const rendimiento = ag.retornoCartera(w);
  const riesgo = ag.riesgoCartera(w);
  const sharpe = ag.fitnessSharpe(w);
  const anual = anualizarMetricas(rendimiento, riesgo);
  const rf = ag.params.riskFreeRate;
  return {
    pesos: w,
    sumaPesos: suma(w),
    rendimiento,
    riesgo,
    sharpe,
    rendimientoAnualTDA: rendimiento * 12,
    rendimientoAnual: anual.rendimiento,
    riesgoAnual: anual.riesgo,
    sharpeAnual: anual.riesgo !== 0 ? (anual.rendimiento - rf) / anual.riesgo : 0,
  };
}
