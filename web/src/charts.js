// Gráficos con Chart.js. Reemplazan a matplotlib del script original.
import Chart from 'chart.js/auto';

export const PALETA = ['#2f6fdb', '#e8833a', '#1f9e74', '#d14b6a', '#8a63d2', '#c9a227', '#3fa7c4', '#7a7f87', '#b8541f', '#5b8c2a'];

const graficos = new Map(); // canvas -> instancia, para poder destruirlos

function colorTexto() {
  return getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#666';
}

function crear(canvas, config) {
  graficos.get(canvas)?.destroy();
  Chart.defaults.color = colorTexto();
  const chart = new Chart(canvas, config);
  graficos.set(canvas, chart);
  return chart;
}

export function destruirTodos() {
  graficos.forEach((c) => c.destroy());
  graficos.clear();
}

const pct = (v, d = 2) => `${(v * 100).toFixed(d)}%`;

// Torta de distribución de la cartera (graficar_cartera).
export function graficoCartera(canvas, activos, pesos) {
  return crear(canvas, {
    type: 'doughnut',
    data: {
      labels: activos,
      datasets: [{ data: pesos.map((p) => +(p * 100).toFixed(2)), backgroundColor: PALETA, borderWidth: 1 }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right' },
        tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.parsed.toFixed(2)}%` } },
      },
    },
  });
}

// Fitness del mejor individuo y promedio por generación.
export function graficoConvergencia(canvas, series) {
  return crear(canvas, {
    type: 'line',
    data: {
      labels: series[0].historial.map((_, i) => i + 1),
      datasets: series.flatMap((s, k) => [
        { label: `${s.nombre} · mejor`, data: s.historial.map((h) => h.mejor), borderColor: PALETA[k], pointRadius: 0, borderWidth: 2 },
        { label: `${s.nombre} · promedio`, data: s.historial.map((h) => h.promedio), borderColor: PALETA[k], borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
      ]),
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { title: { display: true, text: 'Generación' } },
        y: { title: { display: true, text: 'Fitness' } },
      },
    },
  });
}

/**
 * Nube riesgo-rendimiento de la población en una generación (graficar_evolucion).
 * Devuelve una función mostrar(generacion) para animar.
 */
export function graficoEvolucion(canvas, ag, ejecucion, fitnessFn) {
  const { historialPoblaciones, historialMejores } = ejecucion;
  const aPunto = (w) => ({ x: ag.riesgoCartera(w) * 100, y: ag.retornoCartera(w) * 100 });
  const puntos = historialPoblaciones.map((pob) => pob.map(aPunto));

  // Ejes fijos para toda la animación.
  const todos = puntos.flat();
  const xs = todos.map((p) => p.x);
  const ys = todos.map((p) => p.y);
  const rango = (arr) => {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const m = (max - min) * 0.1 || 1;
    return { min: min - m, max: max + m };
  };

  const chart = crear(canvas, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'Población', data: puntos[0], backgroundColor: 'rgba(47,111,219,0.45)', pointRadius: 3 },
        {
          label: 'Mejor individuo',
          data: [],
          backgroundColor: '#e8833a',
          borderColor: '#e8833a',
          pointStyle: 'star',
          pointRadius: 12,
          borderWidth: 3,
        },
        { label: 'Trayectoria del mejor', data: [], showLine: true, borderColor: 'rgba(232,131,58,0.5)', pointRadius: 0, borderWidth: 1 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: { ...rango(xs), title: { display: true, text: 'Volatilidad mensual (%)' } },
        y: { ...rango(ys), title: { display: true, text: 'Rendimiento esperado mensual (%)' } },
      },
      plugins: {
        tooltip: {
          callbacks: { label: (ctx) => `riesgo ${ctx.parsed.x.toFixed(2)}% · rend. ${ctx.parsed.y.toFixed(2)}%` },
        },
      },
    },
  });

  return function mostrar(g) {
    chart.data.datasets[0].data = puntos[g];
    const mejor = historialPoblaciones[g].reduce((a, b) => (fitnessFn(b) > fitnessFn(a) ? b : a));
    chart.data.datasets[1].data = [aPunto(mejor)];
    chart.data.datasets[2].data = historialMejores.slice(0, g + 1).map(aPunto);
    chart.update('none');
  };
}

// Comparación de λ: mejores por generación + mejor global (comparar_lambdas).
export function graficoLambdas(canvas, ag, resultados) {
  const aPunto = (w) => ({ x: ag.riesgoCartera(w) * 100, y: ag.retornoCartera(w) * 100 });
  const datasets = resultados.flatMap((r, k) => [
    {
      label: `λ = ${r.lambda} (evolución)`,
      data: r.historialMejores.map(aPunto),
      backgroundColor: PALETA[k % PALETA.length] + '66',
      pointRadius: 2.5,
    },
    {
      label: `λ = ${r.lambda}`,
      data: [aPunto(r.cartera)],
      backgroundColor: PALETA[k % PALETA.length],
      borderColor: PALETA[k % PALETA.length],
      pointStyle: 'star',
      pointRadius: 13,
      borderWidth: 3,
    },
  ]);
  return crear(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Volatilidad mensual (%)' } },
        y: { title: { display: true, text: 'Rendimiento esperado mensual (%)' } },
      },
      plugins: {
        legend: { labels: { filter: (item) => !item.text.includes('evolución') } },
      },
    },
  });
}

export { pct };
