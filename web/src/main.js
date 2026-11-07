// Controlador de la interfaz: une datos, algoritmo genético y gráficos.
import { crearAG, resumenCartera, DEFAULT_PARAMS } from './ga.js';
import { TICKERS_DEFAULT, obtenerPrecios, calcularRendimientos, estadisticas, parsearCSV, datosDemo } from './data.js';
import { graficoCartera, graficoConvergencia, graficoEvolucion, graficoLambdas, destruirTodos, pct } from './charts.js';

const $ = (sel) => document.querySelector(sel);

// Estado global de la app.
const estado = {
  datos: null, // { activos, fechas, retornos, mu, cov, fuente }
  animacion: null,
};

// ------------------------------------------------------------
// Panel de tickers
// ------------------------------------------------------------

function filaTicker(nombre = '', simbolo = '') {
  const row = document.createElement('div');
  row.className = 'ticker-row';
  row.innerHTML = `
    <input class="t-nombre" placeholder="Nombre" value="${nombre}" />
    <input class="t-simbolo" placeholder="Ticker" value="${simbolo}" />
    <button class="btn-icon" type="button" title="Quitar">×</button>`;
  row.querySelector('button').onclick = () => row.remove();
  $('#tickers').appendChild(row);
}

function leerTickers() {
  const t = {};
  document.querySelectorAll('.ticker-row').forEach((row) => {
    const nombre = row.querySelector('.t-nombre').value.trim();
    const simbolo = row.querySelector('.t-simbolo').value.trim().toUpperCase();
    if (nombre && simbolo) t[nombre] = simbolo;
  });
  return t;
}

Object.entries(TICKERS_DEFAULT).forEach(([n, s]) => filaTicker(n, s));
$('#btn-add-ticker').onclick = () => filaTicker();

// ------------------------------------------------------------
// Panel de parámetros
// ------------------------------------------------------------

const CAMPOS = [
  ['tamPoblacion', 'Tamaño población', 1],
  ['generaciones', 'Generaciones', 1],
  ['probCrossover', 'Prob. crossover', 0.01],
  ['probMutacion', 'Prob. mutación', 0.01],
  ['desviacionMutacion', 'Desvío mutación', 0.01],
  ['lambda', 'λ (aversión riesgo)', 0.05],
  ['riskFreeRate', 'Tasa libre riesgo (mensual)', 0.001],
  ['minInversion', 'Mín. por activo', 0.01],
  ['maxInversion', 'Máx. por activo', 0.01],
  ['seed', 'Semilla (vacío = aleatoria)', 1],
];

$('#params').innerHTML = CAMPOS.map(
  ([k, label, step]) =>
    `<label>${label}<input id="p-${k}" type="number" step="${step}" value="${DEFAULT_PARAMS[k] ?? ''}" /></label>`
).join('');

function leerParametros() {
  const p = {};
  for (const [k] of CAMPOS) {
    const v = $(`#p-${k}`).value;
    p[k] = k === 'seed' ? (v === '' ? null : Number(v)) : Number(v);
  }
  return p;
}

// ------------------------------------------------------------
// Carga de datos
// ------------------------------------------------------------

function setStatus(msg, tipo = '') {
  const el = $('#data-status');
  el.textContent = msg;
  el.className = `status ${tipo}`;
}

function cargarDatos(precios, fuente) {
  const r = calcularRendimientos(precios);
  const { mu, cov } = estadisticas(r.retornos);
  estado.datos = { ...r, mu, cov, fuente };
  setStatus(
    `${fuente}: ${r.activos.length} activos, ${r.retornos.length} rendimientos mensuales (${r.fechas[0]} → ${r.fechas.at(-1)}).`,
    'ok'
  );
  renderDatos();
}

$('#btn-yahoo').onclick = async (e) => {
  const tickers = leerTickers();
  if (Object.keys(tickers).length < 2) return setStatus('Necesitás al menos 2 activos.', 'err');
  e.target.disabled = true;
  setStatus('Descargando de Yahoo Finance…');
  try {
    cargarDatos(await obtenerPrecios(tickers, $('#period').value), 'Yahoo Finance');
  } catch (err) {
    setStatus(`No se pudo descargar: ${err.message}. ¿Corriste "npm run dev"? Probá con datos demo o un CSV.`, 'err');
  } finally {
    e.target.disabled = false;
  }
};

$('#btn-demo').onclick = () => cargarDatos(datosDemo(), 'Datos demo (sintéticos)');

$('#input-csv').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    cargarDatos(parsearCSV(await file.text()), `CSV ${file.name}`);
  } catch (err) {
    setStatus(err.message, 'err');
  }
  e.target.value = '';
};

// Vista previa de los datos cargados.
function renderDatos() {
  pararAnimacion();
  destruirTodos();
  const d = estado.datos;
  const filas = d.activos
    .map((a, i) => `<tr><td>${a}</td><td>${pct(d.mu[i])}</td><td>${pct(Math.sqrt(d.cov[i][i]))}</td></tr>`)
    .join('');
  $('#main').innerHTML = `
    <div class="card">
      <h2>Datos cargados · ${d.fuente}</h2>
      <div class="table-scroll"><table>
        <thead><tr><th>Activo</th><th>Rend. medio mensual</th><th>Volatilidad mensual</th></tr></thead>
        <tbody>${filas}</tbody>
      </table></div>
    </div>`;
}

// ------------------------------------------------------------
// Ejecución
// ------------------------------------------------------------

const NOMBRES = { lambda: 'Fitness Lambda', sharpe: 'Índice de Sharpe' };

function pararAnimacion() {
  if (estado.animacion) clearInterval(estado.animacion);
  estado.animacion = null;
}

document.querySelectorAll('[data-run]').forEach((btn) => {
  btn.onclick = () => {
    if (!estado.datos) return setStatus('Primero cargá datos.', 'err');
    try {
      pararAnimacion();
      destruirTodos();
      const modo = btn.dataset.run;
      if (modo === 'lambda' || modo === 'sharpe') ejecutarSimple(modo);
      else if (modo === 'comparar') compararMetodos();
      else compararLambdas();
    } catch (err) {
      $('#main').innerHTML = `<div class="card"><p class="status err">${err.message}</p></div>`;
    }
  };
});

function nuevoAG(extra = {}) {
  return crearAG(estado.datos, { ...leerParametros(), ...extra });
}

function htmlKpis(r) {
  const kpi = (label, v) => `<div class="kpi"><div class="label">${label}</div><div class="value">${v}</div></div>`;
  return `<div class="kpis">
    ${kpi('Rend. mensual', pct(r.rendimiento))}
    ${kpi('Riesgo mensual', pct(r.riesgo))}
    ${kpi('Rend. anual (TEA)', pct(r.rendimientoAnual))}
    ${kpi('Rend. anual (TDA)', pct(r.rendimientoAnualTDA))}
    ${kpi('Riesgo anual', pct(r.riesgoAnual))}
    ${kpi('Sharpe anual', r.sharpeAnual.toFixed(3))}
  </div>`;
}

function htmlEvolucion(id) {
  return `
    <div class="card">
      <h3>Evolución de la población (riesgo vs. rendimiento)</h3>
      <div class="chart-box tall"><canvas id="ev-${id}"></canvas></div>
      <div class="player">
        <button id="play-${id}" type="button">▶</button>
        <input id="slider-${id}" type="range" min="0" value="0" />
        <span id="gen-${id}" class="gen"></span>
      </div>
    </div>`;
}

// Conecta el slider y el botón play con el gráfico de evolución.
// Si autoplay es false, se muestra directamente la última generación.
function activarEvolucion(id, ag, ejecucion, fitnessFn, autoplay = true) {
  const total = ejecucion.historialPoblaciones.length;
  const mostrar = graficoEvolucion($(`#ev-${id}`), ag, ejecucion, fitnessFn);
  const slider = $(`#slider-${id}`);
  const label = $(`#gen-${id}`);
  const play = $(`#play-${id}`);
  slider.max = total - 1;

  const ir = (g) => {
    slider.value = g;
    label.textContent = `Gen. ${g + 1} / ${total}`;
    mostrar(g);
  };
  slider.oninput = () => {
    pararAnimacion();
    ir(Number(slider.value));
  };
  play.onclick = () => {
    pararAnimacion();
    let g = Number(slider.value) >= total - 1 ? 0 : Number(slider.value);
    estado.animacion = setInterval(() => {
      ir(g);
      if (++g >= total) pararAnimacion();
    }, 120);
  };
  if (autoplay) {
    ir(0);
    play.click(); // arranca animado, como FuncAnimation
  } else {
    ir(total - 1);
  }
}

// Opciones 1 y 2 del menú original.
function ejecutarSimple(modo) {
  const ag = nuevoAG();
  const ej = ag.ejecutar(modo);
  const r = resumenCartera(ag, ej.mejor);
  const titulo = modo === 'lambda' ? `Fitness Lambda (λ = ${ag.params.lambda})` : 'Índice de Sharpe';
  const activos = estado.datos.activos;

  $('#main').innerHTML = `
    <div class="card">
      <h2>Cartera óptima · ${titulo}</h2>
      ${htmlKpis(r)}
      <div class="grid-2">
        <div class="chart-box"><canvas id="pie"></canvas></div>
        <div class="table-scroll"><table>
          <thead><tr><th>Activo</th><th>Peso</th></tr></thead>
          <tbody>${activos.map((a, i) => `<tr><td>${a}</td><td>${pct(r.pesos[i])}</td></tr>`).join('')}</tbody>
          <tfoot><tr><td>Fitness</td><td>${ej.mejorFitness.toFixed(6)}</td></tr></tfoot>
        </table></div>
      </div>
    </div>
    <div class="grid-2">
      ${htmlEvolucion('a')}
      <div class="card"><h3>Convergencia del fitness</h3><div class="chart-box tall"><canvas id="conv"></canvas></div></div>
    </div>`;

  graficoCartera($('#pie'), activos, r.pesos);
  graficoConvergencia($('#conv'), [{ nombre: NOMBRES[modo], historial: ej.historialFitness }]);
  activarEvolucion('a', ag, ej, modo === 'lambda' ? ag.fitnessLambda : ag.fitnessSharpe);
}

// Opción 3: misma población inicial para ambos métodos.
function compararMetodos() {
  const ag = nuevoAG();
  const inicial = ag.generarPoblacion();
  const ejL = ag.ejecutar('lambda', inicial);
  const ejS = ag.ejecutar('sharpe', inicial);
  const rL = resumenCartera(ag, ejL.mejor);
  const rS = resumenCartera(ag, ejS.mejor);
  const activos = estado.datos.activos;

  const fila = (label, a, b) => `<tr><td>${label}</td><td>${a}</td><td>${b}</td></tr>`;

  $('#main').innerHTML = `
    <div class="card">
      <h2>Comparación de métodos (anualizado)</h2>
      <div class="table-scroll"><table>
        <thead><tr><th>Métrica</th><th>Lambda (λ = ${ag.params.lambda})</th><th>Sharpe</th></tr></thead>
        <tbody>
          ${fila('Rendimiento anual (TEA)', pct(rL.rendimientoAnual), pct(rS.rendimientoAnual))}
          ${fila('Riesgo anual', pct(rL.riesgoAnual), pct(rS.riesgoAnual))}
          ${fila('Índice de Sharpe', rL.sharpeAnual.toFixed(4), rS.sharpeAnual.toFixed(4))}
          ${fila('Fitness', ejL.mejorFitness.toFixed(6), ejS.mejorFitness.toFixed(6))}
          ${activos.map((a, i) => fila(a, pct(rL.pesos[i]), pct(rS.pesos[i]))).join('')}
        </tbody>
      </table></div>
    </div>
    <div class="grid-2">
      <div class="card"><h3>Cartera · Lambda</h3><div class="chart-box"><canvas id="pieL"></canvas></div></div>
      <div class="card"><h3>Cartera · Sharpe</h3><div class="chart-box"><canvas id="pieS"></canvas></div></div>
    </div>
    <div class="grid-2">${htmlEvolucion('L')}${htmlEvolucion('S')}</div>
    <div class="card"><h3>Convergencia del fitness</h3><div class="chart-box"><canvas id="conv"></canvas></div></div>`;

  graficoCartera($('#pieL'), activos, rL.pesos);
  graficoCartera($('#pieS'), activos, rS.pesos);
  graficoConvergencia($('#conv'), [
    { nombre: 'Lambda', historial: ejL.historialFitness },
    { nombre: 'Sharpe', historial: ejS.historialFitness },
  ]);
  // Solo una animación corre a la vez: la de Sharpe arranca en la generación final.
  activarEvolucion('S', ag, ejS, ag.fitnessSharpe, false);
  activarEvolucion('L', ag, ejL, ag.fitnessLambda);
}

// Opción 4: varios λ con la misma población inicial.
function compararLambdas() {
  const valores = $('#lambdas-list')
    .value.split(',')
    .map((s) => Number(s.trim()))
    .filter((x) => Number.isFinite(x));
  if (!valores.length) throw new Error('Ingresá al menos un valor de λ.');

  const base = nuevoAG();
  const inicial = base.generarPoblacion();
  const resultados = valores.map((lambda) => {
    const ag = nuevoAG({ lambda });
    const ej = ag.ejecutar('lambda', inicial);
    return { lambda, cartera: ej.mejor, fitness: ej.mejorFitness, historialMejores: ej.historialMejores, ...resumenCartera(ag, ej.mejor) };
  });

  const activos = estado.datos.activos;
  $('#main').innerHTML = `
    <div class="card">
      <h2>Comparación de valores de λ</h2>
      <div class="chart-box tall"><canvas id="lam"></canvas></div>
    </div>
    <div class="card">
      <div class="table-scroll"><table>
        <thead><tr><th>λ</th><th>Rend. mensual</th><th>Riesgo mensual</th><th>Sharpe</th><th>Fitness</th>${activos.map((a) => `<th>${a}</th>`).join('')}</tr></thead>
        <tbody>${resultados
          .map(
            (r) => `<tr><td>${r.lambda}</td><td>${pct(r.rendimiento)}</td><td>${pct(r.riesgo)}</td><td>${r.sharpe.toFixed(4)}</td><td>${r.fitness.toFixed(6)}</td>${r.pesos.map((p) => `<td>${pct(p, 1)}</td>`).join('')}</tr>`
          )
          .join('')}</tbody>
      </table></div>
    </div>`;

  graficoLambdas($('#lam'), base, resultados);
}
