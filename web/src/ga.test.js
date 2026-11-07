import { test } from 'node:test';
import assert from 'node:assert/strict';
import { crearAG } from './ga.js';
import { datosDemo, calcularRendimientos, estadisticas } from './data.js';

const datos = estadisticas(calcularRendimientos(datosDemo()).retornos);

const cumple = (w, min, max) => {
  assert.ok(Math.abs(w.reduce((a, b) => a + b, 0) - 1) < 1e-9, 'suma 1');
  for (const p of w) assert.ok(p >= min - 1e-9 && p <= max + 1e-9, `peso ${p} fuera de rango`);
};

test('las carteras generadas cumplen las restricciones', () => {
  const ag = crearAG(datos, { seed: 1 });
  for (const w of ag.generarPoblacion()) cumple(w, 0.05, 0.25);
});

test('ajustarRestricciones repara carteras inválidas', () => {
  const ag = crearAG(datos, { seed: 1 });
  cumple(ag.ajustarRestricciones([1, 0, 0, 0, 0, 0, 0]), 0.05, 0.25);
  cumple(ag.ajustarRestricciones([-1, 3, 0.2, 0.2, 0.1, 0, 0.9]), 0.05, 0.25);
});

test('el AG mejora el fitness y es reproducible con semilla', () => {
  const a = crearAG(datos, { seed: 7, generaciones: 50 }).ejecutar('sharpe');
  const b = crearAG(datos, { seed: 7, generaciones: 50 }).ejecutar('sharpe');
  assert.deepEqual(a.mejor, b.mejor);
  assert.ok(a.historialFitness.at(-1).mejor >= a.historialFitness[0].mejor);
  cumple(a.mejor, 0.05, 0.25);
});

test('rechaza restricciones imposibles', () => {
  assert.throws(() => crearAG(datos, { maxInversion: 0.1 }));
  assert.throws(() => crearAG(datos, { minInversion: 0.2 }));
});
