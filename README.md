# TPI_AlgGeneticos_Optimizacion-de-Carteras-Financieras
“Optimización de Carteras Financieras mediante Algoritmos Genéticos Canónicos: Un enfoque basado en el Modelo de Markowitz para la asignación óptima del capital. “

## Versión web (JavaScript)

La carpeta [`web/`](web/) tiene el algoritmo migrado a JavaScript con una interfaz interactiva.
El script original en Python (`TPI_EvaluarCartera.py`) queda como referencia.

### Cómo correrla

Requiere [Node.js](https://nodejs.org/) 18 o superior.

```bash
cd web
npm install
npm run dev
```

Abrir la URL que muestra la consola (normalmente http://localhost:5173).

### Qué hace

1. **Datos**: descarga precios mensuales de Yahoo Finance (a través de un proxy de Vite, porque el
   navegador no puede llamar a Yahoo directo), o usa datos demo sintéticos, o un CSV propio
   (primera columna fecha, el resto precios por activo).
2. **Parámetros**: todos los del script (población, generaciones, mutación, crossover, λ, tasa libre
   de riesgo, mínimo/máximo por activo) más una semilla opcional para repetir resultados.
3. **Ejecutar**: las mismas 4 opciones del menú de Python: Fitness Lambda, Índice de Sharpe,
   comparar ambos y comparar varios λ. Muestra cartera óptima, métricas mensuales y anualizadas,
   evolución animada de la población y convergencia del fitness.

### Estructura

| Archivo | Contenido |
| --- | --- |
| `web/src/ga.js` | Algoritmo genético (generación, restricciones, fitness, selección, cruce, mutación) |
| `web/src/data.js` | Descarga de precios, rendimientos, media y covarianza, CSV y datos demo |
| `web/src/charts.js` | Gráficos (Chart.js) |
| `web/src/main.js` | Interfaz |
| `web/src/ga.test.js` | Tests (`npm test`) |
