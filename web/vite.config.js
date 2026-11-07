import { defineConfig } from 'vite';

// El navegador no puede pedir datos a Yahoo Finance directamente (CORS).
// Este proxy redirige /yahoo/... -> https://query1.finance.yahoo.com/...
// Funciona con `npm run dev` y `npm run preview`.
const yahooProxy = {
  '/yahoo': {
    target: 'https://query1.finance.yahoo.com',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/yahoo/, ''),
    headers: { 'User-Agent': 'Mozilla/5.0' },
  },
};

export default defineConfig({
  server: { proxy: yahooProxy },
  preview: { proxy: yahooProxy },
});
