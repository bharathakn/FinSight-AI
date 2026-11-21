import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static SPA build for Vercel (D-022). Plotly is split into its own lazily loaded chunk so the
// first paint (and the "waking the backend" screen) doesn't wait for the 3.5 MB chart library.
export default defineConfig({
  plugins: [react()],
  build: {
    target: "es2020",
    chunkSizeWarningLimit: 5000,   // plotly (~1.4 MB gzipped) is lazy-loaded on first chart, never on first paint
    rollupOptions: { output: { manualChunks: { plotly: ["plotly.js-dist-min"], react: ["react", "react-dom", "react-router-dom"] } } },
  },
  server: { port: 5173 },
  preview: { port: 4173 },
});
