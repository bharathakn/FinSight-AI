import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./theme.css";

// restore the viewer's theme choice (a per-viewer convenience; storage may be unavailable)
try {
  const t = localStorage.getItem("finsight-theme");
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
} catch { /* private mode etc. */ }

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
