import { lazy, Suspense, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { API_BASE } from "./api";
import { Loading, WakeGate } from "./components/ui";

const Analyze = lazy(() => import("./pages/Analyze"));
const Overview = lazy(() => import("./pages/Overview"));
const DataQuality = lazy(() => import("./pages/DataQuality"));
const Transactions = lazy(() => import("./pages/Transactions"));
const Patterns = lazy(() => import("./pages/Patterns"));
const MLLab = lazy(() => import("./pages/MLLab"));
const Clusters = lazy(() => import("./pages/Clusters"));
const Network = lazy(() => import("./pages/Network"));
const RiskScorer = lazy(() => import("./pages/RiskScorer"));
const Platform = lazy(() => import("./pages/Platform"));

const NAV: { group: string; items: [string, string, string][] }[] = [
  { group: "Analyze", items: [["/", "Analyze transactions", "▶"]] },
  { group: "Demo dataset insights", items: [["/overview", "Executive overview", "◧"]] },
  { group: "Data (DMW)", items: [["/data-quality", "Data quality", "▤"], ["/transactions", "Transaction explorer", "☰"], ["/patterns", "Pattern mining lab", "⋈"]] },
  { group: "Models (ML)", items: [["/ml", "ML experiments lab", "◭"], ["/clusters", "Clustering & anomaly", "⁘"], ["/network", "Network explorer", "⟡"], ["/score", "Risk scorer", "◎"]] },
  { group: "Cloud (CCT)", items: [["/platform", "Platform", "▦"]] },
];

function ThemeToggle() {
  const [t, setT] = useState(document.documentElement.dataset.theme ?? "auto");
  const set = (v: string) => {
    if (v === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = v;
    try { if (v === "auto") localStorage.removeItem("finsight-theme"); else localStorage.setItem("finsight-theme", v); } catch { /* ignore */ }
    setT(v);
  };
  return (
    <label className="field" style={{ marginTop: 8 }}>Theme
      <select value={t} onChange={(e) => set(e.target.value)} aria-label="Colour theme">
        <option value="auto">System</option><option value="light">Light</option><option value="dark">Dark</option>
      </select>
    </label>
  );
}

export default function App() {
  return (
    <WakeGate>
      {(health) => (
        <div className="shell">
          <aside className="side">
            <div className="brand"><img src="/favicon.svg" alt="" />FinSight AI</div>
            <nav className="nav" aria-label="Main">
              {NAV.map((g) => (
                <div key={g.group}>
                  <div className="group">{g.group}</div>
                  {g.items.map(([to, label, icon]) => (
                    <NavLink key={to} to={to} end={to === "/"}><span aria-hidden>{icon}</span>{label}</NavLink>
                  ))}
                </div>
              ))}
            </nav>
            <div className="foot">
              <div>API: <span className="num">{String(health.mode)}</span> mode</div>
              <div title={API_BASE} style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{API_BASE.replace(/^https?:\/\//, "")}</div>
              <div>model {String(health.model_version ?? health.version)}</div>
              <ThemeToggle />
            </div>
          </aside>
          <main className="main">
            <Suspense fallback={<Loading what="Loading page" />}>
              <Routes>
                <Route path="/" element={<Analyze />} />
                <Route path="/overview" element={<Overview />} />
                <Route path="/data-quality" element={<DataQuality />} />
                <Route path="/transactions" element={<Transactions />} />
                <Route path="/patterns" element={<Patterns />} />
                <Route path="/ml" element={<MLLab />} />
                <Route path="/clusters" element={<Clusters />} />
                <Route path="/network" element={<Network />} />
                <Route path="/score" element={<RiskScorer />} />
                <Route path="/platform" element={<Platform wakeSeconds={Number(health.wake_seconds)} />} />
                <Route path="*" element={<p>Page not found. <a href="/">Back to the overview</a>.</p>} />
              </Routes>
            </Suspense>
          </main>
        </div>
      )}
    </WakeGate>
  );
}
