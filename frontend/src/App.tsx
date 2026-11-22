import { lazy, Suspense, useState } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { API_BASE } from "./api";
import { Icon, IconName } from "./components/Icon";
import { Brand, Loading, PageHead, Seg, WakeGate } from "./components/ui";
import { getTheme, setTheme } from "./theme";

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

const NAV: { group: string; items: [string, string, IconName][] }[] = [
  { group: "Analyze", items: [["/", "Analyze transactions", "scan"]] },
  { group: "Demo dataset insights", items: [["/overview", "Executive overview", "overview"]] },
  { group: "Data (DMW)", items: [["/data-quality", "Data quality", "quality"], ["/transactions", "Transaction explorer", "list"], ["/patterns", "Pattern mining lab", "pattern"]] },
  { group: "Models (ML)", items: [["/ml", "ML experiments lab", "model"], ["/clusters", "Clustering & anomaly", "cluster"], ["/network", "Network explorer", "network"], ["/score", "Risk scorer", "gauge"]] },
  { group: "Cloud (CCT)", items: [["/platform", "Platform", "cloud"]] },
];

function ThemeToggle() {
  const [value, setValue] = useState(getTheme);
  return (
    <Seg label="Colour theme" value={value} onChange={(v) => { setTheme(v); setValue(v); }}
      options={[{ value: "light", label: "Light" }, { value: "dark", label: "Dark" }, { value: "system", label: "System" }]} />
  );
}

function Sidebar({ health }: { health: Record<string, unknown> }) {
  const [open, setOpen] = useState(false);          // only used on phones, where the menu collapses
  const close = () => setOpen(false);
  return (
    <aside className="side">
      <div className="side-top">
        <Brand onClick={close} />
        <button className="menu-btn" aria-expanded={open} aria-controls="side-body" onClick={() => setOpen(!open)}><Icon name="menu" />Menu</button>
      </div>
      <div id="side-body" className={"side-body" + (open ? " open" : "")}>
        <nav className="nav" aria-label="Main">
          {NAV.map((g) => (
            <div key={g.group}>
              <div className="group">{g.group}</div>
              {g.items.map(([to, label, icon]) => (
                <NavLink key={to} to={to} end={to === "/"} onClick={close}><Icon name={icon} />{label}</NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="foot">
          <div className="meta">API: {String(health.mode)} mode</div>
          <div className="meta" title={API_BASE}>{API_BASE.replace(/^https?:\/\//, "")}</div>
          <div className="meta">model {String(health.model_version ?? health.version)}</div>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}

function AppBar() {
  const { pathname } = useLocation();
  const group = NAV.find((g) => g.items.some(([to]) => to === pathname));
  const item = group?.items.find(([to]) => to === pathname);
  return (
    <div className="appbar">
      <div className="crumb">{group && item ? `${group.group} / ${item[1]}` : "Not found"}</div>
      <div className="pill"><span className="dot" />API connected</div>
    </div>
  );
}

export default function App() {
  return (
    <WakeGate>
      {(health) => (
        <div className="shell">
          <Sidebar health={health} />
          <main className="main">
            <div className="page">
            <AppBar />
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
                <Route path="*" element={<><PageHead title="Page not found" sub="That address is not part of the dashboard." /><Link className="btn primary" to="/">Go to Analyze transactions</Link></>} />
              </Routes>
            </Suspense>
            </div>
          </main>
        </div>
      )}
    </WakeGate>
  );
}
