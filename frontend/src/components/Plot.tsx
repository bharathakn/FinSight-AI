// Plotly wrapper: lazy-loads plotly (its own chunk), applies the FinSight theme from CSS tokens, re-renders on
// light/dark changes, resizes with its container, and keeps the hover layer (tooltips) on by default.
import { useEffect, useRef, useState } from "react";

type PlotlyLib = (typeof import("plotly.js-dist-min"))["default"];
let plotlyPromise: Promise<PlotlyLib> | null = null;
const loadPlotly = () => (plotlyPromise ??= import("plotly.js-dist-min").then((m) => (m.default ?? (m as unknown as PlotlyLib))));

export function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Re-render key that changes when the theme changes (OS setting or the toggle). */
export function useThemeKey(): string {
  const [key, setKey] = useState(() => document.documentElement.dataset.theme ?? "auto");
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setKey((document.documentElement.dataset.theme ?? "auto") + (mq.matches ? "-d" : "-l"));
    update();
    mq.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { mq.removeEventListener("change", update); obs.disconnect(); };
  }, []);
  return key;
}

export const series = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css(`--s${i}`));
export const labelColor = (l: string) => ({ LEGITIMATE: css("--s1"), FRAUDULENT: css("--s2"), BORDERLINE: css("--s3") } as Record<string, string>)[l] ?? css("--muted");

export function baseLayout(extra: Record<string, unknown> = {}): Record<string, unknown> {
  const ink2 = css("--ink-2"), grid = css("--grid"), base = css("--baseline"), surface = css("--surface");
  const axis = { gridcolor: grid, linecolor: base, zerolinecolor: base, tickfont: { color: ink2, size: 11 }, title: { font: { color: ink2, size: 12 } }, automargin: true };
  const e = extra as Record<string, Record<string, unknown> | undefined>;
  return {
    paper_bgcolor: surface, plot_bgcolor: surface, font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", color: css("--ink"), size: 12 },
    margin: { l: 10, r: 14, t: 10, b: 10 }, showlegend: true,
    legend: { orientation: "h", y: -0.2, x: 0, font: { color: ink2, size: 11 } },
    hoverlabel: { bgcolor: surface, bordercolor: base, font: { color: css("--ink"), size: 12 } },
    colorway: series(), bargap: 0.25,
    ...extra,
    xaxis: { ...axis, ...(e.xaxis ?? {}) },
    yaxis: { ...axis, ...(e.yaxis ?? {}) },
  };
}

export function Plot({ data, layout, height = 300, onClick }: {
  data: () => unknown[]; layout?: () => Record<string, unknown>; height?: number; deps?: unknown[];
  onClick?: (pt: { customdata?: unknown; x?: unknown; y?: unknown }) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const theme = useThemeKey();
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    loadPlotly().then((Plotly) => {
      if (!alive || !ref.current) return;
      const el = ref.current as HTMLDivElement & { on?: (ev: string, cb: (e: { points: { customdata?: unknown; x?: unknown; y?: unknown }[] }) => void) => void };
      Plotly.react(el, data() as never, { ...baseLayout(layout ? layout() : {}), height } as never,
        { displaylogo: false, responsive: true, displayModeBar: false } as never);
      if (onClick && el.on) el.on("plotly_click", (e) => e.points?.[0] && onClick(e.points[0]));
    }).catch(() => setFailed(true));
    return () => { alive = false; };
  });
  useEffect(() => () => { if (ref.current) loadPlotly().then((P) => ref.current && P.purge(ref.current)); }, []);
  if (failed) return <div className="error">The chart library could not be loaded.</div>;
  return <div ref={ref} className="plot" style={{ minHeight: height }} data-theme-key={theme} />;
}
