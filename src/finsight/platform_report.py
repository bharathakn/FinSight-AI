"""Phase 8/10 platform measurements → reports/platform_summary.json + reports/figures/platform_01_*.png.

    python -m finsight.platform_report            # after `make up`, `make measure`, `make loadtest`

Collects: warm latency per deployment (reports/platform_<label>.json from deploy/measure.py), locust results
(reports/load_<label>_stats.csv), image sizes (docker image inspect), and the bundle manifest.
"""
from __future__ import annotations

import csv
import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from finsight import config, viz

REP = config.REPORTS_DIR
IMAGES = ["finsight/gateway:local", "finsight/ml:local", "finsight/mining:local", "finsight/graph:local",
          "finsight/sanctions:local", "finsight/monolith:local"]
LOADS = {"remote_ml3_u10": "10 users · 3 ml replicas", "remote_ml1": "30 users · 1 ml replica",
         "remote_ml3": "30 users · 3 ml replicas"}


def image_sizes() -> dict:
    out = {}
    for img in IMAGES:
        try:
            r = subprocess.run(["docker", "image", "inspect", img, "--format", "{{.Size}}"], capture_output=True, text=True, check=True)
            out[img] = round(int(r.stdout.strip()) / 1e6, 1)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            out[img] = None
    return out


def load_results() -> dict:
    res = {}
    for label, name in LOADS.items():
        f = REP / f"load_{label}_stats.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if r["Name"] == "/score-transaction":
                res[label] = {"scenario": name, "requests": int(r["Request Count"]), "failures": int(r["Failure Count"]),
                              "rps_score": round(float(r["Requests/s"]), 1), "p50_ms": float(r["50%"]),
                              "p95_ms": float(r["95%"]), "p99_ms": float(r["99%"])}
            if r["Name"] == "Aggregated" and label in res:
                res[label]["rps_total"] = round(float(r["Requests/s"]), 1)
    return res


def build(startup: dict | None = None) -> dict:
    warm = {lab: json.loads((REP / f"platform_{lab}.json").read_text()) for lab in ("compose-remote", "compose-inprocess")
            if (REP / f"platform_{lab}.json").exists()}
    s = {"warm": warm, "load": load_results(), "images_mb": image_sizes(),
         "bundle": {k: v for k, v in json.loads((config.ARTIFACTS_DIR / "bundle/current/manifest.json").read_text()).items()
                    if k in ("version", "total_mb", "libraries")},
         "startup_s": startup or {}}
    (REP / "platform_summary.json").write_text(json.dumps(s, indent=2))
    return s


def fig_platform(s: dict):
    viz.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.5, 1.3]})
    # (a) warm single-client latency
    ax = axes[0]
    labs = [("compose-inprocess", "inprocess\n(1 container)"), ("compose-remote", "remote\n(5 containers)")]
    x = np.arange(len(labs))
    for k, (q, col) in enumerate((("client_p50_ms", viz.SERIES[0]), ("client_p95_ms", viz.SERIES[1]))):
        v = [s["warm"][lab]["warm"][q] for lab, _ in labs]
        ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=col, label=q.split("_")[1])
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.36, vi + 0.5, f"{vi:.0f}", ha="center", fontsize=8)
    ax.axhline(200, color=viz.MUTED, linewidth=1, linestyle="--")
    ax.text(-0.45, 196, "target p95 200 ms", fontsize=8, color=viz.INK_2, va="top")
    ax.set_xticks(x, [l for _, l in labs]); ax.set_ylabel("ms (client-measured)"); ax.set_ylim(0, 210)
    ax.set_title("Warm /score-transaction, one client", fontsize=10)
    ax.legend(fontsize=8, loc="center right"); ax.grid(axis="x", visible=False)
    # (b) load tests
    ax = axes[1]
    L = [s["load"][k] for k in LOADS if k in s["load"]]
    y = np.arange(len(L))
    for k, (q, col) in enumerate((("p50_ms", viz.SERIES[0]), ("p95_ms", viz.SERIES[1]))):
        ax.barh(y + (k - 0.5) * 0.36, [r[q] for r in L], height=0.34, color=col, label=q.split("_")[0])
    for i, r in enumerate(L):
        ax.text(max(r["p95_ms"], r["p50_ms"]) + 8, i, f"{r['rps_total']:.0f} req/s · {r['failures']} errors", va="center", fontsize=8)
    ax.set_yticks(y, [r["scenario"] for r in L]); ax.invert_yaxis()
    ax.set_xlabel("latency under load (ms)"); ax.set_xlim(0, max(r["p95_ms"] for r in L) * 1.55)
    ax.set_title("Locust load test (remote mode, one 8-core laptop)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", visible=False)
    # (c) memory
    ax = axes[2]
    mem = {"monolith (inprocess)": s["warm"]["compose-inprocess"]["rss_mb"]["gateway"],
           **{f"{k} (remote)": v for k, v in s["warm"]["compose-remote"]["rss_mb"].items()}}
    names = list(mem)
    ax.barh(np.arange(len(names)), list(mem.values()), color=[viz.SERIES[3]] + [viz.SERIES[2]] * (len(names) - 1), height=0.6)
    for i, v in enumerate(mem.values()):
        ax.text(v + 6, i, f"{v:.0f}", va="center", fontsize=8)
    ax.axvline(512, color=viz.MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(np.arange(len(names)), names); ax.set_ylim(len(names) - 0.5, -1.1); ax.set_xlim(0, 560)
    ax.text(505, -0.8, "Render free instance: 512 MB", fontsize=8, color=viz.INK_2, ha="right", va="center")
    ax.set_xlabel("process RSS (MB)"); ax.set_title("Memory per process", fontsize=10); ax.grid(axis="y", visible=False)
    fig.suptitle("Phase 8 platform measurements (Docker Compose, local)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def main():
    import sys

    startup = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    s = build(startup)
    viz.save(fig_platform(s), "platform_01_phase8")
    print(json.dumps({k: s[k] for k in ("images_mb", "startup_s")} | {"load": s["load"]}, indent=2))


if __name__ == "__main__":
    main()
