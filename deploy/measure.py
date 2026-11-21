"""Measure a running FinSight deployment (local Compose, Render, Cloud Run) with one method (D-028).

    python deploy/measure.py --url http://localhost:8000 --label compose-remote
    python deploy/measure.py --url https://finsight-api.onrender.com --label render --cold

--cold   poll /health from now until it answers and report the wait (run it after the service has slept:
         Render free spins down after 15 min idle; Cloud Run scales to zero).
Warm latency: N sequential POST /score-transaction calls over a fixed, seeded sample of transaction IDs
(client-side wall time, so it includes the network), plus the server-side time from the Server-Timing header.
Writes reports/platform_<label>.json.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import httpx

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def pct(v, q):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(q / 100 * (len(v) - 1))))], 2)


def cold_start(c: httpx.Client, timeout_s: float = 180) -> dict:
    t0 = time.perf_counter()
    attempts = 0
    while time.perf_counter() - t0 < timeout_s:
        attempts += 1
        try:
            r = c.get("/health", timeout=timeout_s)
            if r.status_code == 200:
                return {"seconds_to_healthy": round(time.perf_counter() - t0, 2), "attempts": attempts, "health": r.json()}
        except httpx.HTTPError:
            time.sleep(1)
    return {"seconds_to_healthy": None, "attempts": attempts}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--cold", action="store_true")
    a = ap.parse_args(argv)
    out = {"label": a.label, "url": a.url, "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with httpx.Client(base_url=a.url, timeout=120) as c:
        if a.cold:
            out["cold_start"] = cold_start(c)
        ids = [t["transaction_id"] for t in c.get("/transactions", params={"limit": 500, "sort": "transaction_id"}).json()["items"]]
        sample = random.Random(42).choices(ids, k=a.n)
        c.post("/score-transaction", json={"transaction_id": sample[0]})          # warm-up
        client_ms, server_ms, tiers = [], [], {}
        for tx in sample:
            t0 = time.perf_counter()
            r = c.post("/score-transaction", json={"transaction_id": tx})
            client_ms.append((time.perf_counter() - t0) * 1000)
            r.raise_for_status()
            st = r.headers.get("server-timing", "")
            if "dur=" in st:
                server_ms.append(float(st.split("dur=")[1].split(";")[0]))
            tiers[r.json()["tier"]] = tiers.get(r.json()["tier"], 0) + 1
        health = c.get("/health").json()
        status = c.get("/platform/status").json()
    out["warm"] = {"n": a.n, "client_p50_ms": pct(client_ms, 50), "client_p95_ms": pct(client_ms, 95),
                   "client_p99_ms": pct(client_ms, 99), "client_mean_ms": round(statistics.mean(client_ms), 2),
                   "server_p50_ms": pct(server_ms, 50) if server_ms else None,
                   "server_p95_ms": pct(server_ms, 95) if server_ms else None, "tiers": tiers}
    out["mode"] = health.get("mode")
    out["rss_mb"] = {"gateway": health.get("rss_mb"), **{k: v.get("rss_mb") for k, v in status["services"].items()
                                                         if isinstance(v, dict) and "rss_mb" in v}}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"platform_{a.label}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
