# FinSight AI dashboard (Phase 9)

React 18 + Vite 5 + TypeScript single-page app. Every page calls only the public API gateway; it holds no data of its own.
Deployed as static files on **Vercel** (D-022); the API runs on **Render**.

```bash
make serve                 # API on :8000 (repo root, inprocess mode)
make dashboard             # this app on http://localhost:5173
make dashboard-build       # type-check + production build → frontend/dist
```

| Page | Route | API endpoints |
|---|---|---|
| **Analyze transactions** (home) | `/` | `/analyze/schema`, `/analyze/upload`, `/analyze/demo`, `/analyze/simulate`, `/analyses/{id}/…` |
| Executive overview | `/overview` | `/overview`, `/models/compare` |
| Data quality | `/data-quality` | `/data-quality` |
| Transaction explorer | `/transactions` | `/transactions`, `/transactions/{id}` |
| Pattern mining lab | `/patterns` | `/patterns`, `/patterns/benchmark` |
| ML experiments lab | `/ml` | `/models/compare` |
| Clustering & anomaly | `/clusters` | `/clusters` |
| Network explorer | `/network` | `/accounts`, `/network/{account_id}` |
| Risk scorer | `/score` | `/score-transaction` (with what-if overrides) |
| Platform | `/platform` | `/platform/status`, `/platform/benchmarks`, `/health` |

**Configuration:** `VITE_API_BASE_URL` (build-time; see `.env.example`). On Vercel: Project → Settings → Environment
Variables, then redeploy. The API must list the Vercel origin in `FINSIGHT_CORS_ORIGINS`.

**Design notes:**
- **Plotly** is a lazily loaded chunk (~1.4 MB gzipped). The first paint is about 60 KB, so the "waking the backend" screen appears immediately while a sleeping Render instance starts.
- **Colours** come from the validated palette shared with the report figures (`src/theme.css`):
  - Legitimate is blue, fraudulent orange, borderline aqua.
  - Dark mode uses its own stepped values.
  - Tier badges always carry an icon and text, never colour alone.
- **`vercel.json`** rewrites every path to `index.html`, so client-side routes work, and adds immutable caching for hashed assets plus basic security headers.
