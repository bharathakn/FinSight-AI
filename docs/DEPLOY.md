# Deploying FinSight AI (Render + Vercel)

**Order matters:**
1. Render (the API) comes first: Vercel needs the API address when it builds.
2. Vercel (the dashboard) comes second.
3. Then add the Vercel address to Render's CORS list, otherwise browsers block the calls.

**Current live deployment:**
- Dashboard: https://fin-sight-ai-bay.vercel.app
- API: https://finsight-api-r2mw.onrender.com

**Before you start:**
- The code must be on GitHub (`main`).
- A model bundle must be published as a GitHub Release (`make bundle && make publish-artifacts`). Render downloads `releases/latest/download/finsight-artifacts.tar.gz` while it builds.

## A. API on Render (about 10 min)
1. https://dashboard.render.com → sign in with GitHub.
2. **+ New → Blueprint** and connect **bharathakn/FinSight-AI**. If the repo isn't listed, use *Configure GitHub App* to grant access.
3. Blueprint name: `finsight`; branch: `main`; path: `render.yaml`.
4. When asked for **`FINSIGHT_CORS_ORIGINS`**, enter `http://localhost:5173` for now.
5. **Apply.** The first build takes 5–10 minutes: Python packages plus the model bundle. Wait for **Live**.
6. Copy the service URL (`https://finsight-api-xxxx.onrender.com`).
7. Test it: `/health` should return `"status":"ok"`, and `/docs` should show the API documentation.

## B. Dashboard on Vercel (about 3 min)
1. https://vercel.com → **Add New… → Project** → import **FinSight-AI**.
2. **Root Directory:** `frontend` (click *Edit*). **Framework:** Vite. Leave the build settings at their defaults (`vercel.json`).
3. **Environment variable:** `VITE_API_BASE_URL` = the Render URL, with `https` and **no trailing `/`**.
4. **Deploy.** Then copy the production domain from *Overview → Domains* (e.g. `https://fin-sight-ai-bay.vercel.app`). Don't use the long per-deployment link.

## C. Connect them (about 3 min)
1. Render → **finsight-api → Environment** → `FINSIGHT_CORS_ORIGINS` → set it to:
   ```
   https://<your-vercel-domain>,http://localhost:5173
   ```
   The address must match the browser bar exactly: `https`, no trailing `/`, no spaces.
2. **Save, rebuild, and deploy**, then wait for **Live**.
3. Open the Vercel address. "Waking the backend…" may show for up to ~1 minute, then the Analyze page opens.

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| The waking screen shows `http://localhost:8000` | `VITE_API_BASE_URL` was missing when Vercel built the site; Vite bakes it in at build time | Vercel → Settings → Environment Variables → check it → **Deployments → ⋯ → Redeploy** |
| Stuck on "Waking the backend…" even though the Render logs show `GET /health 200` | CORS: the Vercel address isn't in `FINSIGHT_CORS_ORIGINS`, or is spelled differently | Fix it as in step C; check with F12 → Console (a red "CORS" error) |
| A 502 shortly after changing a setting | Render is swapping in the redeployed instance | Wait for **Live** and retry |
| The first demo analysis takes ~30 s | The free instance has a small share of one CPU | Later runs of the same demo are cached and instant; attack simulations take ~10–15 s |
| Slow first visit | The free instance sleeps after 15 min idle | Open the site a minute before a demo |

## Optional
- **Microservices on Render:** New → Blueprint → path `deploy/render-full.yaml` (5 free services, remote mode).
- **Google Cloud Run comparison:** `./deploy/cloudrun.sh <PROJECT_ID>`. It needs billing enabled (see `docs/SETUP.md` §5.1c).
- **Measurements:** `make measure URL=https://… LABEL=render`, plus `COLD=1` after 15 idle minutes.
