# Setup Guide — FinSight AI (Ubuntu 26.04, x86_64)

Already present: `git`, `curl`, `make`, `uv` 0.13.9, project venv `.venv` (Python 3.12, core packages).
Status (2026-11-21): Python env, Docker, Node.js ✅ installed. Still missing: GitHub CLI (`gh`). Deployment target: Vercel + Render.

Run the blocks in order. Each block has a **verify** step.

---

## 1. System packages (one-time)
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg build-essential
```

## 2. Python dependencies (needed now, before Phase 2)
`pyproject.toml` is configured for **CPU-only** PyTorch and XGBoost, which skips about 3 GB of CUDA downloads.
```bash
cd /mnt/data/bharatha_kn/On_Going_Projects/cct/FinSight_ai
uv sync --all-extras          # installs everything from uv.lock into .venv (~1 GB download)
```
If the connection is slow or drops, install in stages (each one can be re-run safely):
```bash
uv sync                               # core: pandas, sklearn, xgboost-cpu, mlxtend, networkx, ...
uv sync --extra viz --extra dev       # jupyter, matplotlib, seaborn, plotly, pytest, locust
uv sync --extra viz --extra dev --extra explain --extra dl     # + shap, torch (CPU), onnx
uv sync --all-extras                  # + fastapi, streamlit, gcsfs, s3fs
```
Register the Jupyter kernel (so notebooks / VS Code use this venv):
```bash
uv run python -m ipykernel install --user --name finsight --display-name "FinSight (py3.12)"
```
**Verify:**
```bash
uv run python -c "import pandas, sklearn, xgboost, mlxtend, networkx, polars, shap, torch, onnxruntime, fastapi, streamlit; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
uv run pytest -q              # expect: 10 passed
```
Expected: `cuda? False`, meaning the CPU build is installed correctly.

In VS Code: `Ctrl+Shift+P` → *Python: Select Interpreter* → `./.venv/bin/python`.

## 3. Docker Engine + Compose (needed by Phase 8)
**Option A: Docker's official repo (recommended)**
```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
**Option B: if `apt-get update` shows a 404 for docker.com (26.04 not yet in Docker's repo), use Ubuntu's packages instead:**
```bash
sudo rm -f /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
```
**Then (both options): run Docker without sudo**
```bash
sudo usermod -aG docker $USER
# log out and back in (or reboot) so the group change applies everywhere, incl. VS Code
```
**Verify:**
```bash
docker run --rm hello-world
docker compose version
```

## 4. GitHub CLI (needed by Phase 8/10 to publish model artifacts as Releases)
```bash
sudo mkdir -p -m 755 /etc/apt/keyrings
wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt-get update && sudo apt-get install -y gh
gh auth login          # choose GitHub.com → HTTPS → login with browser
```
**Verify:** `gh auth status`

Node.js (v24, via nvm) is already installed; it's needed for the React dashboard (Phase 9). No action needed.

> The Google Cloud CLI installed earlier is no longer needed (deployment moved to Vercel + Render, D-022). You can leave it or remove it: `sudo apt-get remove google-cloud-cli`.

---

## 5. Manual tasks only you can do (human-only)

### 5.1 GitHub, Render and Vercel accounts (needed by Phase 10; do them anytime)
1. **GitHub repo:** the code is at `github.com/bharathakn/FinSight-AI`. Make sure all teammates who will deploy have access.
   - **Public repo (simplest):** Render can download the model release during `docker build` with no token.
   - **Private repo:** create a fine-grained GitHub token with *Contents: read-only* on this repo. It's later stored in Render as `GITHUB_TOKEN` (never committed).
   - The dataset stays out of Git (`dataset/` is ignored); only code + small reports are pushed.
2. **Render** (backend): sign up at https://render.com with **"Sign in with GitHub"**, then authorise access to the `FinSight-AI` repo.
   - Free plan; a card is not required for free web services.
   - Nothing else to click yet: in Phase 10 you'll choose *New → Blueprint* and select the repo (it reads `render.yaml`).
3. **Vercel** (frontend): sign up at https://vercel.com with **"Continue with GitHub"** (Hobby plan, free).
   - In Phase 10: *Add New → Project → import FinSight-AI*, set **Root Directory = `frontend`**, and add the env var `VITE_API_BASE_URL` = the Render URL.
4. **Tell the agent** once these exist: the Render service URL (after the first deploy) and the Vercel project URL, so CORS and the frontend config can be set.
5. Optional, only if you want a real object-storage bucket for the CCT "cloud storage" topic: a free **Supabase** project (Storage → S3 access keys) or **Cloudflare R2**. Not required for the deployment to work.

### 5.1b Free-tier facts to know (no action)
- Render free services **sleep after 15 minutes idle**; the first request afterwards takes **~1 minute** (the dashboard shows a "waking the backend" screen). Open the site a minute before a demo.
- Render gives **750 free instance-hours per month** per workspace; sleeping services don't use hours.
- Don't use "keep-alive ping" hacks; Render discourages them.

### 5.2 Course requirements (anytime; these shape the report)
- Each professor's **rubric / evaluation criteria** and **submission format** (report template, page limits, viva/demo format)
- **Deadlines** for each subject (the phase schedule gets planned around them)
- **Syllabus documents** for DMW, ML, CCT, if they differ from the mapping in Plan §12
- Team members and who presents what (if it's a group project)
- Put files in `docs/course/` (PDF or text is fine)

### 5.3 Decisions to confirm later (no action now)
- Public repo or private repo + token (affects the Render build)?
- A custom domain for the demo? (optional; both Vercel and Render support one for free)

---

## 6. Status checklist (tick as you go)
- [x] §1 system packages
- [x] §2 `uv sync --all-extras` + kernel registered + `pytest` passes
- [x] §3 Docker works without sudo (`docker run --rm hello-world`)
- [x] §4 (old) gcloud installed (no longer needed)
- [ ] §4 `gh auth status` works
- [ ] §5.1 Render + Vercel accounts created via GitHub; repo access granted (public repo or token decided)
- [ ] §5.2 Rubrics / deadlines added to `docs/course/`
