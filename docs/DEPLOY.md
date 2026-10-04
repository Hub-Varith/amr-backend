# Deploying DNAgen

**Frontend** (`frontend/`, static Vite build) → **Vercel**.
**API** (Dockerfile) → any host that runs a Docker container with HTTPS: Railway, Render, Fly.io, or the AWS VM.

The API cannot run on Vercel: it needs AMRFinderPlus and Mash (Linux binaries from bioconda), torch, and
background jobs that outlive the request, and genome uploads exceed Vercel's 4.5 MB function body limit.

Deploy the API first; the frontend needs its URL.

---

## 1. API (Docker)

### Get the model
`models/` is gitignored, and the image bakes in `models/all5_run1`. Fetch it first (MODEL_HANDOFF.md section 10).
A host that builds straight from GitHub will not have it — build locally and push the image (below).

### Build and run locally
```bash
cp .env.example .env            # fill in G2M_DATABASE_URL if you want jobs kept across restarts
docker compose up --build       # first build ~10–15 min: AMRFinderPlus DB + reference genomes
curl localhost:8000/ready       # {"ready": true, ...}
```

### Quick public URL from your laptop (what the demo runs on)
Run the container locally and expose it over HTTPS with a localhost.run SSH tunnel:
```bash
docker compose up -d --build
bash scripts/tunnel.sh        # Windows PowerShell: & "C:\Program Files\Git\bin\bash.exe" scripts/tunnel.sh
```
`tunnel.sh` opens the tunnel, repoints the Vercel site at each new `https://<random>.lhr.life` URL
(`scripts/point_frontend.sh`, ~1 min redeploy), reconnects when the tunnel drops, and restarts it when
it hangs (health check every 30 s). Leave its terminal open. The API is only reachable while the laptop
is awake and online; each reconnect means 1–3 min of downtime. To repoint by hand:
`scripts/point_frontend.sh https://<url>`.

Nothing else may listen on `127.0.0.1:8000` (e.g. a `make api` in WSL), or the tunnel reaches that instead
of the container. On networks that allow outbound TCP 7844, a Cloudflare quick tunnel also works:
`docker compose --profile tunnel up -d`, then read the URL from `docker compose logs tunnel`.

### Hugging Face Space (needs HF PRO)
`scripts/deploy_hf_space.sh <user>/dnagen-api` creates a public Docker Space plus a private model repo; the
Space build pulls the model with an `HF_TOKEN` secret (a read token, added in the Space settings).
Docker Spaces require a PRO subscription, even on the free CPU hardware.

### Push and deploy to a cloud host
```bash
docker build -t <registry>/genome2mic-api:latest .
docker push <registry>/genome2mic-api:latest     # Docker Hub, GHCR, or ECR
```
On the host, deploy that image and set:

| Variable | Value |
| --- | --- |
| `G2M_CORS_ORIGINS` | `["https://<your-app>.vercel.app"]` — JSON list; add custom domains |
| `G2M_CORS_ORIGIN_REGEX` | optional, for preview deploys: `https://<your-app>-[a-z0-9-]+\.vercel\.app` |
| `G2M_DATABASE_URL` | Neon pooled connection string (optional; without it jobs are lost on restart) |

The container listens on `$PORT` (default 8000), which Railway/Render/Fly set themselves.
Health check path: `/health`. Readiness: `/ready`.

Sizing: at least 2 GB RAM (torch + AMRFinderPlus). Run **one instance** — jobs execute in the API process.

---

## 2. Frontend (Vercel)

Vercel project: **`dnagen`** in the MHacks team (`mh-acks3`). Production URL: **https://dnagen-app.vercel.app**
(`dnagen.vercel.app` belongs to another account).

`frontend/vercel.json` sets the Vite build and rewrites every path to `index.html` so
`BrowserRouter` routes survive a reload.

**Dashboard:** Add New → Project → import the repo, then:
- **Root Directory:** `frontend`
- **Environment Variable:** `VITE_API_BASE_URL=https://<your-api-host>` (no trailing slash, must be https)
- Deploy

**CLI:**
```bash
cd frontend
npx vercel link
npx vercel env add VITE_API_BASE_URL production
npx vercel --prod
npx vercel alias set <deployment-url> dnagen-app.vercel.app   # only if the alias did not move on its own
```

`VITE_API_BASE_URL` is inlined at build time: redeploy after changing it.

---

## 3. Connect them

1. Put the Vercel URL in the API's `G2M_CORS_ORIGINS` and restart the API.
2. Open the site and check the browser console: a CORS error means the origin is missing or mistyped;
   a "mixed content" error means the API URL is http.
