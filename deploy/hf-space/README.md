---
title: DNAgen API
emoji: 🧬
colorFrom: gray
colorTo: green
sdk: docker
app_port: 8000
pinned: false
short_description: Genome-to-MIC antibiotic susceptibility prediction API
---

# DNAgen API

Backend for [DNAgen](https://dnagen-app.vercel.app): upload an assembled bacterial genome (FASTA) and get a
predicted MIC and likely-active / uncertain / likely-inactive call for each antibiotic.

These are predictions of in-vitro susceptibility, not prescribing advice. Research prototype built at MHacks;
not clinically validated.

Endpoints: `GET /health`, `GET /ready`, `GET /v1/species`, `POST /v1/predict`, `GET /v1/jobs/{id}`,
`GET /v1/jobs/{id}/result`. OpenAPI docs at `/docs`.

Deployed from the amr-backend repo with `scripts/deploy_hf_space.sh`; edit there, not here.
