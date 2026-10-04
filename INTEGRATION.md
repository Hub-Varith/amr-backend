# Genome-to-MIC — Frontend ↔ Backend Integration

**Status:** draft v0.1 · **Last updated:** 2026-10-03

How the frontend connects to the backend while the backend is still being built.
`DATA_CONTRACT.md` defines the data; this file defines the connection. **If the two
disagree, `DATA_CONTRACT.md` wins.** Items marked **PROPOSED** are not agreed yet and
need team sign-off; adopting one means updating `DATA_CONTRACT.md` in the same commit.

---

## 0. The shape in one picture

```
Frontend (browser)
   │  HTTP + JSON, types generated from /openapi.json
   ▼
Orchestrator  = the FastAPI app in src/genome2mic/api/   (exists)
   ├ Public API       /health /ready /v1/species /v1/predict /v1/jobs/...
   ├ Job manager      upload → job → run steps in order → record progress → store report
   ├ Contract guard   every result is validated against the Pydantic schemas
   └ Adapters         one per backend capability, each switchable mock ↔ real
        ├ QcAdapter         → DATA_CONTRACT stage 3 (QC)
        ├ SpeciesAdapter    → stage 3 (Mash species ID)
        ├ NoveltyAdapter    → stage 12 (distance to nearest training genome)
        ├ AmrAdapter        → stages 4–5 (AMRFinderPlus → known-AMR features)
        ├ UnitigAdapter     → stage 8 (query against the fixed unitig set)
        ├ MicModelAdapter   → stages 9–10 (model + conformal band)
        └ CallRules         → stage 12 (breakpoints, overrides, ranking); not mocked
```

Three rules make this work:

1. **The frontend talks only to the orchestrator.** It never calls a model, a tool,
   or a teammate's service directly.
2. **The public API does not change when a backend piece lands.** Swapping a mock
   adapter for a real one is a config change, invisible to the frontend.
3. **Mocked output is always labelled as mocked.** No fake MIC can reach a screen
   without a "simulated data" mark (§6).

---

## 1. Public API (frontend ↔ orchestrator)

Base URL: `http://127.0.0.1:8000` in dev (`make api`). All `/v1` routes go through
`require_auth`, which is a no-op placeholder today.

| Method | Path | Success | Purpose |
| ------ | ---- | ------- | ------- |
| GET | `/health` | 200 `{"status":"ok"}` | Process is up |
| GET | `/ready` | 200 / 503 `ReadyStatus` | Configs parsed and models loaded |
| GET | `/v1/species` | 200 `SpeciesInfo[]` | Species in scope + drugs with a loaded model |
| POST | `/v1/predict` | 202 `JobAccepted` | Upload one FASTA, get a job id |
| GET | `/v1/jobs?limit=20` | 200 `JobState[]` | Recent jobs, newest first (`limit` 1–100). Kept across restarts when `G2M_DATABASE_URL` is set |
| GET | `/v1/jobs/{job_id}` | 200 `JobState` | Job status |
| GET | `/v1/jobs/{job_id}/result` | 200 `PredictionReport` | The report, once status is `done` |
| GET | `/v1/jobs/{job_id}/events` | 200 SSE stream | **PROPOSED** — live progress (§4) |
| GET | `/v1/drugs` | 200 `DrugInfo[]` | **PROPOSED** — display names + spectrum tiers from `drugs.yaml` |

### 1.1 `POST /v1/predict`

`multipart/form-data`:

| Field | Required | Rule |
| ----- | -------- | ---- |
| `file` | yes | Extension `.fasta`, `.fa` or `.fna`, or the same gzipped (`.fasta.gz`, `.fa.gz`, `.fna.gz`); ≤ 50 MB (`G2M_MAX_UPLOAD_BYTES`), and ≤ 50 MB again once unpacked; first non-blank line starts with `>`. A corrupt gzip is 422 |
| `sample_id` | no | 1–128 chars. Defaults to the file name without its FASTA and gzip extensions (`470.7409.fasta.gz` → `470.7409`) |

```json
// 202 Accepted
{
  "job_id": "9f1c2e...",
  "status": "queued",
  "status_url": "http://127.0.0.1:8000/v1/jobs/9f1c2e..."
}
```

The upload is deleted when the job finishes, whether it succeeds or fails.

### 1.2 `GET /v1/jobs/{job_id}` → `JobState`

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `job_id` | str | |
| `sample_id` | str | |
| `status` | `queued` / `running` / `done` / `failed` | |
| `created_at`, `updated_at` | ISO 8601 UTC | |
| `error` | str \| null | Set only when `failed`; quotes the request id |
| `stage` | str \| null | **PROPOSED** — current step id (§4) |
| `stages_done` | list[str] | **PROPOSED** — completed step ids, in order |

### 1.3 `GET /v1/jobs/{job_id}/result` → `PredictionReport`

The shape is `DATA_CONTRACT.md` stage 12, enforced by
`api/schemas/prediction_report.py` and `api/schemas/drug_prediction.py`. Returns
**409** until status is `done`.

```json
{
  "sample_id": "BC-0142",
  "species": "KPNEU",
  "qc_pass": true,
  "nearest_training_distance": 0.004,
  "in_range": true,
  "predictions": [
    {
      "drug": "meropenem",
      "pred_mic": 0.0625, "band_low": 0.03125, "band_high": 0.125,
      "s_breakpoint": 2.0, "r_breakpoint": 8.0,
      "call": "likely_active", "margin_steps": 4,
      "reasons": [], "override": null
    },
    {
      "drug": "ampicillin",
      "pred_mic": null, "band_low": null, "band_high": null,
      "s_breakpoint": null, "r_breakpoint": null,
      "call": "likely_inactive", "margin_steps": null,
      "reasons": [], "override": "natural_resistance"
    }
  ],
  "ranked_active": ["meropenem"],
  "model_version": "mock-0.1",
  "run_id": "abc123",
  "disclaimer": "These are predictions of in-vitro susceptibility, not prescribing advice. ..."
}
```

Validation guarantees the frontend can rely on:

- `pred_mic`, `band_low`, `band_high` are all set or all null, and
  `band_low ≤ pred_mic ≤ band_high`.
- `s_breakpoint ≤ r_breakpoint` when both are set.
- `override` set ⇒ `call == "likely_inactive"`.
- Every drug in `ranked_active` is a `likely_active` prediction.
- `disclaimer` is never blank.

### 1.4 Errors

Every error is RFC 7807 `application/problem+json`:

```json
{
  "type": "about:blank",
  "title": "Request Entity Too Large",
  "status": 413,
  "detail": "File is larger than 52428800 bytes.",
  "instance": "/v1/predict",
  "request_id": "4b7e..."
}
```

| Status | When | Frontend should |
| ------ | ---- | --------------- |
| 404 | Unknown job id | "Job not found"; offer a new upload |
| 409 | Result requested before `done` | Keep polling; not an error to the user |
| 413 | File too large | Show limit before upload; show `detail` |
| 415 | Wrong extension | Show allowed extensions |
| 422 | Not FASTA, or bad form fields (`errors[]` lists them) | Show `detail` |
| 503 | Models not loaded | Disable upload; show the `/ready` state |
| 500 | Unexpected | Generic message + `request_id` |

Every response carries an `X-Request-ID` header. The frontend may send its own
(`[A-Za-z0-9._-]{1,128}`) and should show the id in any error message so a failed
job can be traced in the logs.

---

## 2. Job lifecycle

```
Frontend                         Orchestrator                         Adapters
   │ POST /v1/predict (FASTA)       │                                    │
   │──────────────────────────────▶ │ validate + save upload             │
   │ ◀────────── 202 {job_id} ──────│ create job (queued)                │
   │                                │ ── background ───────────────────▶ │ qc → species → novelty
   │ GET /v1/jobs/{id}  (poll/SSE)  │ status running, stage = ...        │ → amr → unitigs → mic_model
   │ ◀──────── {status, stage} ─────│                                    │ → call_rules
   │                                │ ◀──────── step outputs ────────────│
   │                                │ validate PredictionReport          │
   │                                │ status done (or failed)            │
   │ GET /v1/jobs/{id}/result       │                                    │
   │ ◀──────── PredictionReport ────│                                    │
```

Jobs run in-process with FastAPI `BackgroundTasks` (`services/background_tasks_queue.py`).
Job state lives in memory (`services/job_store.py`): **a restart loses every job.**
Fine for development and the demo. A Redis/RQ worker can replace it behind the
existing `JobQueue` interface without touching the public API.

---

## 3. Adapters (orchestrator ↔ backend)

The adapter interfaces are the **handoff contract with the backend teams**. A
backend piece counts as integrated when a real adapter satisfies its interface and
passes the checks in §7.

Assumption: backend code lives as Python modules in this repo
(`src/genome2mic/features/`, `models/`, …) and adapters call them in-process. If a
piece ships as a separate service instead, its adapter makes HTTP calls with the
same inputs and outputs, plus timeouts and retries.

### 3.1 Step order and short-circuits

| # | Step id | Adapter | Short-circuit |
| - | ------- | ------- | ------------- |
| 1 | `qc` | `QcAdapter` | — (QC failure is reported, not fatal) |
| 2 | `species` | `SpeciesAdapter` | Species not in scope → `species: null`, `in_range: false`, empty `predictions`, stop |
| 3 | `novelty` | `NoveltyAdapter` | — |
| 4 | `amr` | `AmrAdapter` | — |
| 5 | `unitigs` | `UnitigAdapter` | Skipped until stage 8 lands |
| 6 | `mic_model` | `MicModelAdapter` | Skipped for drugs with an override |
| 7 | `calls` | `CallRules` | — |
| 8 | `report` | orchestrator | Validation failure → job `failed` |

Each step writes its output to the job's work directory (`<upload_dir>/<job_id>/`),
following the "every stage writes a file" convention, so a bad report can be
traced to the step that caused it.

### 3.2 Interfaces

Types below are Python signatures. Field names follow `DATA_CONTRACT.md`.

```python
class QcAdapter(Protocol):
    def run(self, fasta_path: Path) -> QcResult: ...

@dataclass(frozen=True)
class QcResult:                      # DATA_CONTRACT stage 3, qc.parquet columns
    n_contigs: int
    total_length: int
    n50: int
    gc_percent: float
    qc_pass: bool
    qc_fail_reason: str | None
```

```python
class SpeciesAdapter(Protocol):
    def identify(self, fasta_path: Path) -> SpeciesResult: ...

@dataclass(frozen=True)
class SpeciesResult:
    species: str | None              # 5-letter key, None if no reference is close enough
    mash_distance: float             # to the nearest species reference
```

```python
class NoveltyAdapter(Protocol):
    def nearest_training(self, fasta_path: Path, species: str) -> NoveltyResult: ...

@dataclass(frozen=True)
class NoveltyResult:
    nearest_training_distance: float  # Mash distance to the nearest QC-passing training genome
    in_range: bool                    # threshold: OPEN QUESTION (§8)
```

```python
class AmrAdapter(Protocol):
    def detect(self, fasta_path: Path, species: str) -> AmrResult: ...

@dataclass(frozen=True)
class AmrHit:
    symbol: str                       # AMRFinderPlus "Element symbol", e.g. "blaKPC-2"
    type: str                         # AMR / POINT / ...
    drug_class: str                   # "Class"
    subclass: str                     # "Subclass"

@dataclass(frozen=True)
class AmrResult:
    features: dict[str, int]          # gene_*, point_*, n_class_* — same names as known_amr.parquet
    hits: list[AmrHit]                # raw hits, used for reasons and strong-marker overrides
```

`AmrAdapter` **must use the same parser and column naming as the training feature
build** (`features/known_amr.py`). A second implementation of the naming rules is
how train/serve skew gets in.

```python
class UnitigAdapter(Protocol):
    def query(self, fasta_path: Path, species: str) -> UnitigResult: ...

@dataclass(frozen=True)
class UnitigResult:
    present_pattern_ids: list[str]    # e.g. ["u_000017", "u_004211"]
```

`UnitigAdapter` **queries** the frozen unitig set for the species. It never builds
or rebuilds it (non-negotiable rule 3).

```python
class MicModelAdapter(Protocol):
    def available(self) -> dict[str, list[str]]: ...       # species -> drugs with a model
    def predict(self, species: str, drug: str,
                amr: AmrResult, unitigs: UnitigResult | None) -> MicResult: ...

@dataclass(frozen=True)
class MicResult:                      # mg/L
    pred_mic: float                   # rounded UP to the next doubling step (rule 9)
    band_low: float                   # 90% conformal band
    band_high: float
    pred_censor: str                  # PROPOSED: "interval" / "left" / "right"
    model: str                        # e.g. "aft_known_unitig"
    run_id: str
```

`MicModelAdapter` owns aligning features to the model's training columns:
a training column missing from the genome becomes `0`; a feature the model never saw
is dropped and logged. The training column list ships with the model (§3.3).

`CallRules` is plain code, not mocked. It reads `configs/breakpoints/*.csv`,
`configs/natural_resistance.csv`, and `configs/drugs.yaml`, and applies the
call logic, the three overrides, and the ranking exactly as written in
`DATA_CONTRACT.md` stage 12. It is written test-first.

### 3.3 Model artifact layout — PROPOSED

`DATA_CONTRACT.md` does not yet say how the model team hands models to the
orchestrator. Proposal:

```
models/<SPECIES>/<drug>/
  model.json         # XGBoost booster
  features.json      # ordered training feature columns (known-AMR + selected u_ ids)
  conformal.json     # {"q_steps": 1.0, "coverage": 0.9, "n_calibration": 412}
  meta.json          # {"model": "aft_known", "run_id": "...", "trained_at": "...", "breakpoint_version": "..."}
```

`models_dir` is set by `G2M_MODELS_DIR` and mounted read-only in Docker
(`make docker-run`).

### 3.4 Switching mock ↔ real

**PROPOSED** setting, read by `api/config.py`:

```bash
G2M_ADAPTERS='{"qc":"real","species":"mock","novelty":"mock","amr":"mock","unitigs":"off","mic_model":"mock"}'
```

| Value | Meaning |
| ----- | ------- |
| `real` | Call the backend implementation |
| `mock` | Return contract-valid synthetic output (§6) |
| `off` | Skip the step (only allowed for `unitigs`) |

Any mix is allowed: real AMRFinderPlus with mocked MICs is a valid demo.
`/ready` reports ready when every adapter set to `real` has loaded.

---

## 4. Progress — PROPOSED

AMRFinderPlus and the unitig query take minutes, so the frontend needs more than
`running`.

**Polling (works today, extended):** `GET /v1/jobs/{id}` adds `stage` and
`stages_done` using the step ids from §3.1.

**Server-sent events:** `GET /v1/jobs/{id}/events`, `text/event-stream`:

```
event: stage
data: {"job_id":"9f1c...","stage":"amr","stages_done":["qc","species","novelty"]}

event: done
data: {"job_id":"9f1c...","result_url":"/v1/jobs/9f1c.../result"}

event: failed
data: {"job_id":"9f1c...","error":"Prediction failed. Quote request id 4b7e... when reporting this."}
```

The stream closes after `done` or `failed`. The frontend falls back to polling if
the stream drops.

---

## 5. Frontend side

### 5.1 Client

- Types generated from the orchestrator's `/openapi.json` with `openapi-typescript`
  into `frontend/src/api/schema.ts`. Regenerate whenever a schema changes; never
  hand-write API types.
- One `api/` module owns every request. Components use hooks, never `fetch`:
  `useReady`, `useSpecies`, `useSubmitGenome`, `useJob`, `useReport`.
- Base URL from `VITE_API_BASE_URL`. In dev, a Vite proxy to `127.0.0.1:8000`
  avoids CORS; otherwise set `G2M_CORS_ORIGINS='["http://localhost:5173"]'`.

### 5.2 Polling rules

- Poll `GET /v1/jobs/{id}` every 1 s for the first 10 s, then every 3 s.
- On `done`, fetch `/result` once. On `failed`, stop and show `error`.
- Stop after 15 minutes and show a timeout with the request id.

### 5.3 Display rules

These come from `CLAUDE.md` and `DATA_CONTRACT.md` and are not style choices.

| Rule | Detail |
| ---- | ------ |
| Disclaimer | Shown on every report view and every export. Not collapsible. Text comes from `report.disclaimer` |
| Call labels | "Likely active", "Uncertain — wait for lab", "Likely inactive". Never "recommended", "use", or "treat with" |
| No dose | Never show or imply a patient dose. MIC is a lab measurement, labelled mg/L |
| Out of range | `in_range == false` → banner: all calls are low confidence |
| Species not covered | `species == null` → no drug table; explain the species is out of scope |
| QC fail | `qc_pass == false` → banner above the results |
| Overrides | `natural_resistance` → "Natural resistance", no MIC. `strong_marker` → show the marker from `reasons` |
| Censored MIC | Show `>32` / `≤0.25` once `pred_censor` exists; until then show the band |
| Ranking | Ranked list follows `ranked_active` order exactly; the frontend never re-ranks |
| MIC numbers | Doubling-step values formatted as lab panels show them (0.06, 0.125, 0.25 …) |

### 5.4 Screens

1. **Upload:** drag-and-drop FASTA, optional sample id, species/drug coverage
   from `/v1/species`. Upload disabled while `/ready` is 503.
2. **Progress:** step timeline from §4, elapsed time.
3. **Report:** header (sample, species, QC, distance, in-range), ranked active list,
   per-drug rows with the MIC band drawn on a log₂ axis next to the S/R breakpoints,
   reasons as chips, disclaimer, print/PDF export.

---

## 6. Mock data rules

Mocks let the frontend run end to end before the backend exists. They must never
pass as real predictions.

1. **PROPOSED** report field `mocked_components: list[str]` lists every adapter
   that ran as `mock` (e.g. `["amr","mic_model"]`). Empty means fully real.
2. If any component is mocked, `model_version` starts with `mock-`.
3. The frontend shows a persistent **SIMULATED DATA** watermark on the report and on
   every export whenever `mocked_components` is non-empty or `model_version` starts
   with `mock-`.
4. Mock output is still validated by the contract guard, so the frontend is built
   against data with the exact real shape.
5. Mocks are deterministic per `sample_id` (seeded) so screenshots and tests are
   reproducible.
6. Test fakes in `tests/api/fakes/` stay separate from runtime mocks and keep their
   `FAKE` labels.

---

## 7. Integrating a backend piece — definition of done

A real adapter replaces a mock when all of these hold:

- [ ] Implements its interface in §3.2 with no changes to the public API
- [ ] Unit test on a small fixture genome checked into `tests/`
- [ ] Output passes the contract guard for at least one KPNEU genome end to end
- [ ] `AmrAdapter`: feature names match `known_amr_columns.csv` from the training build
- [ ] `UnitigAdapter`: queries the frozen set; confirmed it never writes to it
- [ ] `MicModelAdapter`: loads from the §3.3 layout; rounds up; band contains `pred_mic`
- [ ] Failure modes raise clear exceptions (the job runner turns them into `failed`)
- [ ] Switched to `real` in the default config, and the change noted in the PR

---

## 8. Ownership and open questions

| Piece | Owner |
| ----- | ----- |
| Public API, job manager, contract guard, mocks, `CallRules` | Integration (orchestrator) |
| Frontend | Frontend |
| `QcAdapter`, `SpeciesAdapter`, `NoveltyAdapter`, `AmrAdapter`, `UnitigAdapter` real impls | Data team |
| `MicModelAdapter` real impl, model artifacts | Model team |
| Configs (`breakpoints`, `natural_resistance`, `drugs.yaml`) | Whole team, reviewed |

Open:

- [ ] `in_range` threshold: what Mash distance to the nearest training genome counts as "far"?
- [ ] Strong-marker list: which markers force `likely_inactive` for which drugs? Proposed home: `configs/strong_markers.csv` (`species`, `drug`, `marker_pattern`).
- [ ] Breakpoint version shown on the report: needs the `DATA_CONTRACT.md` breakpoint decision. Proposed report fields `breakpoint_standard`, `breakpoint_version`.
- [ ] QC detail on the report: proposed optional `qc` object with the §3.2 `QcResult` fields so a QC failure can say why.
- [ ] Top-level `markers: list[str]` on the report for all detected AMR hits?
- [ ] Are any backend pieces shipping as separate services rather than modules (§3)?

### Proposed contract changes, collected

| Change | Where | Section |
| ------ | ----- | ------- |
| `stage`, `stages_done` on `JobState` | API | §1.2, §4 |
| `GET /v1/jobs/{id}/events` (SSE) | API | §4 |
| `GET /v1/drugs` | API | §1 |
| `pred_censor` on `DrugPrediction` | Stage 12 | §3.2 |
| `mocked_components` on `PredictionReport` | Stage 12 | §6 |
| `qc`, `markers`, `breakpoint_standard`, `breakpoint_version` on `PredictionReport` | Stage 12 | §8 |
| Model artifact layout | New section | §3.3 |
| `G2M_ADAPTERS` setting | API config | §3.4 |
| `configs/strong_markers.csv` | Configs | §8 |
