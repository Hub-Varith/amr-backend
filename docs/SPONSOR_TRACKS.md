# Sponsor Tracks — Implementation Plan

**Status:** draft v0.1 · **Last updated:** 2026-10-03

How to qualify for the MHacks sponsor tracks that fit Breakpoint without bending the
project. Each section covers what the sponsor wants, what we build, the files it
touches, the steps, and how to submit it.

| Priority | Track | Effort | Touches model or data contract? |
| -------- | ----- | ------ | ------------------------------- |
| 1 | FinchNode — personalized healthcare | ~4–6 h | No |
| 2 | Neon — backend | ~3–5 h | No (job store only) |
| 3 | Fetch.ai — ASI:One agent | ~5–8 h | No |
| 4 | ElevenLabs — voice | ~2–3 h | No |
| 5 | Sustainability (MHacks track) | ~1 h, mostly pitch | No |
| — | Notability | ~30 min, no code | No |

**The rule behind every section:** sponsor features sit *around* the prediction and
never feed it. Patient data, voice, agents and storage do not change a predicted MIC,
a call, an override or the ranking. Anything that would change those goes through
`DATA_CONTRACT.md` and a team sign-off first (see `CLAUDE.md`, "Working agreements").

Every new screen, spoken summary and agent reply carries the disclaimer in
`src/genome2mic/api/constants.py` (`DISCLAIMER`). No feature may suggest a dose or say
"use drug X".

---

## 1. FinchNode — patient context next to the report

### What the sponsor wants

An app that makes healthcare easier for clinicians or patients, with a working
FinchNode integration on their synthetic demo records.

### What we build

A **patient context panel** next to the drug report. The clinician picks a synthetic
patient. The panel shows that patient's allergies, recent kidney labs and active
medications. Each drug row in the report gets a flag when an allergy on file matches
the drug's class, for example "Penicillin allergy on file (penicillins)".

The flag is information only. It does not hide the drug, re-rank it or change its call.
The clinician decides.

### FinchNode demo API (checked 2026-10-03)

- Base URL: `https://api.finchnode.com/demo/v1`. No key needed. Rate limit: 120
  requests/min per IP address.
- `GET /scenarios` lists 12 scenarios. Each has an `id`, a `subject` (the patient id)
  and a `persona.displayName`.
- `GET /users/{subject}/records?categories=allergies,labs,medications,conditions`
  returns normalized JSON. The records sit under `data.<category>[]`.
  - Allergy: `substance`, `reaction`, `severity`, `status`, `codes[]` (SNOMED).
  - Lab: `name`, `value`, `unit`, `date`, `referenceRange`, `interpretation`, `codes[]` (LOINC).
  - Medication: `name`, `dosage`, `status`, `startDate`, `codes[]` (RxNorm).
- Every response carries `synthetic: true` and a `meta.disclaimer`. Show both in the UI.

The scenarios that make good demos:

| Scenario `id` | `subject` | Why it's useful |
| ------------- | --------- | --------------- |
| `baseline-adult` | `patient-demo-001` | Penicillin allergy → flags ampicillin, amoxicillin, piperacillin-tazobactam … |
| `polypharmacy-senior` | `patient-demo-polypharmacy` | Sulfonamide allergy → flags trimethoprim-sulfamethoxazole. CKD stage 3 with an eGFR lab. 14 active meds |
| `messy-coding` | `patient-demo-messy-coding` | Allergy recorded as free text ("Amoxicillin allergy") with no code → tests the text matcher |
| `sparse-record` | `patient-demo-sparse` | No allergies, no labs → tests the empty state |
| `source-unavailable` | `patient-demo-source-unavailable` | Source error → tests that the report still renders |

### Backend changes

The report schema (`PredictionReport`) is `extra="forbid"` and part of the data
contract. **Leave it alone.** Add a separate endpoint and let the frontend join the
two responses.

1. **`configs/allergy_classes.csv`** (new). Maps allergen text to the drugs it flags.
   It is display-only, like `configs/drug_reasons.csv`.

   ```csv
   # Allergy flags (display only; never a model input and never changes a call).
   # An allergy matches when its substance or any code display contains match_any (case-insensitive).
   # relation: same_class = the allergen's own class; related_class = shares the beta-lactam ring, clinician to judge.
   match_any,drug_class,relation,drugs
   penicillin|amoxicillin|ampicillin,penicillins,same_class,penicillin|ampicillin|amoxicillin|amoxicillin-clavulanic-acid|ampicillin-sulbactam|oxacillin|piperacillin-tazobactam
   penicillin|amoxicillin|ampicillin,beta-lactams,related_class,cefazolin|cefuroxime|cefoxitin|ceftriaxone|cefotaxime|ceftazidime|cefepime|ertapenem|imipenem|meropenem|doripenem
   sulfonamide|sulfa|sulfamethoxazole,sulfonamides,same_class,trimethoprim-sulfamethoxazole
   cephalosporin|ceftriaxone|cefazolin,cephalosporins,same_class,cefazolin|cefuroxime|cefoxitin|ceftriaxone|cefotaxime|ceftazidime|cefepime|ceftaroline|ceftobiprole
   vancomycin,glycopeptides,same_class,vancomycin|teicoplanin
   fluoroquinolone|ciprofloxacin|levofloxacin,fluoroquinolones,same_class,ciprofloxacin|levofloxacin|moxifloxacin
   ```

   Ask a clinician on the team or a mentor to check these rows before the demo. Label
   `related_class` in the UI as "related class — clinician to assess". Do not word it
   as a contraindication.

2. **`src/genome2mic/integrations/finchnode_client.py`** (new). A thin async client
   with `list_scenarios()` and `get_records(subject, categories)`. Read the base URL
   from `Settings.finchnode_base_url`, default
   `https://api.finchnode.com/demo/v1`. Use a 5 s timeout. On any error, return
   `None` so the report still renders.

3. **`src/genome2mic/api/schemas/patient_context.py`** (new):

   ```python
   class AllergyFlag(BaseModel):
       drug: str                      # matches DrugPrediction.drug
       substance: str                 # as written in the record
       drug_class: str
       relation: Literal["same_class", "related_class"]
       severity: str | None

   class LabValue(BaseModel):
       name: str
       value: float | str | None
       unit: str | None
       date: str | None
       interpretation: str | None

   class PatientContext(BaseModel):
       subject: str
       display_name: str              # "(synthetic)" suffix kept from FinchNode
       synthetic: bool
       allergies: list[str]
       allergy_flags: list[AllergyFlag]
       kidney_labs: list[LabValue]    # latest creatinine and eGFR only, matched on LOINC or name
       active_medications: list[str]
       source_disclaimer: str         # FinchNode meta.disclaimer, shown verbatim
       disclaimer: str = DISCLAIMER
   ```

4. **`src/genome2mic/api/routers/patients.py`** (new). Mount it under `/v1` with the
   same `require_auth` dependency as the other routers:
   - `GET /v1/patients` → the record scenarios as `{subject, display_name, title}`.
     Leave out `behavior`/`session` scenarios, except `source-unavailable` if you
     want to demo the error path.
   - `GET /v1/patients/{subject}/context` → `PatientContext`.

5. **`src/genome2mic/predict/allergy_flags.py`** (new). A pure function
   `flag_allergies(allergies, rules) -> list[AllergyFlag]`. Match on the lowercased
   `substance` plus every `codes[].display`. Skip "No known allergy" entries. Write
   the test first (`tests/predict/test_allergy_flags.py`). Cover these cases:
   penicillin → 7 same-class flags; free-text "Amoxicillin allergy" → penicillins;
   "No known allergy" → nothing; sulfonamide → trimethoprim-sulfamethoxazole only.

6. Add `httpx` to the main `dependencies` in `pyproject.toml`. It is already in the
   `test` extra. Why: it is the async HTTP client for the FinchNode, ElevenLabs and
   agent calls, and it avoids adding a second HTTP library.

### Frontend changes

- A "Patient (synthetic)" dropdown filled from `GET /v1/patients`. Make "None" the
  first option. The report must work with no patient selected.
- A context card: name with the synthetic badge, allergies, latest creatinine/eGFR with
  date and reference range, active medications. Put FinchNode's `source_disclaimer`
  in small print.
- Drug rows: an amber pill such as "Allergy on file: Penicillin (same class)" or a grey
  pill for `related_class`. The call colour, `p_active` and rank order stay the same.
- Kidney labs are shown as context only. Do not add per-drug renal text. That moves
  into dosing, and `CLAUDE.md` forbids it.

### Demo beat (30 s)

Upload a KPNEU genome and pick `baseline-adult`. Piperacillin-tazobactam is likely
active but now carries "Penicillin allergy on file". Say: "The genome says the bug
is susceptible. The record says this patient has an allergy. We put both in front of
the clinician, and the clinician decides."

### Submission

- Devpost: name FinchNode under tools used. Show the `/v1/patients/{subject}/context`
  call in the video.
- Use only the synthetic demo data. Do not connect a real patient record.

---

## 2. Neon — persistent runs, auth, and branches

### What the sponsor wants

A working app that uses Neon's backend tools (Postgres, Auth, branching) as fully as
it reasonably can.

### What we build

Swap `InMemoryJobStore` for a Postgres job store on Neon. Jobs and reports then
survive restarts, and the frontend can show a **run history**. Optionally, add Neon
Auth behind the existing `require_auth` placeholder.

Genomes are **not** stored. `JobRunner` already deletes the upload when a job ends.
Only job state and the validated report JSON go into the database.

### Schema

Write it as `infra/neon/001_init.sql`:

```sql
create table jobs (
    job_id      text primary key,
    sample_id   text not null,
    status      text not null check (status in ('queued','running','done','failed')),
    error       text,
    patient_subject text,                 -- FinchNode subject chosen in the UI, nullable
    created_by  text,                     -- Neon Auth user id, nullable until auth lands
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table reports (
    job_id        text primary key references jobs(job_id) on delete cascade,
    species       text,
    model_version text not null,
    report        jsonb not null          -- PredictionReport.model_dump(mode="json")
);

create index jobs_created_at_idx on jobs (created_at desc);
```

`patient_subject` links a run to a synthetic patient for the history view. It is
never read by the predictor.

### Backend changes

1. **`src/genome2mic/api/services/job_store.py`**. Pull out a `JobStore` `Protocol`
   with the methods `InMemoryJobStore` already has (`create`, `get`, `get_result`,
   `mark_running`, `mark_done`, `mark_failed`). Add `list_recent(limit)`. Update the
   type hints in `deps.py` and `job_runner.py` to use the Protocol.
2. **`src/genome2mic/api/services/postgres_job_store.py`** (new). Implement it with
   `asyncpg` and a pool created in `lifespan`. `get_result` re-validates the stored
   JSON with `PredictionReport.model_validate`, so a bad row fails loudly.
3. **`config.py`**: add `database_url: str | None = None` (`G2M_DATABASE_URL`).
   `main.py` picks `PostgresJobStore` when it is set and `InMemoryJobStore` when it
   is not. Tests and offline dev keep working with no database.
4. **`routers/jobs.py`**: add `GET /v1/jobs?limit=20`, which returns
   `JobState[]`, newest first.
5. Add **`asyncpg`** to `pyproject.toml` under a new `db` extra. Why: it is the async
   Postgres driver, and it fits FastAPI's async handlers without pulling in an ORM.
6. Use Neon's **pooled** connection string (host contains `-pooler`) and keep
   `sslmode=require`. Store it in `.env` (`G2M_DATABASE_URL=...`). Never commit it.
   `/secret` is already git-ignored if you want to keep it there.
7. Tests: `tests/api/test_postgres_job_store.py`, skipped unless
   `G2M_TEST_DATABASE_URL` is set. Point that at a Neon **branch**, not `main`.

### Using more of Neon

These pieces make the "fullest use" case without much extra work:

- **Branching.** `main` holds the demo data. Create one branch per developer or PR for
  tests (`neonctl branches create --name test-$USER`). Show this in the video: a
  migration tested on a branch, then applied to `main`.
- **Neon Auth (optional, ~1–2 h).** Put the body of `require_auth` in
  `deps.py` to work: verify the bearer JWT against Neon Auth's JWKS URL and return the
  user id, which goes into `jobs.created_by`. The run history then filters by user.
  Check the current Neon Auth docs for the JWKS URL and the frontend SDK before you
  start. This plan has not verified them.
- **Read-only demo role.** Create a role with `SELECT` on `jobs`/`reports` so judges
  can see the run history safely.

### Frontend changes

- A "Recent runs" list from `GET /v1/jobs`. Clicking a run loads
  `/v1/jobs/{id}/result`.
- Add a sign-in button only if Neon Auth lands.

### Submission

Devpost: name Neon. In the video, show the Neon console with the `jobs`/`reports`
tables and a branch. Then restart the API and show the history still there.

---

## 3. Fetch.ai — Breakpoint agent on ASI:One

### What the sponsor wants

An agent registered on Agentverse, discoverable through ASI:One, that implements the
Agent Chat Protocol and turns a request into an outcome. They mark down thin
wrappers. Judging: functionality 25%, use of Fetch.ai tech 20%, innovation 20%,
impact 20%, UX 15%. Bonus points for multi-agent collaboration, interactive cards
and good error handling.

### What we build

Two agents that work together:

1. **Breakpoint Orchestrator agent** (user-facing, discoverable on ASI:One).
   - Accepts a genome reference in chat: a BV-BRC genome id (e.g. `573.13159`), an
     NCBI biosample, or a demo genome name.
   - Fetches the FASTA, submits it to our API (`POST /v1/predict`), polls
     `/v1/jobs/{id}`, sends a "running…" update, then replies with the ranked report.
   - If the user names a synthetic patient ("…for patient baseline-adult"), it asks
     the Patient Context agent for flags and merges them into the reply.
2. **Patient Context agent**. Wraps `GET /v1/patients/{subject}/context` (section 1)
   and replies with allergy flags only.

The work is a chain of steps: resolve the genome, download, upload, wait, fetch the
report, combine it with patient context, format. Two agents also earn the
multi-agent bonus. That makes it more than a wrapper.

### Layout

```
agents/
  README.md                 # required badges + agent names and addresses
  requirements.txt          # uagents, uagents-core, httpx
  orchestrator_agent.py
  patient_context_agent.py
  breakpoint_client.py      # calls our FastAPI; no model code here
  formatting.py             # report -> chat text; always appends DISCLAIMER
```

Keep `agents/` out of the `genome2mic` package. It is a client of the API, like the
frontend, so it needs neither the model nor the `[model]` extras.

### Orchestrator skeleton

```python
from datetime import datetime, timezone
from uuid import uuid4

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement, ChatMessage, EndSessionContent, TextContent, chat_protocol_spec,
)

from breakpoint_client import run_prediction, resolve_genome
from formatting import format_report

agent = Agent(name="breakpoint-amr", seed=SEED_FROM_ENV, port=8001, mailbox=True)
chat = Protocol(spec=chat_protocol_spec)


def reply(text: str, end: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(), content=content)


@chat.on_message(ChatMessage)
async def on_chat(ctx: Context, sender: str, msg: ChatMessage) -> None:
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))
    text = " ".join(c.text for c in msg.content if isinstance(c, TextContent))
    genome = await resolve_genome(text)          # BV-BRC id / biosample / demo name
    if genome is None:
        await ctx.send(sender, reply("Send a BV-BRC genome id like 573.13159."))
        return
    await ctx.send(sender, reply(f"Running {genome.sample_id}. This takes a few minutes…"))
    report = await run_prediction(genome)        # POST /v1/predict, then poll
    await ctx.send(sender, reply(format_report(report), end=True))


@chat.on_message(ChatAcknowledgement)
async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement) -> None:
    pass


agent.include(chat, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
```

Check the import paths against the current hackpack before you build. The uAgents
API changes between releases.

### Details that matter

- **Getting a FASTA from an id.** For BV-BRC:
  `https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,<id>)&http_accept=application/dna+fasta&limit(25000)`.
  Cache downloads on disk. For the live demo, prefer the ids in
  `demo_genomes.zip/manifest.csv` and keep their FASTAs local, so the demo doesn't
  depend on BV-BRC being up.
- **Our API must be reachable** from wherever the agent runs. Either deploy it (the
  Docker image and `infra/aws` already exist) or run a tunnel
  (`cloudflared tunnel --url http://127.0.0.1:8000`). Set `G2M_API_URL` for the agent.
- **Polling.** Poll `/v1/jobs/{id}` every 5 s, for up to 10 min. On `failed`, send the
  job's `error` string, which already quotes the request id.
- **Reply format.** Report the drugs at risk first and the likely-active list second,
  in line with "report VME first". For example:

  ```
  Klebsiella pneumoniae · sample 573.13159 · model all5_run1
  Likely inactive: ertapenem (carbapenemase: blaKPC-3), ceftriaxone
  Uncertain — wait for lab: ciprofloxacin
  Likely active, narrowest first: amikacin (2 steps below S), …
  These are predictions of in-vitro susceptibility, not prescribing advice. …
  ```

  `format_report` appends `DISCLAIMER` unconditionally. Add a unit test for that.
- **Out-of-scope inputs.** If the species is not covered or `in_range` is false, say
  so plainly and give no list. If someone asks "what should I give the patient?",
  reply that the system ranks in-vitro activity and the choice stays with the
  clinician.
- **Interactive cards (bonus).** If time allows, send the ranked list as an ASI card
  with one row per drug and its call colour. Do this last.

### Registering and submitting

1. `pip install uagents uagents-core httpx` in a separate venv, not `genome2mic`.
2. Run both agents with `mailbox=True`. Open the Agentverse inspector link that each
   prints, and connect the mailbox.
3. In Agentverse, give each agent a clear description and keywords ("antibiotic
   resistance", "MIC", "bacterial genome", "susceptibility"), so ASI:One routes to it.
4. Test the full flow inside an ASI:One chat. The hackpack requires the whole workflow
   to complete there.
5. `agents/README.md` must carry the badges and both agent names and addresses:

   ```
   ![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
   ![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)
   ```

6. Record a 3–5 min demo video. Register through the MHacks ASI:One Submission Agent
   **and** submit on Devpost. Both are required.

---

## 4. ElevenLabs — spoken report summary

### What the sponsor wants

The best use of ElevenLabs.

### What we build

A "Listen" button on the report that plays a 20–30 s spoken summary. The pitch:
a clinician on the ward can hear the result hands-free.

**The script is built from the report by a fixed template, not by an LLM.** A
template can't make up a drug or drop the disclaimer.

### Backend changes

1. **`src/genome2mic/predict/speech_script.py`** (new). A pure function
   `build_script(report: PredictionReport) -> str`. Write the test first. The
   template:
   - "Predicted results for sample {sample_id}, {species name}."
   - "Likely inactive: {drugs}." Name the override reason if there is one
     ("carbapenemase detected").
   - "Uncertain, wait for the lab: {drugs}."
   - "Likely active, narrowest first: {first 3 of ranked_active}."
   - End with the full `DISCLAIMER`.
   - Never read MIC numbers as amounts ("eight milligrams…" sounds like a dose). Say
     "below the breakpoint" instead.
   - If the species is not covered or `in_range` is false, use one sentence: "This
     genome is outside the model's range; no calls were made," then the disclaimer.
   - Spell drug names out for speech: "piperacillin tazobactam", "trimethoprim
     sulfamethoxazole".
2. **`src/genome2mic/integrations/elevenlabs_client.py`** (new).
   `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}` with header
   `xi-api-key`, body `{"text": ..., "model_id": ...}`. It returns `audio/mpeg`.
   Settings: `elevenlabs_api_key`, `elevenlabs_voice_id`, `elevenlabs_model_id`
   (`G2M_` env vars, empty by default). Check the current model ids in the
   ElevenLabs docs.
3. **`routers/jobs.py`**: add `GET /v1/jobs/{job_id}/summary.mp3`.
   - Return 409 if the job isn't done and 503 if no key is configured.
   - Cache the audio per `job_id`: in memory, or in a Neon `bytea` column if section 2
     landed. Replays then cost no credits.
   - The API key stays on the server. The browser never sees it.
4. Also expose `GET /v1/jobs/{job_id}/summary.txt`, which returns the same script.
   The UI shows it as a transcript under the player, for accessibility and as a
   fallback when audio fails.

### Frontend

A "Listen" button and an `<audio>` element pointed at `summary.mp3`, with the
transcript shown below.

### Stretch (only if everything else is done)

Use the ElevenLabs Conversational AI agent with the report JSON as its knowledge, so a
user can ask "why is meropenem inactive?" Lock its system prompt to explaining the
report only. It must refuse dosing and treatment questions and always end with the
disclaimer. This has a real risk of off-script answers. Skip it unless you have time
to test it hard.

### Submission

Devpost: name ElevenLabs. Play the summary in the demo video.

---

## 5. Sustainability (MHacks track) — stewardship framing

### The angle

Antimicrobial resistance is a sustainability problem: a shared resource, antibiotic
efficacy, gets used up by overuse. Breakpoint already ranks likely-active drugs
**narrowest first** by WHO AWaRe tier (`configs/drugs.yaml`, `tier`). That puts
Access drugs ahead of Reserve drugs, which is what stewardship programmes push for.

### Small build (~1 h)

- Show the AWaRe tier as a badge (Access / Watch / Reserve) on each drug row. The
  data is already in `drugs.yaml`. Expose it through the proposed `GET /v1/drugs`
  (`INTEGRATION.md` §1).
- Add one line on the report: "{n} Access-tier drugs predicted active." Do not claim
  outcomes we haven't measured. No "reduces resistance by X%".

### Pitch line

"Faster answers mean less time on blind broad-spectrum cover, and the ranking puts
the narrowest likely-active drug first."

Enter this only as a secondary track. AI is the primary track.

---

## Appendix A — Notability (no code)

1. Get Notability Pro for at least one teammate.
2. Use it for real work: the pipeline diagram, report-screen wireframes, and notes from
   the planning meeting.
3. Take at least 2 screenshots of that work.
4. Devpost: tag "Notability" and add a short note on how it was used, with the
   screenshots.

## Appendix B — Order of work and owners

| Step | Depends on | Suggested owner |
| ---- | ---------- | --------------- |
| FinchNode endpoint + allergy flags | — | Backend |
| FinchNode panel + drug-row pills | endpoint | Frontend |
| Neon job store + `GET /v1/jobs` | — | Backend |
| Run history UI | Neon store | Frontend |
| Public API URL (deploy or tunnel) | — | Infra |
| Fetch.ai agents | public URL; FinchNode endpoint for the patient step | Whoever is free |
| ElevenLabs summary | — | Backend + Frontend |
| AWaRe badges + `GET /v1/drugs` | — | Frontend |
| Notability screenshots | — | Anyone, during planning |

Before you merge any of these, check:

- [ ] The predictor's output is identical with and without the feature: same calls,
      same `ranked_active`.
- [ ] The disclaimer shows on every new surface: panel, audio, agent reply.
- [ ] Nothing mentions a dose or tells the user to give a drug.
- [ ] No secret (Neon URL, ElevenLabs key, agent seed) is committed.
- [ ] Each new dependency (`httpx`, `asyncpg`, `uagents`) is noted in the PR with the
      reason, as `CLAUDE.md` requires.
