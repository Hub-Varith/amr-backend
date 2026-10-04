-- Job store for the API (docs/SPONSOR_TRACKS.md, section 2).
-- Holds job state and the validated report JSON only. Genomes are never stored.
-- Apply to a test branch first, then to main.

create table if not exists jobs (
    job_id          text primary key,
    sample_id       text not null,
    status          text not null check (status in ('queued', 'running', 'done', 'failed')),
    error           text,
    patient_subject text,                 -- FinchNode subject chosen in the UI; never read by the predictor
    created_by      text,                 -- Neon Auth user id; null until auth lands
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create table if not exists reports (
    job_id        text primary key references jobs (job_id) on delete cascade,
    species       text,
    model_version text not null,
    report        jsonb not null          -- PredictionReport.model_dump(mode="json")
);

create index if not exists jobs_created_at_idx on jobs (created_at desc);
