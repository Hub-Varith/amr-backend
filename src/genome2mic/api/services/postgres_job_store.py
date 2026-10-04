"""Job state in Postgres (Neon). Schema: infra/neon/*.sql, applied with genome2mic.api.services.migrations."""

import uuid

import asyncpg

from genome2mic.api.schemas.job_state import JobState
from genome2mic.api.schemas.job_status import JobStatus
from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.api.services.job_store import INTERRUPTED_ERROR

_STATE_COLUMNS = "job_id, sample_id, status, error, created_at, updated_at"
CONNECT_TIMEOUT_S = 10
COMMAND_TIMEOUT_S = 10


class PostgresJobStore:
    """Keeps job state and the validated report JSON in Postgres. Genomes are never stored.

    Survives restarts. patient_subject and created_by stay null until FinchNode and Neon Auth land.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, database_url: str) -> "PostgresJobStore":
        """Open a pool and check the schema is there, so a missing migration fails at startup, not mid-job."""
        # statement_cache_size=0: Neon's pooled endpoint is PgBouncer in transaction mode, which does not keep
        # asyncpg's named prepared statements across transactions.
        # The timeouts bound how long a request waits on an unreachable database before ResilientJobStore
        # gives up on it.
        pool = await asyncpg.create_pool(
            database_url,
            min_size=1,
            max_size=5,
            statement_cache_size=0,
            timeout=CONNECT_TIMEOUT_S,
            command_timeout=COMMAND_TIMEOUT_S,
        )
        try:
            await pool.execute("select 1 from jobs join reports using (job_id) limit 0")
        except asyncpg.UndefinedTableError as error:
            await pool.close()
            raise RuntimeError(
                "Job tables are missing. Run: python -m genome2mic.api.services.migrations"
            ) from error
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def create(self, sample_id: str) -> JobState:
        row = await self._pool.fetchrow(
            f"insert into jobs (job_id, sample_id, status) values ($1, $2, $3) returning {_STATE_COLUMNS}",
            uuid.uuid4().hex,
            sample_id,
            JobStatus.QUEUED.value,
        )
        return _to_state(row)

    async def get(self, job_id: str) -> JobState | None:
        row = await self._pool.fetchrow(f"select {_STATE_COLUMNS} from jobs where job_id = $1", job_id)
        return _to_state(row) if row else None

    async def get_result(self, job_id: str) -> PredictionReport | None:
        report = await self._pool.fetchval("select report::text from reports where job_id = $1", job_id)
        # Re-validate: a row that no longer fits the schema fails loudly instead of reaching the UI.
        return PredictionReport.model_validate_json(report) if report is not None else None

    async def list_recent(self, limit: int) -> list[JobState]:
        rows = await self._pool.fetch(
            f"select {_STATE_COLUMNS} from jobs order by created_at desc, job_id limit $1", limit
        )
        return [_to_state(row) for row in rows]

    async def mark_running(self, job_id: str) -> None:
        await self._set_status(self._pool, job_id, JobStatus.RUNNING)

    async def mark_done(self, job_id: str, report: PredictionReport) -> None:
        dumped = report.model_dump(mode="json")
        async with self._pool.acquire() as conn, conn.transaction():
            # Status first: it raises KeyError for an unknown job before the report insert hits the foreign key.
            await self._set_status(conn, job_id, JobStatus.DONE)
            await conn.execute(
                "insert into reports (job_id, species, model_version, report) values ($1, $2, $3, $4::jsonb)"
                " on conflict (job_id) do update set species = excluded.species,"
                " model_version = excluded.model_version, report = excluded.report",
                job_id,
                dumped["species"],
                dumped["model_version"],
                report.model_dump_json(),
            )

    async def mark_failed(self, job_id: str, error: str) -> None:
        await self._set_status(self._pool, job_id, JobStatus.FAILED, error)

    async def fail_interrupted(self) -> int:
        result = await self._pool.execute(
            "update jobs set status = $1, error = $2, updated_at = now() where status in ($3, $4)",
            JobStatus.FAILED.value,
            INTERRUPTED_ERROR,
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
        )
        return int(result.split()[-1])  # "UPDATE <n>"

    @staticmethod
    async def _set_status(
        conn: asyncpg.Pool | asyncpg.Connection, job_id: str, status: JobStatus, error: str | None = None
    ) -> None:
        result = await conn.execute(
            "update jobs set status = $2, error = $3, updated_at = now() where job_id = $1",
            job_id,
            status.value,
            error,
        )
        if result == "UPDATE 0":
            raise KeyError(job_id)


def _to_state(row: asyncpg.Record) -> JobState:
    return JobState(
        job_id=row["job_id"],
        sample_id=row["sample_id"],
        status=JobStatus(row["status"]),
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
