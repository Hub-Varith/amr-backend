"""Job state storage: the interface and the in-memory store."""

import uuid
from datetime import UTC, datetime
from typing import Protocol

from genome2mic.api.schemas.job_state import JobState
from genome2mic.api.schemas.job_status import JobStatus
from genome2mic.api.schemas.prediction_report import PredictionReport

INTERRUPTED_ERROR = "Interrupted by an API restart before it finished. Upload the genome again."


class JobStore(Protocol):
    """Where job state and reports live. mark_* raise KeyError for an unknown job_id."""

    async def create(self, sample_id: str) -> JobState: ...

    async def get(self, job_id: str) -> JobState | None: ...

    async def get_result(self, job_id: str) -> PredictionReport | None: ...

    async def list_recent(self, limit: int) -> list[JobState]:
        """Newest first."""
        ...

    async def mark_running(self, job_id: str) -> None: ...

    async def mark_done(self, job_id: str, report: PredictionReport) -> None: ...

    async def mark_failed(self, job_id: str, error: str) -> None: ...

    async def fail_interrupted(self) -> int:
        """Mark every queued or running job failed. Called at startup: jobs run in-process, so none survive a
        restart. Returns the number marked."""
        ...


class InMemoryJobStore:
    """Keeps job state and results in process memory.

    Lost on restart and never evicted. PostgresJobStore is used instead when G2M_DATABASE_URL is set.
    """

    def __init__(self) -> None:
        self._states: dict[str, JobState] = {}
        self._results: dict[str, PredictionReport] = {}

    async def create(self, sample_id: str) -> JobState:
        now = datetime.now(UTC)
        state = JobState(
            job_id=uuid.uuid4().hex,
            sample_id=sample_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        self._states[state.job_id] = state
        return state

    def add(self, state: JobState) -> None:
        """Track a job created elsewhere (ResilientJobStore mirrors database jobs here)."""
        self._states[state.job_id] = state

    async def get(self, job_id: str) -> JobState | None:
        return self._states.get(job_id)

    async def get_result(self, job_id: str) -> PredictionReport | None:
        return self._results.get(job_id)

    async def list_recent(self, limit: int) -> list[JobState]:
        return sorted(self._states.values(), key=lambda state: state.created_at, reverse=True)[:limit]

    async def mark_running(self, job_id: str) -> None:
        self._set_status(job_id, JobStatus.RUNNING)

    async def mark_done(self, job_id: str, report: PredictionReport) -> None:
        self._results[job_id] = report
        self._set_status(job_id, JobStatus.DONE)

    async def mark_failed(self, job_id: str, error: str) -> None:
        self._set_status(job_id, JobStatus.FAILED, error)

    async def fail_interrupted(self) -> int:
        unfinished = [s.job_id for s in self._states.values() if s.status in (JobStatus.QUEUED, JobStatus.RUNNING)]
        for job_id in unfinished:
            self._set_status(job_id, JobStatus.FAILED, INTERRUPTED_ERROR)
        return len(unfinished)

    def _set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        current = self._states[job_id]
        self._states[job_id] = current.model_copy(
            update={"status": status, "error": error, "updated_at": datetime.now(UTC)}
        )
