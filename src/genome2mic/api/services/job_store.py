"""In-memory job state."""

import uuid
from datetime import UTC, datetime

from genome2mic.api.schemas.job_state import JobState
from genome2mic.api.schemas.job_status import JobStatus
from genome2mic.api.schemas.prediction_report import PredictionReport


class InMemoryJobStore:
    """Keeps job state and results in process memory.

    Lost on restart and never evicted. Replace with Redis or a database when a worker queue lands.
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

    async def get(self, job_id: str) -> JobState | None:
        return self._states.get(job_id)

    async def get_result(self, job_id: str) -> PredictionReport | None:
        return self._results.get(job_id)

    async def mark_running(self, job_id: str) -> None:
        self._set_status(job_id, JobStatus.RUNNING)

    async def mark_done(self, job_id: str, report: PredictionReport) -> None:
        self._results[job_id] = report
        self._set_status(job_id, JobStatus.DONE)

    async def mark_failed(self, job_id: str, error: str) -> None:
        self._set_status(job_id, JobStatus.FAILED, error)

    def _set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        current = self._states[job_id]
        self._states[job_id] = current.model_copy(
            update={"status": status, "error": error, "updated_at": datetime.now(UTC)}
        )
