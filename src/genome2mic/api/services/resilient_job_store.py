"""Database job store that keeps the API working when the database is not."""

import logging
from collections.abc import Awaitable

from genome2mic.api.schemas.job_state import JobState
from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.api.services.job_store import InMemoryJobStore, JobStore

logger = logging.getLogger(__name__)


class ResilientJobStore:
    """Every job lives in memory; the database is written best-effort.

    Jobs of this process are always served from memory, so uploads, polling and results keep working when the
    database drops. A failed database write only means that job is not tracked (no run history across restarts).
    The database is read only for older jobs, which need it.
    """

    def __init__(self, database: JobStore, memory: InMemoryJobStore | None = None) -> None:
        self.database = database
        self.memory = memory or InMemoryJobStore()

    async def create(self, sample_id: str) -> JobState:
        try:
            state = await self.database.create(sample_id)
        except Exception as error:
            _warn("create", error)
            return await self.memory.create(sample_id)
        self.memory.add(state)
        return state

    async def get(self, job_id: str) -> JobState | None:
        state = await self.memory.get(job_id)
        if state is not None:
            return state
        try:
            return await self.database.get(job_id)
        except Exception as error:
            _warn("get", error, job_id)
            return None

    async def get_result(self, job_id: str) -> PredictionReport | None:
        if await self.memory.get(job_id) is not None:
            return await self.memory.get_result(job_id)
        try:
            return await self.database.get_result(job_id)
        except Exception as error:
            _warn("get_result", error, job_id)
            return None

    async def list_recent(self, limit: int) -> list[JobState]:
        merged = {state.job_id: state for state in await self.memory.list_recent(limit)}
        try:
            for state in await self.database.list_recent(limit):
                merged.setdefault(state.job_id, state)  # memory holds the newer state of this process's jobs
        except Exception as error:
            _warn("list_recent", error)
        return sorted(merged.values(), key=lambda state: state.created_at, reverse=True)[:limit]

    async def mark_running(self, job_id: str) -> None:
        await self.memory.mark_running(job_id)
        await self._write(self.database.mark_running(job_id), "mark_running", job_id)

    async def mark_done(self, job_id: str, report: PredictionReport) -> None:
        await self.memory.mark_done(job_id, report)
        await self._write(self.database.mark_done(job_id, report), "mark_done", job_id)

    async def mark_failed(self, job_id: str, error: str) -> None:
        await self.memory.mark_failed(job_id, error)
        await self._write(self.database.mark_failed(job_id, error), "mark_failed", job_id)

    async def fail_interrupted(self) -> int:
        return await self.database.fail_interrupted()

    async def close(self) -> None:
        close = getattr(self.database, "close", None)
        if close is not None:
            try:
                await close()
            except Exception as error:
                _warn("close", error)

    @staticmethod
    async def _write(call: Awaitable[None], operation: str, job_id: str) -> None:
        try:
            await call
        except KeyError:
            pass  # created while the database was down, so it was never tracked there
        except Exception as error:
            _warn(operation, error, job_id)


def _warn(operation: str, error: Exception, job_id: str | None = None) -> None:
    # The error type only: asyncpg messages can quote connection details.
    logger.warning(
        "Job database unavailable; continuing without tracking",
        extra={"operation": operation, "job_id": job_id, "error_type": type(error).__name__},
    )
