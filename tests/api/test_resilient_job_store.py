"""The API keeps working when the job database is down: jobs run, they are just not tracked."""

from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import SecretStr

from genome2mic.api.config import Settings
from genome2mic.api.main import open_job_store
from genome2mic.api.schemas.job_status import JobStatus
from genome2mic.api.services.job_store import InMemoryJobStore
from genome2mic.api.services.resilient_job_store import ResilientJobStore
from tests.api.conftest import FAKE_FASTA
from tests.api.fakes.fake_predictor import fake_report


class DownDatabase:
    """FAKE database whose every call fails, like an unreachable Neon."""

    def __getattr__(self, name: str):
        async def fail(*args, **kwargs):
            raise ConnectionError("FAKE database down")

        return fail


class DropsAfterCreate(InMemoryJobStore):
    """FAKE database that accepts create, then goes down."""

    async def mark_running(self, job_id: str) -> None:
        raise ConnectionError("FAKE database down")

    async def mark_done(self, job_id, report) -> None:
        raise ConnectionError("FAKE database down")

    async def get(self, job_id: str):
        raise ConnectionError("FAKE database down")


async def test_job_runs_when_database_is_down(ready_app: FastAPI, ready_client: AsyncClient) -> None:
    ready_app.state.job_store = ResilientJobStore(DownDatabase())

    submit = await ready_client.post("/v1/predict", files={"file": ("genome.fasta", FAKE_FASTA)})
    assert submit.status_code == 202
    job_id = submit.json()["job_id"]

    status = await ready_client.get(f"/v1/jobs/{job_id}")
    result = await ready_client.get(f"/v1/jobs/{job_id}/result")
    history = await ready_client.get("/v1/jobs")

    assert status.json()["status"] == "done"
    assert result.status_code == 200
    assert [job["job_id"] for job in history.json()] == [job_id]


async def test_database_dropping_mid_job_does_not_fail_the_job() -> None:
    database = DropsAfterCreate()
    store = ResilientJobStore(database)
    job = await store.create("FAKE-TEST-ONLY")
    report = fake_report("FAKE-TEST-ONLY")

    await store.mark_running(job.job_id)
    await store.mark_done(job.job_id, report)

    assert (await store.get(job.job_id)).status is JobStatus.DONE
    assert await store.get_result(job.job_id) == report
    assert database._states[job.job_id].status is JobStatus.QUEUED  # its last good write; the rest went untracked


async def test_unknown_job_is_none_when_database_is_down() -> None:
    store = ResilientJobStore(DownDatabase())

    assert await store.get("does-not-exist") is None
    assert await store.get_result("does-not-exist") is None


async def test_history_merges_database_and_memory() -> None:
    database = InMemoryJobStore()
    older = await database.create("FAKE-TEST-OLDER")  # from before a restart
    store = ResilientJobStore(database)
    newer = await store.create("FAKE-TEST-NEWER")

    assert [state.job_id for state in await store.list_recent(10)] == [newer.job_id, older.job_id]


async def test_startup_falls_back_to_memory_when_database_unreachable(settings: Settings) -> None:
    # Port 1 on localhost refuses connections at once, like a blocked or wrong database.
    settings.database_url = SecretStr("postgresql://fake:fake@127.0.0.1:1/fake")

    store = await open_job_store(settings)

    assert isinstance(store, InMemoryJobStore)
