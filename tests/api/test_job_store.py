"""Job store contract, run against every store.

The Postgres run is skipped unless G2M_TEST_DATABASE_URL is set (environment or .env). Point it at a Neon
test branch, never production: the tests write rows, and fail_interrupted touches every unfinished job there.
Rows the tests create have sample ids starting FAKE-TEST- and are deleted afterwards.
"""

from collections.abc import AsyncIterator

import pytest
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from genome2mic.api.schemas.job_status import JobStatus
from genome2mic.api.services.job_store import INTERRUPTED_ERROR, InMemoryJobStore, JobStore
from tests.api.fakes.fake_predictor import fake_report

SAMPLE_PREFIX = "FAKE-TEST-"


class _TestDatabase(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="G2M_TEST_", env_file=".env", extra="ignore")

    database_url: SecretStr | None = None


@pytest.fixture(params=["memory", "postgres"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[JobStore]:
    if request.param == "memory":
        yield InMemoryJobStore()
        return

    database_url = _TestDatabase().database_url
    if database_url is None:
        pytest.skip("G2M_TEST_DATABASE_URL not set")
    pytest.importorskip("asyncpg")
    from genome2mic.api.services.migrations import apply_migrations
    from genome2mic.api.services.postgres_job_store import PostgresJobStore

    url = database_url.get_secret_value()
    await apply_migrations(url)
    store = await PostgresJobStore.connect(url)
    try:
        yield store
    finally:
        await store._pool.execute("delete from jobs where sample_id like $1", f"{SAMPLE_PREFIX}%")
        await store.close()


async def test_create_then_get(store: JobStore) -> None:
    created = await store.create(f"{SAMPLE_PREFIX}A")

    fetched = await store.get(created.job_id)

    assert fetched == created
    assert fetched.status is JobStatus.QUEUED
    assert fetched.error is None


async def test_unknown_job(store: JobStore) -> None:
    assert await store.get("does-not-exist") is None
    assert await store.get_result("does-not-exist") is None
    with pytest.raises(KeyError):
        await store.mark_running("does-not-exist")
    with pytest.raises(KeyError):
        await store.mark_done("does-not-exist", fake_report(f"{SAMPLE_PREFIX}X"))


async def test_done_job_round_trips_the_report(store: JobStore) -> None:
    job = await store.create(f"{SAMPLE_PREFIX}B")
    report = fake_report(f"{SAMPLE_PREFIX}B")

    await store.mark_running(job.job_id)
    assert (await store.get(job.job_id)).status is JobStatus.RUNNING
    assert await store.get_result(job.job_id) is None

    await store.mark_done(job.job_id, report)

    done = await store.get(job.job_id)
    assert done.status is JobStatus.DONE
    assert done.updated_at >= job.updated_at
    assert await store.get_result(job.job_id) == report


async def test_failed_job_keeps_its_error(store: JobStore) -> None:
    job = await store.create(f"{SAMPLE_PREFIX}C")

    await store.mark_failed(job.job_id, "FAKE error")

    failed = await store.get(job.job_id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "FAKE error"
    assert await store.get_result(job.job_id) is None


async def test_list_recent_is_newest_first(store: JobStore) -> None:
    jobs = [await store.create(f"{SAMPLE_PREFIX}L{i}") for i in range(3)]

    recent = await store.list_recent(limit=50)
    ours = [state.job_id for state in recent if state.job_id in {job.job_id for job in jobs}]

    assert ours == [job.job_id for job in reversed(jobs)]
    assert len(await store.list_recent(limit=2)) == 2


async def test_fail_interrupted_fails_only_unfinished_jobs(store: JobStore) -> None:
    queued = await store.create(f"{SAMPLE_PREFIX}Q")
    running = await store.create(f"{SAMPLE_PREFIX}R")
    done = await store.create(f"{SAMPLE_PREFIX}D")
    await store.mark_running(running.job_id)
    await store.mark_done(done.job_id, fake_report(f"{SAMPLE_PREFIX}D"))

    count = await store.fail_interrupted()

    assert count >= 2
    for job in (queued, running):
        state = await store.get(job.job_id)
        assert state.status is JobStatus.FAILED
        assert state.error == INTERRUPTED_ERROR
    assert (await store.get(done.job_id)).status is JobStatus.DONE
