"""Full job lifecycle against the fake predictor."""

from fastapi import FastAPI
from httpx import AsyncClient

from genome2mic.api.config import Settings
from tests.api.conftest import FAKE_FASTA
from tests.api.fakes.fake_predictor import FAKE_REASON, FakePredictor


async def test_job_runs_to_done(ready_client: AsyncClient, settings: Settings, fake_predictor: FakePredictor) -> None:
    submit = await ready_client.post(
        "/v1/predict",
        files={"file": ("genome.fasta", FAKE_FASTA)},
        data={"sample_id": "FAKE-TEST-ONLY"},
    )

    assert submit.status_code == 202
    accepted = submit.json()
    assert accepted["status"] == "queued"
    assert accepted["status_url"].endswith(f"/v1/jobs/{accepted['job_id']}")

    # httpx's ASGI transport returns after background tasks finish, so the job is done here.
    status = await ready_client.get(f"/v1/jobs/{accepted['job_id']}")
    assert status.status_code == 200
    assert status.json()["status"] == "done"
    assert status.json()["error"] is None

    result = await ready_client.get(f"/v1/jobs/{accepted['job_id']}/result")
    assert result.status_code == 200
    report = result.json()
    assert report["sample_id"] == "FAKE-TEST-ONLY"
    assert report["predictions"][0]["reasons"] == [FAKE_REASON]

    assert len(fake_predictor.seen_paths) == 1
    assert not fake_predictor.seen_paths[0].exists()
    assert list(settings.upload_dir.glob("*")) == []


async def test_sample_id_defaults_to_file_stem(ready_client: AsyncClient) -> None:
    submit = await ready_client.post("/v1/predict", files={"file": ("FAKE_SAMPLE.fa", FAKE_FASTA)})
    status = await ready_client.get(f"/v1/jobs/{submit.json()['job_id']}")

    assert status.json()["sample_id"] == "FAKE_SAMPLE"


async def test_failed_job_reports_failure(
    ready_client: AsyncClient, settings: Settings, fake_predictor: FakePredictor
) -> None:
    fake_predictor.should_fail = True

    submit = await ready_client.post("/v1/predict", files={"file": ("genome.fasta", FAKE_FASTA)})
    job_id = submit.json()["job_id"]

    status = await ready_client.get(f"/v1/jobs/{job_id}")
    assert status.json()["status"] == "failed"
    assert submit.headers["X-Request-ID"] in status.json()["error"]
    assert "FAKE failure" not in status.json()["error"]

    result = await ready_client.get(f"/v1/jobs/{job_id}/result")
    assert result.status_code == 409
    assert result.headers["content-type"] == "application/problem+json"

    assert list(settings.upload_dir.glob("*")) == []


async def test_result_is_409_while_queued(ready_app: FastAPI, ready_client: AsyncClient) -> None:
    job = await ready_app.state.job_store.create("FAKE-TEST-ONLY")

    response = await ready_client.get(f"/v1/jobs/{job.job_id}/result")

    assert response.status_code == 409


async def test_unknown_job_is_404(ready_client: AsyncClient) -> None:
    status = await ready_client.get("/v1/jobs/does-not-exist")
    result = await ready_client.get("/v1/jobs/does-not-exist/result")

    assert status.status_code == 404
    assert result.status_code == 404
    assert status.headers["content-type"] == "application/problem+json"


async def test_predict_is_503_when_not_ready(client: AsyncClient, settings: Settings) -> None:
    response = await client.post("/v1/predict", files={"file": ("genome.fasta", FAKE_FASTA)})

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert list(settings.upload_dir.glob("*")) == []
