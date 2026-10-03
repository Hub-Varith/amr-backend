"""Upload checks on POST /v1/predict."""

from pathlib import Path

from httpx import AsyncClient

from genome2mic.api.config import Settings
from tests.api.conftest import FAKE_FASTA, MAX_UPLOAD_BYTES


def _uploaded_files(settings: Settings) -> list[Path]:
    return list(settings.upload_dir.glob("*"))


async def test_rejects_bad_extension(ready_client: AsyncClient, settings: Settings) -> None:
    response = await ready_client.post("/v1/predict", files={"file": ("genome.txt", FAKE_FASTA)})

    assert response.status_code == 415
    assert response.headers["content-type"] == "application/problem+json"
    assert _uploaded_files(settings) == []


async def test_rejects_oversized_file(ready_client: AsyncClient, settings: Settings) -> None:
    oversized = b">FAKE_TEST_CONTIG\n" + b"A" * (MAX_UPLOAD_BYTES + 1)

    response = await ready_client.post("/v1/predict", files={"file": ("genome.fasta", oversized)})

    assert response.status_code == 413
    assert response.headers["content-type"] == "application/problem+json"
    assert _uploaded_files(settings) == []


async def test_rejects_non_fasta_content(ready_client: AsyncClient, settings: Settings) -> None:
    response = await ready_client.post("/v1/predict", files={"file": ("genome.fa", b"\n\nnot a fasta file\n")})

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert _uploaded_files(settings) == []


async def test_rejects_empty_file(ready_client: AsyncClient) -> None:
    response = await ready_client.post("/v1/predict", files={"file": ("genome.fna", b"")})

    assert response.status_code == 422


async def test_accepts_fasta_after_blank_lines(ready_client: AsyncClient) -> None:
    response = await ready_client.post("/v1/predict", files={"file": ("GENOME.FNA", b"\n  \n" + FAKE_FASTA)})

    assert response.status_code == 202


async def test_missing_file_is_problem_json_with_errors(ready_client: AsyncClient) -> None:
    response = await ready_client.post("/v1/predict", data={"sample_id": "FAKE-TEST-ONLY"})

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["errors"][0]["loc"] == ["body", "file"]
