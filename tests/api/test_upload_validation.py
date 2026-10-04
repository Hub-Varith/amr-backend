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


async def test_accepts_gzipped_fasta_and_stores_it_unpacked(ready_client: AsyncClient, settings: Settings) -> None:
    import gzip

    response = await ready_client.post("/v1/predict", files={"file": ("genome.fasta.gz", gzip.compress(FAKE_FASTA))})

    assert response.status_code == 202
    for extension in (".fa.gz", ".FNA.GZ"):
        again = await ready_client.post("/v1/predict", files={"file": (f"g{extension}", gzip.compress(FAKE_FASTA))})
        assert again.status_code == 202


async def test_rejects_gzip_of_a_non_fasta_name(ready_client: AsyncClient, settings: Settings) -> None:
    import gzip

    response = await ready_client.post("/v1/predict", files={"file": ("genome.txt.gz", gzip.compress(FAKE_FASTA))})

    assert response.status_code == 415
    assert _uploaded_files(settings) == []


async def test_rejects_corrupt_gzip(ready_client: AsyncClient, settings: Settings) -> None:
    response = await ready_client.post("/v1/predict", files={"file": ("genome.fasta.gz", b"\x1f\x8b\x08not really gzip")})

    assert response.status_code == 422
    assert _uploaded_files(settings) == []


async def test_rejects_gzip_without_fasta_inside(ready_client: AsyncClient, settings: Settings) -> None:
    import gzip

    response = await ready_client.post("/v1/predict", files={"file": ("genome.fa.gz", gzip.compress(b"hello\n"))})

    assert response.status_code == 422
    assert _uploaded_files(settings) == []


async def test_rejects_gzip_that_unpacks_past_the_limit(ready_client: AsyncClient, settings: Settings) -> None:
    import gzip

    small_but_huge = gzip.compress(b">FAKE_TEST_CONTIG\n" + b"A" * (MAX_UPLOAD_BYTES * 20))
    assert len(small_but_huge) < MAX_UPLOAD_BYTES

    response = await ready_client.post("/v1/predict", files={"file": ("genome.fasta.gz", small_but_huge)})

    assert response.status_code == 413
    assert _uploaded_files(settings) == []


def test_default_sample_id_drops_fasta_and_gzip_extensions() -> None:
    from genome2mic.api.routers.predict import default_sample_id

    assert default_sample_id("470.7409.fasta.gz") == "470.7409"
    assert default_sample_id("BC-0142.FNA") == "BC-0142"
    assert default_sample_id("x.fa.gz") == "x"
    assert default_sample_id(None) == "sample"
