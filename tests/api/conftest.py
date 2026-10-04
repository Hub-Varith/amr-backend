"""Shared fixtures for API tests."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from genome2mic.api.config import Settings
from genome2mic.api.deps import get_predictor, get_registry
from genome2mic.api.main import create_app
from tests.api.fakes.fake_predictor import FakePredictor
from tests.api.fakes.fake_registry import FakeRegistry

MAX_UPLOAD_BYTES = 1024
FAKE_FASTA = b">FAKE_TEST_CONTIG\nACGTACGT\n"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "species.yaml").write_text("fake_test_config: true\n")
    (configs_dir / "drugs.yaml").write_text("fake_test_config: true\n")
    return Settings(
        configs_dir=configs_dir,
        models_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=MAX_UPLOAD_BYTES,
        log_level="WARNING",
        # Never a database from .env: API tests use the in-memory store (test_job_store.py covers Postgres).
        database_url=None,
    )


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    """The real app with the real (stubbed) pipeline. Never ready."""
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def ready_app(app: FastAPI, settings: Settings, fake_predictor: FakePredictor) -> FastAPI:
    """The app with a ready fake registry and the fake predictor."""
    registry = FakeRegistry(settings)
    registry.load()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_predictor] = lambda: fake_predictor
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def ready_client(ready_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=ready_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
