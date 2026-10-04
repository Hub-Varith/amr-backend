"""A corrupt or unloadable model bundle leaves the API up (``/health`` 200) but not ready (``/ready`` 503)."""

import json
import logging
import shutil
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from genome2mic.api.config import Settings
from genome2mic.api.main import create_app
from genome2mic.api.services.model_registry import ModelRegistry
from genome2mic.predict.pipeline import PredictionPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def real_settings(settings: Settings, tmp_path: Path) -> Settings:
    """Settings with the real configs and a FAKE models dir holding a valid manifest."""
    configs_dir = tmp_path / "real_configs"
    shutil.copytree(REPO_ROOT / "configs", configs_dir)
    models_dir = tmp_path / "corrupt_models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text(
        json.dumps({"model_version": "FAKE-CORRUPT-0.1", "run_id": "fakecorrupt", "species": {}, "synthetic": True})
    )
    return settings.model_copy(update={"configs_dir": configs_dir, "models_dir": models_dir})


def _corrupt_reference_sketches(settings: Settings) -> None:
    (settings.models_dir / "reference_sketches.npz").write_bytes(b"FAKE: not an npz archive")


def test_corrupt_sketch_file_leaves_registry_not_ready(real_settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    _corrupt_reference_sketches(real_settings)
    registry = ModelRegistry(real_settings)
    with caplog.at_level(logging.WARNING):
        registry.load()
    assert registry.configs_parsed is True
    assert registry.models_loaded is False
    assert registry.is_ready is False
    assert registry.drugs_by_species == {}
    assert any("Models not loaded" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("FAKE: bad npz"),
        KeyError("params"),
        RuntimeError("FAKE: XGBoostError-like failure"),
        OSError("FAKE: truncated file"),
        MemoryError("FAKE: bundle too large"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_any_loader_exception_leaves_registry_not_ready(
    real_settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    def explode(self: PredictionPipeline) -> None:
        raise error

    monkeypatch.setattr(PredictionPipeline, "load", explode)
    registry = ModelRegistry(real_settings)
    with caplog.at_level(logging.ERROR):
        registry.load()
    assert registry.models_loaded is False
    assert registry.is_ready is False
    logged = [r for r in caplog.records if r.levelno >= logging.ERROR and "Models not loaded" in r.getMessage()]
    assert logged, "the load failure must be logged at ERROR"
    assert logged[0].exc_info is not None


def test_keyboard_interrupt_is_not_swallowed(real_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(self: PredictionPipeline) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(PredictionPipeline, "load", interrupt)
    with pytest.raises(KeyboardInterrupt):
        ModelRegistry(real_settings).load()


async def test_app_starts_with_a_corrupt_bundle_and_reports_not_ready(
    real_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(self: PredictionPipeline) -> None:
        raise RuntimeError("FAKE: corrupt model.ubj")

    monkeypatch.setattr(PredictionPipeline, "load", explode)
    app = create_app(real_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            health = await client.get("/health")
            ready = await client.get("/ready")
    assert health.status_code == 200
    assert ready.status_code == 503
    assert ready.json() == {"ready": False, "configs_parsed": True, "models_loaded": False}


async def test_app_starts_with_a_corrupt_sketch_file(real_settings: Settings) -> None:
    _corrupt_reference_sketches(real_settings)
    app = create_app(real_settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            ready = await client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["models_loaded"] is False
