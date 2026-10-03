"""FastAPI app factory. Run with: uvicorn genome2mic.api.main:create_app --factory"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from genome2mic.api.config import Settings
from genome2mic.api.constants import DISCLAIMER, REQUEST_ID_HEADER
from genome2mic.api.deps import require_auth
from genome2mic.api.errors.problem_handlers import register_problem_handlers
from genome2mic.api.log_format.json_formatter import JsonFormatter
from genome2mic.api.middleware.request_id_middleware import RequestIdMiddleware
from genome2mic.api.routers import health, jobs, predict
from genome2mic.api.services.job_store import InMemoryJobStore
from genome2mic.api.services.model_registry import ModelRegistry
from genome2mic.api.services.predictor import Predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info("API starting", extra={"upload_dir": str(settings.upload_dir)})
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(settings)
    registry.load()
    app.state.registry = registry
    app.state.predictor = Predictor(registry.pipeline)
    app.state.job_store = InMemoryJobStore()
    logger.info("API started", extra={"ready": registry.is_ready})
    yield
    logger.info("API stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass settings in tests; otherwise they come from the environment."""
    settings = settings or Settings()
    JsonFormatter.install(settings.log_level)

    app = FastAPI(title="genome2mic API", description=DISCLAIMER, lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    # Added last so it wraps everything, including CORS preflight responses.
    app.add_middleware(RequestIdMiddleware)
    register_problem_handlers(app)

    app.include_router(health.router)
    app.include_router(predict.router, prefix="/v1", dependencies=[Depends(require_auth)])
    app.include_router(jobs.router, prefix="/v1", dependencies=[Depends(require_auth)])
    return app
