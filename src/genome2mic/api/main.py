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
from genome2mic.api.services.job_store import InMemoryJobStore, JobStore
from genome2mic.api.services.model_registry import ModelRegistry
from genome2mic.api.services.resilient_job_store import ResilientJobStore
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
    app.state.job_store = await open_job_store(settings)
    logger.info("API started", extra={"ready": registry.is_ready})
    try:
        yield
    finally:
        close = getattr(app.state.job_store, "close", None)
        if close is not None:
            await close()
        logger.info("API stopped")


async def open_job_store(settings: Settings) -> JobStore:
    """Postgres when G2M_DATABASE_URL is set and reachable, else in memory.

    The database only adds run history, so it never stops the API: if it is unreachable at startup, or drops
    later (ResilientJobStore), predictions still run and are just not tracked.
    """
    if settings.database_url is None:
        logger.info("Job store: in memory (G2M_DATABASE_URL not set)")
        return InMemoryJobStore()
    try:
        # Imported here so the API runs without the db extra (asyncpg) when no database is configured.
        from genome2mic.api.services.postgres_job_store import PostgresJobStore

        database = await PostgresJobStore.connect(settings.database_url.get_secret_value())
    except Exception as error:
        # The error type only: the message can quote connection details.
        logger.warning(
            "Job store: in memory (database unavailable; jobs will not be tracked)",
            extra={"error_type": type(error).__name__},
        )
        return InMemoryJobStore()
    store = ResilientJobStore(database)
    # Jobs run in this process, so any job left queued or running was cut off by a restart.
    try:
        interrupted = await store.fail_interrupted()
    except Exception as error:
        interrupted = None
        logger.warning("Could not fail interrupted jobs", extra={"error_type": type(error).__name__})
    logger.info("Job store: Postgres", extra={"interrupted_jobs_failed": interrupted})
    return store


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
