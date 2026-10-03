"""FastAPI dependencies. Tests swap these with app.dependency_overrides."""

from typing import Annotated

from fastapi import BackgroundTasks, Depends, Request

from genome2mic.api.config import Settings
from genome2mic.api.services.background_tasks_queue import BackgroundTasksQueue
from genome2mic.api.services.job_queue import JobQueue
from genome2mic.api.services.job_runner import JobRunner
from genome2mic.api.services.job_store import InMemoryJobStore
from genome2mic.api.services.model_registry import ModelRegistry
from genome2mic.api.services.predictor import Predictor
from genome2mic.api.services.upload_validator import UploadValidator


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def get_job_store(request: Request) -> InMemoryJobStore:
    return request.app.state.job_store


SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
PredictorDep = Annotated[Predictor, Depends(get_predictor)]
JobStoreDep = Annotated[InMemoryJobStore, Depends(get_job_store)]


def get_upload_validator(settings: SettingsDep) -> UploadValidator:
    return UploadValidator(upload_dir=settings.upload_dir, max_bytes=settings.max_upload_bytes)


def get_job_queue(background_tasks: BackgroundTasks, predictor: PredictorDep, job_store: JobStoreDep) -> JobQueue:
    return BackgroundTasksQueue(background_tasks, JobRunner(predictor, job_store))


UploadValidatorDep = Annotated[UploadValidator, Depends(get_upload_validator)]
JobQueueDep = Annotated[JobQueue, Depends(get_job_queue)]


async def require_auth() -> None:
    """Placeholder. Every /v1 router already depends on this, so real auth only needs to change this body."""
    return None
