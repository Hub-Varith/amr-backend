"""Liveness and readiness probes."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from genome2mic.api.deps import RegistryDep
from genome2mic.api.schemas.ready_status import ReadyStatus

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyStatus, responses={503: {"model": ReadyStatus}})
async def ready(registry: RegistryDep) -> JSONResponse:
    """Readiness: configs parsed and models loaded. 503 until both are true."""
    status = ReadyStatus(
        ready=registry.is_ready,
        configs_parsed=registry.configs_parsed,
        models_loaded=registry.models_loaded,
    )
    return JSONResponse(status.model_dump(), status_code=200 if status.ready else 503)
