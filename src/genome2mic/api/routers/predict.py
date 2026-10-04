"""Species listing and genome submission."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from genome2mic.api.constants import ALLOWED_FASTA_EXTENSIONS, GZIP_EXTENSION
from genome2mic.api.deps import JobQueueDep, JobStoreDep, RegistryDep, UploadValidatorDep
from genome2mic.api.schemas.job_accepted import JobAccepted
from genome2mic.api.schemas.problem_detail import ProblemDetail
from genome2mic.api.schemas.species_info import SpeciesInfo

router = APIRouter(tags=["predict"])


@router.get("/species", response_model=list[SpeciesInfo])
async def list_species(registry: RegistryDep) -> list[SpeciesInfo]:
    """Species in scope and the drugs that have a loaded model."""
    return registry.species_info()


def default_sample_id(filename: str | None) -> str:
    """The file name without its FASTA (and gzip) extension: 470.7409.fasta.gz -> 470.7409."""
    name = Path(filename or "sample").name
    for extension in (GZIP_EXTENSION, *ALLOWED_FASTA_EXTENSIONS):
        if name.lower().endswith(extension):
            name = name[: -len(extension)]
    return name[:128] or "sample"


@router.post(
    "/predict",
    status_code=202,
    response_model=JobAccepted,
    responses={code: {"model": ProblemDetail} for code in (413, 415, 422, 503)},
)
async def submit_prediction(
    request: Request,
    file: UploadFile,
    registry: RegistryDep,
    validator: UploadValidatorDep,
    job_store: JobStoreDep,
    job_queue: JobQueueDep,
    sample_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
) -> JobAccepted:
    """Upload one genome FASTA. Returns a job id; poll the status URL for the result."""
    if not registry.is_ready:
        raise HTTPException(503, "Models are not loaded. Check GET /ready.")

    fasta_path = await validator.save(file)
    resolved_sample_id = sample_id or default_sample_id(file.filename)
    job = await job_store.create(resolved_sample_id)
    request_id = getattr(request.state, "request_id", None)
    await job_queue.submit(job.job_id, fasta_path, resolved_sample_id, request_id)
    return JobAccepted(
        job_id=job.job_id,
        status=job.status,
        status_url=str(request.url_for("get_job_status", job_id=job.job_id)),
    )
