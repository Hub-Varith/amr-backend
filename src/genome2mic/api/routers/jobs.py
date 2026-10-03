"""Job status and results."""

from fastapi import APIRouter, HTTPException

from genome2mic.api.deps import JobStoreDep
from genome2mic.api.schemas.job_state import JobState
from genome2mic.api.schemas.job_status import JobStatus
from genome2mic.api.schemas.prediction_report import PredictionReport
from genome2mic.api.schemas.problem_detail import ProblemDetail

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobState, responses={404: {"model": ProblemDetail}})
async def get_job_status(job_id: str, job_store: JobStoreDep) -> JobState:
    """Current status of a job."""
    state = await job_store.get(job_id)
    if state is None:
        raise HTTPException(404, "Job not found.")
    return state


@router.get(
    "/jobs/{job_id}/result",
    response_model=PredictionReport,
    responses={404: {"model": ProblemDetail}, 409: {"model": ProblemDetail}},
)
async def get_job_result(job_id: str, job_store: JobStoreDep) -> PredictionReport:
    """The prediction report. 409 until the job is done."""
    state = await job_store.get(job_id)
    if state is None:
        raise HTTPException(404, "Job not found.")
    if state.status is not JobStatus.DONE:
        raise HTTPException(409, f"Job status is '{state.status}'. A result exists only when status is 'done'.")
    return await job_store.get_result(job_id)
