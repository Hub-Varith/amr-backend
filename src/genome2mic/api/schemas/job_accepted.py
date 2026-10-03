"""Response to a new prediction job."""

from pydantic import BaseModel, ConfigDict

from genome2mic.api.schemas.job_status import JobStatus


class JobAccepted(BaseModel):
    """Returned with 202 when a genome upload is queued."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    status: JobStatus
    status_url: str
