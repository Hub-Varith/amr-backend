"""State of one prediction job."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from genome2mic.api.schemas.job_status import JobStatus


class JobState(BaseModel):
    """Status of a prediction job. error is set only when status is failed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    sample_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None
