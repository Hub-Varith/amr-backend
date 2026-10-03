"""Job lifecycle states."""

from enum import StrEnum


class JobStatus(StrEnum):
    """Where a prediction job is in its lifecycle."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
