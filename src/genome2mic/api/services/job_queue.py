"""Job queue interface."""

from abc import ABC, abstractmethod
from pathlib import Path


class JobQueue(ABC):
    """Hands a job to something that runs it. Add a Celery or RQ subclass here and swap it in deps.py."""

    @abstractmethod
    async def submit(self, job_id: str, fasta_path: Path, sample_id: str, request_id: str | None) -> None:
        """Schedule the job. Must return without waiting for the prediction."""
