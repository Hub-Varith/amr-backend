"""In-process job queue."""

from pathlib import Path

from fastapi import BackgroundTasks

from genome2mic.api.services.job_queue import JobQueue
from genome2mic.api.services.job_runner import JobRunner


class BackgroundTasksQueue(JobQueue):
    """Runs jobs in the API process after the response is sent, using FastAPI BackgroundTasks."""

    def __init__(self, background_tasks: BackgroundTasks, runner: JobRunner) -> None:
        self.background_tasks = background_tasks
        self.runner = runner

    async def submit(self, job_id: str, fasta_path: Path, sample_id: str, request_id: str | None) -> None:
        self.background_tasks.add_task(self.runner.run, job_id, fasta_path, sample_id, request_id)
