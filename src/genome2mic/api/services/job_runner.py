"""Runs one prediction job end to end."""

import logging
from pathlib import Path

from genome2mic.api.request_context import request_id_var
from genome2mic.api.services.job_store import InMemoryJobStore
from genome2mic.api.services.predictor import Predictor

logger = logging.getLogger(__name__)


class JobRunner:
    """Runs a job, records the outcome and deletes the upload. A future worker queue calls the same run()."""

    def __init__(self, predictor: Predictor, job_store: InMemoryJobStore) -> None:
        self.predictor = predictor
        self.job_store = job_store

    async def run(self, job_id: str, fasta_path: Path, sample_id: str, request_id: str | None) -> None:
        token = request_id_var.set(request_id)
        logger.info("Job started", extra={"job_id": job_id, "sample_id": sample_id})
        try:
            await self.job_store.mark_running(job_id)
            report = await self.predictor.predict(fasta_path, sample_id)
            await self.job_store.mark_done(job_id, report)
            logger.info("Job finished", extra={"job_id": job_id})
        except Exception:
            # Catch everything: an uncaught error here would leave the job stuck in "running".
            logger.exception("Job failed", extra={"job_id": job_id})
            await self.job_store.mark_failed(
                job_id, f"Prediction failed. Quote request id {request_id} when reporting this."
            )
        finally:
            fasta_path.unlink(missing_ok=True)
            request_id_var.reset(token)
