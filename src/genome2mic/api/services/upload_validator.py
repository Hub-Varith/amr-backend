"""Genome upload checks."""

import logging
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from genome2mic.api.constants import ALLOWED_FASTA_EXTENSIONS, UPLOAD_CHUNK_BYTES

logger = logging.getLogger(__name__)


class UploadValidator:
    """Checks extension, size and FASTA header of an upload, then writes it to the upload directory."""

    def __init__(self, upload_dir: Path, max_bytes: int) -> None:
        self.upload_dir = upload_dir
        self.max_bytes = max_bytes

    async def save(self, upload: UploadFile) -> Path:
        """Return the saved file path, or raise HTTPException 415, 413 or 422."""
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_FASTA_EXTENSIONS:
            raise HTTPException(415, f"File extension must be one of {sorted(ALLOWED_FASTA_EXTENSIONS)}.")

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        destination = self.upload_dir / f"{uuid.uuid4().hex}{suffix}"
        bytes_written = 0
        try:
            with destination.open("wb") as handle:
                while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                    bytes_written += len(chunk)
                    if bytes_written > self.max_bytes:
                        raise HTTPException(413, f"File is larger than {self.max_bytes} bytes.")
                    handle.write(chunk)

            first_line = b""
            with destination.open("rb") as handle:
                for line in handle:
                    if line.strip():
                        first_line = line
                        break
            if not first_line.startswith(b">"):
                raise HTTPException(422, "File is not FASTA: the first non-blank line must start with '>'.")
        except HTTPException as error:
            destination.unlink(missing_ok=True)
            logger.warning("Upload rejected", extra={"status": error.status_code, "reason": error.detail})
            raise

        logger.info("Upload saved", extra={"bytes": bytes_written, "suffix": suffix})
        return destination
