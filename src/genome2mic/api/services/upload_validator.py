"""Genome upload checks."""

import asyncio
import gzip
import logging
import uuid
import zlib
from pathlib import Path

from fastapi import HTTPException, UploadFile

from genome2mic.api.constants import ALLOWED_FASTA_EXTENSIONS, GZIP_EXTENSION, UPLOAD_CHUNK_BYTES

logger = logging.getLogger(__name__)


class UploadValidator:
    """Checks extension, size and FASTA header of an upload, then writes it to the upload directory.

    Gzipped FASTA (`.fasta.gz`, `.fa.gz`, `.fna.gz`) is unpacked on arrival, so every later step
    sees plain FASTA. The size limit applies to the upload and again to the unpacked genome.
    """

    def __init__(self, upload_dir: Path, max_bytes: int) -> None:
        self.upload_dir = upload_dir
        self.max_bytes = max_bytes

    @staticmethod
    def fasta_suffix(filename: str) -> tuple[str, bool]:
        """(FASTA extension, is gzipped). Raises 415 when the name is not FASTA or gzipped FASTA."""
        suffixes = [suffix.lower() for suffix in Path(filename).suffixes]
        gzipped = bool(suffixes) and suffixes[-1] == GZIP_EXTENSION
        if gzipped:
            suffixes = suffixes[:-1]
        if not suffixes or suffixes[-1] not in ALLOWED_FASTA_EXTENSIONS:
            allowed = sorted(ALLOWED_FASTA_EXTENSIONS) + sorted(f"{suffix}{GZIP_EXTENSION}" for suffix in ALLOWED_FASTA_EXTENSIONS)
            raise HTTPException(415, f"File extension must be one of {allowed}.")
        return suffixes[-1], gzipped

    async def save(self, upload: UploadFile) -> Path:
        """Return the saved (plain FASTA) file path, or raise HTTPException 415, 413 or 422."""
        suffix, gzipped = self.fasta_suffix(upload.filename or "")

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        stem = uuid.uuid4().hex
        destination = self.upload_dir / f"{stem}{suffix}"
        received = self.upload_dir / f"{stem}{suffix}{GZIP_EXTENSION}" if gzipped else destination
        bytes_written = 0
        try:
            with received.open("wb") as handle:
                while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                    bytes_written += len(chunk)
                    if bytes_written > self.max_bytes:
                        raise HTTPException(413, f"File is larger than {self.max_bytes} bytes.")
                    handle.write(chunk)
            if gzipped:
                await asyncio.to_thread(self._unpack, received, destination)

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
        finally:
            if gzipped:
                received.unlink(missing_ok=True)

        logger.info("Upload saved", extra={"bytes": bytes_written, "suffix": suffix, "gzipped": gzipped})
        return destination

    def _unpack(self, source: Path, destination: Path) -> None:
        """Stream-decompress, stopping at max_bytes so a small archive cannot fill the disk."""
        unpacked = 0
        try:
            with gzip.open(source, "rb") as reader, destination.open("wb") as writer:
                while chunk := reader.read(UPLOAD_CHUNK_BYTES):
                    unpacked += len(chunk)
                    if unpacked > self.max_bytes:
                        raise HTTPException(413, f"Unpacked genome is larger than {self.max_bytes} bytes.")
                    writer.write(chunk)
        except (OSError, EOFError, zlib.error) as error:
            raise HTTPException(422, "File is not a valid gzip archive.") from error
