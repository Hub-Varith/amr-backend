"""API settings read from the environment."""

import tempfile
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API settings. Set any field with a G2M_-prefixed environment variable, e.g. G2M_MODELS_DIR."""

    model_config = SettingsConfigDict(env_prefix="G2M_", env_file=".env", extra="ignore")

    configs_dir: Path = Path("configs")
    models_dir: Path = Path("models")
    upload_dir: Path = Field(default_factory=lambda: Path(tempfile.gettempdir()) / "genome2mic_uploads")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    cors_origins: list[str] = []
    log_level: str = "INFO"
    # Prediction pipeline (predict/pipeline.py). The tools (mash, amrfinder) must be on PATH.
    model_run: str = "all5_run1"
    references_sketch: Path = Path("data/references/references.msh")
    amrfinder_db: Path | None = None   # default: the database installed with amrfinder
    tool_threads: int = Field(default=4, gt=0)
    keep_work_files: bool = False      # keep each job's step outputs (qc.json, amrfinder.tsv, ...) for debugging
    # Job store. Set it to a Neon pooled connection string to keep jobs and reports across restarts
    # (needs the db extra). Unset: jobs live in memory. Never commit it; keep it in .env.
    database_url: SecretStr | None = None
