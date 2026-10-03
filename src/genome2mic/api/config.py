"""API settings read from the environment."""

import tempfile
from pathlib import Path

from pydantic import Field
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
