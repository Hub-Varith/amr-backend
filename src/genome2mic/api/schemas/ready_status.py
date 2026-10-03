"""Readiness check body."""

from pydantic import BaseModel, ConfigDict


class ReadyStatus(BaseModel):
    """Whether the service can take prediction jobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    configs_parsed: bool
    models_loaded: bool
