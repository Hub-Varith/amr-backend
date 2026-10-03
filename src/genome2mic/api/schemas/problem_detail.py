"""RFC 7807 error body."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ProblemDetail(BaseModel):
    """Error body sent as application/problem+json. request_id and errors are extension members."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None
    request_id: str | None = None
    errors: list[dict[str, Any]] | None = None
