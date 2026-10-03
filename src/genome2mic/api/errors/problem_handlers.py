"""Exception handlers that return RFC 7807 problem+json."""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from genome2mic.api.constants import PROBLEM_JSON, REQUEST_ID_HEADER
from genome2mic.api.schemas.problem_detail import ProblemDetail

logger = logging.getLogger(__name__)


def register_problem_handlers(app: FastAPI) -> None:
    """Attach the problem+json handlers to the app."""
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _problem_response(request, exc.status_code, str(exc.detail), headers=exc.headers)


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Leave out the rejected input: it can echo uploaded content back to the client and into logs.
    errors = []
    for error in exc.errors():
        errors.append({"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]})
    return _problem_response(request, 422, "Request validation failed.", errors=errors)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Starlette runs this outside the request-id middleware, so pass the id by hand.
    request_id = getattr(request.state, "request_id", None)
    logger.error("Unhandled error", exc_info=exc, extra={"request_id": request_id, "path": request.url.path})
    return _problem_response(request, 500, "An unexpected error occurred.")


def _problem_response(
    request: Request,
    status: int,
    detail: str,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    problem = ProblemDetail(
        title=HTTPStatus(status).phrase,
        status=status,
        detail=detail,
        instance=request.url.path,
        request_id=request_id,
        errors=errors,
    )
    response_headers = dict(headers or {})
    if request_id:
        response_headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        problem.model_dump(exclude_none=True),
        status_code=status,
        media_type=PROBLEM_JSON,
        headers=response_headers,
    )
