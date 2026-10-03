"""Request id and access logging."""

import logging
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from genome2mic.api.constants import REQUEST_ID_HEADER, REQUEST_ID_PATTERN
from genome2mic.api.request_context import request_id_var

logger = logging.getLogger(__name__)


class RequestIdMiddleware:
    """Gives every HTTP request an id, logs its start and end, and returns the id in a response header."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # A client-supplied id is kept only if it is safe to write into logs.
        incoming_id = Headers(scope=scope).get(REQUEST_ID_HEADER, "")
        request_id = incoming_id if REQUEST_ID_PATTERN.match(incoming_id) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500
        logger.info("Request started", extra={"method": scope["method"], "path": scope["path"]})

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                if REQUEST_ID_HEADER not in headers:
                    headers.append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.info(
                "Request finished",
                extra={"method": scope["method"], "path": scope["path"], "status": status_code, "duration_ms": duration_ms},
            )
            request_id_var.reset(token)
