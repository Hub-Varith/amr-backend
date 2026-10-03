"""Small HTTP client with retries, shared by the BV-BRC and NCBI downloaders."""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from genome2mic.ingest.constants import HTTP_RETRIES, HTTP_TIMEOUT_SECONDS, HTTP_USER_AGENT

logger = logging.getLogger(__name__)


class HttpClient:
    """GET and POST with a fixed user agent and exponential backoff."""

    def get_text(self, url: str, headers: dict[str, str] | None = None) -> str:
        """Return the response body of a GET request as text."""
        return self._request(url, data=None, headers=headers or {})

    def post_json(self, url: str, body: str, headers: dict[str, str]) -> Any:
        """POST a text body and parse the JSON response."""
        return json.loads(self._request(url, data=body.encode(), headers=headers))

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> Any:
        """GET a URL and parse the JSON response."""
        return json.loads(self.get_text(url, headers))

    @staticmethod
    def _request(url: str, data: bytes | None, headers: dict[str, str]) -> str:
        all_headers = {"User-Agent": HTTP_USER_AGENT, **headers}
        for attempt in range(1, HTTP_RETRIES + 1):
            request = urllib.request.Request(url, data=data, headers=all_headers)
            try:
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                    return response.read().decode("utf-8")
            except (urllib.error.URLError, TimeoutError) as error:
                # A bad request will not succeed on retry; rate limits (429) and server errors might.
                is_client_error = isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429
                if is_client_error or attempt == HTTP_RETRIES:
                    raise
                wait_seconds = 2**attempt
                logger.warning("HTTP request failed, retrying in %ss (attempt %s): %s", wait_seconds, attempt, error)
                time.sleep(wait_seconds)
        raise RuntimeError("unreachable")
