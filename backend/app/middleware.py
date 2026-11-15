"""
ASGI middlewares for request size limits, correlation tracking, and safe request timing.

Enforces route-specific maximum byte limits on incoming request bodies
for protected endpoints before and during body reading, preventing unbounded
memory usage or disk spooling for both JSON and multipart requests.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.services.exceptions import OversizedFileError

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware:
    """
    ASGI middleware that attaches an X-Request-ID correlation ID to every HTTP request
    and logs safe timing and status metadata without logging request bodies or sensitive contents.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(raw=scope.get("headers", []))
        incoming_id = headers.get("x-request-id")
        request_id = incoming_id.strip() if incoming_id and incoming_id.strip() else str(uuid.uuid4())

        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        method = scope.get("method", "")
        path = scope.get("path", "")
        start_time = time.monotonic()

        logger.info(
            "HTTP request started",
            extra={"request_id": request_id, "http_method": method, "http_path": path},
        )

        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
                raw_headers = list(message.get("headers", []))
                has_req_id = any(h[0].lower() == b"x-request-id" for h in raw_headers)
                if not has_req_id:
                    raw_headers.append((b"x-request-id", request_id.encode("ascii", errors="ignore")))
                    message = dict(message)
                    message["headers"] = raw_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            logger.info(
                "HTTP request completed",
                extra={
                    "request_id": request_id,
                    "http_method": method,
                    "http_path": path,
                    "http_status": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )


def format_byte_size(num_bytes: int) -> str:
    """
    Format a byte count into an accurate, human-readable string.

    Ensures small byte counts (e.g. 500 bytes) are not rounded down to '0 MB'.
    """
    if num_bytes >= 1024 * 1024:
        mb = num_bytes / (1024 * 1024)
        return f"{int(mb)} MB" if mb.is_integer() else f"{mb:.1f} MB"
    elif num_bytes >= 1024:
        kb = num_bytes / 1024
        return f"{int(kb)} KB" if kb.is_integer() else f"{kb:.1f} KB"
    elif num_bytes == 1:
        return "1 byte"
    else:
        return f"{num_bytes} bytes"


class RequestSizeLimitMiddleware:
    r"""
    ASGI middleware enforcing route-specific byte limits on incoming request bodies
    before and during body reading.

    Protected routes and limits:
    - POST /api/documents/ingest: max_upload_bytes (e.g. 10 MB)
    - POST /api/generate/flashcards: max_generation_bytes (e.g. ~3 MB raw JSON body limit, allowing worst-case \u00XX escaping for 500 KB documents)
    - POST /api/generate/quiz: max_generation_bytes (e.g. ~3 MB raw JSON body limit, allowing worst-case \u00XX escaping for 500 KB documents)

    Enforcement mechanism:
    1. Checks 'content-length' header if present; rejects early with HTTP 413
       before the request body is read or parsed by downstream handlers.
    2. Wraps the ASGI 'receive' callable to monitor bytes in real-time as they
       arrive over the network, aborting immediately with HTTP 413 if the
       accumulated body size exceeds the route limit, even if 'content-length' is
       missing or misleading.
    3. Other routes and HTTP methods pass through without size-limit handling.
    """

    INGEST_PATH = "/api/documents/ingest"
    FLASHCARDS_PATH = "/api/generate/flashcards"
    QUIZ_PATH = "/api/generate/quiz"

    def __init__(
        self,
        app: ASGIApp,
        max_upload_bytes: int | None = None,
        max_generation_bytes: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.app = app
        if max_upload_bytes is not None:
            self.max_upload_bytes = max_upload_bytes
        elif max_bytes is not None:
            self.max_upload_bytes = max_bytes
        else:
            raise ValueError("max_upload_bytes must be specified.")

        if max_generation_bytes is not None:
            self.max_generation_bytes = max_generation_bytes
        elif max_bytes is not None:
            self.max_generation_bytes = max_bytes
        else:
            self.max_generation_bytes = self.max_upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()
        path = scope.get("path", "")

        # Determine route-specific limit
        route_limit: int | None = None
        if method == "POST":
            if path == self.INGEST_PATH:
                route_limit = self.max_upload_bytes
            elif path in (self.FLASHCARDS_PATH, self.QUIZ_PATH):
                route_limit = self.max_generation_bytes

        # Pass through unaffected if route is not protected
        if route_limit is None:
            await self.app(scope, receive, send)
            return

        limit_str = format_byte_size(route_limit)

        # 1. Early rejection via Content-Length header if present
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    content_length = int(value.decode("latin1"))
                    if content_length > route_limit:
                        response = JSONResponse(
                            status_code=413,
                            content={
                                "detail": f"Request body exceeds the maximum limit of {limit_str}."
                            },
                        )
                        await response(scope, receive, send)
                        return
                except (ValueError, UnicodeDecodeError):
                    pass
                break

        # 2. Real-time byte counting during body streaming
        total_received = 0

        async def bounded_receive() -> Message:
            nonlocal total_received
            message = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                total_received += len(chunk)
                if total_received > route_limit:
                    raise OversizedFileError(
                        f"Request body exceeds the maximum limit of {limit_str}."
                    )
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except OversizedFileError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.message},
            )
            await response(scope, receive, send)
