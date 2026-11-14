"""
Request size limit ASGI middleware.

Enforces route-specific maximum byte limits on incoming request bodies
for protected endpoints before and during body reading, preventing unbounded
memory usage or disk spooling for both JSON and multipart requests.
"""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.services.exceptions import OversizedFileError


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
