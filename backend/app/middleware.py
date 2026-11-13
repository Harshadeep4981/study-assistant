"""
Request size limit ASGI middleware.

Enforces MAX_UPLOAD_SIZE_BYTES on incoming request bodies for the document
ingestion endpoint before and during body reading, preventing unbounded
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
    """
    ASGI middleware enforcing a maximum byte limit on incoming request bodies
    for the document ingestion endpoint (POST /api/documents/ingest).

    1. Checks 'content-length' header if present; rejects early with HTTP 413.
    2. Wraps the ASGI 'receive' callable to monitor bytes in real-time as they
       arrive over the network, aborting immediately with HTTP 413 if the
       accumulated body size exceeds max_bytes, even if 'content-length' is
       missing or misleading.
    3. Other routes and HTTP methods pass through without size-limit handling.
    """

    INGEST_PATH = "/api/documents/ingest"
    INGEST_METHOD = "POST"

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Scope size-limit enforcement strictly to POST /api/documents/ingest
        if (
            scope.get("path") != self.INGEST_PATH
            or scope.get("method", "").upper() != self.INGEST_METHOD
        ):
            await self.app(scope, receive, send)
            return

        limit_str = format_byte_size(self.max_bytes)

        # 1. Early rejection via Content-Length header if present
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    content_length = int(value.decode("latin1"))
                    if content_length > self.max_bytes:
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
                if total_received > self.max_bytes:
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
