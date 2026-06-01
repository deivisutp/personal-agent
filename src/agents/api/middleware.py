"""FastAPI middleware for logging, error handling, and request tracking."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from agents.api.logging_config import get_logger
from agents.api.exceptions import AgentException

logger = get_logger("middleware")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context (ID, timing) to all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.perf_counter()

        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"request_id": request_id},
        )

        try:
            response = await call_next(request)

            duration_ms = (time.perf_counter() - start_time) * 1000

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration_ms:.2f}ms",
                extra={"request_id": request_id, "duration_ms": duration_ms},
            )

            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"error={str(e)} duration={duration_ms:.2f}ms",
                extra={"request_id": request_id, "duration_ms": duration_ms},
                exc_info=True,
            )

            raise


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and return consistent error responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)

        except AgentException as e:
            request_id = getattr(request.state, "request_id", None)

            logger.warning(
                f"Agent exception: {e.message}",
                extra={"request_id": request_id},
            )

            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.message,
                    "details": e.details,
                    "request_id": request_id,
                },
            )

        except Exception as e:
            request_id = getattr(request.state, "request_id", None)

            logger.error(
                f"Unhandled exception: {str(e)}",
                extra={"request_id": request_id},
                exc_info=True,
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "details": {"message": str(e)},
                    "request_id": request_id,
                },
            )


class CORSMiddleware:
    """Simple CORS middleware for development."""

    def __init__(self, app, allow_origins: list[str] = None):
        self.app = app
        self.allow_origins = allow_origins or ["*"]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))

                if "*" in self.allow_origins or origin in self.allow_origins:
                    response_headers.extend([
                        (b"access-control-allow-origin", origin.encode() or b"*"),
                        (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
                        (b"access-control-allow-headers", b"*"),
                        (b"access-control-max-age", b"86400"),
                    ])

                message["headers"] = response_headers

            await send(message)

        await self.app(scope, receive, send_wrapper)
