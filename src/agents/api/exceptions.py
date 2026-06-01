"""Custom exceptions and error handling for the API."""

from typing import Any, Optional
from fastapi import HTTPException, status


class AgentException(Exception):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)


class AgentNotReadyError(AgentException):
    """Raised when an agent is not ready (e.g., Ollama not available)."""

    def __init__(self, agent_name: str, reason: str = "Agent not ready"):
        super().__init__(
            message=f"{agent_name}: {reason}",
            details={"agent": agent_name, "reason": reason},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ConfigurationError(AgentException):
    """Raised when configuration is missing or invalid."""

    def __init__(self, message: str, missing_keys: Optional[list[str]] = None):
        super().__init__(
            message=message,
            details={"missing_keys": missing_keys or []},
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class ExternalServiceError(AgentException):
    """Raised when an external service (GitHub, Azure DevOps) fails."""

    def __init__(self, service: str, message: str, original_error: Optional[str] = None):
        super().__init__(
            message=f"{service} error: {message}",
            details={"service": service, "original_error": original_error},
            status_code=status.HTTP_502_BAD_GATEWAY,
        )


class ValidationError(AgentException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            details={"field": field} if field else {},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class RateLimitError(AgentException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(
            message=message,
            details={"retry_after": retry_after} if retry_after else {},
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


def agent_exception_to_http(exc: AgentException) -> HTTPException:
    """Convert AgentException to HTTPException."""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "error": exc.message,
            "details": exc.details,
        },
    )
