"""Health check and monitoring endpoints."""

from fastapi import APIRouter, Depends
from agents.api.models import HealthResponse
from agents.api.dependencies import get_pr_agent, get_pattern_agent
from agents.pr_review import PRReviewAgent
from agents.pattern import PatternAgent
from agents import __version__

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def health_check(
    pr_agent: PRReviewAgent = Depends(get_pr_agent),
    pattern_agent: PatternAgent = Depends(get_pattern_agent),
) -> HealthResponse:
    """Check the health status of the API and its dependencies."""
    ollama_available = pr_agent.is_ready()

    services = {
        "ollama": ollama_available,
        "chromadb": True,
    }

    status = "healthy" if ollama_available else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        ollama_available=ollama_available,
        model=pr_agent.llm.model_name,
        patterns_indexed=pattern_agent.patterns_count,
        services=services,
    )


@router.get("/ready")
async def readiness_check(
    pr_agent: PRReviewAgent = Depends(get_pr_agent),
) -> dict:
    """Check if the service is ready to accept requests."""
    if not pr_agent.is_ready():
        return {"ready": False, "reason": "Ollama not available"}

    return {"ready": True}


@router.get("/live")
async def liveness_check() -> dict:
    """Check if the service is alive."""
    return {"alive": True}
