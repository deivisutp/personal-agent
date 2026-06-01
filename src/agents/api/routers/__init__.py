"""API routers."""

from agents.api.routers.pr_review import router as pr_review_router
from agents.api.routers.pattern import router as pattern_router
from agents.api.routers.health import router as health_router
from agents.api.routers.dev_assistant import router as dev_assistant_router

__all__ = [
    "pr_review_router",
    "pattern_router",
    "health_router",
    "dev_assistant_router",
]
