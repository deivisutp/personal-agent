"""FastAPI dependencies for agent injection."""

from functools import lru_cache

from agents.pr_review import PRReviewAgent
from agents.pattern import PatternAgent
from agents.dev_assistant import DevAssistantAgent
from agents.core.config import get_settings


@lru_cache()
def get_pr_agent() -> PRReviewAgent:
    """Get or create the PR Review Agent singleton."""
    return PRReviewAgent()


@lru_cache()
def get_pattern_agent() -> PatternAgent:
    """Get or create the Pattern Agent singleton."""
    return PatternAgent()


@lru_cache()
def get_dev_assistant_agent() -> DevAssistantAgent:
    """Get or create the Dev Assistant Agent singleton."""
    settings = get_settings()
    return DevAssistantAgent(
        azure_devops_org=settings.azure_devops.organization,
        azure_devops_project=settings.azure_devops.project,
        azure_devops_pat=settings.azure_devops.pat,
    )


def reset_agents() -> None:
    """Reset agent singletons (useful for testing)."""
    get_pr_agent.cache_clear()
    get_pattern_agent.cache_clear()
    get_dev_assistant_agent.cache_clear()
