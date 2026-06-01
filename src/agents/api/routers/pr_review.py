"""PR Review API endpoints."""

from fastapi import APIRouter, Depends, BackgroundTasks
from agents.api.models import (
    PRReviewRequest,
    GitHubPRReviewRequest,
    AgentResponse,
    ReviewFocus,
)
from agents.api.dependencies import get_pr_agent
from agents.api.exceptions import AgentNotReadyError, ExternalServiceError
from agents.api.logging_config import get_logger
from agents.pr_review import PRReviewAgent, SECURITY_FOCUSED_PROMPT, PERFORMANCE_FOCUSED_PROMPT, with_language
from agents.core.base_agent import AgentContext

router = APIRouter(prefix="/pr-review", tags=["PR Review"])
logger = get_logger("api.pr_review")


@router.post("/review", response_model=AgentResponse)
async def review_diff(
    request: PRReviewRequest,
    agent: PRReviewAgent = Depends(get_pr_agent),
) -> AgentResponse:
    """Review a PR diff and provide feedback.

    This endpoint accepts a raw diff and returns a comprehensive code review.
    """
    if not agent.is_ready():
        raise AgentNotReadyError("PR Review Agent", "Ollama is not available")

    logger.info(f"Reviewing diff with focus: {request.focus}")

    if request.focus == ReviewFocus.SECURITY:
        agent.system_prompt = with_language(SECURITY_FOCUSED_PROMPT, request.language)
    elif request.focus == ReviewFocus.PERFORMANCE:
        agent.system_prompt = with_language(PERFORMANCE_FOCUSED_PROMPT, request.language)
    elif request.language != "en":
        agent.system_prompt = with_language(agent.system_prompt, request.language)

    context = AgentContext(
        user_input=request.diff,
        metadata={
            "pr_title": request.title or "Untitled PR",
            "pr_description": request.description or "",
            "pr_diff": request.diff,
            "files_changed": request.files_changed,
        },
    )

    result = agent.execute(context)

    return AgentResponse(
        success=result.success,
        output=result.output,
        reasoning=result.reasoning,
        metadata=result.metadata,
        suggestions=result.suggestions,
    )


@router.post("/quick-review")
async def quick_review(
    diff: str,
    focus: ReviewFocus = ReviewFocus.GENERAL,
    agent: PRReviewAgent = Depends(get_pr_agent),
) -> dict:
    """Perform a quick review of a diff without full context."""
    if not agent.is_ready():
        raise AgentNotReadyError("PR Review Agent", "Ollama is not available")

    focus_str = None if focus == ReviewFocus.GENERAL else focus.value

    result = agent.quick_review(diff, focus=focus_str)

    return {"review": result}


@router.post("/github", response_model=AgentResponse)
async def review_github_pr(
    request: GitHubPRReviewRequest,
    agent: PRReviewAgent = Depends(get_pr_agent),
) -> AgentResponse:
    """Review a PR directly from GitHub.

    Fetches the PR from GitHub using MCP and performs a comprehensive review.
    Optionally posts the review as a comment on the PR.
    """
    if not agent.is_ready():
        raise AgentNotReadyError("PR Review Agent", "Ollama is not available")

    logger.info(f"Reviewing GitHub PR: {request.owner}/{request.repo}#{request.pr_number}")

    if request.focus == ReviewFocus.SECURITY:
        agent.system_prompt = with_language(SECURITY_FOCUSED_PROMPT, request.language)
    elif request.focus == ReviewFocus.PERFORMANCE:
        agent.system_prompt = with_language(PERFORMANCE_FOCUSED_PROMPT, request.language)
    elif request.language != "en":
        agent.system_prompt = with_language(agent.system_prompt, request.language)

    try:
        result = await agent.review_github_pr(
            owner=request.owner,
            repo=request.repo,
            pr_number=request.pr_number,
            post_comment=request.post_comment,
        )

        return AgentResponse(
            success=result.success,
            output=result.output,
            reasoning=result.reasoning,
            metadata=result.metadata,
            suggestions=result.suggestions,
        )

    except Exception as e:
        logger.error(f"GitHub PR review failed: {e}")
        raise ExternalServiceError("GitHub", str(e))

    finally:
        await agent.close()


@router.get("/github/{owner}/{repo}/prs")
async def list_github_prs(
    owner: str,
    repo: str,
    state: str = "open",
    limit: int = 10,
    agent: PRReviewAgent = Depends(get_pr_agent),
) -> list[dict]:
    """List pull requests from a GitHub repository."""
    try:
        prs = await agent.list_open_prs(owner, repo, limit=limit)
        return prs

    except Exception as e:
        logger.error(f"Failed to list GitHub PRs: {e}")
        raise ExternalServiceError("GitHub", str(e))

    finally:
        await agent.close()
