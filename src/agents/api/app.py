"""FastAPI application factory and configuration."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.api.routers import (
    pr_review_router,
    pattern_router,
    health_router,
    dev_assistant_router,
)
from agents.api.middleware import RequestContextMiddleware, ErrorHandlerMiddleware
from agents.api.logging_config import setup_logging, get_logger
from agents.api.dependencies import (
    get_pr_agent,
    get_pattern_agent,
    get_dev_assistant_agent,
)
from agents.core.config import get_settings
from agents import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger = get_logger("app")

    logger.info(f"Starting Personal Agents API v{__version__}")

    pr_agent = get_pr_agent()
    dev_agent = get_dev_assistant_agent()
    logger.info(f"Dev knowledge chunks indexed: {dev_agent.knowledge_count}")

    pattern_agent = get_pattern_agent()

    if pr_agent.is_ready():
        logger.info(f"Ollama connected: {pr_agent.llm.model_name}")
    else:
        logger.warning("Ollama is not available - some features will be disabled")

    logger.info(f"Patterns indexed: {pattern_agent.patterns_count}")

    yield

    logger.info("Shutting down Personal Agents API")


def create_app(
    title: str = "Personal Agents API",
    debug: bool = False,
    log_level: str = "INFO",
    json_logs: bool = False,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        title: API title.
        debug: Enable debug mode.
        log_level: Logging level.
        json_logs: Use JSON format for logs.

    Returns:
        Configured FastAPI application.
    """
    setup_logging(level=log_level, json_format=json_logs)

    app = FastAPI(
        title=title,
        description="""
# Personal Agents API

AI-powered agents for technical refinement and pattern compliance.

## Features

- **PR Review Agent**: Review pull requests for code quality, security, and performance
- **Pattern Agent**: Evaluate features against company patterns and generate NFRs
- **GitHub Integration**: Fetch and review PRs directly from GitHub
- **Azure DevOps Integration**: Evaluate features and generate NFRs from Azure DevOps

## Authentication

Currently, no authentication is required. For production use, implement
appropriate authentication middleware.
        """,
        version=__version__,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(pr_review_router)
    app.include_router(pattern_router)
    app.include_router(dev_assistant_router)

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": title,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
):
    """Run the API server using uvicorn.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload for development.
        log_level: Uvicorn log level.
    """
    import uvicorn

    uvicorn.run(
        "agents.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_server(reload=True)
