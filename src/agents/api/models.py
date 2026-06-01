"""Pydantic models for API requests and responses."""

from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ReviewFocus(str, Enum):
    """Focus areas for PR review."""

    GENERAL = "general"
    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"


class NFRCategory(str, Enum):
    """NFR categories."""

    PERFORMANCE = "Performance"
    SECURITY = "Security"
    SCALABILITY = "Scalability"
    RELIABILITY = "Reliability"
    MAINTAINABILITY = "Maintainability"
    OBSERVABILITY = "Observability"
    USABILITY = "Usability"


class PRReviewRequest(BaseModel):
    """Request model for PR review."""

    diff: str = Field(..., description="The PR diff content to review")
    title: Optional[str] = Field(None, description="PR title")
    description: Optional[str] = Field(None, description="PR description")
    files_changed: list[str] = Field(default_factory=list, description="List of changed files")
    focus: ReviewFocus = Field(default=ReviewFocus.GENERAL, description="Review focus area")
    language: str = Field(default="en", description="Output language code (e.g., 'pt' for Portuguese, 'es' for Spanish)")


class GitHubPRReviewRequest(BaseModel):
    """Request model for GitHub PR review."""

    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    pr_number: int = Field(..., description="Pull request number", gt=0)
    post_comment: bool = Field(default=False, description="Post review as GitHub comment")
    focus: ReviewFocus = Field(default=ReviewFocus.GENERAL, description="Review focus area")
    language: str = Field(default="en", description="Output language code (e.g., 'pt' for Portuguese, 'es' for Spanish)")


class FeatureEvaluationRequest(BaseModel):
    """Request model for feature evaluation."""

    title: str = Field(..., description="Feature title")
    description: str = Field(..., description="Feature description")
    user_stories: list[str] = Field(default_factory=list, description="Related user stories")
    existing_nfrs: list[str] = Field(default_factory=list, description="Existing NFRs")
    language: str = Field(default="en", description="Output language code (e.g., 'pt' for Portuguese, 'es' for Spanish)")


class AzureFeatureRequest(BaseModel):
    """Request model for Azure DevOps feature evaluation."""

    feature_id: int = Field(..., description="Azure DevOps feature work item ID", gt=0)
    project: Optional[str] = Field(None, description="Project name (uses default if not specified)")
    include_user_stories: bool = Field(default=False, description="Include child user stories")
    language: str = Field(default="en", description="Output language code (e.g., 'pt' for Portuguese, 'es' for Spanish)")


class NFRGenerationRequest(BaseModel):
    """Request model for NFR generation."""

    feature_description: str = Field(..., description="Feature description")
    categories: list[NFRCategory] = Field(
        default_factory=list,
        description="NFR categories to focus on",
    )
    language: str = Field(default="en", description="Output language code (e.g., 'pt' for Portuguese, 'es' for Spanish)")


class AzureNFRRequest(BaseModel):
    """Request model for Azure DevOps NFR generation."""

    feature_id: int = Field(..., description="Azure DevOps feature work item ID", gt=0)
    project: Optional[str] = Field(None, description="Project name")
    categories: list[NFRCategory] = Field(default_factory=list, description="NFR categories")
    create_work_items: bool = Field(default=False, description="Create NFR work items in Azure DevOps")
    language: str = Field(default="en", description="Output language code (e.g., 'pt' for Portuguese, 'es' for Spanish)")


class PatternIngestRequest(BaseModel):
    """Request model for pattern ingestion."""

    content: str = Field(..., description="Pattern content to ingest")
    source_name: str = Field(default="api_input", description="Source identifier")


class WikiIngestRequest(BaseModel):
    """Request model for wiki pattern ingestion."""

    wiki_name: str = Field(..., description="Azure DevOps wiki name")
    path: str = Field(default="/", description="Wiki path to ingest from")
    project: Optional[str] = Field(None, description="Project name")
    recursive: bool = Field(default=True, description="Ingest recursively")


class AgentResponse(BaseModel):
    """Standard response model for agent operations."""

    success: bool
    output: str
    reasoning: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    ollama_available: bool
    model: str
    patterns_indexed: int
    services: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
