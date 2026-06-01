"""Pattern and NFR API endpoints."""

from fastapi import APIRouter, Depends
from agents.api.models import (
    FeatureEvaluationRequest,
    AzureFeatureRequest,
    NFRGenerationRequest,
    AzureNFRRequest,
    PatternIngestRequest,
    WikiIngestRequest,
    AgentResponse,
)
from agents.api.dependencies import get_pattern_agent
from agents.api.exceptions import AgentNotReadyError, ExternalServiceError
from agents.api.logging_config import get_logger
from agents.pattern import PatternAgent
from agents.pr_review import with_language
from agents.core.base_agent import AgentContext

router = APIRouter(prefix="/pattern", tags=["Pattern & NFR"])
logger = get_logger("api.pattern")


@router.post("/evaluate", response_model=AgentResponse)
async def evaluate_feature(
    request: FeatureEvaluationRequest,
    agent: PatternAgent = Depends(get_pattern_agent),
) -> AgentResponse:
    """Evaluate a feature against company patterns and generate NFRs."""
    if not agent.is_ready():
        raise AgentNotReadyError("Pattern Agent", "Ollama is not available")

    logger.info(f"Evaluating feature: {request.title}")

    if request.language != "en":
        agent.system_prompt = with_language(agent.system_prompt, request.language)

    context = AgentContext(
        user_input=request.description,
        metadata={
            "feature_title": request.title,
            "feature_description": request.description,
            "user_stories": request.user_stories,
            "existing_nfrs": request.existing_nfrs,
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


@router.post("/nfrs", response_model=AgentResponse)
async def generate_nfrs(
    request: NFRGenerationRequest,
    agent: PatternAgent = Depends(get_pattern_agent),
) -> AgentResponse:
    """Generate NFRs for a feature description."""
    if not agent.is_ready():
        raise AgentNotReadyError("Pattern Agent", "Ollama is not available")

    logger.info(f"Generating NFRs for feature, categories: {request.categories}")

    if request.language != "en":
        agent.system_prompt = with_language(agent.system_prompt, request.language)

    categories = [c.value for c in request.categories] if request.categories else None

    result = agent.generate_nfrs(request.feature_description, categories=categories)

    return AgentResponse(
        success=True,
        output=result,
        metadata={"categories": categories or []},
    )


@router.post("/ingest", response_model=dict)
async def ingest_patterns(
    request: PatternIngestRequest,
    agent: PatternAgent = Depends(get_pattern_agent),
) -> dict:
    """Ingest pattern content into the vector store."""
    logger.info(f"Ingesting patterns from: {request.source_name}")

    chunks = agent.ingest_text(request.content, source_name=request.source_name)

    return {
        "success": True,
        "chunks_ingested": chunks,
        "total_patterns": agent.patterns_count,
    }


@router.get("/patterns/count")
async def get_patterns_count(
    agent: PatternAgent = Depends(get_pattern_agent),
) -> dict:
    """Get the number of indexed patterns."""
    return {"count": agent.patterns_count}


@router.delete("/patterns")
async def clear_patterns(
    agent: PatternAgent = Depends(get_pattern_agent),
) -> dict:
    """Clear all indexed patterns."""
    agent.vector_store.reset()
    return {"success": True, "message": "All patterns cleared"}


@router.post("/azure/evaluate", response_model=AgentResponse)
async def evaluate_azure_feature(
    request: AzureFeatureRequest,
    agent: PatternAgent = Depends(get_pattern_agent),
) -> AgentResponse:
    """Evaluate an Azure DevOps feature and generate NFRs."""
    if not agent.is_ready():
        raise AgentNotReadyError("Pattern Agent", "Ollama is not available")

    logger.info(f"Evaluating Azure DevOps feature: {request.feature_id}")

    if request.language != "en":
        agent.system_prompt = with_language(agent.system_prompt, request.language)

    try:
        result = await agent.evaluate_azure_feature(
            feature_id=request.feature_id,
            project=request.project,
            include_user_stories=request.include_user_stories,
        )

        return AgentResponse(
            success=result.success,
            output=result.output,
            reasoning=result.reasoning,
            metadata=result.metadata,
            suggestions=result.suggestions,
        )

    except Exception as e:
        logger.error(f"Azure DevOps feature evaluation failed: {e}")
        raise ExternalServiceError("Azure DevOps", str(e))

    finally:
        await agent.close()


@router.post("/azure/nfrs", response_model=AgentResponse)
async def generate_azure_nfrs(
    request: AzureNFRRequest,
    agent: PatternAgent = Depends(get_pattern_agent),
) -> AgentResponse:
    """Generate NFRs for an Azure DevOps feature."""
    if not agent.is_ready():
        raise AgentNotReadyError("Pattern Agent", "Ollama is not available")

    logger.info(f"Generating NFRs for Azure DevOps feature: {request.feature_id}")

    if request.language != "en":
        agent.system_prompt = with_language(agent.system_prompt, request.language)

    try:
        categories = [c.value for c in request.categories] if request.categories else None

        result = await agent.generate_nfrs_for_feature(
            feature_id=request.feature_id,
            project=request.project,
            categories=categories,
            create_work_items=request.create_work_items,
        )

        return AgentResponse(
            success=result.success,
            output=result.output,
            reasoning=result.reasoning,
            metadata=result.metadata,
            suggestions=result.suggestions,
        )

    except Exception as e:
        logger.error(f"Azure DevOps NFR generation failed: {e}")
        raise ExternalServiceError("Azure DevOps", str(e))

    finally:
        await agent.close()


@router.get("/azure/features")
async def list_azure_features(
    state: str = None,
    limit: int = 20,
    agent: PatternAgent = Depends(get_pattern_agent),
) -> list[dict]:
    """List features from Azure DevOps."""
    try:
        features = await agent.list_features(state=state, limit=limit)
        return features

    except Exception as e:
        logger.error(f"Failed to list Azure DevOps features: {e}")
        raise ExternalServiceError("Azure DevOps", str(e))

    finally:
        await agent.close()


@router.post("/azure/wiki/ingest")
async def ingest_wiki_patterns(
    request: WikiIngestRequest,
    agent: PatternAgent = Depends(get_pattern_agent),
) -> dict:
    """Ingest patterns from Azure DevOps wiki."""
    logger.info(f"Ingesting wiki patterns: {request.wiki_name}{request.path}")

    try:
        chunks = await agent.ingest_wiki_patterns(
            wiki_name=request.wiki_name,
            path=request.path,
            project=request.project,
            recursive=request.recursive,
        )

        return {
            "success": True,
            "chunks_ingested": chunks,
            "total_patterns": agent.patterns_count,
        }

    except Exception as e:
        logger.error(f"Wiki ingestion failed: {e}")
        raise ExternalServiceError("Azure DevOps", str(e))

    finally:
        await agent.close()
