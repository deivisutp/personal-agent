"""Personal Agents - AI Agents for PR Review and Pattern/NFR Compliance."""

from agents.core.base_agent import BaseAgent
from agents.core.llm_client import OllamaClient

__version__ = "0.1.0"
__all__ = ["BaseAgent", "OllamaClient"]
