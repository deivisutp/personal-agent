"""Tests for the Ollama LLM client."""

import pytest
from unittest.mock import Mock, patch

from agents.core.llm_client import OllamaClient, ChatResponse
from agents.core.config import OllamaSettings


class TestOllamaClient:
    """Tests for OllamaClient."""

    def test_init_with_defaults(self):
        """Test client initialization with default settings."""
        with patch("agents.core.llm_client.ChatOllama"):
            client = OllamaClient()
            assert client.model_name == "mistral-nemo"

    def test_init_with_custom_model(self):
        """Test client initialization with custom model."""
        with patch("agents.core.llm_client.ChatOllama"):
            client = OllamaClient(model="llama3")
            assert client.model_name == "llama3"

    def test_chat_response_model(self):
        """Test ChatResponse model."""
        response = ChatResponse(
            content="Hello, world!",
            model="mistral-nemo",
            tokens_used=10,
            metadata={"key": "value"},
        )
        assert response.content == "Hello, world!"
        assert response.model == "mistral-nemo"
        assert response.tokens_used == 10

    def test_chat_response_defaults(self):
        """Test ChatResponse with minimal fields."""
        response = ChatResponse(content="Test", model="test-model")
        assert response.tokens_used is None
        assert response.metadata == {}


class TestOllamaClientIntegration:
    """Integration tests for OllamaClient (requires running Ollama)."""

    @pytest.mark.skipif(
        not OllamaClient().is_available(),
        reason="Ollama not available",
    )
    def test_chat_basic(self):
        """Test basic chat functionality."""
        client = OllamaClient()
        response = client.chat("Say 'hello' and nothing else.")
        assert response.content
        assert "hello" in response.content.lower()

    @pytest.mark.skipif(
        not OllamaClient().is_available(),
        reason="Ollama not available",
    )
    def test_chat_with_system_prompt(self):
        """Test chat with system prompt."""
        client = OllamaClient()
        response = client.chat(
            "What are you?",
            system_prompt="You are a helpful pirate. Always say 'Arrr!'",
        )
        assert response.content
