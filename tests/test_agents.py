"""Tests for agent implementations."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from agents.core.base_agent import AgentContext, AgentResult
from agents.pr_review.agent import PRReviewAgent
from agents.pattern.agent import PatternAgent


class TestAgentContext:
    """Tests for AgentContext model."""

    def test_context_basic(self):
        """Test basic context creation."""
        context = AgentContext(user_input="test input")
        assert context.user_input == "test input"
        assert context.metadata == {}
        assert context.history == []

    def test_context_with_metadata(self):
        """Test context with metadata."""
        context = AgentContext(
            user_input="test",
            metadata={"key": "value"},
            history=[{"role": "user", "content": "hello"}],
        )
        assert context.metadata["key"] == "value"
        assert len(context.history) == 1


class TestAgentResult:
    """Tests for AgentResult model."""

    def test_result_success(self):
        """Test successful result."""
        result = AgentResult(
            success=True,
            output="Review complete",
            reasoning="All checks passed",
        )
        assert result.success
        assert result.output == "Review complete"

    def test_result_failure(self):
        """Test failure result."""
        result = AgentResult(
            success=False,
            output="Error occurred",
            reasoning="Connection failed",
        )
        assert not result.success


class TestPRReviewAgent:
    """Tests for PRReviewAgent."""

    @patch("agents.pr_review.agent.OllamaClient")
    def test_init(self, mock_client):
        """Test agent initialization."""
        agent = PRReviewAgent()
        assert agent.name == "PR Review Agent"
        assert "Code Reviewer" in agent.system_prompt

    @patch("agents.pr_review.agent.OllamaClient")
    def test_build_review_prompt(self, mock_client):
        """Test review prompt building."""
        agent = PRReviewAgent()
        prompt = agent._build_review_prompt(
            pr_title="Fix bug",
            pr_description="Fixes issue #123",
            pr_diff="+ added line",
            files_changed=["file.py"],
        )
        assert "Fix bug" in prompt
        assert "Fixes issue #123" in prompt
        assert "+ added line" in prompt
        assert "file.py" in prompt


class TestPatternAgent:
    """Tests for PatternAgent."""

    @patch("agents.pattern.agent.ChromaVectorStore")
    @patch("agents.pattern.agent.OllamaClient")
    def test_init(self, mock_client, mock_store):
        """Test agent initialization."""
        agent = PatternAgent()
        assert agent.name == "Pattern Agent"
        assert "NFR" in agent.system_prompt

    @patch("agents.pattern.agent.ChromaVectorStore")
    @patch("agents.pattern.agent.OllamaClient")
    def test_patterns_count(self, mock_client, mock_store):
        """Test patterns count property."""
        mock_store_instance = MagicMock()
        mock_store_instance.count = 10
        mock_store.return_value = mock_store_instance

        agent = PatternAgent()
        assert agent.patterns_count == 10
