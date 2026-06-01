"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from agents.api.app import create_app
from agents.api.dependencies import reset_agents


@pytest.fixture
def client():
    """Create a test client."""
    reset_agents()
    app = create_app(debug=True)
    return TestClient(app)


@pytest.fixture
def mock_pr_agent():
    """Mock PR Review Agent."""
    with patch("agents.api.dependencies.PRReviewAgent") as mock:
        agent = MagicMock()
        agent.is_ready.return_value = True
        agent.llm.model_name = "mistral-nemo"
        mock.return_value = agent
        yield agent


@pytest.fixture
def mock_pattern_agent():
    """Mock Pattern Agent."""
    with patch("agents.api.dependencies.PatternAgent") as mock:
        agent = MagicMock()
        agent.is_ready.return_value = True
        agent.patterns_count = 10
        mock.return_value = agent
        yield agent


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data

    def test_liveness_check(self, client):
        """Test liveness endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["alive"] is True


class TestPRReviewEndpoints:
    """Tests for PR review endpoints."""

    def test_review_diff_requires_body(self, client):
        """Test that review endpoint requires request body."""
        response = client.post("/pr-review/review")
        assert response.status_code == 422


class TestPatternEndpoints:
    """Tests for pattern endpoints."""

    def test_patterns_count(self, client, mock_pattern_agent):
        """Test patterns count endpoint."""
        reset_agents()
        with patch("agents.api.routers.pattern.get_pattern_agent", return_value=mock_pattern_agent):
            response = client.get("/pattern/patterns/count")
            assert response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""

    def test_not_found(self, client):
        """Test 404 for unknown endpoints."""
        response = client.get("/unknown/endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.delete("/health")
        assert response.status_code == 405
