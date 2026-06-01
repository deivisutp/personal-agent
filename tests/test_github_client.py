"""Tests for GitHub MCP client."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from agents.mcp.github_client import (
    GitHubMCPClient,
    PullRequest,
    PRFile,
    PRComment,
)


class TestPullRequestModel:
    """Tests for PullRequest model."""

    def test_pull_request_basic(self):
        """Test basic PR creation."""
        pr = PullRequest(
            number=123,
            title="Fix bug",
            state="open",
            author="testuser",
            base_branch="main",
            head_branch="fix-bug",
        )
        assert pr.number == 123
        assert pr.title == "Fix bug"
        assert pr.state == "open"

    def test_pull_request_with_files(self):
        """Test PR with files."""
        pr = PullRequest(
            number=1,
            title="Test",
            state="open",
            author="user",
            base_branch="main",
            head_branch="feature",
            files=[
                PRFile(filename="test.py", status="modified", additions=10, deletions=5),
            ],
        )
        assert len(pr.files) == 1
        assert pr.files[0].filename == "test.py"


class TestPRFile:
    """Tests for PRFile model."""

    def test_pr_file_basic(self):
        """Test basic file creation."""
        f = PRFile(filename="src/main.py", status="added")
        assert f.filename == "src/main.py"
        assert f.status == "added"
        assert f.additions == 0
        assert f.deletions == 0

    def test_pr_file_with_patch(self):
        """Test file with patch."""
        f = PRFile(
            filename="test.py",
            status="modified",
            additions=5,
            deletions=2,
            patch="@@ -1,3 +1,5 @@\n+new line",
        )
        assert f.patch is not None
        assert "+new line" in f.patch


class TestGitHubMCPClient:
    """Tests for GitHubMCPClient."""

    def test_init_without_token_raises(self):
        """Test that init without token raises error."""
        with patch("agents.mcp.github_client.get_settings") as mock_settings:
            mock_settings.return_value.github.token = None
            with pytest.raises(ValueError, match="GitHub token required"):
                GitHubMCPClient()

    def test_init_with_token(self):
        """Test init with token parameter."""
        client = GitHubMCPClient(token="test_token")
        assert client._token == "test_token"
        assert not client._connected


class TestGitHubMCPClientAsync:
    """Async tests for GitHubMCPClient."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch.object(GitHubMCPClient, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(GitHubMCPClient, "disconnect", new_callable=AsyncMock) as mock_disconnect:
                async with GitHubMCPClient(token="test") as client:
                    mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
