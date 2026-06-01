"""Tests for Azure DevOps MCP client."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from agents.mcp.azure_devops_client import (
    AzureDevOpsMCPClient,
    WorkItem,
    WikiPage,
    AzureDevOpsProject,
    WorkItemType,
)


class TestWorkItemModel:
    """Tests for WorkItem model."""

    def test_work_item_basic(self):
        """Test basic work item creation."""
        wi = WorkItem(
            id=123,
            title="Implement feature X",
            work_item_type="Feature",
            state="Active",
            description="Feature description",
        )
        assert wi.id == 123
        assert wi.title == "Implement feature X"
        assert wi.work_item_type == "Feature"
        assert wi.state == "Active"

    def test_work_item_with_all_fields(self):
        """Test work item with all fields."""
        wi = WorkItem(
            id=456,
            title="User Story",
            work_item_type="User Story",
            state="New",
            description="As a user...",
            acceptance_criteria="Given... When... Then...",
            assigned_to="John Doe",
            area_path="Project\\Team",
            iteration_path="Sprint 1",
            tags=["frontend", "priority-high"],
            parent_id=123,
            url="https://dev.azure.com/org/project/_workitems/edit/456",
        )
        assert wi.assigned_to == "John Doe"
        assert len(wi.tags) == 2
        assert wi.parent_id == 123


class TestWikiPageModel:
    """Tests for WikiPage model."""

    def test_wiki_page_basic(self):
        """Test basic wiki page creation."""
        page = WikiPage(
            id=1,
            path="/Architecture/Patterns",
            title="Patterns",
            content="# Architecture Patterns\n...",
        )
        assert page.path == "/Architecture/Patterns"
        assert page.title == "Patterns"
        assert "Architecture" in page.content


class TestAzureDevOpsMCPClient:
    """Tests for AzureDevOpsMCPClient."""

    def test_init_without_pat_raises(self):
        """Test that init without PAT raises error."""
        with patch("agents.mcp.azure_devops_client.get_settings") as mock_settings:
            mock_settings.return_value.azure_devops.pat = None
            mock_settings.return_value.azure_devops.organization = "test-org"
            with pytest.raises(ValueError, match="Azure DevOps PAT required"):
                AzureDevOpsMCPClient()

    def test_init_without_org_raises(self):
        """Test that init without organization raises error."""
        with patch("agents.mcp.azure_devops_client.get_settings") as mock_settings:
            mock_settings.return_value.azure_devops.pat = "test-pat"
            mock_settings.return_value.azure_devops.organization = None
            with pytest.raises(ValueError, match="Azure DevOps organization required"):
                AzureDevOpsMCPClient()

    def test_init_with_params(self):
        """Test init with explicit parameters."""
        client = AzureDevOpsMCPClient(
            organization="my-org",
            project="my-project",
            pat="my-pat",
        )
        assert client.organization == "my-org"
        assert client.project == "my-project"
        assert not client._connected


class TestAzureDevOpsMCPClientAsync:
    """Async tests for AzureDevOpsMCPClient."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch.object(AzureDevOpsMCPClient, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(AzureDevOpsMCPClient, "disconnect", new_callable=AsyncMock) as mock_disconnect:
                async with AzureDevOpsMCPClient(
                    organization="org",
                    project="proj",
                    pat="pat",
                ) as client:
                    mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()


class TestWorkItemType:
    """Tests for WorkItemType enum."""

    def test_work_item_types(self):
        """Test work item type values."""
        assert WorkItemType.FEATURE == "Feature"
        assert WorkItemType.USER_STORY == "User Story"
        assert WorkItemType.BUG == "Bug"
        assert WorkItemType.TASK == "Task"
