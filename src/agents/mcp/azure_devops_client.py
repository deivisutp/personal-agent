"""Azure DevOps MCP client for work items and wiki operations."""

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from typing import Any, Optional
from enum import Enum

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from agents.core.config import get_settings

logger = logging.getLogger(__name__)


class WorkItemType(str, Enum):
    """Azure DevOps work item types."""

    EPIC = "Epic"
    FEATURE = "Feature"
    USER_STORY = "User Story"
    TASK = "Task"
    BUG = "Bug"
    ISSUE = "Issue"


class WorkItem(BaseModel):
    """Represents an Azure DevOps work item."""

    id: int
    title: str
    work_item_type: str
    state: str
    description: str = ""
    acceptance_criteria: str = ""
    assigned_to: Optional[str] = None
    area_path: str = ""
    iteration_path: str = ""
    tags: list[str] = Field(default_factory=list)
    parent_id: Optional[int] = None
    children_ids: list[int] = Field(default_factory=list)
    url: str = ""
    custom_fields: dict[str, Any] = Field(default_factory=dict)


class WikiPage(BaseModel):
    """Represents an Azure DevOps wiki page."""

    id: int
    path: str
    title: str
    content: str = ""
    version: str = ""
    url: str = ""


class AzureDevOpsProject(BaseModel):
    """Represents an Azure DevOps project."""

    id: str
    name: str
    description: str = ""
    url: str = ""


class AzureDevOpsMCPClient:
    """Client for interacting with Azure DevOps via MCP server.

    Uses MCP to fetch work items, wiki pages, and project information
    for pattern compliance evaluation and NFR generation.
    """

    def __init__(
        self,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        pat: Optional[str] = None,
    ):
        """Initialize the Azure DevOps MCP client.

        Args:
            organization: Azure DevOps organization name.
            project: Default project name.
            pat: Personal Access Token for authentication.
        """
        settings = get_settings()
        self._organization = organization or settings.azure_devops.organization
        self._project = project or settings.azure_devops.project
        self._pat = pat or settings.azure_devops.pat
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self._connected = False

        if not self._pat:
            raise ValueError(
                "Azure DevOps PAT required. Set AZURE_DEVOPS_PAT in .env or pass pat parameter."
            )
        if not self._organization:
            raise ValueError(
                "Azure DevOps organization required. Set AZURE_DEVOPS_ORG in .env."
            )

    @property
    def organization(self) -> str:
        """Get the organization name."""
        return self._organization

    @property
    def project(self) -> Optional[str]:
        """Get the default project name."""
        return self._project

    async def __aenter__(self) -> "AzureDevOpsMCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to the Azure DevOps MCP server."""
        if self._connected:
            return

        # Build organization URL from org name if not already a URL
        org_url = self._organization
        if org_url and not org_url.startswith("http"):
            org_url = f"https://dev.azure.com/{org_url}"

        # Build environment with current env vars plus Azure DevOps credentials
        env = os.environ.copy()
        env.update({
            "AZURE_DEVOPS_ORG_URL": org_url,
            "AZURE_DEVOPS_PAT": self._pat,
            "AZURE_DEVOPS_AUTH_METHOD": "pat",
        })
        if self._project:
            env["AZURE_DEVOPS_DEFAULT_PROJECT"] = self._project

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@tiberriver256/mcp-server-azure-devops"],
            env=env,
        )

        logger.debug("Starting Azure DevOps MCP server...")

        try:
            self._read, self._write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(self._read, self._write)
            )
            await asyncio.wait_for(self._session.initialize(), timeout=30.0)
            self._connected = True
            logger.debug("Azure DevOps MCP server connected successfully")
        except asyncio.TimeoutError:
            raise ConnectionError("Timeout connecting to Azure DevOps MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to Azure DevOps MCP server: {e}")
            raise ConnectionError(f"Failed to connect to Azure DevOps MCP server: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the Azure DevOps MCP server."""
        if self._connected:
            await self._exit_stack.aclose()
            self._session = None
            self._connected = False

    async def _call_tool(self, name: str, arguments: dict[str, Any], _retry: bool = True) -> Any:
        """Call an MCP tool and return the result.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            _retry: Internal flag to prevent infinite retry loops.

        Returns:
            Tool result content.
        """
        if not self._connected:
            await self.connect()

        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as e:
            error_msg = str(e).lower()
            if _retry and ("connection closed" in error_msg or "closed" in error_msg):
                # Connection was lost, reset state and reconnect
                self._connected = False
                self._session = None
                await self._exit_stack.aclose()
                self._exit_stack = AsyncExitStack()
                await self.connect()
                return await self._call_tool(name, arguments, _retry=False)
            raise

        if result.content:
            for content in result.content:
                if hasattr(content, "text"):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return content.text
        return None

    async def list_tools(self) -> list[str]:
        """List available MCP tools.

        Returns:
            List of tool names.
        """
        if not self._connected:
            await self.connect()

        try:
            result = await self._session.list_tools()
        except Exception as e:
            error_msg = str(e).lower()
            if "connection closed" in error_msg or "closed" in error_msg:
                # Connection was lost, reset state and reconnect
                self._connected = False
                self._session = None
                await self._exit_stack.aclose()
                self._exit_stack = AsyncExitStack()
                await self.connect()
                result = await self._session.list_tools()
            else:
                raise

        return [tool.name for tool in result.tools]

    async def get_work_item(
        self,
        work_item_id: int,
        project: Optional[str] = None,
    ) -> WorkItem:
        """Fetch a work item by ID.

        Args:
            work_item_id: The work item ID.
            project: Project name (uses default if not specified).

        Returns:
            WorkItem object with details.
        """
        project = project or self._project

        data = await self._call_tool(
            "get_work_item",
            {"workItemId": work_item_id},
        )

        if not isinstance(data, dict):
            raise ValueError(f"Failed to get work item {work_item_id}: {data}")

        fields = data.get("fields", {}) if data else {}

        return WorkItem(
            id=data.get("id", work_item_id),
            title=fields.get("System.Title", ""),
            work_item_type=fields.get("System.WorkItemType", ""),
            state=fields.get("System.State", ""),
            description=fields.get("System.Description", "") or "",
            acceptance_criteria=fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", "") or "",
            assigned_to=fields.get("System.AssignedTo", {}).get("displayName") if isinstance(fields.get("System.AssignedTo"), dict) else None,
            area_path=fields.get("System.AreaPath", ""),
            iteration_path=fields.get("System.IterationPath", ""),
            tags=[t.strip() for t in (fields.get("System.Tags") or "").split(";") if t.strip()],
            parent_id=self._extract_parent_id(data.get("relations", [])),
            url=data.get("_links", {}).get("html", {}).get("href", ""),
        )

    def _extract_parent_id(self, relations: list) -> Optional[int]:
        """Extract parent work item ID from relations.

        Args:
            relations: List of work item relations.

        Returns:
            Parent ID if found, None otherwise.
        """
        for rel in relations or []:
            if rel.get("rel") == "System.LinkTypes.Hierarchy-Reverse":
                url = rel.get("url", "")
                if url:
                    try:
                        return int(url.split("/")[-1])
                    except (ValueError, IndexError):
                        pass
        return None

    async def search_work_items(
        self,
        search_text: str,
        project: Optional[str] = None,
        top: int = 50,
    ) -> list[WorkItem]:
        """Search work items.

        Args:
            search_text: The text to search for.
            project: Project name.
            top: Maximum number of results.

        Returns:
            List of WorkItem objects.
        """
        project = project or self._project

        args = {"searchText": search_text, "top": top}
        if project:
            args["projectId"] = project

        result = await self._call_tool(
            "search_work_items",
            args,
        )

        work_items = []
        if isinstance(result, dict) and "results" in result:
            for item in result.get("results", [])[:top]:
                try:
                    # search_work_items returns details, but might need to fetch full item
                    wi_id = item.get("id") or item.get("fields", {}).get("System.Id")
                    if wi_id:
                        wi = await self.get_work_item(int(wi_id), project)
                        work_items.append(wi)
                except Exception:
                    pass

        return work_items

    async def get_features(
        self,
        project: Optional[str] = None,
        state: Optional[str] = None,
        area_path: Optional[str] = None,
        top: int = 20,
    ) -> list[WorkItem]:
        """Get features from the project.

        Args:
            project: Project name.
            state: Filter by state (e.g., "New", "Active", "Resolved").
            area_path: Filter by area path.
            top: Maximum number of results.

        Returns:
            List of Feature work items.
        """
        project = project or self._project

        filters = {"System.WorkItemType": ["Feature"]}
        if state:
            filters["System.State"] = [state]
        if area_path:
            filters["System.AreaPath"] = [area_path]

        args = {"searchText": "*", "filters": filters, "top": top}
        if project:
            args["projectId"] = project

        result = await self._call_tool(
            "search_work_items",
            args,
        )

        work_items = []
        if isinstance(result, dict) and "results" in result:
            for item in result.get("results", [])[:top]:
                try:
                    wi_id = item.get("id") or item.get("fields", {}).get("System.Id")
                    if wi_id:
                        wi = await self.get_work_item(int(wi_id), project)
                        work_items.append(wi)
                except Exception:
                    pass

        return work_items

    async def get_user_stories(
        self,
        feature_id: Optional[int] = None,
        project: Optional[str] = None,
        state: Optional[str] = None,
        top: int = 50,
    ) -> list[WorkItem]:
        """Get user stories, optionally filtered by parent feature.

        Args:
            feature_id: Parent feature ID to filter by.
            project: Project name.
            state: Filter by state.
            top: Maximum number of results.

        Returns:
            List of User Story work items.
        """
        project = project or self._project

        filters = {"System.WorkItemType": ["User Story"]}
        if state:
            filters["System.State"] = [state]

        args = {"searchText": "*", "filters": filters, "top": top}
        if project:
            args["projectId"] = project

        result = await self._call_tool(
            "search_work_items",
            args,
        )

        stories = []
        if isinstance(result, dict) and "results" in result:
            for item in result.get("results", [])[:top]:
                try:
                    wi_id = item.get("id") or item.get("fields", {}).get("System.Id")
                    if wi_id:
                        wi = await self.get_work_item(int(wi_id), project)
                        stories.append(wi)
                except Exception:
                    pass

        if feature_id:
            stories = [s for s in stories if s.parent_id == feature_id]

        return stories

    async def get_wiki_page(
        self,
        wiki_name: str,
        path: str,
        project: Optional[str] = None,
    ) -> WikiPage:
        """Get a wiki page by path.

        Args:
            wiki_name: Name of the wiki.
            path: Page path (e.g., "/Architecture/Patterns").
            project: Project name.

        Returns:
            WikiPage object.
        """
        project = project or self._project

        data = await self._call_tool(
            "get_wiki_page",
            {"wikiId": wiki_name, "pagePath": path, "projectId": project},
        )

        if isinstance(data, str):
            # The tool sometimes returns the raw markdown string directly instead of a structured dict
            return WikiPage(
                id=0,
                path=path,
                title=path.split("/")[-1] if path else wiki_name,
                content=data,
                version="",
                url="",
            )

        if not isinstance(data, dict):
            raise ValueError(f"Failed to get wiki page {path}: {data}")

        # Data contains a 'page' object with the page details
        page = data.get("page", data)

        return WikiPage(
            id=page.get("id", 0),
            path=page.get("path", path),
            title=path.split("/")[-1] if path else "",
            content=page.get("content", ""),
            version=data.get("eTag", ""),
            url=page.get("remoteUrl", ""),
        )

    async def list_wiki_pages(
        self,
        wiki_name: str,
        path: str = "",
        project: Optional[str] = None,
        recursive: bool = True,
    ) -> list[WikiPage]:
        """List wiki pages under a path.

        Args:
            wiki_name: Name of the wiki.
            path: Root path to list from.
            project: Project name.
            recursive: Whether to list recursively.

        Returns:
            List of WikiPage objects (without content).
        """
        project = project or self._project

        data = await self._call_tool(
            "list_wiki_pages",
            {
                "wikiId": wiki_name,
                "projectId": project,
            },
        )

        pages = []
        if data and isinstance(data, list):
            for page in data:
                if isinstance(page, dict):
                    pages.append(
                        WikiPage(
                            id=page.get("id", 0),
                            path=page.get("path", ""),
                            title=page.get("path", "").split("/")[-1],
                            url=page.get("remoteUrl", ""),
                        )
                    )

        # Filter by path since the tool just returns all pages
        if path and path != "/" and path != "":
            filtered_pages = []
            target_path = path if path.startswith("/") else f"/{path}"
            
            for page in pages:
                if recursive:
                    if page.path.startswith(target_path):
                        filtered_pages.append(page)
                else:
                    if page.path == target_path or page.path.startswith(f"{target_path}/") and page.path.count("/") == target_path.count("/") + 1:
                        filtered_pages.append(page)
            pages = filtered_pages

        return pages

    async def search_wiki(
        self,
        wiki_name: str,
        search_text: str,
        project: Optional[str] = None,
        top: int = 10,
    ) -> list[WikiPage]:
        """Search wiki pages by text.

        Args:
            wiki_name: Name of the wiki.
            search_text: Text to search for.
            project: Project name.
            top: Maximum results.

        Returns:
            List of matching WikiPage objects.
        """
        project = project or self._project

        data = await self._call_tool(
            "search_wiki",
            {
                "searchText": search_text,
                "projectId": project,
                "top": top,
            },
        )

        pages = []
        if isinstance(data, dict) and "results" in data:
            for result in data["results"]:
                if isinstance(result, dict):
                    # Azure DevOps search returns 'wiki' object inside result
                    if wiki_name and result.get("wiki", {}).get("name") != wiki_name and result.get("wiki", {}).get("id") != wiki_name:
                        continue
                        
                    pages.append(
                        WikiPage(
                            id=result.get("id", 0),
                            path=result.get("path", ""),
                            title=result.get("fileName", result.get("title", "")),
                            content=result.get("content", ""),
                            url=result.get("url", ""),
                        )
                    )

        return pages

    async def get_projects(self) -> list[AzureDevOpsProject]:
        """List all projects in the organization.

        Returns:
            List of AzureDevOpsProject objects.
        """
        data = await self._call_tool("list_projects", {})

        projects = []
        if isinstance(data, dict) and "value" in data:
            for proj in data["value"]:
                if isinstance(proj, dict):
                    projects.append(
                        AzureDevOpsProject(
                            id=proj.get("id", ""),
                            name=proj.get("name", ""),
                            description=proj.get("description", ""),
                            url=proj.get("url", ""),
                        )
                    )

        return projects

    async def create_work_item(
        self,
        work_item_type: str,
        title: str,
        description: str = "",
        project: Optional[str] = None,
        parent_id: Optional[int] = None,
        fields: Optional[dict[str, Any]] = None,
    ) -> WorkItem:
        """Create a new work item.

        Args:
            work_item_type: Type (e.g., "User Story", "Task").
            title: Work item title.
            description: Description/body.
            project: Project name.
            parent_id: Parent work item ID.
            fields: Additional fields to set.

        Returns:
            Created WorkItem.
        """
        project = project or self._project

        args = {
            "workItemType": work_item_type,
            "title": title,
        }
        
        if project:
            args["projectId"] = project
            
        if description:
            args["description"] = description
            
        if parent_id:
            args["parentId"] = parent_id
            
        if fields:
            # Add specific fields if they exist in schema
            if "System.AssignedTo" in fields:
                args["assignedTo"] = fields.pop("System.AssignedTo")
            if "System.AreaPath" in fields:
                args["areaPath"] = fields.pop("System.AreaPath")
            if "System.IterationPath" in fields:
                args["iterationPath"] = fields.pop("System.IterationPath")
                
            # Remaining fields go to additionalFields
            if fields:
                args["additionalFields"] = fields

        data = await self._call_tool(
            "create_work_item",
            args,
        )

        if not isinstance(data, dict):
            raise ValueError(f"Failed to create work item: {data}")

        return await self.get_work_item(data.get("id"), project)

    async def update_work_item(
        self,
        work_item_id: int,
        fields: dict[str, Any],
        project: Optional[str] = None,
    ) -> WorkItem:
        """Update a work item.

        Args:
            work_item_id: Work item ID.
            fields: Fields to update.
            project: Project name.

        Returns:
            Updated WorkItem.
        """
        project = project or self._project

        args = {"workItemId": work_item_id}
        
        if "System.Title" in fields:
            args["title"] = fields.pop("System.Title")
        if "System.Description" in fields:
            args["description"] = fields.pop("System.Description")
        if "System.AssignedTo" in fields:
            args["assignedTo"] = fields.pop("System.AssignedTo")
        if "System.AreaPath" in fields:
            args["areaPath"] = fields.pop("System.AreaPath")
        if "System.IterationPath" in fields:
            args["iterationPath"] = fields.pop("System.IterationPath")
        if "System.State" in fields:
            args["state"] = fields.pop("System.State")
            
        if fields:
            args["additionalFields"] = fields

        data = await self._call_tool(
            "update_work_item",
            args,
        )

        if not isinstance(data, dict):
            raise ValueError(f"Failed to update work item {work_item_id}: {data}")

        return await self.get_work_item(work_item_id, project)


def run_async(coro):
    """Helper to run async code synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)
