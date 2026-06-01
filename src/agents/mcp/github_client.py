"""GitHub MCP client for PR operations."""

import asyncio
import json
from typing import Any, Optional
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from agents.core.config import get_settings


class PRFile(BaseModel):
    """Represents a file changed in a PR."""

    filename: str
    status: str  # added, removed, modified, renamed
    additions: int = 0
    deletions: int = 0
    patch: Optional[str] = None


class PRComment(BaseModel):
    """Represents a comment on a PR."""

    id: int
    user: str
    body: str
    path: Optional[str] = None
    line: Optional[int] = None
    created_at: str


class PullRequest(BaseModel):
    """Represents a Pull Request."""

    number: int
    title: str
    description: str = ""
    state: str  # open, closed, merged
    author: str
    base_branch: str
    head_branch: str
    files: list[PRFile] = Field(default_factory=list)
    comments: list[PRComment] = Field(default_factory=list)
    diff: str = ""
    url: str = ""


class GitHubMCPClient:
    """Client for interacting with GitHub via MCP server.

    Uses the official @modelcontextprotocol/server-github MCP server
    to fetch PR data, post comments, and manage reviews.
    """

    def __init__(self, token: Optional[str] = None):
        """Initialize the GitHub MCP client.

        Args:
            token: GitHub personal access token. If None, uses settings.
        """
        settings = get_settings()
        self._token = token or settings.github.token
        self._session: Optional[ClientSession] = None
        self._connected = False

        if not self._token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN in .env or pass token parameter."
            )

    async def __aenter__(self) -> "GitHubMCPClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to the GitHub MCP server."""
        if self._connected:
            return

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": self._token},
        )

        self._transport = await stdio_client(server_params).__aenter__()
        self._read, self._write = self._transport
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()
        await self._session.initialize()
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from the GitHub MCP server."""
        if self._session:
            await self._session.__aexit__(None, None, None)
        if hasattr(self, "_transport"):
            await self._transport.__aexit__(None, None, None)
        self._connected = False

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool and return the result.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            Tool result content.
        """
        if not self._connected:
            await self.connect()

        result = await self._session.call_tool(name, arguments)

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

        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def get_pull_request(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> PullRequest:
        """Fetch a pull request with its details.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.

        Returns:
            PullRequest object with details.
        """
        pr_data = await self._call_tool(
            "get_pull_request",
            {"owner": owner, "repo": repo, "pull_number": pr_number},
        )

        files_data = await self._call_tool(
            "get_pull_request_files",
            {"owner": owner, "repo": repo, "pull_number": pr_number},
        )

        diff = await self._call_tool(
            "get_pull_request_diff",
            {"owner": owner, "repo": repo, "pull_number": pr_number},
        )

        files = []
        if files_data:
            for f in files_data:
                files.append(
                    PRFile(
                        filename=f.get("filename", ""),
                        status=f.get("status", "modified"),
                        additions=f.get("additions", 0),
                        deletions=f.get("deletions", 0),
                        patch=f.get("patch"),
                    )
                )

        return PullRequest(
            number=pr_data.get("number", pr_number),
            title=pr_data.get("title", ""),
            description=pr_data.get("body", "") or "",
            state=pr_data.get("state", "open"),
            author=pr_data.get("user", {}).get("login", "unknown"),
            base_branch=pr_data.get("base", {}).get("ref", "main"),
            head_branch=pr_data.get("head", {}).get("ref", ""),
            files=files,
            diff=diff if isinstance(diff, str) else "",
            url=pr_data.get("html_url", ""),
        )

    async def get_pr_comments(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> list[PRComment]:
        """Fetch comments on a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.

        Returns:
            List of PRComment objects.
        """
        comments_data = await self._call_tool(
            "get_pull_request_comments",
            {"owner": owner, "repo": repo, "pull_number": pr_number},
        )

        comments = []
        if comments_data:
            for c in comments_data:
                comments.append(
                    PRComment(
                        id=c.get("id", 0),
                        user=c.get("user", {}).get("login", "unknown"),
                        body=c.get("body", ""),
                        path=c.get("path"),
                        line=c.get("line"),
                        created_at=c.get("created_at", ""),
                    )
                )

        return comments

    async def create_pr_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
    ) -> dict[str, Any]:
        """Create a comment on a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.
            body: Comment body (markdown supported).

        Returns:
            Created comment data.
        """
        return await self._call_tool(
            "create_pull_request_review",
            {
                "owner": owner,
                "repo": repo,
                "pull_number": pr_number,
                "body": body,
                "event": "COMMENT",
            },
        )

    async def create_review_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        path: str,
        line: int,
        side: str = "RIGHT",
    ) -> dict[str, Any]:
        """Create a review comment on a specific line.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: PR number.
            body: Comment body.
            path: File path.
            line: Line number.
            side: Side of the diff (LEFT or RIGHT).

        Returns:
            Created comment data.
        """
        return await self._call_tool(
            "create_review_comment",
            {
                "owner": owner,
                "repo": repo,
                "pull_number": pr_number,
                "body": body,
                "path": path,
                "line": line,
                "side": side,
            },
        )

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None,
    ) -> str:
        """Get the content of a file from the repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File path.
            ref: Git ref (branch, tag, commit). Defaults to default branch.

        Returns:
            File content as string.
        """
        args = {"owner": owner, "repo": repo, "path": path}
        if ref:
            args["ref"] = ref

        result = await self._call_tool("get_file_contents", args)

        if isinstance(result, dict) and "content" in result:
            import base64
            return base64.b64decode(result["content"]).decode("utf-8")

        return result if isinstance(result, str) else ""

    async def search_repositories(
        self,
        query: str,
        per_page: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for repositories.

        Args:
            query: Search query.
            per_page: Results per page.

        Returns:
            List of repository data.
        """
        result = await self._call_tool(
            "search_repositories",
            {"query": query, "perPage": per_page},
        )
        return result.get("items", []) if isinstance(result, dict) else []

    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        per_page: int = 10,
    ) -> list[dict[str, Any]]:
        """List pull requests for a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: PR state (open, closed, all).
            per_page: Results per page.

        Returns:
            List of PR data.
        """
        return await self._call_tool(
            "list_pull_requests",
            {"owner": owner, "repo": repo, "state": state, "perPage": per_page},
        )


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
