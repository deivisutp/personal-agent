"""PR Review Agent implementation."""

import asyncio
from typing import Optional

from agents.core.base_agent import BaseAgent, AgentContext, AgentResult
from agents.core.llm_client import OllamaClient
from agents.mcp.github_client import GitHubMCPClient, PullRequest


class PRReviewAgent(BaseAgent):
    """Agent for reviewing Pull Requests and providing technical refinement feedback.

    This agent analyzes PR diffs, identifies issues, and provides actionable feedback
    for code quality, best practices, and potential bugs.
    """

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        system_prompt: Optional[str] = None,
        github_token: Optional[str] = None,
    ):
        """Initialize the PR Review Agent.

        Args:
            llm_client: Optional pre-configured LLM client.
            system_prompt: Optional custom system prompt.
            github_token: Optional GitHub token. If None, uses settings.
        """
        super().__init__(
            name="PR Review Agent",
            description="Reviews Pull Requests for technical refinement and code quality",
            system_prompt=system_prompt,
            llm_client=llm_client,
        )
        self._github_token = github_token
        self._github_client: Optional[GitHubMCPClient] = None

    def _default_system_prompt(self) -> str:
        """Return the default system prompt for PR review."""
        return """You are an expert Senior Software Engineer and Code Reviewer.
Your role is to review Pull Requests and provide constructive, actionable feedback.

## Your Review Focus Areas:
1. **Code Quality**: Clean code principles, readability, maintainability
2. **Best Practices**: Design patterns, SOLID principles, DRY/KISS
3. **Security**: Potential vulnerabilities, input validation, authentication
4. **Performance**: Algorithmic efficiency, resource usage, caching opportunities
5. **Testing**: Test coverage, edge cases, test quality
6. **Documentation**: Comments, docstrings, README updates

## Review Guidelines:
- Be constructive and specific - explain WHY something is an issue
- Provide code examples when suggesting changes
- Prioritize issues: 🔴 Critical, 🟡 Important, 🟢 Suggestion
- Acknowledge good practices and improvements
- Consider the context and constraints of the change

## Output Format:
Structure your review with:
1. **Summary**: Brief overview of the changes
2. **Strengths**: What's done well
3. **Issues Found**: Categorized by severity
4. **Suggestions**: Optional improvements
5. **Questions**: Clarifications needed from the author

Be thorough but respectful. Your goal is to help improve the code and mentor the developer."""

    def execute(self, context: AgentContext) -> AgentResult:
        """Execute a PR review.

        Args:
            context: Context containing PR diff and metadata.
                Expected metadata keys:
                - pr_title: Title of the PR
                - pr_description: PR description
                - pr_diff: The diff content
                - files_changed: List of changed files

        Returns:
            AgentResult with the review feedback.
        """
        pr_title = context.metadata.get("pr_title", "Untitled PR")
        pr_description = context.metadata.get("pr_description", "")
        pr_diff = context.metadata.get("pr_diff", context.user_input)
        files_changed = context.metadata.get("files_changed", [])

        review_prompt = self._build_review_prompt(
            pr_title=pr_title,
            pr_description=pr_description,
            pr_diff=pr_diff,
            files_changed=files_changed,
        )

        self.log_info(f"Reviewing PR: {pr_title}")
        self.log_info(f"Files changed: {len(files_changed)}")

        try:
            response = self.chat(review_prompt, remember=False)

            return AgentResult(
                success=True,
                output=response.content,
                reasoning="PR review completed successfully",
                metadata={
                    "pr_title": pr_title,
                    "files_reviewed": len(files_changed),
                    "model": response.model,
                },
            )
        except Exception as e:
            self.log_error(f"Review failed: {e}")
            return AgentResult(
                success=False,
                output=f"Failed to complete review: {e}",
                reasoning=str(e),
            )

    def _build_review_prompt(
        self,
        pr_title: str,
        pr_description: str,
        pr_diff: str,
        files_changed: list[str],
    ) -> str:
        """Build the review prompt from PR components.

        Args:
            pr_title: PR title.
            pr_description: PR description.
            pr_diff: The diff content.
            files_changed: List of changed file paths.

        Returns:
            Formatted prompt string.
        """
        files_list = "\n".join(f"- {f}" for f in files_changed) if files_changed else "Not specified"

        return f"""Please review the following Pull Request:

## PR Title
{pr_title}

## PR Description
{pr_description or "No description provided"}

## Files Changed
{files_list}

## Diff
```diff
{pr_diff}
```

Please provide a comprehensive code review following the guidelines in your instructions."""

    def quick_review(self, diff: str, focus: Optional[str] = None) -> str:
        """Perform a quick review of a diff without full context.

        Args:
            diff: The diff content to review.
            focus: Optional specific area to focus on (e.g., "security", "performance").

        Returns:
            Review feedback as string.
        """
        focus_instruction = ""
        if focus:
            focus_instruction = f"\n\nPlease focus specifically on: {focus}"

        prompt = f"""Review this code diff and provide brief, actionable feedback:{focus_instruction}

```diff
{diff}
```"""

        response = self.chat(prompt, remember=False)
        return response.content

    async def _get_github_client(self) -> GitHubMCPClient:
        """Get or create the GitHub MCP client.

        Returns:
            Connected GitHubMCPClient instance.
        """
        if self._github_client is None:
            self._github_client = GitHubMCPClient(token=self._github_token)
            await self._github_client.connect()
        return self._github_client

    async def review_github_pr(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        post_comment: bool = False,
    ) -> AgentResult:
        """Review a GitHub PR by fetching it via MCP.

        Args:
            owner: Repository owner (e.g., "microsoft").
            repo: Repository name (e.g., "vscode").
            pr_number: Pull request number.
            post_comment: If True, post the review as a PR comment.

        Returns:
            AgentResult with the review.
        """
        self.log_info(f"Fetching PR #{pr_number} from {owner}/{repo}...")

        try:
            client = await self._get_github_client()
            pr = await client.get_pull_request(owner, repo, pr_number)

            self.log_info(f"PR: {pr.title}")
            self.log_info(f"Author: {pr.author}")
            self.log_info(f"Files: {len(pr.files)}")

            context = AgentContext(
                user_input=f"Review PR #{pr_number}",
                metadata={
                    "pr_title": pr.title,
                    "pr_description": pr.description,
                    "pr_diff": pr.diff,
                    "files_changed": [f.filename for f in pr.files],
                    "pr_url": pr.url,
                    "pr_author": pr.author,
                    "pr_state": pr.state,
                },
            )

            result = self.execute(context)

            if post_comment and result.success:
                self.log_info("Posting review comment to GitHub...")
                await client.create_pr_comment(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    body=f"## 🤖 AI Code Review\n\n{result.output}",
                )
                result.metadata["comment_posted"] = True
                self.log_success("Review posted to GitHub!")

            return result

        except Exception as e:
            self.log_error(f"Failed to review GitHub PR: {e}")
            return AgentResult(
                success=False,
                output=f"Failed to fetch or review PR: {e}",
                reasoning=str(e),
            )

    def review_github_pr_sync(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        post_comment: bool = False,
    ) -> AgentResult:
        """Synchronous wrapper for review_github_pr.

        Args:
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            post_comment: If True, post the review as a PR comment.

        Returns:
            AgentResult with the review.
        """
        return asyncio.run(
            self.review_github_pr(owner, repo, pr_number, post_comment)
        )

    async def list_open_prs(
        self,
        owner: str,
        repo: str,
        limit: int = 10,
    ) -> list[dict]:
        """List open PRs for a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            limit: Maximum number of PRs to return.

        Returns:
            List of PR summaries.
        """
        client = await self._get_github_client()
        prs = await client.list_pull_requests(owner, repo, state="open", per_page=limit)

        return [
            {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "author": pr.get("user", {}).get("login"),
                "url": pr.get("html_url"),
            }
            for pr in (prs or [])
        ]

    async def close(self) -> None:
        """Close the GitHub MCP client connection."""
        if self._github_client:
            await self._github_client.disconnect()
            self._github_client = None
