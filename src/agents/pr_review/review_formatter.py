"""Formatters for PR review output."""

import json
import re
from typing import Optional
from pydantic import BaseModel, Field


class ReviewIssue(BaseModel):
    """A single issue found in the review."""

    severity: str  # critical, important, suggestion
    category: str  # security, performance, quality, etc.
    file: Optional[str] = None
    line: Optional[int] = None
    title: str
    description: str
    suggestion: Optional[str] = None


class StructuredReview(BaseModel):
    """Structured PR review output."""

    summary: str
    strengths: list[str] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    overall_assessment: str = ""  # approve, request_changes, comment


class ReviewFormatter:
    """Formats review output for different targets."""

    @staticmethod
    def to_markdown(review: StructuredReview) -> str:
        """Format review as Markdown for GitHub comments.

        Args:
            review: Structured review data.

        Returns:
            Markdown formatted string.
        """
        lines = ["## 🔍 Code Review Summary", "", review.summary, ""]

        if review.strengths:
            lines.append("### ✅ Strengths")
            for s in review.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if review.issues:
            lines.append("### 🔎 Issues Found")
            lines.append("")

            critical = [i for i in review.issues if i.severity == "critical"]
            important = [i for i in review.issues if i.severity == "important"]
            suggestions = [i for i in review.issues if i.severity == "suggestion"]

            if critical:
                lines.append("#### 🔴 Critical")
                for issue in critical:
                    lines.extend(ReviewFormatter._format_issue(issue))

            if important:
                lines.append("#### 🟡 Important")
                for issue in important:
                    lines.extend(ReviewFormatter._format_issue(issue))

            if suggestions:
                lines.append("#### 🟢 Suggestions")
                for issue in suggestions:
                    lines.extend(ReviewFormatter._format_issue(issue))

        if review.suggestions:
            lines.append("### 💡 Additional Suggestions")
            for s in review.suggestions:
                lines.append(f"- {s}")
            lines.append("")

        if review.questions:
            lines.append("### ❓ Questions")
            for q in review.questions:
                lines.append(f"- {q}")
            lines.append("")

        if review.overall_assessment:
            emoji = {
                "approve": "✅",
                "request_changes": "🔄",
                "comment": "💬",
            }.get(review.overall_assessment, "💬")
            lines.append(f"### {emoji} Overall: {review.overall_assessment.replace('_', ' ').title()}")

        return "\n".join(lines)

    @staticmethod
    def _format_issue(issue: ReviewIssue) -> list[str]:
        """Format a single issue.

        Args:
            issue: The issue to format.

        Returns:
            List of formatted lines.
        """
        lines = []
        location = ""
        if issue.file:
            location = f" (`{issue.file}"
            if issue.line:
                location += f":{issue.line}"
            location += "`)"

        lines.append(f"**{issue.title}**{location}")
        lines.append(f"  - {issue.description}")

        if issue.suggestion:
            lines.append(f"  - 💡 *Suggestion*: {issue.suggestion}")

        lines.append("")
        return lines

    @staticmethod
    def to_github_review(
        review: StructuredReview,
    ) -> dict:
        """Format review for GitHub API.

        Args:
            review: Structured review data.

        Returns:
            Dict with body and event for GitHub review API.
        """
        body = ReviewFormatter.to_markdown(review)

        if any(i.severity == "critical" for i in review.issues):
            event = "REQUEST_CHANGES"
        elif review.overall_assessment == "approve":
            event = "APPROVE"
        else:
            event = "COMMENT"

        return {
            "body": body,
            "event": event,
        }

    @staticmethod
    def parse_llm_response(response: str) -> StructuredReview:
        """Parse LLM response into structured review.

        Attempts to extract structured data from free-form LLM output.

        Args:
            response: Raw LLM response text.

        Returns:
            StructuredReview object.
        """
        review = StructuredReview(summary="")

        sections = re.split(r'\n##\s+', response)

        for section in sections:
            section_lower = section.lower()

            if "summary" in section_lower[:50]:
                review.summary = section.split("\n", 1)[-1].strip()

            elif "strength" in section_lower[:50]:
                items = re.findall(r'[-*]\s+(.+)', section)
                review.strengths = [i.strip() for i in items]

            elif "issue" in section_lower[:50] or "problem" in section_lower[:50]:
                issues = ReviewFormatter._extract_issues(section)
                review.issues.extend(issues)

            elif "suggestion" in section_lower[:50]:
                items = re.findall(r'[-*]\s+(.+)', section)
                review.suggestions = [i.strip() for i in items]

            elif "question" in section_lower[:50]:
                items = re.findall(r'[-*]\s+(.+)', section)
                review.questions = [i.strip() for i in items]

        if not review.summary:
            review.summary = response[:500] + "..." if len(response) > 500 else response

        return review

    @staticmethod
    def _extract_issues(section: str) -> list[ReviewIssue]:
        """Extract issues from a section.

        Args:
            section: Text section containing issues.

        Returns:
            List of ReviewIssue objects.
        """
        issues = []

        issue_blocks = re.split(r'\n(?=[-*🔴🟡🟢])', section)

        for block in issue_blocks:
            if not block.strip():
                continue

            severity = "suggestion"
            if "🔴" in block or "critical" in block.lower():
                severity = "critical"
            elif "🟡" in block or "important" in block.lower():
                severity = "important"

            lines = block.strip().split("\n")
            title = re.sub(r'^[-*🔴🟡🟢\s]+', '', lines[0]).strip()

            if not title or len(title) < 3:
                continue

            description = " ".join(lines[1:]).strip() if len(lines) > 1 else ""

            file_match = re.search(r'`([^`]+\.\w+)(?::(\d+))?`', block)
            file_path = file_match.group(1) if file_match else None
            line_num = int(file_match.group(2)) if file_match and file_match.group(2) else None

            issues.append(
                ReviewIssue(
                    severity=severity,
                    category="general",
                    file=file_path,
                    line=line_num,
                    title=title,
                    description=description,
                )
            )

        return issues
