"""CLI entry points for agents."""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file before anything else
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from agents.core.config import get_settings
from agents.pr_review.agent import PRReviewAgent
from agents.pr_review.prompts import with_language
from agents.pattern.agent import PatternAgent


console = Console()


def run_pr_agent() -> None:
    """Run the PR Review Agent in interactive mode."""
    console.print(Panel.fit(
        "[bold blue]PR Review Agent[/bold blue]\n"
        "Paste your PR diff and press Enter twice to review.",
        title="🔍 Technical Refinement",
    ))

    agent = PRReviewAgent()

    if not agent.is_ready():
        console.print("[red]Error: Ollama is not available. Please ensure it's running.[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Connected to Ollama ({agent.llm.model_name})[/green]\n")

    while True:
        try:
            console.print("[bold]Enter PR diff (or 'quit' to exit):[/bold]")
            lines = []
            while True:
                line = input()
                if line.lower() == "quit":
                    console.print("[yellow]Goodbye![/yellow]")
                    return
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)

            diff = "\n".join(lines[:-1])

            if not diff.strip():
                console.print("[yellow]No input provided. Try again.[/yellow]")
                continue

            console.print("\n[blue]Analyzing...[/blue]\n")
            result = agent.quick_review(diff)

            console.print(Panel(Markdown(result), title="Review Results", border_style="green"))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def run_pattern_agent() -> None:
    """Run the Pattern Agent in interactive mode."""
    console.print(Panel.fit(
        "[bold blue]Pattern & NFR Agent[/bold blue]\n"
        "Commands: ingest <path>, evaluate <feature>, nfrs <description>, quit",
        title="📋 Pattern Compliance",
    ))

    agent = PatternAgent()

    if not agent.is_ready():
        console.print("[red]Error: Ollama is not available. Please ensure it's running.[/red]")
        sys.exit(1)

    console.print(f"[green]✓ Connected to Ollama ({agent.llm.model_name})[/green]")
    console.print(f"[blue]ℹ Patterns indexed: {agent.patterns_count}[/blue]\n")

    while True:
        try:
            command = console.input("[bold]> [/bold]").strip()

            if not command:
                continue

            if command.lower() == "quit":
                console.print("[yellow]Goodbye![/yellow]")
                break

            if command.lower().startswith("ingest "):
                path = command[7:].strip()
                try:
                    count = agent.ingest_patterns(Path(path))
                    console.print(f"[green]✓ Ingested {count} chunks[/green]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

            elif command.lower().startswith("nfrs "):
                description = command[5:].strip()
                console.print("\n[blue]Generating NFRs...[/blue]\n")
                result = agent.generate_nfrs(description)
                console.print(Panel(Markdown(result), title="Generated NFRs", border_style="green"))

            elif command.lower().startswith("evaluate "):
                feature = command[9:].strip()
                console.print("\n[blue]Evaluating feature...[/blue]\n")
                from agents.core.base_agent import AgentContext
                context = AgentContext(
                    user_input=feature,
                    metadata={"feature_title": feature[:50], "feature_description": feature},
                )
                result = agent.execute(context)
                console.print(Panel(Markdown(result.output), title="Evaluation", border_style="green"))

            else:
                console.print("[yellow]Unknown command. Use: ingest, evaluate, nfrs, or quit[/yellow]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def run_azure_devops_agent() -> None:
    """Run Azure DevOps Pattern Agent in interactive mode."""
    console.print(Panel.fit(
        "[bold blue]Azure DevOps Pattern Agent[/bold blue]\n"
        "Evaluate features and generate NFRs from Azure DevOps.",
        title="📋 Azure DevOps Integration",
    ))

    settings = get_settings()
    if not settings.azure_devops.pat:
        console.print("[red]Error: AZURE_DEVOPS_PAT not set in .env file[/red]")
        console.print("Add your Azure DevOps configuration to .env:")
        console.print("  AZURE_DEVOPS_ORG=your_organization")
        console.print("  AZURE_DEVOPS_PAT=your_personal_access_token")
        console.print("  AZURE_DEVOPS_PROJECT=your_project")
        return

    agent = PatternAgent()

    if not agent.is_ready():
        console.print("[red]Error: Ollama is not available. Please ensure it's running.[/red]")
        return

    console.print(f"[green]✓ Connected to Ollama ({agent.llm.model_name})[/green]")
    console.print(f"[green]✓ Azure DevOps configured[/green]")
    console.print(f"[blue]ℹ Patterns indexed: {agent.patterns_count}[/blue]\n")

    console.print("Commands: list [state], eval <id>, nfrs <id>, wiki <name>, quit")
    console.print("Options: --lang <code> (e.g., --lang pt-br for Brazilian Portuguese)\n")

    async def azure_loop():
        try:
            while True:
                cmd = console.input("[bold]> [/bold]").strip()

                if not cmd:
                    continue

                if cmd.lower() == "quit":
                    break

                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                try:
                    if action == "list":
                        state = arg or None
                        console.print(f"\n[blue]Fetching features...[/blue]")
                        features = await agent.list_features(state=state, limit=10)

                        table = Table(title=f"Features ({len(features)})")
                        table.add_column("ID", style="cyan")
                        table.add_column("Title")
                        table.add_column("State", style="green")

                        for f in features:
                            table.add_row(str(f["id"]), f["title"], f["state"])

                        console.print(table)
                        console.print()

                    elif action == "eval":
                        if not arg:
                            console.print("[red]Please provide a feature ID[/red]")
                            continue
                        
                        # Parse feature ID and optional arguments
                        parts = arg.split()
                        feature_id = int(parts[0])
                        include_stories = "--include-stories" in parts
                        
                        # Parse language option
                        lang = "en"
                        if "--lang" in parts:
                            lang_idx = parts.index("--lang")
                            if lang_idx + 1 < len(parts):
                                lang = parts[lang_idx + 1]
                        
                        if lang != "en":
                            agent.system_prompt = with_language(agent._default_system_prompt(), lang)
                        
                        console.print(f"\n[blue]Evaluating feature #{feature_id}...[/blue]")
                        result = await agent.evaluate_azure_feature(feature_id, include_user_stories=include_stories)

                        if result.success:
                            console.print(Panel(
                                Markdown(result.output),
                                title=f"Evaluation: Feature #{feature_id}",
                                border_style="green"
                            ))
                        else:
                            console.print(f"[red]Error: {result.output}[/red]")

                    elif action == "nfrs":
                        if not arg:
                            console.print("[yellow]Usage: nfrs <feature_id> [--include-stories] [--lang <code>][/yellow]")
                            continue
                        
                        # Parse feature ID and optional arguments
                        parts = arg.split()
                        feature_id = int(parts[0])
                        include_stories = "--include-stories" in parts
                        
                        # Parse language option
                        lang = "en"
                        if "--lang" in parts:
                            lang_idx = parts.index("--lang")
                            if lang_idx + 1 < len(parts):
                                lang = parts[lang_idx + 1]
                        
                        if lang != "en":
                            agent.system_prompt = with_language(agent._default_system_prompt(), lang)

                        console.print(f"\n[blue]Fetching feature #{feature_id}...[/blue]")
                        # Use evaluate_azure_feature to get context, then generate NFRs
                        result = await agent.evaluate_azure_feature(feature_id, include_user_stories=include_stories)

                        if result.success:
                            console.print(Panel(
                                Markdown(result.output),
                                title=f"NFRs: Feature #{feature_id}",
                                border_style="green"
                            ))

                            create = console.input("\n[bold]Create work items in Azure DevOps? (y/N):[/bold] ").strip().lower()
                            if create == "y":
                                result = await agent.generate_nfrs_for_feature(
                                    feature_id, create_work_items=True
                                )
                                created = result.metadata.get("created_work_items", [])
                                console.print(f"[green]✓ Created {len(created)} work items[/green]")
                        else:
                            console.print(f"[red]Error: {result.output}[/red]")

                    elif action == "wiki":
                        if not arg:
                            console.print("[yellow]Usage: wiki <wiki_name> [path][/yellow]")
                            continue
                        
                        parts = arg.split()
                        wiki_name = parts[0]
                        path = parts[1] if len(parts) > 1 else ""
                        
                        console.print(f"\n[blue]Ingesting wiki: {wiki_name} at path '{path}'...[/blue]")
                        chunks = await agent.ingest_wiki_patterns(wiki_name, path=path)
                        console.print(f"[green]✓ Ingested {chunks} chunks[/green]")
                        console.print(f"[blue]ℹ Total patterns: {agent.patterns_count}[/blue]\n")

                    else:
                        console.print("[yellow]Commands: list, eval, nfrs, wiki, quit[/yellow]")

                except ValueError as e:
                    console.print(f"[yellow]Invalid input: {e}[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

        finally:
            await agent.close()

    asyncio.run(azure_loop())
    console.print("[yellow]Goodbye![/yellow]")


def run_github_pr_review() -> None:
    """Run GitHub PR review in interactive mode."""
    console.print(Panel.fit(
        "[bold blue]GitHub PR Review Agent[/bold blue]\n"
        "Review PRs directly from GitHub repositories.",
        title="🐙 GitHub Integration",
    ))

    settings = get_settings()
    if not settings.github.token:
        console.print("[red]Error: GITHUB_TOKEN not set in .env file[/red]")
        console.print("Add your GitHub Personal Access Token to .env:")
        console.print("  GITHUB_TOKEN=ghp_your_token_here")
        return

    agent = PRReviewAgent()

    if not agent.is_ready():
        console.print("[red]Error: Ollama is not available. Please ensure it's running.[/red]")
        return

    console.print(f"[green]✓ Connected to Ollama ({agent.llm.model_name})[/green]")
    console.print("[green]✓ GitHub token configured[/green]\n")

    async def review_loop():
        try:
            while True:
                console.print("[bold]Enter PR (owner/repo#number) or 'quit':[/bold]")
                pr_input = input("> ").strip()

                if pr_input.lower() == "quit":
                    break

                try:
                    if "#" not in pr_input or "/" not in pr_input:
                        console.print("[yellow]Format: owner/repo#number (e.g., microsoft/vscode#12345)[/yellow]")
                        continue

                    repo_part, pr_num = pr_input.split("#")
                    owner, repo = repo_part.split("/")
                    pr_number = int(pr_num)

                    console.print(f"\n[blue]Fetching PR #{pr_number} from {owner}/{repo}...[/blue]")

                    result = await agent.review_github_pr(owner, repo, pr_number)

                    if result.success:
                        console.print(Panel(
                            Markdown(result.output),
                            title=f"Review: {owner}/{repo}#{pr_number}",
                            border_style="green"
                        ))

                        post = console.input("\n[bold]Post review to GitHub? (y/N):[/bold] ").strip().lower()
                        if post == "y":
                            client = await agent._get_github_client()
                            await client.create_pr_comment(
                                owner, repo, pr_number,
                                f"## 🤖 AI Code Review\n\n{result.output}"
                            )
                            console.print("[green]✓ Review posted to GitHub![/green]")
                    else:
                        console.print(f"[red]Error: {result.output}[/red]")

                except ValueError:
                    console.print("[yellow]Invalid format. Use: owner/repo#number[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")

                console.print()

        finally:
            await agent.close()

    asyncio.run(review_loop())
    console.print("[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "pattern":
            run_pattern_agent()
        elif sys.argv[1] == "github":
            run_github_pr_review()
        elif sys.argv[1] == "azure":
            run_azure_devops_agent()
        else:
            console.print("Usage: python -m agents.cli [pattern|github|azure]")
    else:
        run_pr_agent()
