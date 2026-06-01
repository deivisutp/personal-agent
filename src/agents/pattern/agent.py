"""Pattern and NFR Agent implementation."""

import asyncio
from pathlib import Path
from typing import Optional

from agents.core.base_agent import BaseAgent, AgentContext, AgentResult
from agents.core.llm_client import OllamaClient
from agents.rag.vector_store import ChromaVectorStore
from agents.rag.document_loader import DocumentLoader
from agents.mcp.azure_devops_client import AzureDevOpsMCPClient, WorkItem, WikiPage


class PatternAgent(BaseAgent):
    """Agent for evaluating features against company patterns and generating NFRs.

    This agent uses RAG to search through internal documentation and patterns,
    then evaluates features and helps write consistent non-functional requirements.
    """

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        system_prompt: Optional[str] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        azure_devops_org: Optional[str] = None,
        azure_devops_project: Optional[str] = None,
        azure_devops_pat: Optional[str] = None,
    ):
        """Initialize the Pattern Agent.

        Args:
            llm_client: Optional pre-configured LLM client.
            system_prompt: Optional custom system prompt.
            vector_store: Optional pre-configured vector store for patterns.
            azure_devops_org: Azure DevOps organization name.
            azure_devops_project: Azure DevOps project name.
            azure_devops_pat: Azure DevOps Personal Access Token.
        """
        super().__init__(
            name="Pattern Agent",
            description="Evaluates features against company patterns and generates NFRs",
            system_prompt=system_prompt,
            llm_client=llm_client,
        )

        self._vector_store = vector_store or ChromaVectorStore(
            collection_name="company_patterns"
        )
        self._document_loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)

        self._azure_devops_org = azure_devops_org
        self._azure_devops_project = azure_devops_project
        self._azure_devops_pat = azure_devops_pat
        self._azure_client: Optional[AzureDevOpsMCPClient] = None

    @property
    def vector_store(self) -> ChromaVectorStore:
        """Get the vector store."""
        return self._vector_store

    @property
    def patterns_count(self) -> int:
        """Get the number of indexed patterns."""
        return self._vector_store.count

    def _default_system_prompt(self) -> str:
        """Return the default system prompt for pattern evaluation."""
        return """You are an expert Software Architect specializing in enterprise patterns and non-functional requirements.
Your role is to evaluate features against company standards and help write consistent NFRs.

## Your Responsibilities:
1. **Pattern Compliance**: Evaluate if features align with established patterns
2. **NFR Generation**: Write clear, measurable non-functional requirements
3. **Gap Analysis**: Identify missing requirements or pattern violations
4. **Best Practices**: Recommend industry-standard approaches

## NFR Categories to Consider:
- **Performance**: Response times, throughput, resource utilization
- **Scalability**: Horizontal/vertical scaling, load handling
- **Security**: Authentication, authorization, data protection, compliance
- **Reliability**: Availability, fault tolerance, disaster recovery
- **Maintainability**: Code quality, documentation, modularity
- **Observability**: Logging, monitoring, alerting, tracing
- **Usability**: Accessibility, internationalization, UX standards

## NFR Writing Guidelines:
- Be SPECIFIC and MEASURABLE (use numbers, percentages, SLAs)
- Include acceptance criteria
- Consider edge cases and failure scenarios
- Reference relevant company patterns when applicable
- Use consistent terminology

## Output Format:
When generating NFRs, structure them as:
```
NFR-[ID]: [Title]
Category: [Category]
Priority: [Critical/High/Medium/Low]
Description: [Clear description]
Acceptance Criteria:
  - [Measurable criterion 1]
  - [Measurable criterion 2]
Related Patterns: [Pattern references if any]
```

Be thorough and ensure all NFRs are actionable and verifiable."""

    def execute(self, context: AgentContext) -> AgentResult:
        """Execute pattern evaluation and NFR generation.

        Args:
            context: Context containing feature details.
                Expected metadata keys:
                - feature_title: Title of the feature
                - feature_description: Feature description
                - user_stories: Optional list of user stories
                - existing_nfrs: Optional existing NFRs to consider

        Returns:
            AgentResult with evaluation and generated NFRs.
        """
        feature_title = context.metadata.get("feature_title", "Untitled Feature")
        feature_description = context.metadata.get("feature_description", context.user_input)
        user_stories = context.metadata.get("user_stories", [])
        existing_nfrs = context.metadata.get("existing_nfrs", [])

        self.log_info(f"Evaluating feature: {feature_title}")

        # Truncate search query to avoid context errors in vector DB
        search_query = feature_description
        if len(search_query) > 2000:
            search_query = search_query[:2000]

        relevant_patterns = self._retrieve_relevant_patterns(search_query)
        self.log_info(f"Found {len(relevant_patterns)} relevant patterns")

        evaluation_prompt = self._build_evaluation_prompt(
            feature_title=feature_title,
            feature_description=feature_description,
            user_stories=user_stories,
            existing_nfrs=existing_nfrs,
            relevant_patterns=relevant_patterns,
        )

        try:
            response = self.chat(evaluation_prompt, remember=False)

            return AgentResult(
                success=True,
                output=response.content,
                reasoning="Feature evaluation completed successfully",
                metadata={
                    "feature_title": feature_title,
                    "patterns_referenced": len(relevant_patterns),
                    "model": response.model,
                },
                suggestions=[p["content"][:100] + "..." for p in relevant_patterns[:3]],
            )
        except Exception as e:
            self.log_error(f"Evaluation failed: {e}")
            return AgentResult(
                success=False,
                output=f"Failed to complete evaluation: {e}",
                reasoning=str(e),
            )

    def _retrieve_relevant_patterns(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Retrieve relevant patterns from the vector store.

        Args:
            query: Search query (usually feature description).
            n_results: Maximum number of patterns to retrieve.

        Returns:
            List of pattern dictionaries with content and metadata.
        """
        if self._vector_store.count == 0:
            return []

        results = self._vector_store.search(query, n_results=n_results)

        return [
            {
                "content": r.content,
                "metadata": r.metadata,
                "relevance_score": 1 - r.score,
            }
            for r in results
        ]

    def _build_evaluation_prompt(
        self,
        feature_title: str,
        feature_description: str,
        user_stories: list[str],
        existing_nfrs: list[str],
        relevant_patterns: list[dict],
    ) -> str:
        """Build the evaluation prompt.

        Args:
            feature_title: Feature title.
            feature_description: Feature description.
            user_stories: List of user stories.
            existing_nfrs: Existing NFRs.
            relevant_patterns: Retrieved patterns from vector store.

        Returns:
            Formatted prompt string.
        """
        stories_text = "\n".join(f"- {s}" for s in user_stories) if user_stories else "None provided"
        nfrs_text = "\n".join(f"- {n}" for n in existing_nfrs) if existing_nfrs else "None provided"

        patterns_text = ""
        if relevant_patterns:
            patterns_text = "\n\n## Relevant Company Patterns\n"
            for i, p in enumerate(relevant_patterns, 1):
                source = p["metadata"].get("source", "Unknown")
                patterns_text += f"\n### Pattern {i} (from {source})\n{p['content']}\n"
        else:
            patterns_text = "\n\n## Relevant Company Patterns\nNo patterns indexed yet. Proceeding with general best practices.\n"

        return f"""Please evaluate the following feature and generate appropriate Non-Functional Requirements:

## Feature Title
{feature_title}

## Feature Description
{feature_description}

## User Stories
{stories_text}

## Existing NFRs
{nfrs_text}
{patterns_text}

## Your Task
1. Evaluate if this feature aligns with the company patterns shown above
2. Identify any gaps or potential issues
3. Generate comprehensive NFRs for this feature
4. Prioritize the NFRs based on criticality

Please provide your analysis and NFR recommendations."""

    def ingest_patterns(
        self,
        source: Path | str,
        recursive: bool = True,
    ) -> int:
        """Ingest pattern documents into the vector store.

        Args:
            source: Path to file or directory containing patterns.
            recursive: Whether to search subdirectories.

        Returns:
            Number of chunks ingested.
        """
        source = Path(source)

        if source.is_file():
            chunks = self._document_loader.load_file(source)
        elif source.is_dir():
            chunks = self._document_loader.load_directory(source, recursive=recursive)
        else:
            raise ValueError(f"Source not found: {source}")

        if not chunks:
            self.log_warning("No documents found to ingest")
            return 0

        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._vector_store.add_documents(documents=documents, metadatas=metadatas)

        self.log_success(f"Ingested {len(chunks)} chunks from {source}")
        return len(chunks)

    def ingest_text(
        self,
        text: str,
        source_name: str = "manual_input",
    ) -> int:
        """Ingest raw text as a pattern.

        Args:
            text: Pattern text content.
            source_name: Name to identify this pattern source.

        Returns:
            Number of chunks ingested.
        """
        chunks = self._document_loader.load_text(
            text,
            metadata={"source": source_name},
        )

        documents = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._vector_store.add_documents(documents=documents, metadatas=metadatas)

        self.log_success(f"Ingested {len(chunks)} chunks from {source_name}")
        return len(chunks)

    def generate_nfrs(
        self,
        feature_description: str,
        categories: Optional[list[str]] = None,
    ) -> str:
        """Generate NFRs for a feature description.

        Args:
            feature_description: Description of the feature.
            categories: Optional specific NFR categories to focus on.

        Returns:
            Generated NFRs as formatted string.
        """
        category_focus = ""
        if categories:
            category_focus = f"\n\nFocus specifically on these NFR categories: {', '.join(categories)}"

        relevant_patterns = self._retrieve_relevant_patterns(feature_description)

        patterns_context = ""
        if relevant_patterns:
            patterns_context = "\n\nRelevant patterns to consider:\n"
            for p in relevant_patterns[:3]:
                patterns_context += f"- {p['content'][:200]}...\n"

        prompt = f"""Generate comprehensive Non-Functional Requirements for this feature:

{feature_description}
{patterns_context}
{category_focus}

Provide well-structured, measurable NFRs following the format in your instructions."""

        response = self.chat(prompt, remember=False)
        return response.content

    async def _get_azure_client(self) -> AzureDevOpsMCPClient:
        """Get or create the Azure DevOps MCP client.

        Returns:
            Connected AzureDevOpsMCPClient instance.
        """
        if self._azure_client is None:
            self._azure_client = AzureDevOpsMCPClient(
                organization=self._azure_devops_org,
                project=self._azure_devops_project,
                pat=self._azure_devops_pat,
            )
            await self._azure_client.connect()
        return self._azure_client

    async def evaluate_azure_feature(
        self,
        feature_id: int,
        project: Optional[str] = None,
        include_user_stories: bool = False,
    ) -> AgentResult:
        """Evaluate an Azure DevOps feature and generate NFRs.

        Args:
            feature_id: Azure DevOps feature work item ID.
            project: Project name (uses default if not specified).
            include_user_stories: Whether to fetch child user stories. Defaults to False.

        Returns:
            AgentResult with evaluation and NFRs.
        """
        self.log_info(f"Fetching feature #{feature_id} from Azure DevOps...")

        try:
            client = await self._get_azure_client()
            feature = await client.get_work_item(feature_id, project)

            self.log_info(f"Feature: {feature.title}")
            self.log_info(f"State: {feature.state}")

            # Truncate long texts to avoid context length issues
            MAX_TEXT_LENGTH = 4000
            
            desc = feature.description
            if desc and len(desc) > MAX_TEXT_LENGTH:
                desc = desc[:MAX_TEXT_LENGTH] + "\n\n... (Description truncated due to length)"
                
            acc_criteria = feature.acceptance_criteria
            if acc_criteria and len(acc_criteria) > MAX_TEXT_LENGTH:
                acc_criteria = acc_criteria[:MAX_TEXT_LENGTH] + "\n\n... (Acceptance criteria truncated due to length)"

            user_stories = []
            if include_user_stories:
                stories = await client.get_user_stories(
                    feature_id=feature_id,
                    project=project,
                )
                
                # Limit number of stories
                MAX_STORIES = 20
                if len(stories) > MAX_STORIES:
                    stories = stories[:MAX_STORIES]
                    
                user_stories = [f"{s.title}: {(s.description or '')[:200]}" for s in stories]
                self.log_info(f"User Stories: {len(user_stories)}")

            context = AgentContext(
                user_input=f"Evaluate feature #{feature_id}",
                metadata={
                    "feature_title": feature.title,
                    "feature_description": desc,
                    "acceptance_criteria": acc_criteria,
                    "user_stories": user_stories,
                    "area_path": feature.area_path,
                    "tags": feature.tags,
                    "work_item_url": feature.url,
                },
            )

            result = self.execute(context)
            result.metadata["azure_devops_id"] = feature_id
            result.metadata["work_item_url"] = feature.url

            return result

        except Exception as e:
            self.log_error(f"Failed to evaluate Azure DevOps feature: {e}")
            return AgentResult(
                success=False,
                output=f"Failed to fetch or evaluate feature: {e}",
                reasoning=str(e),
            )

    def evaluate_azure_feature_sync(
        self,
        feature_id: int,
        project: Optional[str] = None,
        include_user_stories: bool = False,
    ) -> AgentResult:
        """Synchronous wrapper for evaluate_azure_feature.

        Args:
            feature_id: Azure DevOps feature work item ID.
            project: Project name (uses default if not specified).
            include_user_stories: Whether to fetch child user stories. Defaults to False.

        Returns:
            AgentResult with evaluation and NFRs.
        """
        return asyncio.run(
            self.evaluate_azure_feature(feature_id, project, include_user_stories)
        )

    async def ingest_wiki_patterns(
        self,
        wiki_name: str,
        path: str = "",
        project: Optional[str] = None,
        recursive: bool = True,
    ) -> int:
        """Ingest patterns from Azure DevOps wiki into vector store.

        Args:
            wiki_name: Name of the wiki.
            path: Root path to ingest from.
            project: Project name.
            recursive: Whether to ingest recursively.

        Returns:
            Number of chunks ingested.
        """
        self.log_info(f"Fetching wiki pages from {wiki_name}{path}...")

        try:
            client = await self._get_azure_client()

            pages = await client.list_wiki_pages(
                wiki_name=wiki_name,
                path=path,
                project=project,
                recursive=recursive,
            )

            self.log_info(f"Found {len(pages)} wiki pages")

            total_chunks = 0
            for page in pages:
                try:
                    full_page = await client.get_wiki_page(
                        wiki_name=wiki_name,
                        path=page.path,
                        project=project,
                    )

                    if full_page.content:
                        chunks = self._document_loader.load_text(
                            full_page.content,
                            metadata={
                                "source": f"wiki:{wiki_name}{page.path}",
                                "wiki": wiki_name,
                                "path": page.path,
                                "url": full_page.url,
                            },
                        )

                        documents = [c.content for c in chunks]
                        metadatas = [c.metadata for c in chunks]
                        self._vector_store.add_documents(documents=documents, metadatas=metadatas)

                        total_chunks += len(chunks)

                except Exception as e:
                    self.log_warning(f"Failed to ingest {page.path}: {e}")

            self.log_success(f"Ingested {total_chunks} chunks from wiki")
            return total_chunks

        except Exception as e:
            self.log_error(f"Failed to ingest wiki: {e}")
            raise

    def ingest_wiki_patterns_sync(
        self,
        wiki_name: str,
        path: str = "",
        project: Optional[str] = None,
        recursive: bool = True,
    ) -> int:
        """Synchronous wrapper for ingest_wiki_patterns."""
        return asyncio.run(
            self.ingest_wiki_patterns(wiki_name, path, project, recursive)
        )

    async def list_features(
        self,
        project: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """List features from Azure DevOps.

        Args:
            project: Project name.
            state: Filter by state (e.g., "New", "Active").
            limit: Maximum number of features.

        Returns:
            List of feature summaries.
        """
        client = await self._get_azure_client()
        features = await client.get_features(project=project, state=state, top=limit)

        return [
            {
                "id": f.id,
                "title": f.title,
                "state": f.state,
                "area_path": f.area_path,
                "url": f.url,
            }
            for f in features
        ]

    async def generate_nfrs_for_feature(
        self,
        feature_id: int,
        project: Optional[str] = None,
        categories: Optional[list[str]] = None,
        create_work_items: bool = False,
    ) -> AgentResult:
        """Generate NFRs for an Azure DevOps feature.

        Args:
            feature_id: Feature work item ID.
            project: Project name.
            categories: NFR categories to focus on.
            create_work_items: If True, create NFR work items in Azure DevOps.

        Returns:
            AgentResult with generated NFRs.
        """
        self.log_info(f"Generating NFRs for feature #{feature_id}...")

        try:
            client = await self._get_azure_client()
            feature = await client.get_work_item(feature_id, project)

            feature_context = f"""
Feature: {feature.title}

Description:
{feature.description}

Acceptance Criteria:
{feature.acceptance_criteria}

Tags: {', '.join(feature.tags) if feature.tags else 'None'}
Area: {feature.area_path}
"""

            nfrs_text = self.generate_nfrs(feature_context, categories)

            result = AgentResult(
                success=True,
                output=nfrs_text,
                reasoning="NFRs generated successfully",
                metadata={
                    "feature_id": feature_id,
                    "feature_title": feature.title,
                    "categories": categories or [],
                },
            )

            if create_work_items:
                self.log_info("Creating NFR work items in Azure DevOps...")
                nfr_items = self._parse_nfrs_from_text(nfrs_text)

                created_ids = []
                for nfr in nfr_items:
                    try:
                        wi = await client.create_work_item(
                            work_item_type="User Story",
                            title=f"[NFR] {nfr['title']}",
                            description=nfr['description'],
                            project=project,
                            parent_id=feature_id,
                            fields={
                                "System.Tags": f"NFR;{nfr.get('category', 'General')}",
                            },
                        )
                        created_ids.append(wi.id)
                    except Exception as e:
                        self.log_warning(f"Failed to create NFR work item: {e}")

                result.metadata["created_work_items"] = created_ids
                self.log_success(f"Created {len(created_ids)} NFR work items")

            return result

        except Exception as e:
            self.log_error(f"Failed to generate NFRs: {e}")
            return AgentResult(
                success=False,
                output=f"Failed to generate NFRs: {e}",
                reasoning=str(e),
            )

    def _parse_nfrs_from_text(self, text: str) -> list[dict]:
        """Parse NFRs from generated text.

        Args:
            text: Generated NFR text.

        Returns:
            List of NFR dictionaries with title, description, category.
        """
        import re

        nfrs = []
        nfr_pattern = r'NFR-\d+:\s*(.+?)(?:\n|$)'
        category_pattern = r'Category:\s*(.+?)(?:\n|$)'
        desc_pattern = r'Description:\s*(.+?)(?=\n(?:Acceptance|Priority|Related|NFR-|\Z))'

        matches = re.finditer(nfr_pattern, text, re.IGNORECASE)

        for match in matches:
            title = match.group(1).strip()
            start_pos = match.end()

            remaining = text[start_pos:start_pos + 500]

            cat_match = re.search(category_pattern, remaining, re.IGNORECASE)
            category = cat_match.group(1).strip() if cat_match else "General"

            desc_match = re.search(desc_pattern, remaining, re.IGNORECASE | re.DOTALL)
            description = desc_match.group(1).strip() if desc_match else ""

            nfrs.append({
                "title": title,
                "category": category,
                "description": description,
            })

        return nfrs

    async def close(self) -> None:
        """Close the Azure DevOps MCP client connection."""
        if self._azure_client:
            await self._azure_client.disconnect()
            self._azure_client = None
