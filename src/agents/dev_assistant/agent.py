"""Dev Assistant Agent: conversational RAG over internal engineering documentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.core.base_agent import AgentContext, AgentResult, BaseAgent
from agents.core.llm_client import OllamaClient
from agents.dev_assistant.knowledge_loader import KnowledgeLoader
from agents.dev_assistant.retrieval import HybridRetriever, llm_rerank
from agents.dev_assistant.manifest import (
    DirectorySource,
    FileSource,
    KnowledgeManifest,
    KnowledgeSource,
    WikiSource,
    source_metadata,
)
from agents.dev_assistant.prompts import ANSWER_TEMPLATE, DEV_ASSISTANT_SYSTEM_PROMPT
from agents.dev_assistant.session_store import ChatMessage, ChatSessionStore
from agents.mcp.azure_devops_client import AzureDevOpsMCPClient
from agents.rag.vector_store import ChromaVectorStore, SearchResult


DEFAULT_COLLECTION = "dev_knowledge"


class DevAssistantAgent(BaseAgent):
    """Conversational assistant grounded in the user's internal engineering docs."""

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        system_prompt: Optional[str] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        session_store: Optional[ChatSessionStore] = None,
        azure_devops_org: Optional[str] = None,
        azure_devops_project: Optional[str] = None,
        azure_devops_pat: Optional[str] = None,
        history_window: int = 15,
        hybrid_candidate_k: int = 20,
        rerank: bool = True,
        retrieval_top_k: int = 6,
    ):
        super().__init__(
            name="Dev Assistant",
            description="Answers internal dev questions using the company knowledge base",
            system_prompt=system_prompt,
            llm_client=llm_client,
        )
        self._vector_store = vector_store or ChromaVectorStore(
            collection_name=DEFAULT_COLLECTION
        )
        self._loader = KnowledgeLoader()
        self._sessions = session_store or ChatSessionStore(
            db_path=Path("./data/chat.db")
        )
        self._azure_devops_org = azure_devops_org
        self._azure_devops_project = azure_devops_project
        self._azure_devops_pat = azure_devops_pat
        self._azure_client: Optional[AzureDevOpsMCPClient] = None
        self.history_window = history_window
        self.retrieval_top_k = retrieval_top_k
        self.hybrid_candidate_k = hybrid_candidate_k
        self.rerank = rerank
        self._retriever = HybridRetriever(self._vector_store)

    # -- Required overrides -------------------------------------------------

    def _default_system_prompt(self) -> str:
        return DEV_ASSISTANT_SYSTEM_PROMPT

    def execute(self, context: AgentContext) -> AgentResult:
        """Single-shot Q&A. For multi-turn use ``ask_in_session``."""
        question = context.user_input
        filters = context.metadata.get("filters") or None
        results = self._retrieve(question, filters=filters)
        prompt = self._build_prompt(question, results)
        response = self.chat(prompt, remember=False)
        return AgentResult(
            success=True,
            output=response.content,
            reasoning=f"Retrieved {len(results)} chunks",
            metadata={
                "sources": [self._source_summary(r) for r in results],
                "model": response.model,
            },
        )

    # -- Public API ---------------------------------------------------------

    @property
    def vector_store(self) -> ChromaVectorStore:
        return self._vector_store

    @property
    def sessions(self) -> ChatSessionStore:
        return self._sessions

    @property
    def knowledge_count(self) -> int:
        return self._vector_store.count

    def stats(self) -> dict[str, Any]:
        """Best-effort breakdown of indexed chunks. Cheap aggregate."""
        return {
            "total_chunks": self._vector_store.count,
            "collection": self._vector_store.collection_name,
        }

    def ask_in_session(
        self,
        session_id: str,
        question: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> ChatMessage:
        """Append the user message, run RAG, persist the assistant message, return it."""
        session = self._sessions.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        self._sessions.add_message(session_id, "user", question)
        # Auto-title if first user message
        if session.message_count == 0:
            short = question.strip().splitlines()[0][:60] or "New chat"
            self._sessions.rename_session(session_id, short)

        results = self._retrieve(question, filters=filters)
        prompt = self._build_prompt(question, results)

        history = self._history_to_messages(
            self._sessions.get_recent_history(session_id, self.history_window)[:-1]
        )
        response = self._client.chat(
            message=prompt,
            system_prompt=self._system_prompt,
            history=history,
        )
        sources = [self._source_summary(r) for r in results]
        return self._sessions.add_message(
            session_id, "assistant", response.content, sources=sources
        )

    # -- Ingestion ----------------------------------------------------------

    def ingest_file(
        self,
        path: Path | str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        chunks = self._loader.load_file(Path(path), extra_metadata=extra_metadata)
        return self._add_chunks(chunks)

    def ingest_directory(
        self,
        path: Path | str,
        recursive: bool = True,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        chunks = self._loader.load_directory(
            Path(path), recursive=recursive, extra_metadata=extra_metadata
        )
        return self._add_chunks(chunks)

    def ingest_text(
        self,
        text: str,
        source_name: str = "manual_input",
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        meta: dict[str, Any] = {"source": f"manual:{source_name}", "doc_type": "manual"}
        if extra_metadata:
            meta.update(extra_metadata)
        chunks = self._loader.load_markdown(text, meta)
        return self._add_chunks(chunks)

    async def ingest_wiki(
        self,
        wiki_name: str,
        path: str = "",
        project: Optional[str] = None,
        recursive: bool = True,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        client = await self._get_azure_client()
        pages = await client.list_wiki_pages(
            wiki_name=wiki_name, path=path, project=project, recursive=recursive
        )
        total = 0
        for page in pages:
            try:
                full = await client.get_wiki_page(
                    wiki_name=wiki_name, path=page.path, project=project
                )
                if not full.content:
                    continue
                chunks = self._loader.load_wiki_page(
                    wiki_name=wiki_name,
                    path=page.path,
                    content=full.content,
                    url=full.url,
                    extra_metadata=extra_metadata,
                )
                total += self._add_chunks(chunks)
            except Exception as exc:  # noqa: BLE001
                self.log_warning(f"Failed to ingest {page.path}: {exc}")
        self.log_success(f"Ingested {total} chunks from wiki:{wiki_name}{path}")
        return total

    async def ingest_manifest(self, manifest: KnowledgeManifest) -> dict[str, Any]:
        report: dict[str, Any] = {"total_chunks": 0, "items": []}
        for src in manifest.parsed():
            extra = source_metadata(src)
            chunks = 0
            try:
                if isinstance(src, WikiSource):
                    chunks = await self.ingest_wiki(
                        wiki_name=src.wiki_name,
                        path=src.path,
                        project=src.project,
                        recursive=src.recursive,
                        extra_metadata=extra,
                    )
                elif isinstance(src, DirectorySource):
                    chunks = self.ingest_directory(
                        src.path, recursive=src.recursive, extra_metadata=extra
                    )
                elif isinstance(src, FileSource):
                    chunks = self.ingest_file(src.path, extra_metadata=extra)
                report["items"].append({"source": src.dict(), "chunks": chunks})
                report["total_chunks"] += chunks
            except Exception as exc:  # noqa: BLE001
                self.log_error(f"Failed source {src}: {exc}")
                report["items"].append({"source": src.dict(), "error": str(exc)})
        return report

    # -- Internals ----------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        rerank: Optional[bool] = None,
    ) -> list[SearchResult]:
        if self._vector_store.count == 0:
            return []
        q = query if len(query) <= 2000 else query[:2000]
        where = self._build_where(filters)
        candidates = self._retriever.search(
            q, top_k=self.hybrid_candidate_k, where=where
        )
        if not candidates:
            return []
        do_rerank = self.rerank if rerank is None else rerank
        if do_rerank and len(candidates) > self.retrieval_top_k:
            return llm_rerank(
                self._client, q, candidates, final_k=self.retrieval_top_k
            )
        return candidates[: self.retrieval_top_k]

    @staticmethod
    def _build_where(filters: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not filters:
            return None
        # Chroma where clause: simple equality filters only here.
        clean = {k: v for k, v in filters.items() if v not in (None, "", [])}
        return clean or None

    def _build_prompt(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            context_block = "(no matching internal documentation found)"
        else:
            blocks: list[str] = []
            for r in results:
                meta = r.metadata or {}
                header_parts = [
                    f"doc_type={meta.get('doc_type', 'unknown')}",
                ]
                if meta.get("layer"):
                    header_parts.append(f"layer={meta['layer']}")
                if meta.get("language"):
                    header_parts.append(f"language={meta['language']}")
                if meta.get("heading_path"):
                    header_parts.append(f"section={meta['heading_path']}")
                header_parts.append(f"source={meta.get('source', 'unknown')}")
                blocks.append(
                    f"### [{' | '.join(header_parts)}]\n{r.content.strip()}"
                )
            context_block = "\n\n".join(blocks)
        return ANSWER_TEMPLATE.format(context_block=context_block, question=question)

    @staticmethod
    def _source_summary(r: SearchResult) -> dict[str, Any]:
        meta = r.metadata or {}
        return {
            "source": meta.get("source", "unknown"),
            "doc_type": meta.get("doc_type"),
            "layer": meta.get("layer"),
            "heading_path": meta.get("heading_path"),
            "url": meta.get("url"),
            "score": round(1 - r.score, 4),
        }

    @staticmethod
    def _history_to_messages(messages: list[ChatMessage]) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for m in messages:
            if m.role == "user":
                out.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                out.append(AIMessage(content=m.content))
        return out

    def _add_chunks(self, chunks: list) -> int:
        if not chunks:
            return 0
        documents = [c.content for c in chunks]
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks]
        self._vector_store.add_documents(documents=documents, metadatas=metadatas)
        return len(chunks)

    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Chroma requires str/int/float/bool values."""
        clean: dict[str, Any] = {}
        for k, v in (meta or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, tuple, set)):
                clean[k] = ",".join(str(x) for x in v)
            else:
                clean[k] = str(v)
        return clean

    # -- Streaming chat -----------------------------------------------------

    async def astream_in_session(
        self,
        session_id: str,
        question: str,
        filters: Optional[dict[str, Any]] = None,
    ):
        """Async generator yielding events for SSE.

        Yields dicts of one of these shapes:
          {"type": "sources", "sources": [...]}
          {"type": "delta",   "text": "..."}
          {"type": "done",    "message_id": "...", "content": "..."}
          {"type": "error",   "error": "..."}
        Persists user message immediately and assistant message on completion.
        """
        session = self._sessions.get_session(session_id)
        if session is None:
            yield {"type": "error", "error": f"Unknown session: {session_id}"}
            return

        # Persist user message + auto-title on first turn
        self._sessions.add_message(session_id, "user", question)
        if session.message_count == 0:
            short = question.strip().splitlines()[0][:60] or "New chat"
            self._sessions.rename_session(session_id, short)

        try:
            results = self._retrieve(question, filters=filters)
            sources = [self._source_summary(r) for r in results]
            yield {"type": "sources", "sources": sources}

            prompt = self._build_prompt(question, results)
            history = self._history_to_messages(
                self._sessions.get_recent_history(session_id, self.history_window)[:-1]
            )

            buffer: list[str] = []
            async for delta in self._client.astream_chat(
                message=prompt,
                system_prompt=self._system_prompt,
                history=history,
            ):
                buffer.append(delta)
                yield {"type": "delta", "text": delta}

            full = "".join(buffer)
            saved = self._sessions.add_message(
                session_id, "assistant", full, sources=sources
            )
            yield {"type": "done", "message_id": saved.id, "content": full}
        except Exception as e:  # noqa: BLE001
            self.log_error(f"streaming failed: {e}")
            yield {"type": "error", "error": str(e)}

    # -- Knowledge browsing -------------------------------------------------

    def list_chunks(
        self,
        offset: int = 0,
        limit: int = 50,
        filters: Optional[dict[str, Any]] = None,
        preview_chars: int = 240,
    ) -> dict[str, Any]:
        """Return a paginated, filtered listing of indexed chunks for the UI."""
        # pylint: disable=protected-access
        coll = self._vector_store._collection
        where = self._build_where(filters)
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        # Chroma supports limit/offset on get()
        kwargs["limit"] = max(1, min(limit, 200))
        kwargs["offset"] = max(0, offset)
        try:
            data = coll.get(**kwargs)
        except Exception:
            # Some Chroma versions reject unknown params; fall back to no offset
            data = coll.get(include=["documents", "metadatas"], where=where)

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{} for _ in ids]

        items: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            meta = metas[i] or {}
            preview = (doc or "").strip().replace("\n", " ")
            if len(preview) > preview_chars:
                preview = preview[:preview_chars] + "…"
            items.append(
                {
                    "id": ids[i] if i < len(ids) else None,
                    "source": meta.get("source"),
                    "doc_type": meta.get("doc_type"),
                    "layer": meta.get("layer"),
                    "language": meta.get("language"),
                    "heading_path": meta.get("heading_path"),
                    "url": meta.get("url"),
                    "has_code": meta.get("has_code"),
                    "preview": preview,
                    "length": len(doc or ""),
                }
            )
        return {
            "total": self._vector_store.count,
            "returned": len(items),
            "offset": offset,
            "limit": limit,
            "items": items,
        }

    def get_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        """Get a single chunk's full content + metadata."""
        result = self._vector_store.get_document(chunk_id)
        if not result:
            return None
        return {
            "id": result.document_id,
            "content": result.content,
            "metadata": result.metadata,
        }

    async def _get_azure_client(self) -> AzureDevOpsMCPClient:
        if self._azure_client is None:
            self._azure_client = AzureDevOpsMCPClient(
                organization=self._azure_devops_org,
                project=self._azure_devops_project,
                pat=self._azure_devops_pat,
            )
            await self._azure_client.connect()
        return self._azure_client

    async def close(self) -> None:
        if self._azure_client:
            await self._azure_client.disconnect()
            self._azure_client = None
