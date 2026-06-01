"""Hybrid retrieval (BM25 + dense vectors) and optional LLM re-ranking.

Design:
- The dense index is the existing Chroma collection.
- The BM25 index is built lazily over the same collection's documents and cached.
  Call `invalidate()` after any ingestion to force a rebuild on next query.
- Hybrid fusion uses Reciprocal Rank Fusion (RRF):
      score(doc) = sum_over_rankers( 1 / (k + rank_in_ranker) )
  This is robust to score-scale differences between BM25 and cosine.
- Optional LLM re-rank takes the top_k candidates and asks the LLM to return the
  best `final_k` indices. Falls back gracefully on parse errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from agents.core.llm_client import OllamaClient
from agents.rag.vector_store import ChromaVectorStore, SearchResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize for BM25. Keep symbol-friendly tokens (CamelCase, snake_case).

    Strategy: extract word/identifier-like runs, lowercase them. Also split
    CamelCase identifiers into their parts so `WCPanelAction` matches both
    `wcpanelaction` and `wc`, `panel`, `action`.
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        low = raw.lower()
        tokens.append(low)
        # Split CamelCase / mixedCase
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", raw)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts if p)
    return tokens


@dataclass
class _BM25Index:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    bm25: BM25Okapi


class HybridRetriever:
    """Hybrid BM25 + dense retriever over a `ChromaVectorStore`."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        rrf_k: int = 60,
    ):
        self._vs = vector_store
        self._rrf_k = rrf_k
        self._index: Optional[_BM25Index] = None

    def invalidate(self) -> None:
        """Drop the cached BM25 index. Call after any ingestion."""
        self._index = None

    def _build_index(self) -> Optional[_BM25Index]:
        # pylint: disable=protected-access
        coll = self._vs._collection
        if coll.count() == 0:
            return None
        # Chroma's `get()` with no ids returns everything.
        data = coll.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{} for _ in ids]
        if not docs:
            return None
        tokenized = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized)
        return _BM25Index(ids=ids, documents=docs, metadatas=metas, bm25=bm25)

    def _ensure_index(self) -> Optional[_BM25Index]:
        if self._index is None:
            self._index = self._build_index()
        return self._index

    @staticmethod
    def _matches_filter(meta: dict[str, Any], where: Optional[dict[str, Any]]) -> bool:
        if not where:
            return True
        for k, v in where.items():
            if meta.get(k) != v:
                return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Run BM25 + vector search and fuse with RRF; return up to `top_k` results."""
        if self._vs.count == 0:
            return []

        # --- Dense ---
        dense_n = max(top_k, 30)
        dense_hits = self._vs.search(query, n_results=dense_n, where=where)

        # --- BM25 ---
        bm25_results: list[SearchResult] = []
        idx = self._ensure_index()
        if idx is not None:
            tokens = _tokenize(query)
            if tokens:
                scores = idx.bm25.get_scores(tokens)
                # Filter by metadata if provided
                ranked = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True,
                )
                for i in ranked:
                    if scores[i] <= 0:
                        break
                    meta = idx.metadatas[i] or {}
                    if not self._matches_filter(meta, where):
                        continue
                    bm25_results.append(
                        SearchResult(
                            content=idx.documents[i],
                            metadata=meta,
                            score=-float(scores[i]),  # not used, lower=better convention
                            document_id=idx.ids[i],
                        )
                    )
                    if len(bm25_results) >= dense_n:
                        break

        # --- Fuse with RRF ---
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, hit in enumerate(dense_hits):
            rrf_scores[hit.document_id] = rrf_scores.get(hit.document_id, 0.0) + 1.0 / (
                self._rrf_k + rank + 1
            )
            result_map[hit.document_id] = hit

        for rank, hit in enumerate(bm25_results):
            rrf_scores[hit.document_id] = rrf_scores.get(hit.document_id, 0.0) + 1.0 / (
                self._rrf_k + rank + 1
            )
            # Prefer the dense version if both exist (it carries the cosine distance).
            result_map.setdefault(hit.document_id, hit)

        ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[SearchResult] = []
        for doc_id, fused in ordered[:top_k]:
            r = result_map[doc_id]
            # Repurpose the score field to encode RRF (higher = better → store as 1-fused).
            out.append(
                SearchResult(
                    content=r.content,
                    metadata=r.metadata,
                    score=max(0.0, 1.0 - fused),
                    document_id=r.document_id,
                )
            )
        return out


# --- Optional LLM re-rank --------------------------------------------------


_RERANK_PROMPT = """You are a relevance ranking assistant.
Given a user QUESTION and a numbered list of CANDIDATE snippets from internal
engineering docs, return the indices of the {final_k} MOST relevant snippets to
answer the question, ordered from most to least relevant.

Respond with a single JSON array of integers, nothing else. Example: [3, 1, 7, 2, 5]

QUESTION:
{question}

CANDIDATES:
{candidates}
"""


def _candidate_block(results: list[SearchResult], snippet_chars: int = 360) -> str:
    parts: list[str] = []
    for i, r in enumerate(results):
        meta = r.metadata or {}
        header_bits = [f"doc_type={meta.get('doc_type', 'unknown')}"]
        if meta.get("layer"):
            header_bits.append(f"layer={meta['layer']}")
        if meta.get("heading_path"):
            header_bits.append(f"section={meta['heading_path']}")
        snippet = (r.content or "").strip().replace("\n", " ")
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars] + "…"
        parts.append(f"[{i}] ({' | '.join(header_bits)}) {snippet}")
    return "\n".join(parts)


_INT_LIST_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def llm_rerank(
    llm: OllamaClient,
    question: str,
    candidates: list[SearchResult],
    final_k: int = 5,
) -> list[SearchResult]:
    """Use the LLM to pick the best `final_k` candidates. Safe fallback on errors."""
    if len(candidates) <= final_k:
        return candidates

    prompt = _RERANK_PROMPT.format(
        final_k=final_k,
        question=question[:1500],
        candidates=_candidate_block(candidates),
    )
    try:
        resp = llm.chat(prompt)
        text = resp.content.strip()
        # Try direct JSON parse first
        order: list[int] = []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                order = [int(x) for x in data if isinstance(x, (int, float))]
        except (ValueError, TypeError):
            m = _INT_LIST_RE.search(text)
            if m:
                order = [int(x.strip()) for x in m.group(1).split(",")]
        # Deduplicate, drop out-of-range
        seen: set[int] = set()
        clean: list[int] = []
        for i in order:
            if 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                clean.append(i)
            if len(clean) == final_k:
                break
        if not clean:
            return candidates[:final_k]
        # Pad with remaining top hits if model returned fewer than final_k
        if len(clean) < final_k:
            for i in range(len(candidates)):
                if i not in seen:
                    clean.append(i)
                    if len(clean) == final_k:
                        break
        return [candidates[i] for i in clean]
    except Exception:  # noqa: BLE001
        return candidates[:final_k]
