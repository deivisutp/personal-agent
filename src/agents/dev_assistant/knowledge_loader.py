"""Knowledge loader: markdown-structure-aware splitter that preserves code fences.

Designed for internal engineering docs (wiki pages, READMEs, code, PL/SQL files).
Produces clean DocumentChunk objects with rich metadata so downstream retrieval can
filter by doc_type / layer / language / component.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from agents.rag.document_loader import DocumentChunk


# --- Cleaning rules ---------------------------------------------------------

_NOISE_PATTERNS = [
    re.compile(r"\[\[_TOC_\]\]", re.IGNORECASE),
    re.compile(r"<!--.*?-->", re.DOTALL),
    re.compile(r"^\s*:::\s*toc\s*:::\s*$", re.MULTILINE | re.IGNORECASE),
]

# A line that contains only an image reference / nothing useful for retrieval
_IMAGE_ONLY = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")


def clean_markdown(text: str) -> str:
    """Strip wiki noise (TOC markers, HTML comments, image-only lines)."""
    cleaned = text
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    out_lines: list[str] = []
    for line in cleaned.splitlines():
        if _IMAGE_ONLY.match(line):
            continue
        out_lines.append(line.rstrip())

    # Collapse 3+ blank lines to 2
    result = "\n".join(out_lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result


# --- Markdown section parsing ----------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```([^\n]*)$", re.MULTILINE)


@dataclass
class _Section:
    heading_path: list[str]
    level: int
    content: str
    has_code: bool = False
    code_languages: set[str] = field(default_factory=set)


def _split_sections(text: str) -> list[_Section]:
    """Split markdown into sections by headings, keeping fenced code blocks intact.

    Returns a list of sections; each section's content includes its own heading line
    so retrieval results show clear context.
    """
    if not text.strip():
        return []

    # Build heading index but ignore matches inside fenced code blocks.
    fences: list[tuple[int, int]] = []
    in_fence = False
    fence_start = 0
    for m in _FENCE_RE.finditer(text):
        if not in_fence:
            in_fence = True
            fence_start = m.start()
        else:
            in_fence = False
            fences.append((fence_start, m.end()))

    def _inside_fence(pos: int) -> bool:
        for s, e in fences:
            if s <= pos < e:
                return True
        return False

    headings: list[tuple[int, int, str]] = []  # (start_index, level, title)
    for m in _HEADING_RE.finditer(text):
        if _inside_fence(m.start()):
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        headings.append((m.start(), level, title))

    if not headings:
        return [_section_from_block(text, heading_path=[], level=0)]

    sections: list[_Section] = []
    path_stack: list[tuple[int, str]] = []  # [(level, title)]
    boundaries = [h[0] for h in headings] + [len(text)]

    # If there's preamble before the first heading, keep it as an unlabeled section.
    if headings[0][0] > 0:
        preamble = text[: headings[0][0]].strip()
        if preamble:
            sections.append(_section_from_block(preamble, heading_path=[], level=0))

    for i, (start, level, title) in enumerate(headings):
        end = boundaries[i + 1]
        block = text[start:end].strip()

        # Maintain heading path stack (pop deeper-or-equal levels)
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, title))
        heading_path = [t for _, t in path_stack]

        sections.append(_section_from_block(block, heading_path=heading_path, level=level))

    return sections


def _section_from_block(block: str, heading_path: list[str], level: int) -> _Section:
    has_code = "```" in block
    languages: set[str] = set()
    if has_code:
        for fm in _FENCE_RE.finditer(block):
            lang = fm.group(1).strip().lower()
            if lang:
                languages.add(lang)
    return _Section(
        heading_path=heading_path,
        level=level,
        content=block,
        has_code=has_code,
        code_languages=languages,
    )


# --- Chunk packing ----------------------------------------------------------

def _split_section_safely(content: str, target_size: int) -> list[str]:
    """Split a section that exceeds target_size while keeping code fences intact.

    Strategy: walk paragraphs; if a paragraph contains an opened fence, keep accumulating
    until the fence closes before considering a split.
    """
    if len(content) <= target_size:
        return [content]

    paragraphs = re.split(r"\n{2,}", content)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    in_fence = False

    for para in paragraphs:
        fences = para.count("```")
        para_len = len(para) + 2

        if buf_len + para_len > target_size and buf and not in_fence:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

        buf.append(para)
        buf_len += para_len
        if fences % 2 == 1:
            in_fence = not in_fence

    if buf:
        chunks.append("\n\n".join(buf).strip())

    # Hard fallback: if a single chunk is still > 2x target, slice on lines.
    final: list[str] = []
    for c in chunks:
        if len(c) <= target_size * 2:
            final.append(c)
        else:
            lines = c.splitlines()
            cur: list[str] = []
            cur_len = 0
            for ln in lines:
                if cur_len + len(ln) + 1 > target_size and cur:
                    final.append("\n".join(cur))
                    cur = []
                    cur_len = 0
                cur.append(ln)
                cur_len += len(ln) + 1
            if cur:
                final.append("\n".join(cur))
    return [c for c in final if c.strip()]


# --- Public loader ----------------------------------------------------------

# Default doc_type inferred from file extension when none provided
_EXTENSION_DOC_TYPE: dict[str, dict[str, Any]] = {
    ".md": {"doc_type": "doc", "language": "markdown"},
    ".txt": {"doc_type": "doc", "language": "text"},
    ".sql": {"doc_type": "db_model", "language": "sql"},
    ".pks": {"doc_type": "plsql_object", "language": "plsql"},
    ".pkb": {"doc_type": "plsql_object", "language": "plsql"},
    ".plsql": {"doc_type": "plsql_object", "language": "plsql"},
    ".java": {"doc_type": "backend_code", "language": "java"},
    ".kt": {"doc_type": "backend_code", "language": "kotlin"},
    ".ts": {"doc_type": "frontend_code", "language": "typescript"},
    ".tsx": {"doc_type": "frontend_code", "language": "typescript"},
    ".js": {"doc_type": "frontend_code", "language": "javascript"},
    ".jsx": {"doc_type": "frontend_code", "language": "javascript"},
    ".py": {"doc_type": "backend_code", "language": "python"},
    ".yaml": {"doc_type": "config", "language": "yaml"},
    ".yml": {"doc_type": "config", "language": "yaml"},
    ".json": {"doc_type": "config", "language": "json"},
}


class KnowledgeLoader:
    """Smarter loader: markdown-section-aware, code-fence-safe, metadata-rich."""

    SUPPORTED_EXTENSIONS = set(_EXTENSION_DOC_TYPE.keys())

    def __init__(self, target_chunk_size: int = 1400, min_chunk_size: int = 120):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size

    # -- Markdown / wiki text ------------------------------------------------

    def load_markdown(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        cleaned = clean_markdown(text)
        if not cleaned:
            return []

        sections = _split_sections(cleaned)
        chunks: list[DocumentChunk] = []
        index = 0

        for sec in sections:
            for piece in _split_section_safely(sec.content, self.target_chunk_size):
                piece = piece.strip()
                # Skip tiny scraps only if they have NO heading and NO code (likely noise)
                if (
                    len(piece) < self.min_chunk_size
                    and not sec.has_code
                    and not sec.heading_path
                ):
                    continue

                meta = dict(base_metadata)
                if sec.heading_path:
                    meta["heading_path"] = " > ".join(sec.heading_path)
                if sec.has_code:
                    meta["has_code"] = True
                    if sec.code_languages:
                        meta["code_languages"] = ",".join(sorted(sec.code_languages))
                meta["chunk_index"] = index

                chunks.append(
                    DocumentChunk(
                        content=piece,
                        metadata=meta,
                        chunk_index=index,
                        total_chunks=0,
                    )
                )
                index += 1

        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    # -- Source code / PL/SQL ------------------------------------------------

    def load_code(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """Code files: keep them whole when small, otherwise slice on blank lines."""
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []

        if len(text) <= self.target_chunk_size * 2:
            return [
                DocumentChunk(
                    content=text,
                    metadata={**base_metadata, "chunk_index": 0},
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

        pieces = _split_section_safely(text, self.target_chunk_size)
        chunks = [
            DocumentChunk(
                content=p,
                metadata={**base_metadata, "chunk_index": i},
                chunk_index=i,
                total_chunks=len(pieces),
            )
            for i, p in enumerate(pieces)
        ]
        return chunks

    # -- Files / dirs --------------------------------------------------------

    def load_file(
        self,
        path: Path,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> list[DocumentChunk]:
        path = Path(path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        ext = path.suffix.lower()
        defaults = _EXTENSION_DOC_TYPE.get(ext, {})
        meta: dict[str, Any] = {
            "source": f"file:{path.as_posix()}",
            "filename": path.name,
            "extension": ext,
            **defaults,
        }
        if extra_metadata:
            meta.update(extra_metadata)

        if ext == ".md":
            return self.load_markdown(text, meta)
        return self.load_code(text, meta)

    def load_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extra_metadata: Optional[dict[str, Any]] = None,
        excludes: Iterable[str] = (".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"),
    ) -> list[DocumentChunk]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        excludes_set = {e.lower() for e in excludes}
        chunks: list[DocumentChunk] = []
        glob = "**/*" if recursive else "*"
        for fp in directory.glob(glob):
            if not fp.is_file():
                continue
            if any(part.lower() in excludes_set for part in fp.parts):
                continue
            try:
                chunks.extend(self.load_file(fp, extra_metadata=extra_metadata))
            except Exception as exc:  # noqa: BLE001
                print(f"[knowledge_loader] skipped {fp}: {exc}")
        return chunks

    # -- Wiki page (already-fetched content) ---------------------------------

    def load_wiki_page(
        self,
        wiki_name: str,
        path: str,
        content: str,
        url: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> list[DocumentChunk]:
        meta: dict[str, Any] = {
            "source": f"wiki:{wiki_name}{path}",
            "wiki": wiki_name,
            "path": path,
            "doc_type": "wiki",
            "language": "markdown",
        }
        if url:
            meta["url"] = url
        if extra_metadata:
            meta.update(extra_metadata)
        return self.load_markdown(content, meta)
