"""Document loader and chunking utilities for RAG."""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of a document with metadata."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0
    total_chunks: int = 1


class DocumentLoader:
    """Load and chunk documents for RAG ingestion."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the document loader.

        Args:
            chunk_size: Target size for each chunk in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, file_path: Path) -> list[DocumentChunk]:
        """Load and chunk a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of DocumentChunk objects.

        Raises:
            ValueError: If file type is not supported.
            FileNotFoundError: If file doesn't exist.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        content = file_path.read_text(encoding="utf-8")
        base_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix,
        }

        return self._chunk_text(content, base_metadata)

    def load_directory(
        self,
        directory: Path,
        recursive: bool = True,
        extensions: Optional[set[str]] = None,
    ) -> list[DocumentChunk]:
        """Load and chunk all supported files in a directory.

        Args:
            directory: Path to the directory.
            recursive: Whether to search subdirectories.
            extensions: Specific extensions to include. Defaults to all supported.

        Returns:
            List of DocumentChunk objects from all files.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        extensions = extensions or self.SUPPORTED_EXTENSIONS
        chunks = []

        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    file_chunks = self.load_file(file_path)
                    chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return chunks

    def load_text(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[DocumentChunk]:
        """Load and chunk raw text.

        Args:
            text: Text content to chunk.
            metadata: Optional metadata to attach.

        Returns:
            List of DocumentChunk objects.
        """
        return self._chunk_text(text, metadata or {})

    def _chunk_text(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk.
            base_metadata: Metadata to attach to each chunk.

        Returns:
            List of DocumentChunk objects.
        """
        if len(text) <= self.chunk_size:
            return [
                DocumentChunk(
                    content=text,
                    metadata=base_metadata,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                break_point = self._find_break_point(text, start, end)
                if break_point > start:
                    end = break_point

            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        metadata={**base_metadata, "chunk_index": chunk_index},
                        chunk_index=chunk_index,
                        total_chunks=0,
                    )
                )
                chunk_index += 1

            start = end - self.chunk_overlap
            if start >= len(text) - self.chunk_overlap:
                break

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a natural break point (paragraph, sentence, or word boundary).

        Args:
            text: Full text.
            start: Start position of current chunk.
            end: Proposed end position.

        Returns:
            Adjusted end position at a natural break.
        """
        search_start = max(start, end - 200)
        search_text = text[search_start:end]

        for sep in ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]:
            last_sep = search_text.rfind(sep)
            if last_sep != -1:
                return search_start + last_sep + len(sep)

        return end
