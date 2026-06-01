"""Knowledge manifest: declarative list of sources to ingest into the dev knowledge base.

Example knowledge_manifest.yaml:

    sources:
      - kind: wiki
        wiki_name: "Architecture"
        path: "/Backend"
        doc_type: backend_structure
        layer: backend
        recursive: true
      - kind: directory
        path: "C:/repos/my-app/docs"
        doc_type: how_to
        layer: backend
      - kind: directory
        path: "C:/repos/my-app/db"
        doc_type: plsql_object
        layer: database
      - kind: file
        path: "C:/repos/my-app/CONVENTIONS.md"
        doc_type: backend_structure
        layer: backend
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class WikiSource(BaseModel):
    kind: Literal["wiki"] = "wiki"
    wiki_name: str
    path: str = ""
    project: Optional[str] = None
    recursive: bool = True
    doc_type: str = "wiki"
    layer: Optional[str] = None
    component: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class DirectorySource(BaseModel):
    kind: Literal["directory"] = "directory"
    path: str
    recursive: bool = True
    doc_type: str = "doc"
    layer: Optional[str] = None
    component: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class FileSource(BaseModel):
    kind: Literal["file"] = "file"
    path: str
    doc_type: str = "doc"
    layer: Optional[str] = None
    component: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


KnowledgeSource = WikiSource | DirectorySource | FileSource


class KnowledgeManifest(BaseModel):
    sources: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("sources")
    @classmethod
    def _validate(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for entry in v:
            kind = entry.get("kind")
            if kind not in ("wiki", "directory", "file"):
                raise ValueError(f"Invalid source kind: {kind!r}")
        return v

    def parsed(self) -> list[KnowledgeSource]:
        out: list[KnowledgeSource] = []
        for entry in self.sources:
            kind = entry["kind"]
            if kind == "wiki":
                out.append(WikiSource(**entry))
            elif kind == "directory":
                out.append(DirectorySource(**entry))
            elif kind == "file":
                out.append(FileSource(**entry))
        return out

    @classmethod
    def from_yaml(cls, path: Path) -> "KnowledgeManifest":
        import yaml  # local import to make YAML optional at install time

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)


def source_metadata(src: KnowledgeSource) -> dict[str, Any]:
    """Build the base metadata dict that should be attached to chunks from this source."""
    meta: dict[str, Any] = {"doc_type": src.doc_type}
    if src.layer:
        meta["layer"] = src.layer
    if src.component:
        meta["component"] = src.component
    if src.tags:
        meta["tags"] = ",".join(src.tags)
    return meta
