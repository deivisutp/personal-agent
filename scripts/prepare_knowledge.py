"""Ingest the dev knowledge base from a manifest YAML.

Usage:
    python scripts/prepare_knowledge.py knowledge_manifest.yaml [--dry-run]

In --dry-run mode, no embeddings are computed; the script just prints the chunks
that would be ingested so you can review and clean the inputs first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Make src/ importable when running as a plain script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from agents.dev_assistant.agent import DevAssistantAgent  # noqa: E402
from agents.dev_assistant.knowledge_loader import KnowledgeLoader  # noqa: E402
from agents.dev_assistant.manifest import (  # noqa: E402
    DirectorySource,
    FileSource,
    KnowledgeManifest,
    WikiSource,
    source_metadata,
)


def _dry_run(manifest: KnowledgeManifest) -> None:
    loader = KnowledgeLoader()
    total = 0
    for src in manifest.parsed():
        extra = source_metadata(src)
        if isinstance(src, FileSource):
            chunks = loader.load_file(Path(src.path), extra_metadata=extra)
        elif isinstance(src, DirectorySource):
            chunks = loader.load_directory(
                Path(src.path), recursive=src.recursive, extra_metadata=extra
            )
        elif isinstance(src, WikiSource):
            print(f"[dry-run] wiki sources are skipped (require Azure DevOps): {src.wiki_name}{src.path}")
            continue
        else:
            continue
        total += len(chunks)
        print(f"\n=== {src.kind}: {getattr(src, 'path', getattr(src, 'wiki_name', ''))} -> {len(chunks)} chunks ===")
        for c in chunks[:3]:
            print(json.dumps(c.metadata, indent=2, ensure_ascii=False))
            preview = c.content[:300].replace("\n", " ")
            print(f"  preview: {preview}...")
    print(f"\nTOTAL chunks (dry-run): {total}")


async def _real_run(manifest: KnowledgeManifest) -> None:
    agent = DevAssistantAgent()
    report = await agent.ingest_manifest(manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    await agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the dev knowledge base")
    parser.add_argument("manifest", type=str, help="Path to knowledge_manifest.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Don't embed; preview chunks")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = KnowledgeManifest.from_yaml(manifest_path)

    if args.dry_run:
        _dry_run(manifest)
    else:
        asyncio.run(_real_run(manifest))


if __name__ == "__main__":
    main()
