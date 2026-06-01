"""SQLite-backed chat session and message store for the Dev Assistant."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    created_at: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatSessionStore:
    """Persistent store for chat sessions and messages.

    Schema is created on demand. Safe for use from a single FastAPI process.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, created_at);
                """
            )

    # -- Sessions ----------------------------------------------------------

    def create_session(self, title: Optional[str] = None) -> ChatSession:
        sid = str(uuid.uuid4())
        now = _now()
        title = title or "New chat"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, title, now, now),
            )
        return ChatSession(id=sid, title=title, created_at=now, updated_at=now, message_count=0)

    def list_sessions(self, limit: int = 100) -> list[ChatSession]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            ChatSession(
                id=r["id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["msg_count"],
            )
            for r in rows
        ]

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count
                FROM sessions s WHERE s.id = ?
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return ChatSession(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["msg_count"],
        )

    def rename_session(self, session_id: str, title: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # -- Messages ----------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict[str, Any]]] = None,
    ) -> ChatMessage:
        mid = str(uuid.uuid4())
        now = _now()
        sources_json = json.dumps(sources or [], ensure_ascii=False)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mid, session_id, role, content, sources_json, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return ChatMessage(
            id=mid,
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
            sources=sources or [],
        )

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> list[ChatMessage]:
        query = (
            "SELECT id, session_id, role, content, sources_json, created_at "
            "FROM messages WHERE session_id = ? ORDER BY created_at ASC"
        )
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            query += " LIMIT ?"
            params = (session_id, limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            ChatMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                sources=json.loads(r["sources_json"] or "[]"),
            )
            for r in rows
        ]

    def get_recent_history(
        self, session_id: str, max_messages: int = 12
    ) -> list[ChatMessage]:
        """Return the last `max_messages` messages, oldest first, for LLM context."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, sources_json, created_at FROM (
                    SELECT * FROM messages WHERE session_id = ?
                    ORDER BY created_at DESC LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (session_id, max_messages),
            ).fetchall()
        return [
            ChatMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                sources=json.loads(r["sources_json"] or "[]"),
            )
            for r in rows
        ]
