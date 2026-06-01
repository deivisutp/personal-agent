"""Dev Assistant API: chat sessions, messages, ingestion, web UI."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from agents.api.dependencies import get_dev_assistant_agent
from agents.api.exceptions import AgentNotReadyError, ExternalServiceError
from agents.api.logging_config import get_logger
from agents.dev_assistant import DevAssistantAgent
from agents.dev_assistant.manifest import KnowledgeManifest

router = APIRouter(prefix="/dev-assistant", tags=["Dev Assistant"])
logger = get_logger("api.dev_assistant")

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
# Build the Jinja2 Environment with cache disabled. Jinja2's default LRUCache
# trips a TypeError on Python 3.14 (`cannot use 'tuple' as a dict key`). Templates
# are tiny and loaded from disk; the perf cost of cache_size=0 is negligible.
_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    cache_size=0,
)
_templates = Jinja2Templates(env=_jinja_env)


# ---------- Models ---------------------------------------------------------


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str
    filters: Optional[dict[str, Any]] = None


class IngestTextRequest(BaseModel):
    content: str
    source_name: str = "manual_input"
    doc_type: Optional[str] = "manual"
    layer: Optional[str] = None
    component: Optional[str] = None


class IngestPathRequest(BaseModel):
    path: str
    recursive: bool = True
    doc_type: Optional[str] = "doc"
    layer: Optional[str] = None
    component: Optional[str] = None


class IngestWikiRequest(BaseModel):
    wiki_name: str
    path: str = ""
    project: Optional[str] = None
    recursive: bool = True
    doc_type: Optional[str] = "wiki"
    layer: Optional[str] = None
    component: Optional[str] = None


class IngestManifestRequest(BaseModel):
    manifest_path: Optional[str] = None
    manifest: Optional[dict[str, Any]] = None


# ---------- Sessions -------------------------------------------------------


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> SessionResponse:
    s = agent.sessions.create_session(title=request.title)
    return SessionResponse(**s.dict())


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> list[SessionResponse]:
    return [SessionResponse(**s.dict()) for s in agent.sessions.list_sessions()]


@router.get("/sessions/{session_id}", response_model=list[MessageResponse])
async def get_session_messages(
    session_id: str,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> list[MessageResponse]:
    if agent.sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return [MessageResponse(**m.dict()) for m in agent.sessions.get_messages(session_id)]


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict:
    agent.sessions.delete_session(session_id)
    return {"success": True}


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def post_message(
    session_id: str,
    request: AskRequest,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> MessageResponse:
    if not agent.is_ready():
        raise AgentNotReadyError("Dev Assistant", "Ollama is not available")
    if agent.sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    msg = agent.ask_in_session(session_id, request.question, filters=request.filters)
    return MessageResponse(**msg.dict())


@router.get("/sessions/{session_id}/messages/stream")
async def stream_message(
    session_id: str,
    question: str = Query(..., description="User question"),
    doc_type: Optional[str] = Query(None),
    layer: Optional[str] = Query(None),
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> StreamingResponse:
    """Server-Sent Events endpoint that streams the answer token by token.

    Event types: ``sources``, ``delta``, ``done``, ``error``.
    """
    if not agent.is_ready():
        raise AgentNotReadyError("Dev Assistant", "Ollama is not available")
    if agent.sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    filters: dict[str, Any] = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if layer:
        filters["layer"] = layer

    async def event_stream():
        try:
            async for evt in agent.astream_in_session(
                session_id, question, filters=filters or None
            ):
                payload = json.dumps(evt, ensure_ascii=False)
                yield f"event: {evt['type']}\ndata: {payload}\n\n"
        except Exception as e:  # noqa: BLE001
            err = json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- Stats / Ingestion ---------------------------------------------


@router.get("/stats")
async def stats(
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict[str, Any]:
    return agent.stats()


@router.delete("/knowledge")
async def clear_knowledge(
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict:
    agent.vector_store.reset()
    return {"success": True, "message": "Dev knowledge collection cleared"}


@router.get("/knowledge/list")
async def list_knowledge(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    doc_type: Optional[str] = Query(None),
    layer: Optional[str] = Query(None),
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict[str, Any]:
    """Paginated, filtered listing of indexed chunks (id + metadata + preview)."""
    filters: dict[str, Any] = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if layer:
        filters["layer"] = layer
    return agent.list_chunks(offset=offset, limit=limit, filters=filters or None)


@router.get("/knowledge/{chunk_id}")
async def get_knowledge_chunk(
    chunk_id: str,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict[str, Any]:
    chunk = agent.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk


@router.post("/ingest/text")
async def ingest_text(
    request: IngestTextRequest,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict:
    extra = {
        k: v
        for k, v in {
            "doc_type": request.doc_type,
            "layer": request.layer,
            "component": request.component,
        }.items()
        if v
    }
    chunks = agent.ingest_text(request.content, source_name=request.source_name, extra_metadata=extra)
    return {"success": True, "chunks_ingested": chunks, "total": agent.knowledge_count}


@router.post("/ingest/path")
async def ingest_path(
    request: IngestPathRequest,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict:
    p = Path(request.path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {request.path}")
    extra = {
        k: v
        for k, v in {
            "doc_type": request.doc_type,
            "layer": request.layer,
            "component": request.component,
        }.items()
        if v
    }
    if p.is_file():
        chunks = agent.ingest_file(p, extra_metadata=extra)
    else:
        chunks = agent.ingest_directory(p, recursive=request.recursive, extra_metadata=extra)
    return {"success": True, "chunks_ingested": chunks, "total": agent.knowledge_count}


@router.post("/ingest/wiki")
async def ingest_wiki(
    request: IngestWikiRequest,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict:
    extra = {
        k: v
        for k, v in {
            "doc_type": request.doc_type,
            "layer": request.layer,
            "component": request.component,
        }.items()
        if v
    }
    try:
        chunks = await agent.ingest_wiki(
            wiki_name=request.wiki_name,
            path=request.path,
            project=request.project,
            recursive=request.recursive,
            extra_metadata=extra,
        )
        return {"success": True, "chunks_ingested": chunks, "total": agent.knowledge_count}
    except Exception as e:
        logger.error(f"Wiki ingestion failed: {e}")
        raise ExternalServiceError("Azure DevOps", str(e))
    finally:
        await agent.close()


@router.post("/ingest/manifest")
async def ingest_manifest(
    request: IngestManifestRequest = Body(...),
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> dict:
    if not request.manifest and not request.manifest_path:
        raise HTTPException(
            status_code=400, detail="Provide either manifest_path or manifest"
        )
    try:
        if request.manifest_path:
            manifest = KnowledgeManifest.from_yaml(Path(request.manifest_path))
        else:
            manifest = KnowledgeManifest(**(request.manifest or {}))
        return await agent.ingest_manifest(manifest)
    finally:
        await agent.close()


# ---------- Web Ingest UI (HTMX) ------------------------------------------


@router.get("/ui/ingest", response_class=HTMLResponse, include_in_schema=False)
async def ingest_ui(
    request: Request,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
):
    from agents.core.config import get_settings  # local import to avoid cycles
    settings = get_settings()
    default_wiki = ""
    default_project = (
        getattr(settings.azure_devops, "project", "") or "" if hasattr(settings, "azure_devops") else ""
    )
    default_manifest = str(Path("knowledge_manifest.yaml").resolve())
    return _templates.TemplateResponse(
        request,
        "dev_assistant_ingest.html",
        {
            "stats": agent.stats(),
            "default_wiki": default_wiki,
            "default_project": default_project,
            "default_manifest_path": default_manifest,
        },
    )


def _ingest_result_html(success: bool, summary: str, details: Optional[dict[str, Any]] = None) -> str:
    klass = "good" if success else "bad"
    body = html.escape(summary)
    if details:
        try:
            body += "\n\n" + html.escape(json.dumps(details, indent=2, ensure_ascii=False))
        except Exception:
            pass
    return f'<div class="result {klass}">{body}</div>'


@router.post("/ui/ingest/wiki", response_class=HTMLResponse, include_in_schema=False)
async def ui_ingest_wiki(
    wiki_name: str = Form(...),
    path: str = Form(""),
    project: str = Form(""),
    recursive: Optional[str] = Form(None),
    doc_type: str = Form("wiki"),
    layer: str = Form(""),
    component: str = Form(""),
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> HTMLResponse:
    extra = {k: v for k, v in {
        "doc_type": doc_type, "layer": layer or None, "component": component or None
    }.items() if v}
    try:
        chunks = await agent.ingest_wiki(
            wiki_name=wiki_name,
            path=path or "",
            project=project or None,
            recursive=bool(recursive),
            extra_metadata=extra or None,
        )
        return HTMLResponse(_ingest_result_html(
            True,
            f"OK — ingested {chunks} chunks from {wiki_name}{path or '/'}.",
            {"total_chunks": agent.knowledge_count, "applied_metadata": extra},
        ))
    except Exception as e:  # noqa: BLE001
        logger.exception("wiki ingest failed")
        return HTMLResponse(_ingest_result_html(False, f"FAILED: {e}"), status_code=200)
    finally:
        await agent.close()


@router.post("/ui/ingest/path", response_class=HTMLResponse, include_in_schema=False)
async def ui_ingest_path(
    path: str = Form(...),
    recursive: Optional[str] = Form(None),
    doc_type: str = Form("doc"),
    layer: str = Form(""),
    component: str = Form(""),
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> HTMLResponse:
    p = Path(path)
    if not p.exists():
        return HTMLResponse(_ingest_result_html(False, f"Path not found: {p}"))
    extra = {k: v for k, v in {
        "doc_type": doc_type, "layer": layer or None, "component": component or None
    }.items() if v}
    try:
        if p.is_dir():
            chunks = agent.ingest_directory(
                p, recursive=bool(recursive), extra_metadata=extra or None
            )
        else:
            chunks = agent.ingest_file(p, extra_metadata=extra or None)
        return HTMLResponse(_ingest_result_html(
            True,
            f"OK — ingested {chunks} chunks from {p}.",
            {"total_chunks": agent.knowledge_count, "applied_metadata": extra},
        ))
    except Exception as e:  # noqa: BLE001
        logger.exception("path ingest failed")
        return HTMLResponse(_ingest_result_html(False, f"FAILED: {e}"))


@router.post("/ui/ingest/manifest", response_class=HTMLResponse, include_in_schema=False)
async def ui_ingest_manifest(
    manifest_path: str = Form(...),
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
) -> HTMLResponse:
    mp = Path(manifest_path)
    if not mp.exists():
        return HTMLResponse(_ingest_result_html(False, f"Manifest not found: {mp}"))
    try:
        manifest = KnowledgeManifest.from_yaml(mp)
        report = await agent.ingest_manifest(manifest)
        return HTMLResponse(_ingest_result_html(
            True,
            f"Manifest ingested — total chunks now: {agent.knowledge_count}.",
            report,
        ))
    except Exception as e:  # noqa: BLE001
        logger.exception("manifest ingest failed")
        return HTMLResponse(_ingest_result_html(False, f"FAILED: {e}"))
    finally:
        await agent.close()


# ---------- Web Chat UI (HTMX) --------------------------------------------


@router.get("/ui/knowledge", response_class=HTMLResponse, include_in_schema=False)
async def knowledge_ui(
    request: Request,
    offset: int = 0,
    limit: int = 50,
    doc_type: Optional[str] = None,
    layer: Optional[str] = None,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
):
    filters: dict[str, Any] = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if layer:
        filters["layer"] = layer
    listing = agent.list_chunks(offset=offset, limit=limit, filters=filters or None)
    return _templates.TemplateResponse(
        request,
        "dev_assistant_knowledge.html",
        {
            "listing": listing,
            "doc_type": doc_type or "",
            "layer": layer or "",
            "offset": offset,
            "limit": limit,
        },
    )


@router.get("/ui/knowledge/{chunk_id}", response_class=HTMLResponse, include_in_schema=False)
async def knowledge_chunk_ui(
    chunk_id: str,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
):
    chunk = agent.get_chunk(chunk_id)
    if not chunk:
        return HTMLResponse('<div class="empty">Chunk not found.</div>', status_code=404)
    meta = chunk["metadata"] or {}
    meta_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in meta.items()
    )
    body = html.escape(chunk["content"])
    return HTMLResponse(
        f"""
        <div class="chunk-detail">
          <table class="meta">{meta_rows}</table>
          <pre class="content">{body}</pre>
        </div>
        """
    )


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def chat_ui(
    request: Request,
    session_id: Optional[str] = None,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
):
    sessions = agent.sessions.list_sessions()
    active_id = session_id or (sessions[0].id if sessions else None)
    messages = agent.sessions.get_messages(active_id) if active_id else []
    stats = agent.stats()
    return _templates.TemplateResponse(
        request,
        "dev_assistant_chat.html",
        {
            "sessions": sessions,
            "active_id": active_id,
            "messages": messages,
            "stats": stats,
        },
    )


@router.post("/ui/sessions", response_class=HTMLResponse, include_in_schema=False)
async def ui_create_session(
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
):
    s = agent.sessions.create_session()
    # Trigger client to navigate to the new session
    return HTMLResponse(
        content="",
        headers={"HX-Redirect": f"/dev-assistant/ui?session_id={s.id}"},
    )


@router.post("/ui/sessions/{session_id}/delete", response_class=HTMLResponse, include_in_schema=False)
async def ui_delete_session(
    session_id: str,
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
):
    agent.sessions.delete_session(session_id)
    return HTMLResponse(content="", headers={"HX-Redirect": "/dev-assistant/ui"})


@router.post("/ui/sessions/{session_id}/messages", response_class=HTMLResponse, include_in_schema=False)
async def ui_post_message(
    session_id: str,
    question: str = Form(...),
    doc_type: str = Form(""),
    layer: str = Form(""),
    agent: DevAssistantAgent = Depends(get_dev_assistant_agent),
):
    if not agent.is_ready():
        return HTMLResponse(
            content=_render_error_bubble("Ollama is not available. Start it and reload."),
            status_code=503,
        )
    if agent.sessions.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    filters: dict[str, Any] = {}
    if doc_type:
        filters["doc_type"] = doc_type
    if layer:
        filters["layer"] = layer

    user_html = _render_bubble("user", question, sources=None)

    try:
        msg = agent.ask_in_session(session_id, question, filters=filters or None)
        assistant_html = _render_bubble("assistant", msg.content, sources=msg.sources)
    except Exception as e:  # noqa: BLE001
        logger.exception("ask_in_session failed")
        assistant_html = _render_error_bubble(f"Error: {e}")

    return HTMLResponse(content=user_html + assistant_html)


# ---------- HTML rendering helpers ----------------------------------------


def _render_bubble(role: str, content: str, sources: Optional[list[dict[str, Any]]]) -> str:
    css_class = "msg-user" if role == "user" else "msg-assistant"
    body = html.escape(content)
    sources_html = ""
    if sources:
        items = []
        for s in sources:
            label = s.get("source", "unknown")
            doc_type = s.get("doc_type") or ""
            layer = s.get("layer") or ""
            heading = s.get("heading_path") or ""
            score = s.get("score")
            url = s.get("url")
            text = label
            if doc_type:
                text = f"[{doc_type}] {text}"
            if layer:
                text = f"{text}  ·  {layer}"
            if heading:
                text = f"{text}  ·  {heading}"
            if score is not None:
                text = f"{text}  ·  {score}"
            text = html.escape(text)
            if url:
                items.append(f'<li><a href="{html.escape(url)}" target="_blank" rel="noopener">{text}</a></li>')
            else:
                items.append(f"<li>{text}</li>")
        sources_html = (
            '<details class="sources"><summary>Sources ({n})</summary><ul>{items}</ul></details>'
        ).format(n=len(items), items="".join(items))
    return (
        f'<div class="msg {css_class}">'
        f'<div class="role">{role}</div>'
        f'<div class="content" data-markdown="1">{body}</div>'
        f'{sources_html}'
        f'</div>'
    )


def _render_error_bubble(text: str) -> str:
    safe = html.escape(text)
    return f'<div class="msg msg-error"><div class="role">error</div><div class="content">{safe}</div></div>'
    return f'<div class="msg msg-error"><div class="role">error</div><div class="content">{safe}</div></div>'
