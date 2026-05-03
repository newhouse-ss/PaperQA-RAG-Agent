import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

from rag_agent.config import ensure_google_api_key
from rag_agent.graph_builder import build_graph
from rag_agent.models import get_embeddings_model
from rag_agent.cache import SemanticCache

logger = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    message: str
    timeout_s: int = Field(default=60, ge=1)


class Citation(BaseModel):
    source: str
    title: Optional[str] = None
    page: Optional[int] = None
    chunk: Optional[str] = None
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    cached: bool = False


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return "\n".join(parts).strip()
    return str(content)


def _get_message_content(m: Any) -> Any:
    if isinstance(m, dict):
        return m.get("content")
    return getattr(m, "content", None)


CIT_BLOCK_RE = re.compile(r"\[CITATION\s+\d+\](.*?)(?=\n\n\[CITATION|\Z)", re.S)
FIELD_RE = re.compile(r"^(SOURCE|TITLE|PAGE|CHUNK|SNIPPET):\s*(.*)$", re.M)


def _to_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _collect_last_citations(messages: list) -> List[Citation]:
    """
    Parse citations from the *last tool output* only.
    Skip final assistant answers that may contain '[CITATION x]' but lack 'SOURCE:' fields.
    """
    for m in reversed(messages):
        content = _get_message_content(m)
        if not content:
            continue

        text = _extract_text(content)
        if "[CITATION" not in text or "SOURCE:" not in text:
            continue

        blocks = CIT_BLOCK_RE.findall(text)
        if not blocks:
            continue

        out: List[Citation] = []
        for b in blocks:
            fields = dict(FIELD_RE.findall(b))
            src = (fields.get("SOURCE") or "").strip()
            if not src:
                continue
            out.append(
                Citation(
                    source=src,
                    title=(fields.get("TITLE") or "").strip() or None,
                    page=_to_int(fields.get("PAGE") or ""),
                    chunk=(fields.get("CHUNK") or "").strip() or None,
                    snippet=(fields.get("SNIPPET") or "").strip() or None,
                )
            )
        return out

    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_google_api_key()
    app.state.graph = await asyncio.to_thread(build_graph)
    app.state.cache = SemanticCache(
        embeddings_model=get_embeddings_model(),
        similarity_threshold=0.92,
        ttl_seconds=3600,
    )
    logger.info("Semantic cache initialised  threshold=0.92  ttl=3600s")
    yield


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Citation-grounded RAG Agent", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda req, exc: JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"}),
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/cache/stats")
def cache_stats():
    return {"entries": app.state.cache.size}


@app.delete("/v1/cache")
def cache_clear():
    app.state.cache.clear()
    return {"status": "cleared"}


@app.post("/v1/chat", response_model=ChatResponse, response_model_exclude_none=True)
@limiter.limit("30/minute")
async def chat(request: Request, req: ChatRequest):
    trace_id = str(uuid.uuid4())

    # --- semantic cache lookup ---
    hit = app.state.cache.get(req.message)
    if hit is not None:
        answer, cached_citations = hit
        citations = [Citation(**c) if isinstance(c, dict) else c for c in cached_citations]
        return ChatResponse(
            trace_id=trace_id, answer=answer, citations=citations, cached=True
        )

    # --- cache miss: run the full agent graph ---
    initial_state = {"messages": [{"role": "user", "content": req.message}]}

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(app.state.graph.invoke, initial_state),
            timeout=req.timeout_s,
        )
        messages = result.get("messages", [])
        citations = _collect_last_citations(messages)

        last = messages[-1] if messages else ""
        answer = _extract_text(getattr(last, "content", last))

        # store in cache
        app.state.cache.put(
            query=req.message,
            answer=answer,
            citations=[c.model_dump() for c in citations],
        )

        return ChatResponse(trace_id=trace_id, answer=answer, citations=citations)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Timeout after {req.timeout_s}s")
    except Exception as e:
        logger.exception("Internal error in /v1/chat")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e!r}")
