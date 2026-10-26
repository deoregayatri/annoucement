"""
neaps_api.py
============
FastAPI router exposing NEAPS ingestion and query endpoints.

Endpoints:
  POST /neaps/start-ingestion   - trigger document ingestion for a company
  POST /neaps/query             - query the NEAPS knowledge base
  GET  /neaps/status/{symbol}   - processing stats for a company

All connection parameters come from settings (.env).
"""

import logging
import mimetypes
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/neaps", tags=["NEAPS Ingestion"])

# Dedicated executor keeps ingestion off the request thread.
_ingestion_executor = ThreadPoolExecutor(
    max_workers=max(2, int(getattr(settings, "max_workers_cpu", 2))),
    thread_name_prefix="neaps-ingestion",
)
_ingestion_tasks: Dict[str, Dict[str, Any]] = {}
_ingestion_tasks_lock = threading.Lock()


def _resolve_neaps_document_root() -> Path:
    """
    Always resolves to the configured NEAPS local storage path.
    Uses NEAPS_LOCAL_STORAGE from .env exclusively — does NOT fall back
    to document_root or documents_dir so NEAPS downloads always serve
    from /mnt/Announcement/neaps_document (or whatever is configured).
    """
    return Path(settings.neaps_local_storage).expanduser().resolve()


def _set_task_state(task_id: str, **kwargs) -> None:
    with _ingestion_tasks_lock:
        task = _ingestion_tasks.get(task_id)
        if task is None:
            return
        task.update(kwargs)


def _run_ingestion_task(task_id: str, symbol: str) -> None:
    _set_task_state(task_id, status="running", started_at=time.time())
    try:
        from ingestion.neaps_ingestion_service import run_ingestion

        stats = run_ingestion(symbol)
        _set_task_state(
            task_id,
            status="completed",
            completed_at=time.time(),
            stats=stats,
            error=None,
        )
    except Exception as exc:
        logger.error(
            f"[NeapsIngestionTask] Failed task_id={task_id} symbol='{symbol}': {exc}",
            exc_info=True,
        )
        _set_task_state(
            task_id,
            status="failed",
            completed_at=time.time(),
            error=str(exc),
        )


def _submit_ingestion_task(symbol: str) -> str:
    with _ingestion_tasks_lock:
        for existing_id, task in _ingestion_tasks.items():
            # Reuse an active task per symbol to avoid duplicate concurrent ingest for same dataset.
            if task.get("symbol") == symbol and task.get("status") in {
                "queued",
                "running",
            }:
                return existing_id
        task_id = str(uuid.uuid4())
        _ingestion_tasks[task_id] = {
            "task_id": task_id,
            "symbol": symbol,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "stats": {},
            "error": None,
        }
    _ingestion_executor.submit(_run_ingestion_task, task_id, symbol)
    return task_id


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class IngestionRequest(BaseModel):
    company_symbol: str = Field(..., description="NSE/BSE company symbol, e.g. 'TCS'")


class IngestionResponse(BaseModel):
    success: bool
    status: str = "completed"
    task_id: Optional[str] = None
    symbol: str
    total: int = 0
    skipped: int = 0
    downloaded: int = 0
    failed_dl: int = 0
    ingested: int = 0
    failed_proc: int = 0
    total_chunks: int = 0
    message: str = ""


class QueryRequest(BaseModel):
    company: str = Field(..., description="Company symbol to query")
    question: str = Field(..., description="Natural language question")
    top_k: int = Field(default=10, ge=1, le=100)
    final_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    success: bool
    company: str
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []
    chunks_used: int = 0


class StatusResponse(BaseModel):
    symbol: str
    oracle_available: bool
    summary: Dict[str, Any] = {}
    recent: List[Dict[str, Any]] = []
    qdrant_total_vectors: int = 0
    qdrant_active_vectors: int = 0
    qdrant_collection: str = ""


class IngestionTaskStatusResponse(BaseModel):
    success: bool
    task_id: str
    symbol: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    stats: Dict[str, Any] = {}
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start-ingestion", response_model=IngestionResponse)
async def start_ingestion(request: IngestionRequest):
    """
    Trigger full ingestion pipeline for a company symbol.

    The ingestion runs asynchronously and immediately returns a task id.
    """
    symbol = request.company_symbol.strip().upper()
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_symbol must not be empty",
        )

    try:
        task_id = _submit_ingestion_task(symbol)
        return IngestionResponse(
            success=True,
            status="ingestion started",
            task_id=task_id,
            symbol=symbol,
            message=f"Ingestion started for '{symbol}'",
        )
    except Exception as exc:
        logger.error(
            f"[/neaps/start-ingestion] Failed for '{symbol}': {exc}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(exc)}",
        )


@router.post("/query", response_model=QueryResponse)
async def query_neaps(request: QueryRequest):
    """
    Query the NEAPS knowledge base for a specific company.

    Runs hybrid retrieval  reranking  LLM answer generation.
    """
    symbol = request.company.strip().upper()

    try:
        # 1. Hybrid retrieval
        from retrieval.neaps_hybrid_search import search as neaps_search

        results = neaps_search(
            query=request.question,
            company_symbol=symbol,
            top_k=request.top_k,
            final_top_k=request.final_k
            * 2,  # over-retrieve for reranker (reduced to 2x)
        )

        # 2. Rerank
        from retrieval.neaps_reranker import rerank

        reranked = rerank(
            query=request.question,
            results=results,
            top_k=request.final_k,
        )

        if not reranked:
            return QueryResponse(
                success=True,
                company=symbol,
                question=request.question,
                answer="Not found in provided documents.",
                sources=[],
                chunks_used=0,
            )

        # 3. Build context and call LLM
        context_parts = []
        sources: List[Dict[str, Any]] = []

        for i, chunk in enumerate(reranked):
            meta = chunk.get("metadata", {}) or {}
            file_name = meta.get("file_name") or meta.get("filename") or "Unknown"
            chunk_id = (
                meta.get("chunk_id")
                or chunk.get("id")
                or chunk.get("chunk_id")
                or f"chunk-{i + 1}"
            )
            date = (
                meta.get("ann_date")
                or meta.get("announcement_date")
                or meta.get("date")
                or "Unknown"
            )
            source = (
                meta.get("source")
                or meta.get("source_type")
                or chunk.get("source_type")
                or "Unknown"
            )
            context_parts.append(
                f"[Document: {file_name} | Chunk: {chunk_id} | Date: {date} | Source: {source}]\n"
                f"{chunk.get('text', '')}"
            )
            sources.append(
                {
                    "ann_seq": meta.get("ann_seq"),
                    "filename": file_name,
                    "chunk_id": str(chunk_id),
                    "source": source,
                    "ann_date": date,
                    "score": chunk.get("score", 0),
                    "metadata": meta,
                }
            )

        context = "\n\n---\n\n".join(context_parts)

        answer = await _generate_answer(
            question=request.question,
            company=symbol,
            context=context,
        )

        return QueryResponse(
            success=True,
            company=symbol,
            question=request.question,
            answer=answer,
            sources=sources,
            chunks_used=len(reranked),
        )

    except Exception as exc:
        logger.error(f"[/neaps/query] Failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(exc)}",
        )


@router.get("/download/{filename:path}")
async def download_file(filename: str):
    """
    Serve a NEAPS document by fetching it on-demand from the SFTP server.

    Files are NOT kept on disk after ingestion, so this endpoint downloads
    the requested file fresh from SFTP into a temporary directory, streams
    it to the browser, then cleans up the temp dir automatically.

    Path traversal is blocked: only the basename of the supplied path is used.
    """
    import asyncio
    import shutil
    import tempfile
    from fastapi.background import BackgroundTasks
    from fastapi.responses import FileResponse

    # Accept only the bare filename — never trust caller-supplied sub-paths
    bare_name = Path(filename).name
    if not bare_name or bare_name in (".", ".."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="neaps_dl_"))
    tmp_path = tmp_dir / bare_name

    try:
        from ingestion.neaps_sftp_fetcher import get_neaps_sftp_fetcher

        fetcher = get_neaps_sftp_fetcher()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            fetcher.download_one_to,
            bare_name,
            tmp_path,
        )
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.error(f"[/neaps/download] SFTP fetch failed for '{bare_name}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{bare_name}' could not be retrieved from the SFTP server.",
        )

    media_type = mimetypes.guess_type(bare_name)[0] or "application/octet-stream"

    # BackgroundTask cleans up the temp dir after the file is fully sent
    background = BackgroundTasks()
    background.add_task(shutil.rmtree, str(tmp_dir), True)

    return FileResponse(
        path=str(tmp_path),
        media_type=media_type,
        filename=bare_name,
        background=background,
    )


@router.get("/dashboard")
async def get_neaps_dashboard():
    """
    Single unified endpoint for the frontend Stats button.

    Returns:
      live    — real-time in-memory task counters (FETCHED / RUNNING / QUEUED / FAILED / COMPLETED)
      oracle_available — True when Oracle is reachable
      summary — aggregate counts from AI_NEAPS_TRACKER (all companies)
      history — last 50 processed documents (AI_NEAPS_TRACKER, most-recent first)
      qdrant  — Qdrant collection totals
    """
    # --- Live in-memory task counters -----------------------------------------
    with _ingestion_tasks_lock:
        tasks_snapshot = list(_ingestion_tasks.values())

    live = {
        "fetched": len(tasks_snapshot),
        "running": sum(1 for t in tasks_snapshot if t.get("status") == "running"),
        "queued": sum(1 for t in tasks_snapshot if t.get("status") == "queued"),
        "failed": sum(1 for t in tasks_snapshot if t.get("status") == "failed"),
        "completed": sum(1 for t in tasks_snapshot if t.get("status") == "completed"),
    }

    # --- AI_NEAPS_TRACKER data (Oracle) ----------------------------------------
    oracle_available = False
    summary: Dict[str, Any] = {}
    history: list = []
    try:
        from ingestion.neaps_duplicate_checker import get_neaps_duplicate_checker

        checker = get_neaps_duplicate_checker()
        tracker = checker.get_all_stats()
        oracle_available = tracker.get("oracle_available", False)
        summary = tracker.get("summary", {})
        # get_all_stats returns last 20 — extend to 50 via direct query
        history = tracker.get("recent", [])
    except Exception as exc:
        logger.warning(f"[/neaps/dashboard] Oracle tracker unavailable: {exc}")

    # --- Qdrant collection info ------------------------------------------------
    qdrant_info: Dict[str, Any] = {}
    try:
        from vector_store.neaps_qdrant_store import get_neaps_qdrant_store

        qdrant_info = get_neaps_qdrant_store().get_collection_info()
    except Exception as exc:
        logger.warning(f"[/neaps/dashboard] Qdrant unavailable: {exc}")

    return {
        "live": live,
        "oracle_available": oracle_available,
        "summary": summary,
        "history": history,
        "qdrant": {
            "collection": qdrant_info.get("collection", ""),
            "total_vectors": qdrant_info.get("total_vectors", 0),
            "active_vectors": qdrant_info.get("active_vectors", 0),
            "dimension": qdrant_info.get("dimension", 0),
            "status": qdrant_info.get("status", "unknown"),
        },
    }


@router.get("/company-stats")
async def get_company_stats(symbols: Optional[str] = None):
    """
    Return per-company processing statistics from Oracle AI_NEAPS_TRACKER.

    This is the primary endpoint for the frontend Stats modal.

    Query params:
        symbols (optional): comma-separated company symbols to filter,
                            e.g. ?symbols=TCS,HDFCBANK
                            Omit to return stats for ALL companies.

    Returns:
        {
          "oracle_available": bool,
          "rows": [
            {
              "company_symbol": str,
              "total":          int,   // total docs seen
              "processed":      int,   // COMPLETED
              "skipped":        int,   // SKIPPED (non-PDF / duplicate)
              "failed":         int    // FAILED
            }, ...
          ]
        }
    """
    symbol_list: Optional[List[str]] = None
    if symbols:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]

    try:
        from ingestion.neaps_duplicate_checker import get_neaps_duplicate_checker

        result = get_neaps_duplicate_checker().get_company_stats(symbols=symbol_list)
        return result
    except Exception as exc:
        logger.error(f"[/neaps/company-stats] Failed: {exc}", exc_info=True)
        return {"oracle_available": False, "rows": [], "error": str(exc)}


@router.get("/companies")
async def list_companies():
    """List DISSEMINATED company symbols from Oracle (last 24 months)."""
    try:
        from ingestion.neaps_oracle_client import get_neaps_oracle_client

        symbols = get_neaps_oracle_client().get_unique_symbols()
        return {"success": True, "companies": symbols, "total": len(symbols)}
    except Exception as exc:
        logger.error(f"[/neaps/companies] Failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Batch ingestion (multiple symbols at once)
# ---------------------------------------------------------------------------


class IngestionBatchRequest(BaseModel):
    company_symbols: List[str] = Field(
        ..., description="List of company symbols to ingest, e.g. ['TCS', 'ZOTA']"
    )


class IngestionBatchResponse(BaseModel):
    success: bool
    status: str = "completed"
    symbols: List[str]
    task_ids: Dict[str, str] = {}
    total: int = 0
    skipped: int = 0
    downloaded: int = 0
    failed_dl: int = 0
    ingested: int = 0
    failed_proc: int = 0
    total_chunks: int = 0
    per_company: List[dict] = []
    message: str = ""


@router.post("/start-ingestion-batch", response_model=IngestionBatchResponse)
async def start_ingestion_batch(request: IngestionBatchRequest):
    """
    Trigger ingestion for multiple company symbols in sequence.

    Each symbol is processed through the full pipeline:
    Oracle metadata -> duplicate check -> SFTP download -> embed -> Qdrant upsert.
    Returns combined totals plus a per-company breakdown.
    """
    symbols = [s.strip().upper() for s in request.company_symbols if s.strip()]
    if not symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_symbols must contain at least one non-empty symbol",
        )

    per_company: List[dict] = []
    task_ids: Dict[str, str] = {}
    for sym in symbols:
        task_id = _submit_ingestion_task(sym)
        task_ids[sym] = task_id
        per_company.append(
            {
                "symbol": sym,
                "success": True,
                "task_id": task_id,
                "status": "ingestion started",
            }
        )

    return IngestionBatchResponse(
        success=True,
        status="ingestion started",
        symbols=symbols,
        task_ids=task_ids,
        per_company=per_company,
        message=f"Batch ingestion started for {len(symbols)} symbol(s)",
    )


@router.get("/task-status/{task_id}", response_model=IngestionTaskStatusResponse)
async def get_ingestion_task_status(task_id: str):
    with _ingestion_tasks_lock:
        task = _ingestion_tasks.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task not found: {task_id}",
        )
    return IngestionTaskStatusResponse(
        success=task.get("status") != "failed",
        task_id=task["task_id"],
        symbol=task["symbol"],
        status=task["status"],
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        stats=task.get("stats", {}),
        error=task.get("error"),
    )


# ---------------------------------------------------------------------------
# LLM helper (reuses existing Ollama/ChatOllama setup)
# ---------------------------------------------------------------------------


async def _generate_answer(question: str, company: str, context: str) -> str:
    """Call the configured LLM to generate an answer from retrieved context."""
    try:
        from agents.llm_client import call_llm_async
        import json

        prompt = f"""You are a financial document intelligence assistant.
Use ONLY the provided documents for {company}.
Answer clearly and concisely.
Cite document name and chunk number for every fact.
Do not invent data.
If the answer is not available, respond exactly with: Not found in provided documents.
Citation format for every factual statement:
(Document: <file_name>, Chunk: <chunk_id>)

QUESTION: {question}

OFFICIAL DOCUMENTS:
{context}

ANSWER:"""

        answer = await call_llm_async(prompt)
        return answer or "Not found in provided documents."

    except Exception as exc:
        logger.error(f"[NeapsAPI] LLM answer generation failed: {exc}")
        return f"[Answer generation unavailable: {exc}]"
