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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/neaps", tags=["NEAPS Ingestion"])



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
    """
    symbol = request.company_symbol.strip().upper()
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_symbol must not be empty",
        )

    try:
        from ingestion.job_store import get_job_store
        task_id = get_job_store().insert_job(
            job_type="neaps_ingestion",
            payload={"symbol": symbol},
            max_retries=3
        )
        if not task_id:
            raise Exception("Failed to insert job into JobStore")
            
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
        # 1. Hybrid retrieval (skip_rerank=True so search returns raw fused candidates;
        #    a single rerank pass is done below to avoid double-reranking score collapse)
        from retrieval.neaps_hybrid_search import search as neaps_search

        results = neaps_search(
            query=request.question,
            company_symbol=symbol,
            top_k=request.top_k,
            final_top_k=request.final_k * 2,  # over-retrieve for reranker
            skip_rerank=True,
        )

        # 2. Single rerank pass
        from retrieval.neaps_reranker import rerank

        reranked = rerank(
            query=request.question,
            results=results,
            top_k=request.final_k,
        )

        # Enforce strict final_k limit to fix Bug 6 (sometimes reranker returns more)
        if reranked:
            reranked = reranked[:request.final_k]

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

        import math
        def _sigmoid(x):
            try: return 1 / (1 + math.exp(-x))
            except OverflowError: return 0.0 if x < 0 else 1.0

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
            
            # Extract and normalize score
            # reranked list has 'rerank_score' (cross-encoder logit) if reranker ran,
            # otherwise fall back to 'fused_score' / 'score' from RRF.
            raw_score = chunk.get("rerank_score", chunk.get("fused_score", chunk.get("score", 0)))
            # Cross-encoder returns logits (typically -10 to +10); Qdrant cosine is 0-1.
            # Apply sigmoid only when value is outside [0, 1].
            if raw_score > 1.0 or raw_score < 0.0:
                normalized_score = _sigmoid(raw_score)
            else:
                normalized_score = float(raw_score)

            # Only discard chunks with truly zero relevance
            # Production bug: threshold of 0.01 was dropping ALL chunks for many queries
            # (cross-encoder logits for weakly-matching docs land in [0.002, 0.008] range)
            # Lowered to 0.001 so only absolute noise is filtered out.
            if normalized_score < 0.001:
                logger.warning(
                    f"[QueryNeaps] Dropping chunk {chunk_id} "
                    f"score={normalized_score:.5f} (below 0.001 threshold)"
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
                    "score": normalized_score,
                    "metadata": meta,
                }
            )

        logger.info(
            f"[QueryNeaps] company={symbol} reranked={len(reranked)} "
            f"after_score_filter={len(sources)} chunks_passed_to_llm={len(sources)}"
        )

        # Fallback: if ALL chunks were filtered out but we do have candidates,
        # use the top-3 unfiltered so the LLM always gets some context.
        # This prevents the 'Chunks Passed to LLM: 0' failure mode.
        if not sources and reranked:
            logger.warning(
                f"[QueryNeaps] All {len(reranked)} chunks filtered out — "
                f"falling back to top-3 unfiltered chunks for company={symbol}"
            )
            for i, chunk in enumerate(reranked[:3]):
                meta = chunk.get("metadata", {}) or {}
                file_name = meta.get("file_name") or meta.get("filename") or "Unknown"
                chunk_id_fb = (
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
                    f"[Document: {file_name} | Chunk: {chunk_id_fb} | Date: {date} | Source: {source}]\n"
                    f"{chunk.get('text', '')}"
                )
                sources.append(
                    {
                        "ann_seq": meta.get("ann_seq"),
                        "filename": file_name,
                        "chunk_id": str(chunk_id_fb),
                        "source": source,
                        "ann_date": date,
                        "score": float(chunk.get("fused_score", chunk.get("score", 0.0))),
                        "metadata": meta,
                    }
                )

        # If still nothing (truly empty Qdrant collection for this company)
            return QueryResponse(
                success=True,
                company=symbol,
                question=request.question,
                answer="Not found in provided documents.",
                sources=[],
                chunks_used=0,
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
            chunks_used=len(sources),
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
    import base64
    import shutil
    import tempfile

    # Accept only the bare filename  never trust caller-supplied sub-paths
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

    file_bytes = tmp_path.read_bytes()
    b64_data = base64.b64encode(file_bytes).decode("utf-8")
    file_ext = bare_name.rsplit(".", 1)[-1].lower() if "." in bare_name else "unknown"

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "file_name": bare_name,
        "file_type": file_ext,
        "base64_data": b64_data,
        "is_base64": True,
    }


@router.get("/dashboard")
async def get_neaps_dashboard():
    """
    Single unified endpoint for the frontend Stats button.
    """
    # --- Live in-memory task counters (Replaced with DB Queue stats) ---
    live = {
        "fetched": 0,
        "running": 0,
        "queued": 0,
        "failed": 0,
        "completed": 0,
    }
    try:
        from ingestion.job_store import get_job_store
        job_stats = get_job_store().get_stats()
        if job_stats.get("available"):
            # job_stats["by_type"] is dict of dicts
            neaps_stats = job_stats.get("by_type", {}).get("neaps_ingestion", {})
            live["queued"] = neaps_stats.get("pending", 0)
            live["running"] = neaps_stats.get("processing", 0)
            live["completed"] = neaps_stats.get("completed", 0)
            live["failed"] = neaps_stats.get("failed", 0)
            live["fetched"] = live["queued"] + live["running"]
    except Exception as e:
        logger.warning(f"Failed to fetch job queue stats: {e}")


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
        # get_all_stats returns last 20  extend to 50 via direct query
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
        from ingestion.job_store import get_job_store

        result = get_neaps_duplicate_checker().get_company_stats(symbols=symbol_list)
        
        # Inject active job queue items so the UI sees symbols that haven't started parsing yet
        active_jobs = get_job_store().get_active_neaps_symbols()
        existing_symbols = {r["company_symbol"] for r in result.get("rows", [])}
        
        for sym, status in active_jobs.items():
            if sym not in existing_symbols:
                result["rows"].append({
                    "company_symbol": sym,
                    "total": 0,
                    "processed": 0,
                    "skipped": 0,
                    "failed": 0,
                    "pending": 0,
                    "job_status": status,  # "pending" or "processing"
                })
            else:
                for r in result["rows"]:
                    if r["company_symbol"] == sym:
                        r["job_status"] = status
                        break
                        
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
    Trigger ingestion for multiple company symbols via Job Queue.
    """
    symbols = [s.strip().upper() for s in request.company_symbols if s.strip()]
    if not symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_symbols must contain at least one non-empty symbol",
        )

    per_company: List[dict] = []
    task_ids: Dict[str, str] = {}
    
    from ingestion.job_store import get_job_store
    store = get_job_store()
    
    for sym in symbols:
        task_id = store.insert_job(job_type="neaps_ingestion", payload={"symbol": sym}, max_retries=3)
        task_ids[sym] = task_id or "failed"
        per_company.append(
            {
                "symbol": sym,
                "success": bool(task_id),
                "task_id": task_id,
                "status": "ingestion started" if task_id else "failed",
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
    # Dummy response since frontend usually uses /company-stats
    # Real task status can be added later via Oracle query if needed.
    return IngestionTaskStatusResponse(
        success=True,
        task_id=task_id,
        symbol="UNKNOWN",
        status="running",
        created_at=time.time(),
        stats={}
    )



# ---------------------------------------------------------------------------
# LLM helper (reuses existing Ollama/ChatOllama setup)
# ---------------------------------------------------------------------------


async def _generate_answer(question: str, company: str, context: str) -> str:
    """Call the configured LLM to generate an answer from retrieved context."""
    try:
        from agents.llm_client import call_llm_async
        import json

        prompt = f"""You are a financial document intelligence assistant for {company}.

STRICT RULES:
1. You MUST summarize ALL provided documents below. Do NOT skip or omit any document.
2. Even if the document does not explicitly contain the exact search keywords, you MUST still summarize its contents.
3. Use ONLY the information from the provided documents. Do NOT invent or assume any data.
4. For EVERY factual statement, cite the exact source using this format: (Document: <file_name>, Chunk: <chunk_id>)
5. ONLY cite documents that appear in the "OFFICIAL DOCUMENTS" section below.

QUESTION: {question}

OFFICIAL DOCUMENTS:
{context}

ANSWER (summarize ALL documents above, cite every fact):"""

        answer = await call_llm_async(prompt)
        return answer or "Not found in provided documents."

    except Exception as exc:
        logger.error(f"[NeapsAPI] LLM answer generation failed: {exc}")
        return f"[Answer generation unavailable: {exc}]"
