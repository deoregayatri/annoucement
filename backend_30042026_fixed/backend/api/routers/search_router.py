"""
Search router - unified POST /search endpoint.
Includes guardrails, two-layer retrieval, NEAPS fallback, and optional LangGraph (graph mode).
"""

import asyncio
import functools
import json
import logging
import os
import re
import threading
import time as _time
import uuid

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from api.schemas import SearchRequest, SearchResponse, SearchResult
from config import settings

# ---------------------------------------------------------------------------
# Akamai keepalive heartbeat interval (seconds).
# Akamai has a 5-minute (300s) idle-connection timeout. Sending a whitespace
# byte every HEARTBEAT_INTERVAL_S seconds keeps the connection alive regardless
# of how long the LLM takes to respond.
# ---------------------------------------------------------------------------
_AKAMAI_HEARTBEAT_INTERVAL_S: float = float(
    getattr(settings, "akamai_heartbeat_interval_s", 30)
)

def _format_ms(ms_val: float) -> str:
    if not ms_val: return "0s"
    sec = ms_val / 1000.0
    if sec < 60: return f"{sec:.2f}s"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}m {s:.0f}s"

logger = logging.getLogger(__name__)
query_logger = logging.getLogger("queries")

router = APIRouter()

from agents.llm_client import ThreadSafeSemaphore

# Semaphore limits concurrent Ollama calls; prevents LLM cluster saturation.
LLM_CONCURRENCY_LIMIT = getattr(settings, "llm_concurrency_limit", 15)
llm_semaphore = ThreadSafeSemaphore(LLM_CONCURRENCY_LIMIT)

# ---------------------------------------------------------------------------
# Query task registry tracks active RAG-graph query tasks for cancellation.
# ISOLATED from neaps_api._ingestion_tasks (ingestion is never affected).
# ---------------------------------------------------------------------------
_query_tasks: dict = {}
_query_tasks_lock = threading.Lock()


def _register_query_task(task_id: str) -> threading.Event:
    """Register a new active query task; returns its cancellation Event."""
    cancel_event = threading.Event()
    with _query_tasks_lock:
        _query_tasks[task_id] = {
            "task_id": task_id,
            "status": "running",
            "created_at": _time.time(),
            "cancel_event": cancel_event,
        }
    return cancel_event


def _finish_query_task(task_id: str, status_val: str = "completed") -> None:
    """Mark a query task as finished (completed/failed/cancelled)."""
    with _query_tasks_lock:
        task = _query_tasks.get(task_id)
        if task and task.get("status") not in {"completed", "failed", "cancelled"}:
            task["status"] = status_val


def _pick_page(metadata: dict) -> str | None:
    for key in ("page", "page_number", "page_no", "source_page"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _pick_announcement_date(metadata: dict) -> str | None:
    for key in ("ann_created_date", "ann_date", "announcement_date", "date"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _strip_web_intelligence(answer: str) -> str:
    if not answer:
        return answer
    lines = answer.splitlines()
    out = []
    skip = False
    for line in lines:
        marker = line.strip().lower()
        if marker == "web intelligence":
            skip = True
            continue
        if (
            skip
            and re.match(r"^[A-Z][A-Z\s]+$", line.strip())
            and len(line.strip()) > 4
        ):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out).strip()


def _enrich_citations(answer: str, sources: list[dict]) -> str:
    if not answer:
        return answer
    if not sources:
        return answer

    source_map: dict[tuple[str, str], dict] = {}
    for src in sources:
        metadata = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
        filename = (
            metadata.get("filename")
            or metadata.get("file_name")
            or src.get("title")
            or ""
        )
        chunk_number = str(
            metadata.get("chunk_number") or src.get("chunk_id") or ""
        ).strip()
        if filename and chunk_number:
            source_map[(str(filename).strip().lower(), chunk_number)] = metadata

    pattern = re.compile(r"\(Document:\s*([^,\)]+),\s*Chunk:\s*([^\),]+)(?:,[^\)]*)?\)")

    def repl(match: re.Match) -> str:
        filename = match.group(1).strip()
        chunk_number = match.group(2).strip()
        metadata = source_map.get((filename.lower(), chunk_number))
        if not metadata:
            return f"(Document: {filename}, Chunk: {chunk_number})"
        parts = [f"Document: {filename}", f"Chunk: {chunk_number}"]
        page = _pick_page(metadata)
        ann_date = _pick_announcement_date(metadata)
        if page:
            parts.append(f"Page: {page}")
        if ann_date:
            parts.append(f"Announcement Date: {ann_date}")
        return f"({', '.join(parts)})"

    return pattern.sub(repl, answer)


# ---------------------------------------------------------------------------
# Internal core function that performs the full search + LLM answer pipeline.
# It is called from the streaming wrapper below so the logic stays unchanged.
# ---------------------------------------------------------------------------
async def _run_search_core(request: SearchRequest) -> SearchResponse:
    """
    Execute the full search pipeline and return a SearchResponse.

    Vocabulary-aware hybrid search with optional LLM answer generation.
    Two-layer strategy:
      Layer 1 - precise, no expansion.
      Layer 2 - dictionary-expanded, triggered when confidence < 0.5 or 0 results.
      NEAPS fallback - searches official filings when both layers return nothing.

    Raises HTTPException on validation/server errors.
    """

    import time
    from datetime import datetime

    try:
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        mode = "graph" if getattr(request, "use_graph", False) else "hybrid"
        logger.info(
            f"[SearchFlow:{trace_id}] START mode={mode} top_k={request.top_k} "
            f"companies={request.companies or []} query='{(request.query or '')[:120]}'"
        )

        # ------ Query deduplication / sanitisation -------------------------
        raw_query = (request.query or "").strip()
        if "\n" in raw_query:
            seen, parts = set(), []
            for part in raw_query.split("\n"):
                pc = part.strip().lower()
                if pc and pc not in seen:
                    seen.add(pc)
                    parts.append(part.strip())
            clean_query = " ".join(parts) if parts else raw_query
        else:
            clean_query = raw_query

        from security.input_guardrails import get_guardrail

        guardrail_result = get_guardrail().validate(clean_query)
        logger.info(
            f"[SearchFlow:{trace_id}] Guardrail allowed={guardrail_result.allowed} "
            f"warnings={len(guardrail_result.warnings or [])}"
        )

        if not guardrail_result.allowed:
            query_logger.warning(
                f"Search blocked: trace={trace_id} threat={guardrail_result.threat_type}"
            )
            raise HTTPException(
                status_code=guardrail_result.http_status,
                detail=guardrail_result.reason,
            )

        request_query = guardrail_result.sanitized_query
        if request_query != raw_query:
            logger.info(f"[SearchFlow:{trace_id}] Query sanitized by guardrails")

        # ------ Build filter conditions ------------------------------------
        filter_conditions: dict = {}
        if request.companies:
            filter_conditions["company_symbol"] = {"$in": request.companies}
        if request.date_from or request.date_to:
            date_filter: dict = {}
            for attr, op in [("date_from", "$gte"), ("date_to", "$lte")]:
                val = getattr(request, attr, None)
                if val:
                    try:
                        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                        date_filter[op] = dt.timestamp()
                    except (ValueError, AttributeError):
                        logger.warning(f"Invalid {attr} format: {val}")
            if date_filter:
                filter_conditions["announcement_date_ts"] = date_filter
        if request.filters:
            filter_conditions.update(request.filters)
        logger.info(f"[SearchFlow:{trace_id}] Filters built: {filter_conditions or {}}")
        # ADD after: logger.info(f"[SearchFlow:{trace_id}] Filters built: ...")
        # search_mode = getattr(request, "search_mode", "announcements") or "announcements"
        # logger.info(f"[SearchFlow:{trace_id}] search_mode={search_mode}")

        # ------ Graph mode (formerly /search/v2) ---------------------------
        if getattr(request, "use_graph", False):
            try:
                # ---- Query Classification: bypass Gardle for simple queries ----
                from api.routers.query_classifier import classify_query

                classification = classify_query(request_query)
                logger.info(
                    f"[SearchFlow:{trace_id}] Classification={classification.category} "
                    f"confidence={classification.confidence:.2f}"
                )

                if classification.category in ("small_talk", "basic_info"):
                    # Return direct answer  Gardle is NOT triggered
                    logger.info(
                        f"[SearchFlow:{trace_id}] Bypassing Gardle "
                        f"(category={classification.category})"
                    )
                    elapsed_ms = (_time.time() - start_time) * 1000
                    return SearchResponse(
                        query=request.query,
                        results=[],
                        total_results=0,
                        expanded_query=None,
                        retrieval_time_ms=elapsed_ms,
                        chunks_used=0,
                        answer=classification.direct_answer,
                        confidence=1.0,
                        executive_summary=classification.direct_answer,
                        neaps_intelligence=None,
                        sources=[],
                        reasoning="Direct bypass",
                        #generation_time_ms="0s",
                        score_breakdown={},
                        latencies_ms={"classifier_ms": elapsed_ms},
                        query_id=trace_id,
                        guardrail_warning=(
                            ", ".join(guardrail_result.warnings)
                            if guardrail_result.warnings
                            else None
                        ),
                    )
                # ----------------------------------------------------------------

                logger.info(f"[SearchFlow:{trace_id}] Graph mode begin")
                from agents.rag_graph import run_rag_query

                company_id = request.company_id or (
                    request.companies[0] if request.companies else None
                )
                filters_for_graph = dict(filter_conditions)
                if company_id:
                    filters_for_graph["company_symbol"] = company_id

                # Register this query task for stop-button cancellation support
                cancel_event = _register_query_task(trace_id)

                loop = asyncio.get_running_loop()
                try:
                    result = await loop.run_in_executor(
                        None,
                        functools.partial(
                            run_rag_query,
                            query=request_query,
                            company_id=company_id,
                            filters=filters_for_graph or None,
                        ),
                    )
                finally:
                    # Always clean up the task registry on completion or error
                    _finish_query_task(trace_id, "completed")

                # Check if cancelled during execution
                if cancel_event.is_set():
                    logger.info(
                        f"[SearchFlow:{trace_id}] Query was cancelled by user during execution"
                    )
                    elapsed_ms = (_time.time() - start_time) * 1000
                    return SearchResponse(
                        query=request.query,
                        results=[],
                        total_results=0,
                        expanded_query=None,
                        retrieval_time_ms=elapsed_ms,
                        answer="Stopped by user.",
                        confidence=0.0,
                        executive_summary="Stopped by user.",
                        neaps_intelligence=None,
                        sources=[],
                        latencies_ms={"cancelled_ms": elapsed_ms},
                        query_id=trace_id,
                        guardrail_warning=None,
                    )
                logger.info(
                    f"[SearchFlow:{trace_id}] Graph mode completed "
                    f"query_id={result.get('query_id', '')} "
                    f"sources_raw={len(result.get('sources') or [])}"
                )

                latencies = result.get("latencies_ms", {}) or {}
                retrieval_ms = latencies.get("total_ms") or (
                    sum(latencies.values()) if latencies else 0
                )

                raw_sources = result.get("sources") or []
                neaps_sources = [
                    src
                    for src in raw_sources
                    if str(src.get("source_type", "")).lower() == "neaps_document"
                    or str((src.get("metadata") or {}).get("source", "")).lower()
                    == "neaps_document"
                ]
                if neaps_sources:
                    raw_sources = neaps_sources
                mapped_sources = []
                for idx, src in enumerate(raw_sources):
                    chunk_id_raw = (
                        src.get("chunk_id")
                        or src.get("id")
                        or src.get("ann_seq")
                        or f"source-{idx}"
                    )
                    chunk_number = str(idx + 1)
                    meta_in = src.get("metadata", {})
                    metadata = dict(meta_in) if isinstance(meta_in, dict) else {}
                    metadata.setdefault("company_symbol", src.get("company_symbol"))
                    metadata.setdefault(
                        "filename",
                        (meta_in.get("filename") if isinstance(meta_in, dict) else None)
                        or (
                            meta_in.get("file_name")
                            if isinstance(meta_in, dict)
                            else None
                        )
                        or src.get("title"),
                    )
                    download_candidate = None
                    if isinstance(meta_in, dict):
                        download_candidate = (
                            meta_in.get("download_url")
                            or meta_in.get("ANN_SERVER_FILE_ATTCH")
                            or meta_in.get("ANN_FILE_ATTCH")
                        )
                    download_candidate = (
                        download_candidate or src.get("download_url") or src.get("url")
                    )
                    if download_candidate:
                        filename = os.path.basename(str(download_candidate))
                        metadata["download_url"] = f"/neaps/download/{filename}"
                    metadata["chunk_id_raw"] = str(chunk_id_raw)
                    metadata["chunk_number"] = chunk_number
                    metadata["page"] = (
                        metadata.get("page")
                        or metadata.get("page_number")
                        or metadata.get("page_no")
                        or metadata.get("source_page")
                    )
                    metadata["ann_created_date"] = (
                        metadata.get("ann_created_date")
                        or metadata.get("ann_date")
                        or metadata.get("announcement_date")
                        or metadata.get("date")
                    )

                    mapped_sources.append(
                        {
                            "chunk_id": chunk_number,
                            "text": src.get("text", "") or src.get("summary", ""),
                            "score": float(
                                src.get("rerank_score")
                                if src.get("rerank_score") is not None
                                else src.get(
                                    "score", src.get("fused_score", 0.0) or 0.0
                                )
                            ),
                            "metadata": metadata,
                        }
                    )

                max_sources = min(request.top_k or 8, 8)
                mapped_sources = mapped_sources[:max_sources]
                formatted_summary = _strip_web_intelligence(
                    result.get("executive_summary") or ""
                )
                formatted_summary = _enrich_citations(formatted_summary, mapped_sources)
                confidence = float(result.get("neaps_confidence", 0.0))
                top_sources = [
                    f"{(s.get('metadata') or {}).get('filename', 'Unknown')}#C{s.get('chunk_id')}"
                    for s in mapped_sources[:3]
                ]
                logger.info(
                    f"[SearchFlow:{trace_id}] Graph sources mapped={len(mapped_sources)} "
                    f"retrieval_ms={retrieval_ms:.0f} confidence={confidence:.3f} "
                    f"top_sources={top_sources}"
                )
                logger.info(
                    f"[SearchFlow:{trace_id}] Graph answer_preview='{(formatted_summary or '')[:220]}'"
                )

                return SearchResponse(
                    query=request.query,
                    results=[],
                    total_results=len(mapped_sources),
                    expanded_query=None,
                    retrieval_time_ms=retrieval_ms,

                    answer=formatted_summary,
                    confidence=confidence,
                    executive_summary=formatted_summary,
                    neaps_intelligence=result.get("neaps_intelligence"),
                    sources=mapped_sources,
                    latencies_ms=latencies,
                    query_id=result.get("query_id"),
                    guardrail_warning=(
                        ", ".join(guardrail_result.warnings)
                        if guardrail_result.warnings
                        else None
                    ),
                )
            except Exception as exc:
                logger.error(f"[Search] Graph mode failed: {exc}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Graph search failed: {str(exc)}",
                )

        # ------ Answer generator -------------------------------------------
        answer_gen = None
        if settings.use_answer_generation:
            try:
                from core.answer_generator import get_answer_generator

                answer_gen = get_answer_generator()
            except Exception as e:
                logger.warning(f"Could not load answer generator: {e}")

        from retrieval.hybrid_retriever import get_hybrid_retriever

        retriever = get_hybrid_retriever()
        loop = asyncio.get_running_loop()
        logger.info(f"[SearchFlow:{trace_id}] Hybrid mode components ready")

        # ------ Layer 1: precise search ------------------------------------
        logger.info(
            f"[SearchFlow:{trace_id}] Layer1 retrieve begin query='{request_query[:80]}'"
        )
        # ADD before the retrieve_result call (in the hybrid section, not graph section)
        # retriever_filters = dict(filter_conditions) if filter_conditions else {}
        # retriever_filters["search_mode"] = search_mode
        retrieve_result = await loop.run_in_executor(
            None,
            functools.partial(
                retriever.retrieve,
                query=request_query,
                top_k=request.top_k,
                filters=filter_conditions or None,   # ← was filter_conditions or None

            ),
        )
        results = retrieve_result.get("all_chunks", [])
        # ADD after: results = retrieve_result.get("all_chunks", [])
        # news_chunks_raw = retrieve_result.get("news_chunks", [])
        # news_answer = None
        # news_results_out = None
        logger.info(
            f"[SearchFlow:{trace_id}] Layer1 retrieve done results={len(results)} "
            f"latencies={retrieve_result.get('latencies_ms', {})}"
        )

        answer = None
        confidence = 0.0
        sources = None
        reasoning = None
        generation_time_ms = 0
        score_breakdown = None
        layer_used = "layer_1_precise"

        if answer_gen and results:
            try:
                async with llm_semaphore:
                    ar = await loop.run_in_executor(
                        None,
                        functools.partial(
                            answer_gen.generate_answer,
                            query=request_query,
                            chunks=results,
                            max_chunks=min(request.top_k, 8),
                        ),
                    )
                answer = ar.get("answer")
                confidence = ar.get("confidence", 0.0)
                sources = ar.get("sources")
                reasoning = ar.get("reasoning")
                generation_time_ms = ar.get("generation_time_ms", 0)
                score_breakdown = ar.get("score_breakdown")
                logger.info(
                    f"[SearchFlow:{trace_id}] Layer1 answer confidence={confidence:.2f}"
                )
            except Exception as e:
                logger.error(
                    f"[SearchFlow:{trace_id}] Layer1 answer generation failed: {e}"
                )

        # ------ Layer 2: conditional expansion -----------------------------
        should_expand = False
        if settings.is_query_expansion_enabled:
            if not results:
                logger.info("[Search] Layer 1 -> 0 results, triggering expansion.")
                should_expand = True
            elif confidence < 0.5:
                logger.info(
                    f"[Search] Layer 1 confidence low ({confidence:.2f}), expanding."
                )
                should_expand = True

        if should_expand:
            logger.info(f"[SearchFlow:{trace_id}] Layer2 expansion triggered")
            expanded_request_query = request_query
            if settings.is_query_expansion_enabled:
                try:
                    from vocabulary.query_expander import get_query_expander

                    expanded_obj = get_query_expander().expand_query(
                        request_query, getattr(request, "content_type", None)
                    )
                    expanded_request_query = expanded_obj.boolean_query
                except Exception as e:
                    logger.warning(
                        f"[SearchFlow:{trace_id}] Query expansion failed (using original): {e}"
                    )

            rr2 = await loop.run_in_executor(
                None,
                functools.partial(
                    retriever.retrieve,
                    query=expanded_request_query,
                    top_k=request.top_k,
                    filters=filter_conditions or None,
                ),
            )
            results2 = rr2.get("all_chunks", [])
            logger.info(
                f"[SearchFlow:{trace_id}] Layer2 retrieve done results={len(results2)} "
                f"expanded_query='{expanded_request_query[:120]}'"
            )
            if results2 and answer_gen:
                try:
                    async with llm_semaphore:
                        ar2 = await loop.run_in_executor(
                            None,
                            functools.partial(
                                answer_gen.generate_answer,
                                query=request_query,
                                chunks=results2,
                                max_chunks=min(request.top_k, 8),
                            ),
                        )
                    conf2 = ar2.get("confidence", 0.0)
                    if conf2 >= confidence or not results:
                        results = results2
                        answer = ar2.get("answer")
                        confidence = conf2
                        sources = ar2.get("sources")
                        reasoning = ar2.get("reasoning")
                        generation_time_ms += ar2.get("generation_time_ms", 0)
                        score_breakdown = ar2.get("score_breakdown")
                        layer_used = "layer_2_expanded"
                except Exception as e:
                    logger.error(
                        f"[SearchFlow:{trace_id}] Layer2 answer generation failed: {e}"
                    )
            elif results2:
                results = results2
                layer_used = "layer_2_expanded_no_llm"

        # ------ NEAPS fallback ---------------------------------------------
        if not results:
            logger.info(f"[SearchFlow:{trace_id}] NEAPS fallback begin")
            try:
                from retrieval.neaps_hybrid_search import search as neaps_search

                # Fix: correctly gather all company symbols
                company_symbol = None
                if filter_conditions.get("company_symbol"):
                    cs = filter_conditions["company_symbol"]
                    if isinstance(cs, dict) and "$in" in cs:
                        symbols = [s for s in (cs["$in"] or []) if s]
                        company_symbol = symbols if symbols else None
                    else:
                        company_symbol = cs

                neaps_results = await loop.run_in_executor(
                    None,
                    functools.partial(
                        neaps_search,
                        query=request_query,
                        company_symbol=company_symbol,
                        top_k=20,
                        final_top_k=request.top_k,
                    ),
                )
                logger.info(
                    f"[SearchFlow:{trace_id}] NEAPS fallback results={len(neaps_results or [])}"
                )
                if neaps_results:
                    results = [
                        {
                            "id": r.get("id", ""),
                            "text": r.get("text", ""),
                            "metadata": r.get("metadata", {}),
                            "score": r.get("score", 0.0),
                            "fused_score": r.get("score", 0.0),
                            "source_type": "neaps_document",
                        }
                        for r in neaps_results
                    ]
                    layer_used = "layer_neaps_fallback"
                    if answer_gen:
                        try:
                            async with llm_semaphore:
                                arn = await loop.run_in_executor(
                                    None,
                                    functools.partial(
                                        answer_gen.generate_answer,
                                        query=request_query,
                                        chunks=results,
                                        max_chunks=min(request.top_k, 8),
                                    ),
                                )
                            answer = arn.get("answer")
                            confidence = arn.get("confidence", 0.0)
                            sources = arn.get("sources")
                            reasoning = arn.get("reasoning")
                            generation_time_ms += arn.get("generation_time_ms", 0)
                        except Exception as e:
                            logger.error(
                                f"[SearchFlow:{trace_id}] NEAPS answer generation failed: {e}"
                            )
            except Exception as e:
                logger.warning(
                    f"[SearchFlow:{trace_id}] NEAPS fallback failed (non-critical): {e}"
                )

        # ADD before: # ------ Build response
        # if news_chunks_raw and answer_gen:
        #     try:
        #         async with llm_semaphore:
        #             news_ar = await loop.run_in_executor(
        #                 None,
        #                 functools.partial(
        #                     answer_gen.generate_answer,
        #                     query=request_query,
        #                     chunks=news_chunks_raw,
        #                     max_chunks=min(request.top_k, 8),
        #                 ),
        #             )
        #         news_answer = news_ar.get("answer")
        #         logger.info(
        #             f"[SearchFlow:{trace_id}] News answer generated confidence={news_ar.get('confidence', 0):.2f}")
        #     except Exception as e:
        #         logger.warning(f"[SearchFlow:{trace_id}] News answer generation failed (non-fatal): {e}")
        # if news_chunks_raw:
        #     news_results_out = [
        #         SearchResult(
        #             id=str(r.get("id", "")),
        #             text=r.get("text", ""),
        #             metadata=r.get("metadata", {}),
        #             vector_score=float(r.get("vector_score", r.get("score", 0.0))),
        #             bm25_score=float(r.get("bm25_score", 0.0)),
        #             dictionary_boost=0.0,
        #             fused_score=float(r.get("fused_score", r.get("score", 0.0))),
        #             rerank_score=(
        #                 float(r["rerank_score"]) if r.get("rerank_score") is not None else None
        #             ),
        #         )
        #         for r in news_chunks_raw
        #     ]


        # ------ Build response ---------------------------------------------
        retrieval_time_ms = (time.time() - start_time) * 1000
        expanded_query = None
        if layer_used == "layer_2_expanded" and settings.is_query_expansion_enabled:
            try:
                from vocabulary.query_expander import get_query_expander

                expander = get_query_expander()
                expanded = expander.expand_query(
                    request.query, getattr(request, "content_type", None)
                )
                expanded_query = {
                    "original": expanded.original_query,
                    "expanded_terms": expanded.expanded_terms,
                    "boolean_query": expanded.boolean_query,
                    "mappings_count": len(expanded.mappings_used),
                }
            except Exception:
                pass

        query_logger.info(
            f"Search trace={trace_id} ({layer_used}): query='{request.query[:50]}' "
            f"results={len(results)} time={retrieval_time_ms:.0f}ms"
        )
        logger.info(
            f"[SearchFlow:{trace_id}] COMPLETE layer={layer_used} results={len(results)} "
            f"answer={'yes' if bool(answer) else 'no'} retrieval_ms={retrieval_time_ms:.0f} "
            f"confidence={confidence:.3f} sources={len(sources or [])}"
        )
        if answer:
            logger.info(
                f"[SearchFlow:{trace_id}] Hybrid answer_preview='{answer[:220]}'"
            )

        # Final defensive slice to guarantee we never over-return what UI requested
        results = results[:request.top_k]

        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    id=str(r.get("id", "")),
                    text=r.get("text", ""),
                    metadata=r.get("metadata", {}),
                    vector_score=float(r.get("vector_score", r.get("score", 0.0))),
                    bm25_score=float(r.get("bm25_score", 0.0)),
                    dictionary_boost=float(r.get("dictionary_boost", 0.0)),
                    fused_score=float(r.get("fused_score", r.get("score", 0.0))),
                    rerank_score=(
                        float(r["rerank_score"])
                        if r.get("rerank_score") is not None
                        else None
                    ),
                )
                for r in results
            ],
            total_results=len(results),
            expanded_query=expanded_query,
            retrieval_time_ms=retrieval_time_ms,
            answer=answer,
            confidence=confidence,
            sources=sources,
            reasoning=reasoning,
            generation_time_ms=generation_time_ms,
            score_breakdown=score_breakdown,
            guardrail_warning=(
                ", ".join(guardrail_result.warnings)
                if guardrail_result.warnings
                else None
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        query_logger.error(f"Search failed for query '{request.query[:50]}': {e}")
        logger.error(f"[Search] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# Streaming endpoint — keeps Akamai alive via periodic heartbeat bytes.
#
# HOW IT WORKS:
#   1. The search pipeline runs as a background asyncio.Task.
#   2. While the task runs, we yield a single space+newline (b" \n") every
#      HEARTBEAT_INTERVAL_S seconds. Akamai sees data flowing and resets
#      its idle-connection timer — no 5-minute cut.
#   3. When the task finishes we yield the final JSON-encoded SearchResponse
#      as the last chunk and close the stream.
#
# NO FRONTEND CHANGES REQUIRED:
#   Content-Type is kept as application/json.
#   RFC 8259 (JSON spec) explicitly allows leading whitespace before the
#   JSON value: ws = *(SP / HTAB / LF / CR). This means the full response
#   body " \n \n {...json...}" is 100% valid JSON.
#   The browser/Axios buffers all chunks and parses the final body — the
#   heartbeat whitespace is silently ignored by every standard JSON parser.
#
# AKAMAI CONFIGURATION NOTE:
#   This works as long as Akamai forwards bytes from the origin to the client
#   progressively (not buffering the full body). If Akamai is in full-buffer
#   mode for this path, ask your Akamai team to enable chunked/streaming
#   pass-through for /api/v1/search.
# ---------------------------------------------------------------------------
@router.post("/search")
async def search(request: SearchRequest) -> StreamingResponse:
    """
    Vocabulary-aware hybrid search with Akamai-safe streaming keepalive.

    Streams periodic whitespace bytes (valid JSON leading whitespace per RFC 8259)
    to prevent Akamai's 5-minute idle timeout from dropping long LLM queries.
    The final chunk is the complete JSON-encoded SearchResponse.
    The response content-type remains application/json — NO frontend changes needed.
    """

    async def _stream_with_heartbeat():
        """Async generator: whitespace heartbeats while LLM runs, then final JSON."""
        # Launch the full search pipeline as a concurrent task.
        # Use get_running_loop() (not get_event_loop()) — always returns the
        # correct loop for the current coroutine; get_event_loop() is deprecated
        # in Python ≥ 3.10 and can return a closed/different loop under workers.
        loop = asyncio.get_running_loop()
        task: asyncio.Task = loop.create_task(_run_search_core(request))

        # Pump whitespace heartbeat bytes until the task completes.
        # These bytes are valid JSON leading whitespace (RFC 8259 §2) so the
        # frontend's response.json() / JSON.parse() will still parse correctly.
        while not task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(task), timeout=_AKAMAI_HEARTBEAT_INTERVAL_S
                )
            except asyncio.TimeoutError:
                # Task still running — send whitespace keepalive to Akamai
                logger.debug(
                    "[Search][Akamai-heartbeat] Sending keepalive for query='%s'",
                    (request.query or "")[:60],
                )
                yield b"   \r\n"   # spaces + CRLF = valid JSON ws, invisible to parsers
            except Exception:
                # Task raised — exit loop, surface below
                break

        # Retrieve result (raises if task failed)
        try:
            response_obj: SearchResponse = task.result()
        except HTTPException as http_exc:
            # Return a valid JSON error object — frontend reads this as the body
            error_payload = json.dumps(
                {"error": True, "status_code": http_exc.status_code, "detail": http_exc.detail}
            )
            yield error_payload.encode("utf-8")
            return
        except Exception as exc:
            logger.error("[Search] Core pipeline failed: %s", exc, exc_info=True)
            error_payload = json.dumps({"error": True, "detail": str(exc)})
            yield error_payload.encode("utf-8")
            return

        # Final chunk: the complete SearchResponse JSON (no trailing newline)
        yield response_obj.model_dump_json().encode("utf-8")

    return StreamingResponse(
        _stream_with_heartbeat(),
        # Keep application/json so the frontend's response.json() / Axios
        # parses the response without any modification.
        media_type="application/json",
        headers={
            # Tell proxies not to buffer — required for heartbeat bytes to flow
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",   # disables Nginx proxy buffering if present
        },
    )