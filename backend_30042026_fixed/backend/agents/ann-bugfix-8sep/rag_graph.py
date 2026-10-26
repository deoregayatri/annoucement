"""
LangGraph RAG Agent - NEAPS-only, single pipeline.

4-node linear graph:

  planner -> neaps_retriever -> neaps_answer -> executive_summary -> END

No parallel branches. No concurrent writes. Single retrieval pass.
"""

import logging
import time
import uuid
import math
import os as _os

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from agents.llm_utils import _call_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class RagGraphState(TypedDict, total=False):
    # Input
    query: str
    company_id: Optional[str]
    filters: Optional[Dict[str, Any]]

    # Retrieval
    query_id: str
    neaps_chunks: List[Dict[str, Any]]
    neaps_confidence: float

    # Answers
    neaps_answer: str
    executive_summary: str

    # Output metadata
    sources: List[Dict[str, Any]]
    latencies_ms: Annotated[Dict[str, float], lambda a, b: {**a, **b}]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node 1: Planner
# ---------------------------------------------------------------------------


def planner_node(state: RagGraphState) -> RagGraphState:
    """Parse query, normalise company filter, generate query_id."""
    t0 = time.perf_counter()

    query = state.get("query", "")
    company_id = state.get("company_id") or (state.get("filters", {}) or {}).get(
        "company_symbol"
    )
    query_id = state.get("query_id") or str(uuid.uuid4())

    logger.info(f"[PlannerNode] query_id={query_id[:8]} company={company_id}")

    return {
        "query_id": query_id,
        "company_id": company_id,
        "latencies_ms": {"planner_ms": (time.perf_counter() - t0) * 1000},
    }


# ---------------------------------------------------------------------------
# Node 2: NEAPS Retriever (dense + BM25 + RRF + rerank)
# ---------------------------------------------------------------------------


def neaps_retriever_node(state: RagGraphState) -> RagGraphState:
    """Hybrid search: dense Qdrant + BM25 sparse + RRF fusion + cross-encoder rerank."""
    t0 = time.perf_counter()

    company = state.get("company_id") or (state.get("filters", {}) or {}).get(
        "company_symbol"
    )

    if not company:
        logger.debug("[NeapsRetriever] No company context - skipping.")
        return {
            "neaps_chunks": [],
            "neaps_confidence": 0.0,
            "latencies_ms": {"neaps_retriever_ms": (time.perf_counter() - t0) * 1000},
        }

    if not company:
        logger.debug("[NeapsRetriever] No company context - skipping.")
        return {
            "neaps_chunks": [],
            "neaps_confidence": 0.0,
            "latencies_ms": {"neaps_retriever_ms": (time.perf_counter() - t0) * 1000},
        }

    try:
        from retrieval.neaps_hybrid_search import search as neaps_search

        reranked = neaps_search(
            query=state["query"],
            company_symbol=company,
            top_k=40,
            final_top_k=10,
        )

        # Compute confidence from top result
        top_rerank = 0.0
        if reranked:
            top = reranked[0]
            raw_rerank = top.get("rerank_score")
            dense_score = top.get("dense_score")

            if raw_rerank is not None and abs(float(raw_rerank)) > 0.1:
                # Real cross-encoder score (typically > 0.5 or < -0.5)
                top_rerank = 1.0 / (1.0 + math.exp(-float(raw_rerank)))
            elif dense_score is not None:
                # Qdrant dense cosine similarity - already in [0, 1] range
                top_rerank = max(0.0, min(1.0, float(dense_score)))
            else:
                # Fallback: normalize fused_score relative to its result set max
                fused_scores = [
                    float(c.get("score", c.get("fused_score", 0.0)))
                    for c in reranked
                    if c.get("score") or c.get("fused_score")
                ]
                if fused_scores:
                    max_fused = max(fused_scores)
                    top_fused = float(top.get("score", top.get("fused_score", 0.0)))
                    top_rerank = top_fused / max_fused if max_fused > 0 else 0.0
                else:
                    top_rerank = 0.0

        raw_display = reranked[0].get("rerank_score") if reranked else None
        logger.info(
            f"[NeapsRetriever] company='{company}' chunks={len(reranked)} "
            f"confidence={top_rerank:.3f} rerank={raw_display}"
        )

        return {
            "neaps_chunks": reranked,
            "neaps_confidence": top_rerank,
            "latencies_ms": {"neaps_retriever_ms": (time.perf_counter() - t0) * 1000},
        }

    except Exception as exc:
        logger.error(f"[NeapsRetriever] Failed: {exc}", exc_info=True)
        return {
            "neaps_chunks": [],
            "neaps_confidence": 0.0,
            "error": str(exc),
            "latencies_ms": {"neaps_retriever_ms": (time.perf_counter() - t0) * 1000},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_context(chunks: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Build LLM context string from chunk list, capped at max_chars."""
    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("full_text") or chunk.get("text", "")
        meta = chunk.get("metadata", {})

        file_name = (
            meta.get("file_name")
            or meta.get("filename")
            or meta.get("title")
            or "Unknown"
        )
        chunk_number = i + 1
        page = (
            meta.get("page")
            or meta.get("page_number")
            or meta.get("page_no")
            or meta.get("source_page")
        )
        announcement_date = (
            meta.get("ann_created_date")
            or meta.get("ann_date")
            or meta.get("announcement_date")
            or meta.get("date")
        )

        header_fields = [f"Document: {file_name}", f"Chunk: {chunk_number}"]
        if page not in (None, ""):
            header_fields.append(f"Page: {page}")
        if announcement_date not in (None, ""):
            header_fields.append(f"Announcement Date: {announcement_date}")
        header = f"[{' | '.join(header_fields)}]\n"

        entry = header + text + "\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Node 3: Unified Answer (single LLM call — replaces neaps_answer + exec_summary)
# ---------------------------------------------------------------------------
# Performance fix: previously two sequential LLM calls (~24s total) were made:
#   neaps_answer_node    → LLM call #1 (~12s) : answer from context
#   executive_summary_node → LLM call #2 (~12s) : re-format call #1's output
# Now: one combined prompt produces the full structured answer directly.
# Saving: ~12s per query on /api/v1/search path.
# ---------------------------------------------------------------------------


def unified_answer_node(state: RagGraphState) -> RagGraphState:
    """Single LLM call: build context, answer query, cite sources, produce executive summary."""
    t0 = time.perf_counter()

    neaps_chunks = state.get("neaps_chunks") or []
    if not neaps_chunks:
        return {
            "neaps_answer": "Not found in provided documents.",
            "executive_summary": "Not found in provided documents.",
            "sources": [],
            "latencies_ms": {"unified_answer_ms": (time.perf_counter() - t0) * 1000},
        }

    context = _build_context(neaps_chunks)
    query = state.get("query", "")

    # Build valid chunk IDs list to prevent citation hallucination (Bug 4 fix)
    valid_chunk_ids = []
    for i, chunk in enumerate(neaps_chunks):
        meta = chunk.get("metadata", {})
        cid = meta.get("chunk_id") or chunk.get("id") or f"chunk-{i+1}"
        valid_chunk_ids.append(str(cid))

    prompt = f"""You are a financial document intelligence assistant.\
 Use ONLY the provided documents. Do not invent data.

STRICT RULES:
1. You MUST summarize ALL provided documents. Do NOT skip or omit any.
2. ONLY cite from this list of valid chunk IDs: {valid_chunk_ids}
   Do NOT reference any chunk ID not in that list.
3. Cite every fact using: (Document: <file_name>, Chunk: <id>, Page: <page>, Date: <date>)
4. If information is missing, respond exactly: Not found in provided documents.

QUERY: {query}

OFFICIAL NEAPS FILINGS:
{context}

Respond in this exact structure:

ANSWER TO THE QUERY
<Direct answer with citations>

EVIDENCE FROM ANNOUNCEMENTS
"<Exact supporting quote>"
(Document: <filename>, Chunk: <id>, Page: <page>, Date: <date>)

FINAL COMPLIANCE ASSESSMENT
<Short factual conclusion>

ANSWER:"""

    llm_t0 = time.perf_counter()
    answer = _call_llm(prompt)
    llm_ms = (time.perf_counter() - llm_t0) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    # Fallback if LLM returned empty / error
    if (not answer) or ("could not generate" in answer.lower()):
        answer = "No coherent summary could be generated from the retrieved announcements."

    logger.info(
        f"[UnifiedAnswer] Generated {len(answer)} chars llm={llm_ms:.0f}ms total={total_ms:.0f}ms"
    )

    # Build sources from neaps_chunks
    sources = []
    for idx, chunk in enumerate(neaps_chunks):
        meta = chunk.get("metadata", {})

        ann_server_file = (
            meta.get("ANN_SERVER_FILE_ATTCH")
            or meta.get("ANN_FILE_ATTCH")
            or meta.get("file_name")
            or meta.get("filename")
            or ""
        )
        sftp_filename = (
            _os.path.basename(str(ann_server_file)) if ann_server_file else ""
        )
        display_filename = (
            meta.get("file_name") or meta.get("filename") or meta.get("title")
        )

        sources.append(
            {
                "title": meta.get("title", chunk.get("title", "")),
                "company_symbol": meta.get(
                    "company_symbol", chunk.get("company_symbol", "")
                ),
                "source_type": chunk.get("source_type", "neaps_document"),
                "url": meta.get("url", ""),
                "text": chunk.get("text", ""),
                "metadata": {
                    "filename": display_filename,
                    "file_name": display_filename,
                    "chunk_id": meta.get("chunk_id")
                    or chunk.get("id")
                    or chunk.get("chunk_id"),
                    "chunk_number": idx + 1,
                    "page": (
                        meta.get("page")
                        or meta.get("page_number")
                        or meta.get("page_no")
                        or meta.get("source_page")
                    ),
                    "ann_date": meta.get("ann_date") or meta.get("announcement_date"),
                    "ann_created_date": meta.get("ann_created_date"),
                    "date": meta.get("ann_date") or meta.get("announcement_date"),
                    "source": (
                        meta.get("source")
                        or meta.get("source_type")
                        or chunk.get("source_type", "neaps_document")
                    ),
                    "company_symbol": meta.get(
                        "company_symbol", chunk.get("company_symbol", "")
                    ),
                    "download_url": f"/neaps/download/{sftp_filename}"
                    if sftp_filename
                    else None,
                },
                "score": chunk.get("score", chunk.get("fused_score", 0.0)),
                "rerank_score": chunk.get("rerank_score"),
            }
        )

    return {
        "neaps_answer": answer,
        "executive_summary": answer,
        "sources": sources,
        "latencies_ms": {
            "unified_answer_ms": total_ms,
            "unified_llm_ms": llm_ms,
            # Legacy keys kept so callers don't break
            "neaps_answer_ms": total_ms,
            "exec_summary_ms": 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_rag_graph():
    """
    Compile and return the LangGraph StateGraph.

    3-node linear topology (was 4-node before performance fix):

      planner -> neaps_retriever -> unified_answer -> END

    The original neaps_answer_node + executive_summary_node (2 sequential LLM calls
    totalling ~24s) have been merged into unified_answer_node (1 LLM call, ~12s).
    The merged prompt includes direct answer + evidence + compliance assessment in
    one pass, with explicit chunk ID injection to prevent citation hallucination.
    """
    from langgraph.graph import StateGraph, END

    builder = StateGraph(RagGraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("neaps_retriever", neaps_retriever_node)
    builder.add_node("unified_answer", unified_answer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "neaps_retriever")
    builder.add_edge("neaps_retriever", "unified_answer")
    builder.add_edge("unified_answer", END)

    graph = builder.compile()
    logger.info("[RagGraph] LangGraph compiled - 3 nodes, linear topology (1 LLM call per query).")
    return graph


# ---------------------------------------------------------------------------
# Cached graph instance
# ---------------------------------------------------------------------------

_rag_graph = None


def get_rag_graph():
    """Get or build the compiled RAG graph (cached singleton)."""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_rag_query(
    query: str,
    company_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a query through the RAG graph.

    Returns:
        {
            "neaps_intelligence": str,
            "executive_summary":  str,
            "sources":            [...],
            "neaps_confidence":   float,
            "query_id":           str,
            "latencies_ms":       {...},
        }
    """
    t0 = time.perf_counter()
    logger.info(
        f"[RagGraphRun] START company={company_id} "
        f"filters={(filters or {})} query='{(query or '')[:120]}'"
    )

    _input = {"query": query, "company_id": company_id, "filters": filters or {}}

    graph = get_rag_graph()

    try:
        result = graph.invoke(_input)
    except Exception as _inv_exc:
        logger.error(f"[RagGraphRun] invoke failed: {_inv_exc}", exc_info=True)
        raise

    latencies = result.get("latencies_ms", {}) or {}
    total_ms = latencies.get("total_ms") or (time.perf_counter() - t0) * 1000
    logger.info(
        f"[RagGraphRun] COMPLETE query_id={result.get('query_id', '')} "
        f"chunks={len(result.get('neaps_chunks') or [])} "
        f"sources={len(result.get('sources') or [])} total_ms={total_ms:.0f}"
    )

    return {
        "neaps_intelligence": result.get("neaps_answer", ""),
        "executive_summary": result.get("executive_summary", ""),
        "sources": result.get("sources", []),
        "neaps_confidence": float(result.get("neaps_confidence", 0.0) or 0.0),
        "query_id": result.get("query_id", ""),
        "latencies_ms": result.get("latencies_ms", {}),
    }
