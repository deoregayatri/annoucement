"""
LangGraph RAG Agent - NEAPS-only, single pipeline.

4-node linear graph:

  planner -> neaps_retriever -> neaps_answer -> executive_summary -> END

No parallel branches. No concurrent writes. Single retrieval pass.
"""

import logging
import re
import time
import uuid
import math
import os as _os

from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

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
    neaps_context: str
    context_chunk_indices: List[int]

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

        # NOTE: neaps_hybrid_search.search() already calls _safe_normalize_query()
        # internally. Do NOT expand the query here  that would cause double-expansion
        # (router → rag_graph → neaps_hybrid_search). Pass the original query directly.
        raw_query = state["query"]

        # FIX: When no company filter is set, still search  include USER_UPLOAD docs
        # (uploaded CVs / reports that have company_symbol = USER_UPLOAD).
        # If a company IS set, also union USER_UPLOAD chunks so uploaded docs are found.
        if company:
            # Search company-specific + uploaded docs
            company_results = neaps_search(
                query=raw_query,
                company_symbol=company,
                top_k=40,
                final_top_k=10,
            )
            upload_results = neaps_search(
                query=raw_query,
                company_symbol="USER_UPLOAD",
                top_k=20,
                final_top_k=5,
            )
            # Merge, deduplicate by id, sort by score
            seen_ids = set()
            merged = []
            for r in (company_results + upload_results):
                rid = r.get("id")
                if rid and rid not in seen_ids:
                    seen_ids.add(rid)
                    merged.append(r)
            merged.sort(key=lambda x: x.get("score", x.get("fused_score", 0.0)), reverse=True)
            reranked = merged[:10]
        else:
            # No company filter  search across all documents (no filter = global)
            logger.debug("[NeapsRetriever] No company context  searching all documents including USER_UPLOAD")
            reranked = neaps_search(
                query=raw_query,
                company_symbol=None,
                top_k=40,
                final_top_k=10,
            )

        # Compute confidence from top result
        top_rerank = 0.0
        if reranked:
            top = reranked[0]
            raw_rerank = top.get("rerank_score")
            dense_score = top.get("dense_score")
            fused_score = top.get("score", top.get("fused_score", 0.0))

            if raw_rerank is not None:
                # Use cross-encoder rerank score if available
                # The rerank score might already be in percentage format
                # Normalize it to 0-1 range first, then apply sigmoid if needed
                raw_rerank_float = float(raw_rerank)
                # If the score is > 1, it's likely already a percentage, so normalize it
                if raw_rerank_float > 1.0:
                    raw_rerank_float = raw_rerank_float / 100.0
                # Apply sigmoid to convert to probability (0-1 range)
                top_rerank = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-raw_rerank_float))))
            elif dense_score is not None:
                # Qdrant dense cosine similarity - already in [0, 1] range
                top_rerank = max(0.0, min(1.0, float(dense_score)))
            elif fused_score > 0.0:
                # Fallback: use fused score directly if it's positive
                # Normalize by the maximum score in the result set
                fused_scores = [
                    float(c.get("score", c.get("fused_score", 0.0)))
                    for c in reranked
                    if c.get("score") or c.get("fused_score")
                ]
                if fused_scores:
                    max_fused = max(fused_scores)
                    top_rerank = fused_score / max_fused if max_fused > 0 else 0.0
            else:
                # If all scores are 0, check if we have any valid chunks at all
                # Even with 0 scores, if we retrieved chunks, we should have some baseline confidence
                if len(reranked) > 0:
                    # Use a small baseline confidence if we retrieved chunks but scores are 0
                    # Keep as decimal (0-1) - will be converted to percentage in run_rag_query
                    top_rerank = 0.1  # 10% baseline confidence for retrieved chunks

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


def _build_context(
    chunks: List[Dict[str, Any]], max_chars: int = 8000
) -> Tuple[str, List[int]]:
    """
    Build LLM context string from chunk list, capped at max_chars.

    Returns:
        (context_str, included_indices) - included_indices are the 0-based
        positions (within `chunks`) that actually made it into the context
        before the max_chars cap was hit. Chunks retrieved but dropped here
        never reached the LLM, so they must not be presented as "sources"
        for the generated answer.
    """
    parts = []
    included_indices: List[int] = []
    total = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
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
        included_indices.append(i)
        total += len(entry)

    return "\n---\n".join(parts), included_indices


# ---------------------------------------------------------------------------
# Grounding helpers - catch the model computing/estimating figures that
# aren't literally stated in the source documents (e.g. turning "12-15%
# growth on Rs. 1500 crores" into an invented "Rs. 1800-2250 crores" range).
# ---------------------------------------------------------------------------

_FIN_NUMBER_RE = re.compile(
    r"(?:rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(crores?|lakhs?|%|percent)",
    re.IGNORECASE,
)


def _extract_financial_numbers(text: str) -> set:
    """Extract normalized (digits, unit) financial figures from text."""
    found = set()
    for m in _FIN_NUMBER_RE.finditer(text or ""):
        digits = m.group(1).replace(",", "")
        unit_raw = m.group(2).lower()
        if unit_raw.startswith("crore"):
            unit = "crore"
        elif unit_raw.startswith("lakh"):
            unit = "lakh"
        else:
            unit = "%"
        found.add((digits, unit))
    return found


def _find_ungrounded_numbers(answer_text: str, context_text: str) -> List[str]:
    """
    Return financial figures present in `answer_text` but absent from
    `context_text` - i.e. numbers the model likely computed/estimated
    rather than read directly from the source documents.
    """
    answer_nums = _extract_financial_numbers(answer_text)
    context_nums = _extract_financial_numbers(context_text)
    missing = answer_nums - context_nums
    return [f"{digits} {unit}" for digits, unit in missing]


# ---------------------------------------------------------------------------
# Node 3: NEAPS Answer (LLM)
# ---------------------------------------------------------------------------


def neaps_answer_node(state: RagGraphState) -> RagGraphState:
    """Generate answer from NEAPS chunks via Ollama LLM."""
    t0 = time.perf_counter()

    neaps_chunks = state.get("neaps_chunks") or []
    if not neaps_chunks:
        return {
            "neaps_answer": "Not found in provided documents.",
            "latencies_ms": {"neaps_answer_ms": (time.perf_counter() - t0) * 1000},
        }

    context, context_chunk_indices = _build_context(neaps_chunks)
    prompt = f"""You are a financial document intelligence assistant. Use ONLY the provided documents. Do not invent, calculate, or estimate any figure. Cite sources: (Document: <file_name>, Chunk: <number>, Page: <page>, Date: <ann_created_date>).

STRICT RULES FOR NUMBERS:
- Only state a number (amount, percentage, date, count) if it appears verbatim in the documents below.
- Never compute a derived figure from a percentage or partial value (e.g. do NOT turn "12-15% growth on Rs. 1500 crores" into a computed range like "Rs. 1800-2250 crores"). If the documents don't state the resulting figure directly, say the figure is not explicitly stated and quote the underlying data instead.
- If information is missing, respond exactly: Not found in provided documents.

QUERY: {state["query"]}

OFFICIAL NEAPS FILINGS:

{context}

ANSWER:"""

    llm_t0 = time.perf_counter()
    from config import settings

    answer = _call_llm(prompt, temperature=settings.ollama_temperature_neaps)
    llm_ms = (time.perf_counter() - llm_t0) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[NeapsAnswer] Generated {len(answer)} chars llm={llm_ms:.0f}ms total={total_ms:.0f}ms "
        f"context_chunks={len(context_chunk_indices)}/{len(neaps_chunks)}"
    )

    return {
        "neaps_answer": answer,
        "neaps_context": context,
        "context_chunk_indices": context_chunk_indices,
        "latencies_ms": {"neaps_answer_ms": total_ms, "neaps_llm_ms": llm_ms},
    }


# ---------------------------------------------------------------------------
# Node 4: Executive Summary
# ---------------------------------------------------------------------------

EXEC_SUMMARY_PROMPT = """You are a financial document intelligence assistant. Use ONLY the SOURCE DOCUMENTS below as ground truth. Do not invent, calculate, or estimate any figure that is not explicitly written in the SOURCE DOCUMENTS - not even a plausible-looking derived number (e.g. never turn a growth percentage plus a base figure into a computed total/range).

The DRAFT ANSWER was generated by another assistant from the same SOURCE DOCUMENTS - restructure and polish it, but re-verify every number and claim in it against the SOURCE DOCUMENTS. If the DRAFT ANSWER contains a figure that does not appear in the SOURCE DOCUMENTS, drop that figure and state plainly that it is not explicitly stated, quoting the underlying data instead.

Cite sources using: (Document: <file_name>, Chunk: <number>, Page: <page>, Date: <ann_created_date>).

If the query is unrelated to company disclosures, respond: "This system analyzes corporate announcements and disclosure patterns. The query is outside the scope of this surveillance system."

Use this structure:

ANSWER TO THE QUERY
<Answer text - only facts verifiable in the SOURCE DOCUMENTS>

EVIDENCE FROM ANNOUNCEMENTS
"<Exact supporting quote>"
(Document: <filename>, Chunk: <number>, Page: <page>, Date: <ann_created_date>)

FINAL COMPLIANCE ASSESSMENT
<Short factual conclusion, no computed/estimated figures>

Do not output internal IDs or UUID values. If information is missing, respond exactly: Not found in provided documents.

Query: "{query}"

SOURCE DOCUMENTS:

{context}

DRAFT ANSWER:

{neaps_answer}
"""


def executive_summary_node(state: RagGraphState) -> RagGraphState:
    """Synthesise final executive answer and build sources list."""
    t0 = time.perf_counter()

    neaps_answer = state.get("neaps_answer", "")
    query = state.get("query", "")
    context = state.get("neaps_context", "")

    from config import settings

    prompt = EXEC_SUMMARY_PROMPT.format(
        query=query, context=context, neaps_answer=neaps_answer
    )

    llm_t0 = time.perf_counter()
    summary = _call_llm(prompt, temperature=settings.ollama_temperature_neaps)
    llm_ms = (time.perf_counter() - llm_t0) * 1000

    # Grounding check: catch figures the model computed/estimated rather than
    # read directly from the source documents (e.g. a growth % turned into a
    # fabricated absolute revenue range). Retry once with an explicit list of
    # the offending figures before falling back.
    ungrounded = _find_ungrounded_numbers(summary, context)
    if ungrounded:
        logger.warning(
            f"[ExecSummary] Ungrounded figures detected: {ungrounded} - retrying with stricter constraint"
        )
        strict_prompt = (
            prompt
            + f"\n\nIMPORTANT: Your previous draft included these figures, which do NOT appear verbatim in the SOURCE DOCUMENTS: {', '.join(ungrounded)}. Remove or replace them - state only what is explicitly written in the SOURCE DOCUMENTS."
        )
        retry_t0 = time.perf_counter()
        summary = _call_llm(strict_prompt, temperature=settings.ollama_temperature_neaps)
        llm_ms += (time.perf_counter() - retry_t0) * 1000
        still_ungrounded = _find_ungrounded_numbers(summary, context)
        if still_ungrounded:
            logger.warning(
                f"[ExecSummary] Still ungrounded after retry: {still_ungrounded}"
            )

    # Fallback to neaps_answer if summary is empty or failed
    if (not summary) or ("could not generate" in summary.lower()):
        summary = (
            neaps_answer
            or "No coherent summary could be generated from the retrieved announcements."
        )

    exec_ms = (time.perf_counter() - t0) * 1000
    neaps_chunks = state.get("neaps_chunks", [])
    context_chunk_indices = set(state.get("context_chunk_indices") or [])

    logger.info(
        f"[ExecSummary] Generated {len(summary)} chars chunks={len(neaps_chunks)} "
        f"llm={llm_ms:.0f}ms total={exec_ms:.0f}ms"
    )

    # Build sources ONLY from chunks that actually reached the LLM's context
    # (neaps_chunks may include more candidates than fit under the context
    # char cap - those never-seen chunks must not be shown as "evidence").
    sources = []
    for idx, chunk in enumerate(neaps_chunks):
        if context_chunk_indices and idx not in context_chunk_indices:
            continue

        chunk_number = idx + 1
        # Was this specific chunk actually referenced in the final answer's
        # citations? Lets the UI distinguish primary evidence from other
        # context the model considered but didn't directly cite.
        cited = bool(
            re.search(rf"Chunk:\s*{chunk_number}\b", summary)
            or re.search(rf"Chunk:\s*{chunk_number}\b", neaps_answer)
        )

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
                "cited": cited,
                "metadata": {
                    "filename": display_filename,
                    "file_name": display_filename,
                    "chunk_id": meta.get("chunk_id")
                    or chunk.get("id")
                    or chunk.get("chunk_id"),
                    "chunk_number": chunk_number,
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
        "executive_summary": summary,
        "sources": sources,
        "latencies_ms": {"exec_summary_ms": exec_ms, "exec_llm_ms": llm_ms},
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_rag_graph():
    """
    Compile and return the LangGraph StateGraph.

    Linear topology - no parallel branches, no concurrent writes:

      planner -> neaps_retriever -> neaps_answer -> executive_summary -> END
    """
    from langgraph.graph import StateGraph, END

    builder = StateGraph(RagGraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("neaps_retriever", neaps_retriever_node)
    builder.add_node("neaps_answer", neaps_answer_node)
    builder.add_node("executive_summary", executive_summary_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "neaps_retriever")
    builder.add_edge("neaps_retriever", "neaps_answer")
    builder.add_edge("neaps_answer", "executive_summary")
    builder.add_edge("executive_summary", END)

    graph = builder.compile()
    logger.info("[RagGraph] LangGraph compiled - 4 nodes, linear topology.")
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
 