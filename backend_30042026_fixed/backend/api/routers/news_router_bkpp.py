"""
News search router — POST /news/search
Dedicated endpoint for searching the company_descriptions (news) corpus.

FIXES applied:
  1. NewsSearchRequest: company_symbol accepts str or List[str].
  2. No-data response includes n_dense / n_sparse context for debugging.
  3. NewsChunkSource includes full_description (stitched article text).
  4. URL dedup preserves sources with no URL (no_url_sources list).
  5. KEYWORD RELEVANCE GATE — two-level fix:

     LEVEL 1 — CHUNK-LEVEL GATE (before dedup):
       Each raw source dict from _extract_sources is mapped back to its original
       hit to scan the FULL chunk text (~1500 chars), not just the 300-char preview. 
       Before URL-dedup, we filter chunks that DO NOT contain any query keyword. 
       This ensures the surviving chunk per URL actually mentions the query.

     LEVEL 2 — MULTI-KEYWORD AND LOGIC for multi-word proper nouns:
       For queries like "Hinduja Group", both "hinduja" AND "group" would
       individually pass because "group" appears in almost every financial
       article. Fix: if the query is a multi-word proper noun (2+ non-stop keywords),
       we require ALL keywords to appear in the text (AND logic).

     LEVEL 3 — full_description gate (after dedup):
       After URL dedup and fetch_full_descriptions, surviving sources are
       re-checked against the full stitched article text using the same logic.
"""

import asyncio
import functools
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from config import settings

logger       = logging.getLogger(__name__)
query_logger = logging.getLogger("queries")
router       = APIRouter(prefix="/news", tags=["News Search"])

LLM_CONCURRENCY_LIMIT = getattr(settings, "llm_concurrency_limit", 15)
llm_semaphore         = asyncio.Semaphore(LLM_CONCURRENCY_LIMIT)

# Words that must not be used as standalone relevance keywords.
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to",
    "for", "is", "are", "was", "were", "has", "have", "had", "by",
    "with", "from", "that", "this", "it", "its", "be", "been",
    "about", "which", "their", "what", "how", "when", "where", "who",
    "do", "did", "not", "as", "also", "than", "then", "into", "can",
    "will", "would", "should", "could", "may", "might", "shall",
    "news", "company", "limited", "ltd", "pvt", "inc",
    # Generic financial terms that appear in virtually every article —
    # must not be used as distinguishing keywords.
    "group", "fund", "capital", "national", "india", "indian",
    "market", "sector", "global", "management", "services",
    "holdings", "ventures", "enterprises", "solutions",
    # Added Corporate Suffixes
    "research", "technologies", "tech", "analytics", "consulting", 
    "bank", "corporation", "corp", "trust", "partners"
}


def _extract_keywords(query: str) -> List[str]:
    """
    Extract meaningful keywords from a query.
    Returns lowercase tokens that are not stop-words and are >= 4 chars.
    Short tokens (3 chars) are kept only if they look like a ticker/acronym
    (all uppercase in original query).
    """
    tokens_raw  = re.findall(r"[a-zA-Z0-9]+", query)
    keywords = []
    for t in tokens_raw:
        tl = t.lower()
        if tl in _STOP_WORDS:
            continue
        # Keep 3-char uppercase tokens (tickers like NSE, BSE, SBI)
        if len(t) == 3 and t.isupper():
            keywords.append(tl)
            continue
        if len(tl) >= 4:
            keywords.append(tl)
    return keywords


def _text_passes_gate(text: str, keywords: List[str]) -> bool:
    """
    Relevance gate logic:
      - 0 keywords  → always pass (gate disabled)
      - 1 keyword   → OR  logic: text must contain that keyword
      - 2+ keywords → AND logic: text must contain ALL keywords
        (prevents generic words like "group" matching unrelated articles)
    Case-insensitive substring match.
    """
    if not keywords:
        return True
    text_lower = text.lower()
    if len(keywords) == 1:
        return keywords[0] in text_lower
    # AND: all keywords must be present
    return all(kw in text_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class NewsSearchRequest(BaseModel):
    query:          str                             = Field(..., description="Search query")
    top_k:          Optional[int]                   = Field(default=5, ge=1, le=50)
    company_symbol: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "Single symbol 'POLICYBZR' or list ['POLICYBZR','LICI']. "
            "Maps directly to the chip value sent from the UI."
        ),
    )
    use_reranker:   Optional[bool]                  = Field(default=True)


class NewsChunkSource(BaseModel):
    """One retrieved news source — deduplicated per URL for frontend display."""
    chunk_id:         str
    chunk_number:     int
    text_preview:     str            = Field(description="First 300 chars of top-scoring chunk")
    url:              Optional[str]  = Field(description="Original NSE/news article URL")
    company_symbol:   Optional[str]
    company_name:     Optional[str]
    chunk_index:      Optional[int]  = Field(description="0-based chunk index")
    total_chunks:     Optional[int]  = Field(description="Total chunks for this article")
    fused_score:      float
    vector_score:     Optional[float] = None
    bm25_score:       Optional[float] = None
    rerank_score:     Optional[float] = None
    source_type:      str             = "news"
    full_description: Optional[str]   = Field(
        default=None,
        description=(
            "Complete article text stitched from all Qdrant chunks for this URL "
            "in chunk_index order. Use this to display the full article in the UI."
        ),
    )


class NewsSearchResponse(BaseModel):
    query_id:           str
    query:              str
    answer:             Optional[str]   = Field(description="LLM-generated synthesised answer")
    confidence:         Optional[float] = Field(description="Answer confidence 0-1")
    sources:            List[NewsChunkSource]
    total_results:      int
    retrieval_time_ms:  float
    generation_time_ms: float
    chunks_used:        int
    guardrail_warning:  Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/search", response_model=NewsSearchResponse)
async def news_search(request: NewsSearchRequest):
    trace_id   = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        f"[NewsSearch:{trace_id}] START query='{request.query[:100]}' "
        f"top_k={request.top_k} company_symbol={request.company_symbol}"
    )

    try:
        # ------------------------------------------------------------------
        # 1. Input guardrails
        # ------------------------------------------------------------------
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
            f"[NewsSearch:{trace_id}] Guardrail allowed={guardrail_result.allowed} "
            f"warnings={len(guardrail_result.warnings or [])}"
        )
        if not guardrail_result.allowed:
            query_logger.warning(
                f"NewsSearch blocked: trace={trace_id} threat={guardrail_result.threat_type}"
            )
            raise HTTPException(
                status_code=guardrail_result.http_status,
                detail=guardrail_result.reason,
            )

        request_query = guardrail_result.sanitized_query
        if request_query != raw_query:
            logger.info(f"[NewsSearch:{trace_id}] Query sanitized by guardrails")

        # Extract keywords once — used in both chunk-level and URL-level gates
        query_keywords = _extract_keywords(request_query)
        logger.info(
            f"[NewsSearch:{trace_id}] Gate keywords={query_keywords} "
            f"logic={'AND' if len(query_keywords) >= 2 else 'OR' if query_keywords else 'OFF'}"
        )

        # ------------------------------------------------------------------
        # 2. News answer generator
        # ------------------------------------------------------------------
        answer_gen = None
        if settings.use_answer_generation:
            try:
                from core.news_answer_generator import get_news_answer_generator
                answer_gen = get_news_answer_generator()
            except Exception as e:
                logger.warning(f"[NewsSearch:{trace_id}] Could not load NewsAnswerGenerator: {e}")

        # ------------------------------------------------------------------
        # 3. Embed query
        # ------------------------------------------------------------------
        from core.embeddings import get_embedding_model
        embedder     = get_embedding_model()
        loop         = asyncio.get_running_loop()
        query_vector = await loop.run_in_executor(
            None, embedder.embed_query, request_query
        )

        # ------------------------------------------------------------------
        # 4. Company filter (Strictly Single Symbol)
        # ------------------------------------------------------------------
        raw_sym = request.company_symbol
        company_symbols = None

        if isinstance(raw_sym, list):
            # Enforce single-variable filter: isolate and keep only the first valid symbol.
            valid_syms = [s.strip().upper() for s in raw_sym if s and s.strip()]
            if valid_syms:
                company_symbols = [valid_syms[0]]
        elif isinstance(raw_sym, str) and raw_sym.strip():
            company_symbols = [raw_sym.strip().upper()]

        logger.info(
            f"[NewsSearch:{trace_id}] company_filter="
            f"{company_symbols if company_symbols else 'NONE (search all)'}"
        )

        # ------------------------------------------------------------------
        # 5. Hybrid retrieval
        # ------------------------------------------------------------------
        from retrieval.news_retriever import get_news_retriever
        retriever = get_news_retriever()

        logger.info(f"[NewsSearch:{trace_id}] Hybrid mode components ready")
        logger.info(f"[NewsSearch:{trace_id}] Layer1 retrieve begin query='{request_query[:80]}'")

        # Fetch a large candidate pool internally to protect exact matches from dense noise
        fetch_k = max(request.top_k * 4, 25)

        retrieve_result: Dict[str, Any] = await loop.run_in_executor(
            None,
            functools.partial(
                retriever.search_with_stats,
                query=request_query,
                query_vector=query_vector,
                top_k=fetch_k,
                company_symbols=company_symbols,
                use_reranker=request.use_reranker,
            ),
        )

        hits          = retrieve_result.get("hits", [])
        n_dense       = retrieve_result.get("n_dense", 0)
        n_sparse      = retrieve_result.get("n_sparse", 0)
        n_unique      = retrieve_result.get("n_unique", 0)
        n_final       = retrieve_result.get("n_final", len(hits))
        n_dense_only  = retrieve_result.get("n_dense_only", 0)
        n_sparse_only = retrieve_result.get("n_sparse_only", 0)
        n_overlapped  = retrieve_result.get("n_overlapped", 0)
        latencies     = retrieve_result.get("latencies_ms", {})
        retrieval_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[NewsSearch:{trace_id}] Layer1 retrieve done results={len(hits)} "
            f"latencies={latencies}"
        )

        print("=" * 18 + " HYBRID SEARCH PIPELINE " + "=" * 18)
        print(
            f"Query: '{request_query[:60]}' | "
            f"Filter: {', '.join(company_symbols) if company_symbols else 'ALL'}"
        )
        print(
            f"Layer 1 (Retrieval): Fetched {n_dense} Dense + "
            f"{n_sparse} Sparse = {n_unique} Unique Chunks"
        )
        print(f"Layer 2 (RRF Fusion & Rerank): Trimmed down to Final Top {n_final}")
        print(
            f"LLM Context Provenance: {n_dense_only} Dense-only | "
            f"{n_sparse_only} Sparse-only | {n_overlapped} Overlapped"
        )
        print("=" * 60)

        # ------------------------------------------------------------------
        # 6. No-data early return
        # ------------------------------------------------------------------
        if not hits:
            company_label = ", ".join(company_symbols) if company_symbols else None

            if n_dense == 0 and n_sparse == 0:
                reason = "No matching articles found in the news database"
            elif n_dense > 0 and n_sparse == 0:
                reason = "Articles found but relevance score was too low to return results"
            else:
                reason = "No relevant articles found after ranking"

            detail = (
                f"{reason} for query '{request_query}'"
                + (f" filtered by: {company_label}" if company_label else "")
                + ". Try a different query or verify that data has been ingested for this company."
            )
            logger.info(
                f"[NewsSearch:{trace_id}] No hits — dense={n_dense} sparse={n_sparse} — {detail}"
            )
            return NewsSearchResponse(
                query_id=           trace_id,
                query=              request.query,
                answer=             detail,
                confidence=         0.0,
                sources=            [],
                total_results=      0,
                retrieval_time_ms=  retrieval_time_ms,
                generation_time_ms= 0.0,
                chunks_used=        0,
                guardrail_warning=  (
                    ", ".join(guardrail_result.warnings)
                    if guardrail_result.warnings else None
                ),
            )

        # ------------------------------------------------------------------
        # 6.5 PRE-LLM KEYWORD GATE
        # ------------------------------------------------------------------
        valid_hits = []
        for hit in hits:
            chunk_text = hit.get("text", "")
            if _text_passes_gate(chunk_text, query_keywords):
                valid_hits.append(hit)
            else:
                logger.info(
                    f"[NewsSearch:{trace_id}] PRE-LLM GATE dropped chunk "
                    f"{hit.get('id')} — missing keywords {query_keywords}"
                )
        
        # Trim the valid hits back down to the actual top_k requested
        hits = valid_hits[:request.top_k]

        if not hits:
            logger.info(f"[NewsSearch:{trace_id}] PRE-LLM GATE dropped all chunks. Returning no data.")
            detail = f"No relevant articles found matching the exact keywords for query '{request_query}'."
            return NewsSearchResponse(
                query_id=           trace_id,
                query=              request.query,
                answer=             detail,
                confidence=         0.0,
                sources=            [],
                total_results=      0,
                retrieval_time_ms=  retrieval_time_ms,
                generation_time_ms= 0.0,
                chunks_used=        0,
                guardrail_warning=  (", ".join(guardrail_result.warnings) if guardrail_result.warnings else None),
            )

        # ------------------------------------------------------------------
        # 7. LLM answer generation
        # ------------------------------------------------------------------
        answer             = None
        confidence         = 0.0
        generation_time_ms = 0.0
        chunks_used        = 0
        raw_sources        = None

        if answer_gen and hits:
            try:
                async with llm_semaphore:
                    ar = await loop.run_in_executor(
                        None,
                        functools.partial(
                            answer_gen.generate_answer,
                            query=request_query,
                            chunks=hits,
                            max_chunks=min(request.top_k, 8),
                        ),
                    )
                answer             = ar.get("answer")
                confidence         = ar.get("confidence", 0.0)
                raw_sources        = ar.get("sources")
                generation_time_ms = ar.get("generation_time_ms", 0.0)
                chunks_used        = ar.get("chunks_used", len(hits))
                logger.info(
                    f"[NewsSearch:{trace_id}] Layer1 answer confidence={confidence:.2f}"
                )

                # --- UX FIX: Clear sources if LLM explicitly rejected them ---
                if confidence == 0.0 and answer and "not found" in answer.lower():
                    logger.info(f"[NewsSearch:{trace_id}] LLM rejected context. Wiping irrelevant sources.")
                    hits = []          
                    raw_sources = []   
                # -------------------------------------------------------------

            except Exception as e:
                logger.error(
                    f"[NewsSearch:{trace_id}] Layer1 answer generation failed: {e}"
                )

        # ------------------------------------------------------------------
        # 8. COMPLETE log
        # ------------------------------------------------------------------
        query_logger.info(
            f"NewsSearch trace={trace_id}: query='{request.query[:50]}' "
            f"results={len(hits)} time={retrieval_time_ms:.0f}ms"
        )
        logger.info(
            f"[NewsSearch:{trace_id}] COMPLETE layer=layer_1_precise results={len(hits)} "
            f"answer={'yes' if bool(answer) else 'no'} "
            f"retrieval_ms={retrieval_time_ms:.0f} "
            f"confidence={confidence:.3f} sources={len(raw_sources or [])}"
        )
        if answer:
            logger.info(f"[NewsSearch:{trace_id}] Hybrid answer_preview='{answer[:220]}'")

        # ------------------------------------------------------------------
        # 9. Build source dicts from hits (prefer raw_sources from answer_gen)
        # ------------------------------------------------------------------
        source_dicts = raw_sources or []

        if not source_dicts:
            for i, hit in enumerate(hits, 1):
                meta = hit.get("metadata") or {}
                source_dicts.append({
                    "chunk_id":       str(hit.get("id", f"chunk-{i}")),
                    "chunk_number":   i,
                    "text_preview":   (hit.get("text", "")[:300] + "..."),
                    "url":            meta.get("url"),
                    "company_symbol": meta.get("company_symbol") or meta.get("symbol"),
                    "company_name":   (
                        meta.get("matched_company_name")
                        or meta.get("csv_company_name")
                    ),
                    "chunk_index":    meta.get("chunk_index"),
                    "total_chunks":   meta.get("total_chunks"),
                    "fused_score":    float(hit.get("fused_score", hit.get("score", 0.0))),
                    "vector_score":   hit.get("vector_score"),
                    "bm25_score":     hit.get("bm25_score"),
                    "rerank_score":   hit.get("rerank_score"),
                    "source_type":    "news",
                })

        # ------------------------------------------------------------------
        # 10. LEVEL 1 GATE — chunk-level, before URL dedup
        # ------------------------------------------------------------------
        chunk_gated: List[Dict] = []
        
        # Create a quick lookup map from the raw hits to get the FULL chunk text 
        # (up to ~1500 chars) instead of relying on the 300-char text_preview.
        hit_text_map = {str(h.get("id")): h.get("text", "") for h in hits}

        for s in source_dicts:
            url      = s.get("url")
            chunk_id = s.get("chunk_id")
            
            # Fetch the full chunk text from the map. Fallback to preview just in case.
            full_chunk_text = hit_text_map.get(chunk_id, s.get("text_preview", ""))

            if not url:
                # No URL → keep unconditionally
                chunk_gated.append(s)
                continue

            # Check the FULL chunk text, not just the first 300 characters
            if _text_passes_gate(full_chunk_text, query_keywords):
                chunk_gated.append(s)
            else:
                logger.info(
                    f"[NewsSearch:{trace_id}] CHUNK GATE dropped chunk_id={chunk_id} "
                    f"url={url!r} — keywords {query_keywords} not in full_chunk_text"
                )

        logger.info(
            f"[NewsSearch:{trace_id}] Chunk gate: "
            f"{len(source_dicts)} chunks → {len(chunk_gated)} after gate"
        )

        # ------------------------------------------------------------------
        # 11. URL dedup — keep highest fused_score chunk per URL
        # ------------------------------------------------------------------
        seen_urls:      Dict[str, Dict] = {}
        no_url_sources: List[Dict]      = []

        for s in chunk_gated:
            url = s.get("url")
            if not url:
                no_url_sources.append(s)
                continue
            existing = seen_urls.get(url)
            if existing is None or s.get("fused_score", 0.0) > existing.get("fused_score", 0.0):
                seen_urls[url] = s

        unique_source_dicts = list(seen_urls.values()) + no_url_sources

        # ------------------------------------------------------------------
        # 12. Fetch full descriptions — stitch all Qdrant chunks per URL
        # ------------------------------------------------------------------
        unique_urls = [s["url"] for s in unique_source_dicts if s.get("url")]
        full_descriptions: Dict[str, str] = {}

        if unique_urls:
            try:
                full_descriptions = await loop.run_in_executor(
                    None,
                    functools.partial(retriever.fetch_full_descriptions, unique_urls),
                )
                logger.info(
                    f"[NewsSearch:{trace_id}] Fetched full descriptions for "
                    f"{len(full_descriptions)} URLs"
                )
            except Exception as e:
                logger.warning(
                    f"[NewsSearch:{trace_id}] fetch_full_descriptions failed: {e} — "
                    f"sources will have null full_description"
                )

        # ------------------------------------------------------------------
        # 13. LEVEL 2 GATE — full article text, after fetch_full_descriptions
        # ------------------------------------------------------------------
        final_source_dicts: List[Dict] = []
        for s in unique_source_dicts:
            url       = s.get("url")
            full_text = full_descriptions.get(url, "") if url else ""

            if not url or not full_text:
                # No URL or fetch failed → keep (don't drop due to infra failure)
                final_source_dicts.append(s)
                continue

            if _text_passes_gate(full_text, query_keywords):
                final_source_dicts.append(s)
            else:
                logger.info(
                    f"[NewsSearch:{trace_id}] FULL DESC GATE dropped url={url!r} — "
                    f"keywords {query_keywords} not in full article text"
                )

        logger.info(
            f"[NewsSearch:{trace_id}] Full desc gate: "
            f"{len(unique_source_dicts)} sources → {len(final_source_dicts)} after gate"
        )

        # ------------------------------------------------------------------
        # 14. Build typed sources list with full_description attached
        # ------------------------------------------------------------------
        typed_sources: List[NewsChunkSource] = []
        for i, s in enumerate(final_source_dicts, 1):
            url = s.get("url")
            typed_sources.append(NewsChunkSource(
                chunk_id=         str(s.get("chunk_id", f"chunk-{i}")),
                chunk_number=     i,
                text_preview=     s.get("text_preview", s.get("text", "")[:300] + "..."),
                url=              url,
                company_symbol=   s.get("company_symbol"),
                company_name=     s.get("company_name") or s.get("matched_company_name"),
                chunk_index=      s.get("chunk_index"),
                total_chunks=     s.get("total_chunks"),
                fused_score=      float(s.get("fused_score", s.get("score", 0.0))),
                vector_score=     s.get("vector_score"),
                bm25_score=       s.get("bm25_score"),
                rerank_score=     s.get("rerank_score"),
                source_type=      "news",
                full_description= full_descriptions.get(url) if url else None,
            ))

        return NewsSearchResponse(
            query_id=           trace_id,
            query=              request.query,
            answer=             answer,
            confidence=         confidence,
            sources=            typed_sources,
            total_results=      len(typed_sources),
            retrieval_time_ms=  retrieval_time_ms,
            generation_time_ms= generation_time_ms,
            chunks_used=        chunks_used,
            guardrail_warning=  (
                ", ".join(guardrail_result.warnings)
                if guardrail_result.warnings else None
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        query_logger.error(f"NewsSearch failed for query '{request.query[:50]}': {exc}")
        logger.error(f"[NewsSearch:{trace_id}] Failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"News search failed: {str(exc)}",
        )