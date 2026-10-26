"""
News search router — POST /news/search
Dedicated endpoint for searching the company_descriptions (news) corpus.
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

def _format_ms(ms_val: float) -> str:
    if not ms_val: return "0s"
    sec = ms_val / 1000.0
    if sec < 60: return f"{sec:.2f}s"
    m = int(sec // 60)
    s = sec % 60
    return f"{m}m {s:.0f}s"

logger       = logging.getLogger(__name__)
query_logger = logging.getLogger("queries")
router       = APIRouter(prefix="/news", tags=["News Search"])

from agents.llm_client import ThreadSafeSemaphore

LLM_CONCURRENCY_LIMIT = getattr(settings, "llm_concurrency_limit", 15)
llm_semaphore         = ThreadSafeSemaphore(LLM_CONCURRENCY_LIMIT)

# --- THE FIX: ADDED CORPORATE SUFFIXES TO STOP WORDS ---
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to",
    "for", "is", "are", "was", "were", "has", "have", "had", "by",
    "with", "from", "that", "this", "it", "its", "be", "been",
    "about", "which", "their", "what", "how", "when", "where", "who",
    "do", "did", "not", "as", "also", "than", "then", "into", "can",
    "will", "would", "should", "could", "may", "might", "shall",
    "news", "company", "limited", "ltd", "pvt", "inc",
    "group", "fund", "capital", "national", "india", "indian",
    "market", "sector", "global", "management", "services",
    "holdings", "ventures", "enterprises", "solutions",
    # Added to prevent aggressive filtering on names like "Greyhound Research"
    "research", "technologies", "tech", "analytics", "consulting", 
    "bank", "corporation", "corp", "trust", "partners"
}


def _extract_keywords(query: str) -> List[str]:
    """
    Extract gating terms from the query.

    Strategy
    --------
    * Consecutive title-cased / ALL-CAPS tokens (not stop words) are collapsed
      into a single phrase — e.g. "Virgin Atlantic" → "virgin atlantic".
      Phrase matching prevents "virgin" + "atlantic" independently matching
      unrelated chunks (e.g. "Atlantic operations" + "Virgin Islands").
    * Only PROPER-NOUN phrases (multi-word) are used as mandatory gates.
      Single common nouns like "partnership" or "deal" are intentionally
      excluded because articles often use synonyms ("agreement", "contract",
      "alliance") — gating on them would wrongly filter out the correct chunks.
    * Single-word proper nouns (e.g. ticker symbols like "TCS") are kept.
    * If no proper-noun phrases are found, fall back to all content words so
      the gate still provides some signal.
    """
    tokens_raw = re.findall(r"[a-zA-Z0-9]+", query)

    proper_phrases: List[str] = []   # multi-word proper nouns → mandatory gate
    single_proper:  List[str] = []   # single proper tokens (tickers, brand names)
    common_words:   List[str] = []   # fallback only

    i = 0
    while i < len(tokens_raw):
        t  = tokens_raw[i]
        tl = t.lower()

        if tl in _STOP_WORDS:
            i += 1
            continue

        # A token is a proper noun candidate when:
        #   • It is ALL-CAPS and short (2-4 chars) — ticker / acronym, e.g. TCS
        #   • OR it starts with a capital AND forms a multi-word run with the
        #     next token (checked lazily below)
        # Plain sentence-initial capitalisation (e.g. "Partnership") is treated
        # as a common word — only the phrase-building step below elevates it.
        is_all_caps_short = t.isupper() and 2 <= len(t) <= 4
        is_title_case     = t[0].isupper() and not t.isupper()  # e.g. "Virgin"

        if is_all_caps_short or is_title_case:
            # Greedily consume adjacent proper tokens into one phrase
            phrase_parts = [t]
            j = i + 1
            while j < len(tokens_raw):
                nt  = tokens_raw[j]
                ntl = nt.lower()
                if ntl in _STOP_WORDS:
                    break
                if nt[0].isupper() or (nt.isupper() and len(nt) >= 2):
                    phrase_parts.append(nt)
                    j += 1
                else:
                    break

            if len(phrase_parts) > 1:
                # Confirmed multi-word proper noun — always include as gate phrase
                proper_phrases.append(" ".join(p.lower() for p in phrase_parts))
                i = j
            elif is_all_caps_short:
                # Ticker/acronym — include as a single gate term
                single_proper.append(tl)
                i += 1
            else:
                # Single title-case token with no adjacent proper noun —
                # ambiguous (could be sentence-initial capitalisation) → treat
                # as a common word so we don't over-filter with synonyms.
                if len(tl) >= 4:
                    common_words.append(tl)
                i += 1
        else:
            if len(tl) >= 4:
                common_words.append(tl)
            i += 1

    # Gate priority: proper phrases first, then single proper tokens.
    # Common nouns are only used when nothing better is available.
    if proper_phrases:
        return proper_phrases + single_proper  # e.g. ["virgin atlantic", "tcs"]
    if single_proper:
        return single_proper
    return common_words  # last resort (pure common-word query)


def _text_passes_gate(text: str, keywords: List[str]) -> bool:
    """
    Return True only if every keyword/phrase appears in the chunk text.
    Phrase entries (e.g. "virgin atlantic") are checked as exact substrings,
    ensuring the words appear together rather than independently scattered.
    """
    if not keywords:
        return True
    text_lower = text.lower()
    return all(kw in text_lower for kw in keywords)


class NewsSearchRequest(BaseModel):
    query:          str                             = Field(..., description="Search query")
    top_k:          Optional[int]                   = Field(default=5, ge=1, le=50)
    company_symbol: Optional[Union[str, List[str]]] = Field(default=None)
    use_reranker:   Optional[bool]                  = Field(default=True)


class NewsChunkSource(BaseModel):
    chunk_id:         str
    chunk_number:     int
    text_preview:     str
    url:              Optional[str]
    article_date:     Optional[str]   = None
    company_symbol:   Optional[str]
    company_name:     Optional[str]
    chunk_index:      Optional[int]
    total_chunks:     Optional[int]
    fused_score:      float
    vector_score:     Optional[float] = None
    bm25_score:       Optional[float] = None
    rerank_score:     Optional[float] = None
    source_type:      str             = "news"
    full_description: Optional[str]   = None


class NewsSearchResponse(BaseModel):
    query_id:           str
    query:              str
    answer:             Optional[str]
    confidence:         Optional[float]
    sources:            List[NewsChunkSource]
    total_results:      int
    retrieval_time_ms:  Union[float, str]
    generation_time_ms: Union[float, str]
    chunks_used:        int
    guardrail_warning:  Optional[str] = None


@router.post("/search", response_model=NewsSearchResponse)
async def news_search(request: NewsSearchRequest):
    trace_id   = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        f"[NewsSearch:{trace_id}] START query='{request.query[:100]}' "
        f"top_k={request.top_k} company_symbol={request.company_symbol}"
    )

    try:
        # 1. Input guardrails
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
        if not guardrail_result.allowed:
            raise HTTPException(
                status_code=guardrail_result.http_status,
                detail=guardrail_result.reason,
            )

        request_query = guardrail_result.sanitized_query
        query_keywords = _extract_keywords(request_query)

        # 2. News answer generator
        answer_gen = None
        if settings.use_answer_generation:
            try:
                from core.news_answer_generator import get_news_answer_generator
                answer_gen = get_news_answer_generator()
            except Exception as e:
                logger.warning(f"[NewsSearch:{trace_id}] Could not load NewsAnswerGenerator: {e}")

        # 3. Embed query
        from core.embeddings import get_embedding_model
        embedder     = get_embedding_model()
        loop         = asyncio.get_running_loop()
        query_vector = await loop.run_in_executor(None, embedder.embed_query, request_query)

        # --- THE FIX: STRICT SINGLE SYMBOL FILTER ---
        raw_sym = request.company_symbol
        company_symbols = None

        if isinstance(raw_sym, list):
            valid_syms = [s.strip().upper() for s in raw_sym if s and s.strip()]
            if valid_syms:
                company_symbols = [valid_syms[0]] # Forces exactly 1 symbol
        elif isinstance(raw_sym, str) and raw_sym.strip():
            company_symbols = [raw_sym.strip().upper()]

        # 5. Hybrid retrieval
        from retrieval.news_retriever import get_news_retriever
        retriever = get_news_retriever()

        # --- THE FIX: DEEP POOL FETCH ---
        # Fetch up to 25 chunks internally so exact matches don't get lost
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
        retrieval_time_ms = (time.time() - start_time) * 1000

        # 6. No-data early return
        if not hits:
            return NewsSearchResponse(
                query_id=trace_id, query=request.query, answer="No matching articles found.",
                confidence=0.0, sources=[], total_results=0, retrieval_time_ms=_format_ms(retrieval_time_ms),
                generation_time_ms="0s", chunks_used=0, guardrail_warning=None
            )

        # --- PRE-LLM KEYWORD & CONFIDENCE GATE ---
        import math
        def _sigmoid(x):
            try: return 1 / (1 + math.exp(-x))
            except OverflowError: return 0.0 if x < 0 else 1.0

        valid_hits = []
        for hit in hits:
            # Filter out low confidence sources (< 10%)
            raw_score = hit.get("rerank_score", hit.get("fused_score", 0))
            if raw_score > 1.0 or raw_score < 0.0:
                normalized_score = _sigmoid(raw_score)
            else:
                normalized_score = float(raw_score)
                
            if normalized_score < 0.1:
                continue
                
            chunk_text = hit.get("text", "")
            if _text_passes_gate(chunk_text, query_keywords):
                valid_hits.append(hit)

        if not valid_hits:
            return NewsSearchResponse(
                query_id=trace_id, query=request.query, answer="No relevant articles found matching the exact keywords.",
                confidence=0.0, sources=[], total_results=0, retrieval_time_ms=_format_ms(retrieval_time_ms),
                generation_time_ms="0s", chunks_used=0, guardrail_warning=None
            )

        # --- URL DEDUPLICATION ---
        seen_urls: Dict[str, Dict] = {}
        no_url_hits: List[Dict] = []
        for h in valid_hits:
            meta = h.get("metadata") or {}
            url = meta.get("url")
            score = float(h.get("fused_score", h.get("score", 0.0)))
            if not url:
                no_url_hits.append(h)
                continue
            if url not in seen_urls or score > float(seen_urls[url].get("fused_score", seen_urls[url].get("score", 0.0))):
                seen_urls[url] = h

        unique_hits = list(seen_urls.values()) + no_url_hits

        # --- FETCH FULL DESCRIPTIONS ---
        unique_urls = [(h.get("metadata") or {}).get("url") for h in unique_hits if (h.get("metadata") or {}).get("url")]
        full_descriptions: Dict[str, str] = {}
        if unique_urls:
            try:
                full_descriptions = await loop.run_in_executor(
                    None, functools.partial(retriever.fetch_full_descriptions, unique_urls)
                )
            except Exception:
                pass

        # --- LEVEL 2 GATE (FULL DESCRIPTIONS) ---
        final_candidate_hits: List[Dict] = []
        for h in unique_hits:
            meta = h.get("metadata") or {}
            url = meta.get("url")
            full_text = full_descriptions.get(url, "") if url else ""
            if not url or not full_text:
                final_candidate_hits.append(h)
                continue
            if _text_passes_gate(full_text, query_keywords):
                final_candidate_hits.append(h)

        # --- SLICE TO EXACT TOP_K ---
        selected_hits = final_candidate_hits[:request.top_k]

        if not selected_hits:
            return NewsSearchResponse(
                query_id=trace_id, query=request.query, answer="No relevant articles found matching the exact keywords.",
                confidence=0.0, sources=[], total_results=0, retrieval_time_ms=_format_ms(retrieval_time_ms),
                generation_time_ms="0s", chunks_used=0, guardrail_warning=None
            )

        # --- BUILD TYPED SOURCES ---
        final_display_sources: List[NewsChunkSource] = []
        for i, s in enumerate(selected_hits, 1):
            meta = s.get("metadata") or {}
            url = meta.get("url")
            article_date = (
                meta.get("article_date")
                or meta.get("date")
                or meta.get("published_date")
                or meta.get("ann_date")
                or meta.get("announcement_date")
            )
            raw_score = s.get("rerank_score", s.get("fused_score", 0.0))
            if raw_score > 1.0 or raw_score < 0.0:
                normalized_score = _sigmoid(raw_score)
            else:
                normalized_score = float(raw_score)

            final_display_sources.append(NewsChunkSource(
                chunk_id=         str(s.get("id", f"chunk-{i}")),
                chunk_number=     i,
                text_preview=     (s.get("text", "")[:300] + "..."),
                url=              url,
                article_date=     article_date,
                company_symbol=   meta.get("company_symbol") or meta.get("symbol"),
                company_name=     meta.get("matched_company_name") or meta.get("csv_company_name"),
                chunk_index=      meta.get("chunk_index"),
                total_chunks=     meta.get("total_chunks"),
                fused_score=      normalized_score,
                vector_score=     s.get("vector_score"),
                bm25_score=       s.get("bm25_score"),
                rerank_score=     s.get("rerank_score"),
                source_type=      "news",
                full_description= full_descriptions.get(url) if url else None,
            ))

        # --- 7. LLM ANSWER GENERATION (EXACT SYNCHRONIZED CHUNKS) ---
        answer             = None
        confidence         = 0.0
        generation_time_ms = 0.0
        chunks_used        = len(selected_hits)

        if answer_gen and selected_hits:
            try:
                async with llm_semaphore:
                    ar = await loop.run_in_executor(
                        None,
                        functools.partial(
                            answer_gen.generate_answer,
                            query=request_query,
                            chunks=selected_hits,
                            max_chunks=len(selected_hits),
                        ),
                    )
                answer             = ar.get("answer")
                confidence         = ar.get("confidence", 0.0)
                generation_time_ms = ar.get("generation_time_ms", 0.0)

                # LLM UX refusal / 0% confidence wipe: Never show news if confidence is 0%
                if confidence <= 0.0 or (answer and "not found" in answer.lower()):
                    final_display_sources = []
                    chunks_used = 0

            except Exception as e:
                logger.error(f"[NewsSearch:{trace_id}] Answer generation failed: {e}")

        return NewsSearchResponse(
            query_id=           trace_id,
            query=              request.query,
            answer=             answer,
            confidence=         confidence,
            sources=            final_display_sources,
            total_results=      len(final_display_sources),
            retrieval_time_ms=  _format_ms(retrieval_time_ms),
            generation_time_ms= _format_ms(generation_time_ms),
            chunks_used=        chunks_used,
            guardrail_warning=  None
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"News search failed: {str(exc)}",
        )