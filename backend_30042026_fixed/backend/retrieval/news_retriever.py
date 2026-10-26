"""
NewsRetriever — dedicated retriever for the company_descriptions Qdrant collection.
All retrieval events logged to logs/news_search.log via "news.search" logger.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from rank_bm25 import BM25Okapi

from config import settings

# All search/retrieval events go to news_search.log
logger = logging.getLogger("news.search")


import math as _math
import re as _re
from datetime import datetime
from typing import Optional as _Optional

# ---------------------------------------------------------------------------
# _NEWS_HEADING_RE  — compiled once at module level
#
# Detects lines that act as headings or section boundaries in news articles.
# Used to (a) split article text at logical boundaries before chunking, and
# (b) lower the dense-search score_threshold for open/exploratory queries.
#
# Alternatives and example matching lines:
#
#  ① Markdown headings             →  # Breaking News  /  ## Key Takeaways  /  ### Background
#  ② ALL-CAPS section titles       →  BREAKING NEWS  /  KEY POINTS  /  MARKET UPDATE
#  ③ **Bold** markdown titles      →  **Key Highlights**  /  **Market Reaction**
#  ④ Byline  "By <Name>"           →  By Rahul Sharma  /  By Our Correspondent  /  By ANI
#  ⑤ Dateline  "CITY, Mon D —"    →  NEW DELHI, Sept 5 —  /  MUMBAI, Mar 31 —  /  LONDON, Jan 12 (Reuters) —
#  ⑥ Cross-reference labels        →  Also Read:  /  Read More:  /  Related:  /  See Also:
#  ⑦ Numbered sub-sections         →  1. Background  /  Q1:  /  Point 3)  /  A.  Overview
#  ⑧ Quote-attribution lines       →  "..." said Amit Shah  /  "..." said Finance Minister
#  ⑨ Titled sections (colon)       →  Market Reaction:  /  Expert View:  /  What it means:
#  ⑩ Question-style lines (orig.)  →  What is the new policy?  /  How will this affect markets?
# ---------------------------------------------------------------------------
_NEWS_HEADING_RE = _re.compile(
    r"""(?x)^(
        \#{1,6}\s.+                                                     # ① ## Markdown heading
        |[A-Z][A-Z\s\-]{4,}:?\s*$                                       # ② ALL-CAPS TITLE (e.g. BREAKING NEWS)
        |\*\*[A-Z][A-Za-z\s]{3,}\*\*                                    # ③ **Bold Title**
        |By\s+[A-Z][A-Za-z][\w\s\.]{2,}                                 # ④ By Rahul Sharma / By ANI
        |[A-Z][A-Z\s]{2,},\s*[A-Za-z]{3}\.?\s+\d{1,2}\s*[\u2014\-\(]  # ⑤ NEW DELHI, Sept 5 —
        |(Also\s+Read|Read\s+More|Related|See\s+Also|More\s+on):\s*     # ⑥ Also Read: / Related:
        |\d+[\.\)]\s+[A-Z][A-Za-z\s]{2,}                                # ⑦ 1. Background / Point 3)
        |[A-Z]\.\s+[A-Z][A-Za-z\s]{2,}                                  # ⑦ A. Overview / B. Analysis
        |Q\d+[:\.]?\s+[A-Z]                                              # ⑦ Q1: What is... / Q2. How...
        |["\u201c].{0,120}["\u201d]\s+said\s+[A-Z][A-Za-z\s,\.]{3,}    # ⑧ "..." said Finance Minister
        |[A-Z][A-Za-z\s]{3,}:\s*$                                        # ⑨ Market Reaction: / Expert View:
        |\s*(what|how|where|when|who|which|why|is|are|does|did|can|could)\b  # ⑩ question-style (original)
    )""",
    _re.MULTILINE | _re.VERBOSE | _re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Date parsing helper — handles formats found in the news corpus
# ---------------------------------------------------------------------------
_DATE_FORMATS = [
    "%d %b %Y %I:%M %p",    # "31 Mar 2025 06:36 PM"  ← seen in CSV
    "%d %b %Y %H:%M",       # "31 Mar 2025 18:36"
    "%d %b %Y",             # "31 Mar 2025"
    "%Y-%m-%dT%H:%M:%S",    # ISO 8601
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def _parse_article_date(date_str: str) -> _Optional[datetime]:
    """Parse mentioned_date string → datetime. Returns None if unparseable."""
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Last-resort: try first 11 chars (covers "31 Mar 2025 ...")
    try:
        return datetime.strptime(s[:11].strip(), "%d %b %Y")
    except ValueError:
        return None


def _recency_score(date_str: str, decay_days: int = 120) -> float:
    """
    Return a 0→1 recency score using exponential decay.
    Articles from today = 1.0; articles decay_days old ≈ 0.37;
    very old articles approach 0.
    Default decay_days=120 (~4 months) suits quarterly-results news cycles.
    """
    dt = _parse_article_date(date_str)
    if dt is None:
        return 0.5   # neutral score for undated articles
    days_old = max(0, (datetime.now() - dt).days)
    return _math.exp(-days_old / decay_days)


def _bm25_tokenize(text: str) -> list[str]:
    """
    Shared tokenizer used for BOTH index build and query time.
    Steps:
      1. Lowercase
      2. Normalise curly/smart apostrophes → straight apostrophe
      3. Remove apostrophes (so "entrust's" → "entrusts")
      4. Replace every non-alphanumeric char with a space
      5. Split on whitespace, drop empty tokens
    """
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("'", "")
    text = _re.sub(r"[^a-z0-9]+", " ", text)
    return [t for t in text.split() if t]

NEWS_COLLECTION = "company_descriptions_05sep2026"
_instance = None


def get_news_retriever() -> "NewsRetriever":
    global _instance
    if _instance is None:
        _instance = NewsRetriever()
    return _instance


# ---------------------------------------------------------------------------
# RRF fusion helper
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense: List[Dict],
    sparse: List[Dict],
    k: int = 60,
    dense_w:   float = 0.4,
    sparse_w:  float = 0.6,
    # Recency blend weight: 15% recency, 85% relevance.
    # Uses RELATIVE recency within the result set (newest article = 1.0,
    # oldest = 0.0) so articles from the same news cycle are correctly ranked
    # even when all of them are months/years old relative to today.
    recency_w: float = 0.15,
) -> List[Dict]:
    scores:   Dict[str, float] = {}
    meta_map: Dict[str, Dict]  = {}
    dense_keys:  set = set()
    sparse_keys: set = set()

    for rank, hit in enumerate(dense):
        key = hit.get("text", "")[:200]
        scores[key] = scores.get(key, 0.0) + dense_w / (k + rank + 1)
        meta_map[key] = hit
        dense_keys.add(key)

    for rank, hit in enumerate(sparse):
        key = hit.get("text", "")[:200]
        scores[key] = scores.get(key, 0.0) + sparse_w / (k + rank + 1)
        if key not in meta_map:
            meta_map[key] = hit
        sparse_keys.add(key)

    # --- Relative recency within this result set ---
    # Parse all article dates, find newest/oldest, normalise to [0,1].
    # Relative ranking avoids the problem where all articles are "equally old"
    # from today's perspective (e.g. March vs April 2025 results in June 2026).
    all_dates: Dict[str, _Optional[datetime]] = {}
    for key, hit in meta_map.items():
        meta     = hit.get("metadata") or {}
        date_str = meta.get("mentioned_date", "")
        all_dates[key] = _parse_article_date(date_str)

    valid_dates = [d for d in all_dates.values() if d is not None]
    if len(valid_dates) >= 2:
        newest     = max(valid_dates)
        oldest     = min(valid_dates)
        span_secs  = max(1, (newest - oldest).total_seconds())
        def _rel_recency(key: str) -> float:
            dt = all_dates.get(key)
            if dt is None:
                return 0.5   # neutral for undated
            return (dt - oldest).total_seconds() / span_secs  # 0=oldest, 1=newest
    else:
        def _rel_recency(key: str) -> float:  # type: ignore[misc]
            return 0.5   # only one date → no relative ordering possible

    max_rel = max(scores.values()) if scores else 1.0
    if max_rel == 0:
        max_rel = 1.0

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, rel_score in fused:
        item     = dict(meta_map[key])
        rec      = _rel_recency(key)
        blended  = (1 - recency_w) * (rel_score / max_rel) + recency_w * rec
        item["fused_score"] = blended

        in_dense  = key in dense_keys
        in_sparse = key in sparse_keys
        if in_dense and in_sparse:
            item["_provenance"] = "overlapped"
        elif in_dense:
            item["_provenance"] = "dense_only"
        else:
            item["_provenance"] = "sparse_only"
        results.append(item)

    results.sort(key=lambda x: x["fused_score"], reverse=True)
    return results


class NewsRetriever:

    def __init__(self):
        self.client     = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection = NEWS_COLLECTION
        self._bm25:       Optional[BM25Okapi] = None
        self._bm25_texts: List[str]           = []
        self._bm25_meta:  List[Dict]          = []
        self._reranker    = None
        logger.info("NewsRetriever initialised — collection=%s", self.collection)

    # ------------------------------------------------------------------
    # BM25 — lazy build from Qdrant scroll
    # ------------------------------------------------------------------

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        logger.info("Building in-memory BM25 from Qdrant scroll...")
        offset = None
        doc_count = 0
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=500, offset=offset, with_payload=True,
            )
            for p in points:
                payload = p.payload or {}
                text = payload.get("text") or payload.get("chunk_text", "")
                if not text:
                    continue

                symbol  = payload.get("company_symbol") or payload.get("symbol") or ""
                company = payload.get("matched_company_name") or payload.get("csv_company_name") or ""
                # FIX: include mentioned_date in BM25 prefix so queries like
                # "April 2025 gas prices" or "Q4 results March" match by date via BM25.
                # Format: "RELIANCE Reliance Industries 25 Apr 2025 <article text>"
                raw_date = payload.get("mentioned_date") or ""
                dt = _parse_article_date(raw_date)
                date_str = dt.strftime("%-d %b %Y") if dt else ""
                prefix  = " ".join(filter(None, [symbol, company, date_str]))
                bm25_text = f"{prefix} {text}" if prefix else text

                self._bm25_texts.append(bm25_text)
                self._bm25_meta.append({
                    "id":                   str(p.id),
                    "url":                  payload.get("url"),
                    "company_symbol":       payload.get("company_symbol") or payload.get("symbol"),
                    "matched_company_name": payload.get("matched_company_name"),
                    "chunk_index":          payload.get("chunk_index"),
                    "total_chunks":         payload.get("total_chunks"),
                    "mentioned_date":       payload.get("mentioned_date"),
                    "source_type":          "news",
                })
                doc_count += 1
            offset = next_offset
            if not points or offset is None:
                break
        if self._bm25_texts:
            self._bm25 = BM25Okapi([_bm25_tokenize(t) for t in self._bm25_texts])
        logger.info("BM25 ready — %d docs indexed", doc_count)

    def invalidate_bm25(self):
        self._bm25       = None
        self._bm25_texts = []
        self._bm25_meta  = []
        logger.info("BM25 index invalidated — will rebuild on next search")

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        company_symbols: Optional[List[str]] = None,
        raw_query: str = "",
    ) -> List[Dict]:
        qdrant_filter = None
        if company_symbols:
            qdrant_filter = Filter(must=[FieldCondition(
                key="company_symbol",
                match=MatchAny(any=[s.upper() for s in company_symbols]),
            )])
        try:
            # Widen threshold for question-style or heading-matched queries —
            # these are exploratory and benefit from a broader result set.
            score_threshold = 0.30 if (raw_query and _NEWS_HEADING_RE.match(raw_query)) else 0.45

            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            logger.debug(
                "Dense search returned %d hits (filter=%s threshold=%.2f)",
                len(hits), company_symbols, score_threshold,
            )
        except Exception as exc:
            logger.error("Dense search failed: %s", exc, exc_info=True)
            return []

        return [
            {
                "id":           str(h.id),
                "text":         (h.payload or {}).get("text") or (h.payload or {}).get("chunk_text", ""),
                "score":        h.score,
                "vector_score": h.score,
                "bm25_score":   0.0,
                "metadata": {
                    "url":                  (h.payload or {}).get("url"),
                    "company_symbol":       (h.payload or {}).get("company_symbol") or (h.payload or {}).get("symbol"),
                    "matched_company_name": (h.payload or {}).get("matched_company_name"),
                    "csv_company_name":     (h.payload or {}).get("csv_company_name"),
                    "chunk_index":          (h.payload or {}).get("chunk_index"),
                    "total_chunks":         (h.payload or {}).get("total_chunks"),
                    "mentioned_date":       (h.payload or {}).get("mentioned_date"),
                    "source_type":          "news",
                },
            }
            for h in hits
        ]

    # ------------------------------------------------------------------
    # BM25 sparse search
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        company_symbols: Optional[List[str]] = None,
    ) -> List[Dict]:
        self._ensure_bm25()
        if not self._bm25:
            logger.info("BM25 index empty — skipping sparse search")
            return []

        symbols_upper = ([s.upper() for s in company_symbols] if company_symbols else None)
        scores = self._bm25.get_scores(_bm25_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            if score <= 0:
                continue
            meta = self._bm25_meta[idx]
            if symbols_upper:
                stored_sym = (meta.get("company_symbol") or "").upper()
                if stored_sym not in symbols_upper:
                    continue
            results.append({
                "id":           meta.get("id") or f"bm25_{idx}",
                "text":         self._bm25_texts[idx],
                "score":        float(score),
                "vector_score": 0.0,
                "bm25_score":   float(score),
                "metadata":     {**meta, "source_type": "news"},
            })

        logger.debug("BM25 search returned %d hits (filter=%s)", len(results), company_symbols)
        if not results:
            logger.info("BM25 0 results for query=%r — dense-only fusion will apply", query[:60])
        return results

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank_results(self, query: str, fused: List[Dict], top_k: int) -> List[Dict]:
        if self._reranker is None:
            try:
                from core.reranker import get_reranker
                self._reranker = get_reranker()
            except Exception:
                self._reranker = False
        if not self._reranker:
            logger.debug("Reranker not available — returning fused top-%d", top_k)
            return fused[:top_k]
        try:
            logger.debug("Reranking %d fused hits with MS-Marco", len(fused))
            ranked = self._reranker.rerank_with_documents(
                query=query, documents=[dict(h) for h in fused],
                text_key="text", top_k=top_k,
            )
            for h in ranked:
                h["source_collection"] = "news"
            return ranked
        except Exception as exc:
            logger.warning("Reranker failed (%s) — returning fused top-%d", exc, top_k)
            return fused[:top_k]

    # ------------------------------------------------------------------
    # Fetch full descriptions — batched MatchAny (O(1) RTT vs O(N) RTT old)
    # ------------------------------------------------------------------

    def fetch_full_descriptions(self, urls: List[str]) -> Dict[str, str]:
        valid_urls = [u for u in urls if u]
        result: Dict[str, str] = {u: "" for u in valid_urls}
        if not valid_urls:
            return result

        try:
            buckets: Dict[str, list] = {u: [] for u in valid_urls}
            offset = None
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="url", match=MatchAny(any=valid_urls))
                    ]),
                    limit=500,
                    offset=offset,
                    with_payload=True,
                )
                for p in points:
                    url = (p.payload or {}).get("url")
                    if url and url in buckets:
                        buckets[url].append(p)
                offset = next_offset
                if not points or offset is None:
                    break

            for url, chunks in buckets.items():
                if not chunks:
                    logger.warning("No chunks found in Qdrant for url=%r", url)
                    continue
                chunks.sort(key=lambda p: (p.payload or {}).get("chunk_index", 0))
                full_text = " ".join(
                    (p.payload or {}).get("text") or (p.payload or {}).get("chunk_text", "")
                    for p in chunks
                )
                result[url] = full_text
                logger.debug("Stitched %d chunks (%d chars) for url=%r", len(chunks), len(full_text), url)

        except Exception as exc:
            logger.warning("fetch_full_descriptions (batched) failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def search_with_stats(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        company_symbols: Optional[List[str]] = None,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        t_total   = time.perf_counter()
        latencies: Dict[str, float] = {}

        t0    = time.perf_counter()
        dense = self._dense_search(query_vector, top_k=top_k * 2, company_symbols=company_symbols, raw_query=query)
        latencies["dense_ms"] = (time.perf_counter() - t0) * 1000

        t0     = time.perf_counter()
        sparse = self._bm25_search(query, top_k=top_k * 2, company_symbols=company_symbols)
        latencies["sparse_ms"] = (time.perf_counter() - t0) * 1000

        n_dense  = len(dense)
        n_sparse = len(sparse)

        if not dense and not sparse:
            logger.info("Both dense and sparse returned 0 results — query=%r filter=%s", query[:60], company_symbols)

        t0    = time.perf_counter()
        fused = _rrf_fuse(dense, sparse)
        latencies["fusion_ms"] = (time.perf_counter() - t0) * 1000

        n_unique      = len(fused)
        n_dense_only  = sum(1 for h in fused if h.get("_provenance") == "dense_only")
        n_sparse_only = sum(1 for h in fused if h.get("_provenance") == "sparse_only")
        n_overlapped  = sum(1 for h in fused if h.get("_provenance") == "overlapped")

        fused = fused[: top_k * 2]

        t0 = time.perf_counter()
        if use_reranker:
            hits = self._rerank_results(query, fused, top_k=top_k)
        else:
            hits = fused[:top_k]
        latencies["rerank_ms"] = (time.perf_counter() - t0) * 1000
        latencies["total_ms"]  = (time.perf_counter() - t_total) * 1000

        logger.info(
            "search_with_stats query=%r dense=%d sparse=%d unique=%d final=%d total_ms=%.0f",
            query[:60], n_dense, n_sparse, n_unique, len(hits), latencies["total_ms"],
        )

        return {
            "hits":          hits,
            "n_dense":       n_dense,
            "n_sparse":      n_sparse,
            "n_unique":      n_unique,
            "n_final":       len(hits),
            "n_dense_only":  n_dense_only,
            "n_sparse_only": n_sparse_only,
            "n_overlapped":  n_overlapped,
            "latencies_ms":  latencies,
        }

    # ------------------------------------------------------------------
    # Legacy API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        company_symbols: Optional[List[str]] = None,
        use_reranker: bool = True,
    ) -> List[Dict]:
        result = self.search_with_stats(
            query=query, query_vector=query_vector, top_k=top_k,
            company_symbols=company_symbols, use_reranker=use_reranker,
        )
        return result["hits"]