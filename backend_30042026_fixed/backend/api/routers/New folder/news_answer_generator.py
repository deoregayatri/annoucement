"""
NewsAnswerGenerator — answer generation tuned for the company_descriptions corpus.

Key differences from the shared AnswerGenerator:
  - Context header uses URL + company_symbol + chunk_index (no filename/page/ann_date)
  - LLM prompt cites sources as (Source: <url>, Company: <symbol>, Chunk: N of M)
  - _extract_sources returns news-native fields for the UI
  - _format_context never emits "Unknown Document" or "Page: Not mentioned"
"""
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate

from config import settings

logger = logging.getLogger(__name__)


# NEWS_ANSWER_PROMPT = """\
# You are a financial news intelligence assistant.
# Use ONLY the provided news articles below.
# Answer clearly and concisely, synthesising across sources where relevant.
# For every factual statement cite the source using this format:
#   (Source: <url>, Company: <company_symbol>, Chunk: <chunk_index> of <total_chunks>)
# Do not invent data. If information is not present respond exactly:
#   Not found in the provided news articles.

# Question:
# {query}

# News articles:
# {context}

# Answer:"""

NEWS_ANSWER_PROMPT = """\
You are a financial news intelligence assistant.

STRICT RULES:
1. You MUST summarize ALL provided news articles below. Do NOT skip or omit any article.
2. Even if an article does not explicitly contain the exact search keywords, you MUST still summarize its contents.
3. For EVERY factual statement, cite the exact source using this format: (Source: <url>) or [Article <number>].
4. ONLY cite articles that appear in the "PROVIDED NEWS ARTICLES" section below. Do NOT cite or refer to any article number that is not in the provided list.
5. Use ONLY the information from the provided articles. Do NOT invent, assume, or extrapolate any data.
6. If none of the articles contain relevant information, respond exactly: Not found in the provided news articles.

Question:
{query}

PROVIDED NEWS ARTICLES:
{context}

Answer (summarize ALL provided articles above, citing each fact):"""

class NewsAnswerGenerator:
    """
    Generate coherent answers from news chunks retrieved from company_descriptions.
    Shares the Ollama backend with the main AnswerGenerator but uses a
    news-appropriate prompt and context formatter.
    """

    def __init__(self):
        from langchain_community.chat_models import ChatOllama

        self.model_name = settings.ollama_model_answer
        self.base_url   = settings.ollama_base_url

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=settings.ollama_temperature_answer,
        )
        self.answer_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=NEWS_ANSWER_PROMPT,
        )
        logger.info(f"[NewsAnswerGenerator] Initialised model={self.model_name}")

    # ------------------------------------------------------------------
    # Context formatting — news-native fields only
    # ------------------------------------------------------------------
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text         = chunk.get("text", "").strip()
            meta         = chunk.get("metadata") or {}
            company_name = (
                meta.get("matched_company_name")
                or meta.get("csv_company_name")
                or meta.get("company_symbol")
                or ""
            )
            url = meta.get("url") or ""
            header_items = [f"Article {i}"]
            if company_name:
                header_items.append(f"Company: {company_name}")
            if url:
                header_items.append(f"Source: {url}")
            label = f"[{' | '.join(header_items)}]"
            parts.append(f"{label}\n{text}")
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
    def _call_llm(self, query: str, context: str) -> str:
        try:
            prompt_text = self.answer_prompt.format(query=query, context=context)
            response    = self.llm.invoke(prompt_text)
            answer      = response.content if hasattr(response, "content") else str(response)
            return answer.strip()
        except Exception as exc:
            logger.error(f"[NewsAnswerGenerator] LLM call failed: {exc}")
            return "An error occurred while generating the answer."

    # ------------------------------------------------------------------
    # Source extraction — news-native shape for UI
    # ------------------------------------------------------------------
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        seen    = set()
        for idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("id", f"chunk-{idx}")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            meta = chunk.get("metadata") or {}
            sources.append({
                "chunk_id":      str(chunk_id),
                "chunk_number":  idx,
                "text_preview":  (chunk.get("text", "")[:300] + "..."),
                "url":           meta.get("url"),
                "company_symbol":meta.get("company_symbol") or meta.get("symbol"),
                "company_name":  meta.get("matched_company_name") or meta.get("csv_company_name"),
                "chunk_index":   meta.get("chunk_index"),
                "total_chunks":  meta.get("total_chunks"),
                "fused_score":   chunk.get("fused_score", chunk.get("score", 0.0)),
                "vector_score":  chunk.get("vector_score"),
                "bm25_score":    chunk.get("bm25_score"),
                "rerank_score":  chunk.get("rerank_score"),
                "source_type":   "news",
            })
        return sources

    # ------------------------------------------------------------------
    # Confidence — identical logic to AnswerGenerator
    # ------------------------------------------------------------------
    def _calculate_confidence(
        self, query: str, answer: str, chunks: List[Dict[str, Any]]
    ) -> Tuple[float, str, Dict[str, float]]:
        score_breakdown: Dict[str, float] = {}
        if chunks:
            score_breakdown = {
                "vector_score": sum(c.get("vector_score", 0.0) for c in chunks) / len(chunks),
                "bm25_score":   sum(c.get("bm25_score",   0.0) for c in chunks) / len(chunks),
                "fused_score":  sum(c.get("fused_score", c.get("score", 0.0)) for c in chunks) / len(chunks),
            }

        refusal_phrases = [
            "not found in the provided news articles",
            "cannot find information",
            "i cannot find",
            "unable to find",
            "no information found",
        ]
        if any(p in answer.lower() for p in refusal_phrases):
            return 0.0, "No relevant information found in news context", score_breakdown

        rerank_scores = [
            float(c["rerank_score"]) for c in chunks if c.get("rerank_score") is not None
        ]
        if rerank_scores:
            top_rerank = max(rerank_scores)
            confidence = 1.0 / (1.0 + math.exp(-top_rerank))
            logger.info(
                f"[NewsAnswerGenerator] confidence_source=rerank "
                f"top_rerank={top_rerank:.4f} confidence={confidence:.4f}"
            )
            return confidence, "Derived from top reranker score", score_breakdown

        fused_scores = [float(c.get("fused_score", c.get("score", 0.0))) for c in chunks]
        if fused_scores:
            confidence = max(0.0, min(1.0, max(fused_scores)))
            logger.info(
                f"[NewsAnswerGenerator] confidence_source=fused "
                f"top_fused={max(fused_scores):.4f} confidence={confidence:.4f}"
            )
            return confidence, "Derived from top retrieval fusion score", score_breakdown

        return 0.0, "No scored chunks available", score_breakdown

    # ------------------------------------------------------------------
    # Public API — drop-in replacement for answer_gen.generate_answer()
    # ------------------------------------------------------------------
    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate a news-style answer.

        Returns dict with:
          answer, confidence, sources, reasoning,
          generation_time_ms, chunks_used, score_breakdown
        """
        start_time = datetime.now()
        logger.info(
            f"[NewsAnswerGenerator] START query='{query[:120]}' "
            f"chunks_in={len(chunks)} max_chunks={max_chunks}"
        )

        top_chunks = chunks[:max_chunks]

        if not top_chunks:
            logger.warning("[NewsAnswerGenerator] No chunks provided")
            return {
                "answer":             "I don't have enough information to answer this question.",
                "confidence":         0.0,
                "sources":            [],
                "reasoning":          "No context chunks available",
                "generation_time_ms": 0,
                "chunks_used":        0,
                "score_breakdown":    {},
            }

        context  = self._format_context(top_chunks)
        answer   = self._call_llm(query, context)
        sources  = self._extract_sources(top_chunks)
        confidence, reasoning, score_breakdown = self._calculate_confidence(
            query=query, answer=answer, chunks=top_chunks
        )

        generation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"Answer generated: confidence={confidence:.2f}, "
            f"chunks_used={len(top_chunks)}, time={generation_time_ms:.0f}ms"
        )
        logger.info(f"[NewsAnswerGenerator] answer_preview='{answer[:220]}'")

        return {
            "answer":             answer,
            "confidence":         confidence,
            "sources":            sources,
            "reasoning":          reasoning,
            "generation_time_ms": generation_time_ms,
            "chunks_used":        len(top_chunks),
            "score_breakdown":    score_breakdown,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_news_answer_gen_instance: Optional[NewsAnswerGenerator] = None


def get_news_answer_generator() -> NewsAnswerGenerator:
    global _news_answer_gen_instance
    if _news_answer_gen_instance is None:
        _news_answer_gen_instance = NewsAnswerGenerator()
    return _news_answer_gen_instance