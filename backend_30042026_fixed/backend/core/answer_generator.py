"""
Answer generation module using Ollama (qwen3:30b-a3b by default).
Generates coherent answers from retrieved chunks with confidence scoring.
"""

import logging
import math
import re as _re
import threading
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_core.prompts import PromptTemplate

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Qwen3 / DeepSeek-R1 style  <think>…</think>  stripping
# ---------------------------------------------------------------------------
_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL | _re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> chain-of-thought blocks from LLM output.

    Qwen3 (qwen3:30b-a3b) wraps its internal reasoning in these tags.
    When the model uses extended thinking, ChatOllama can return an AIMessage
    whose .content is the *raw* string including the think block, or sometimes
    an empty string while the thinking block is in additional_kwargs.
    """
    return _THINK_RE.sub("", text).strip()


class AnswerGenerator:
    """
    Generate coherent answers from retrieved document chunks using LLM reasoning.
    Includes confidence scoring and source attribution.
    """

    ANSWER_GENERATION_PROMPT = """You are a financial document intelligence assistant.

STRICT RULES:
1. You MUST summarize the content of ALL provided documents below. Do NOT skip or omit any document.
2. Even if a document does not explicitly contain the exact search keywords, you MUST still summarize its relevant contents.
3. Use ONLY the provided documents. Do NOT invent or assume any data.
4. Cite document name and chunk number for every factual statement.
5. Only if truly NONE of the documents contain any relevant information should you respond exactly: Not found in provided documents.

Required citation format for every factual statement:
(Document: <file_name>, Chunk: <number>, Page: <page>, Announcement Date: <ann_created_date>)

Question:
{query}

Retrieved context:
{context}

Answer (summarize ALL documents above):"""

    CONFIDENCE_SCORING_PROMPT = """Evaluate the quality and confidence of the following answer based on the provided context.

Context chunks:
{context}

Question: {query}

Generated Answer: {answer}

Rate the answer on the following criteria (0-10 scale):
1. **Relevance**: Does the answer directly address the question?
2. **Support**: Is the answer well-supported by the context?
3. **Completeness**: Does it fully answer the question?
4. **Accuracy**: Is the information factually correct based on context?

Provide your assessment in this exact format:
Relevance: [score]/10
Support: [score]/10
Completeness: [score]/10
Accuracy: [score]/10
Overall Confidence: [0.0-1.0]
Reasoning: [brief explanation]

Assessment:"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        min_confidence: Optional[float] = None
    ):
        """
        Initialize answer generator.

        Args:
            model_name: Ollama model name (default from settings.ollama_model_answer)
            base_url: Ollama base URL (default from settings.ollama_base_url)
            min_confidence: Minimum confidence threshold (default from settings.min_answer_confidence)
        """
        self.min_confidence = min_confidence or settings.min_answer_confidence

        # Use Local Ollama
        from langchain_community.chat_models import ChatOllama

        self.model_name = model_name or settings.ollama_model_answer
        self.base_url = base_url or settings.ollama_base_url

        try:
            # num_predict caps output tokens → prevents runaway <think> generation
            # request_timeout ensures the call doesn't block forever
            # think=False disables qwen3's extended thinking mode (biggest latency driver)
            extra_body: Dict[str, Any] = {}
            if "qwen3" in (self.model_name or "").lower():
                extra_body["think"] = False   # disable chain-of-thought for qwen3

            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=settings.ollama_temperature_answer,
                num_ctx=settings.ollama_num_ctx,
                num_predict=getattr(settings, "ollama_num_predict", 1024),
                request_timeout=getattr(settings, "ollama_timeout", 120.0),
                **(({"extra_body": extra_body}) if extra_body else {}),
            )
            logger.info(
                f"Initialized answer generator with Ollama: {self.model_name} "
                f"num_ctx={settings.ollama_num_ctx} "
                f"num_predict={getattr(settings, 'ollama_num_predict', 1024)} "
                f"think={'off' if extra_body.get('think') is False else 'on'}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}")
            raise

        # Create prompt templates
        self.answer_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=self.ANSWER_GENERATION_PROMPT
        )

        self.confidence_prompt = PromptTemplate(
            input_variables=["context", "query", "answer"],
            template=self.CONFIDENCE_SCORING_PROMPT
        )

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: Retrieved document chunks with scores

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("full_text") or chunk.get("text", "")

            # Extract metadata for context
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown Document")
            # Fallback to document_date if announcement_date missing (backward compatibility)
            date = meta.get("announcement_date") or meta.get("document_date", "Unknown Date")
            chunk_number = i
            page = (
                meta.get("page")
                or meta.get("page_number")
                or meta.get("page_no")
                or meta.get("source_page")
            )
            ann_created = (
                meta.get("ann_created_date")
                or meta.get("ann_date")
                or meta.get("announcement_date")
                or meta.get("document_date")
            )
            source = meta.get("source") or meta.get("source_type", "unknown")
            header_parts = [f"Document: {filename}", f"Chunk: {chunk_number}", f"Date: {date}", f"Source: {source}"]
            if page not in (None, ""):
                header_parts.append(f"Page: {page}")
            if ann_created not in (None, ""):
                header_parts.append(f"Announcement Date: {ann_created}")
            context_header = f"[{' | '.join(header_parts)}]"

            context_parts.append(f"{context_header}\nContent:\n{text}\n")

        return "\n---\n".join(context_parts)

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        Generate answer from retrieved chunks with adaptive context window.

        Args:
            query: User query
            chunks: Retrieved document chunks (sorted by relevance)
            max_chunks: Initial maximum chunks to use for context (default: 10)

        Returns:
            Dict with answer, confidence, sources, and metadata
        """
        try:
            start_time = datetime.now()
            logger.info(f"[AnswerGenerator] START query='{(query or '')[:120]}' chunks_in={len(chunks)} max_chunks={max_chunks}")

            # Use top-k chunks
            top_chunks = chunks[:max_chunks]

            if not top_chunks:
                logger.warning("No chunks provided for answer generation")
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "confidence": 0.0,
                    "sources": [],
                    "reasoning": "No context chunks available",
                    "generation_time_ms": 0,
                    "chunks_used": 0,
                    "score_breakdown": {}
                }

            # Format context
            context = self._format_context(top_chunks)

            # Generate answer
            answer = self._generate_answer_with_llm(query, context)

            # Extract sources
            sources = self._extract_sources(top_chunks)

            # Calculate confidence with refusal detection
            confidence, reasoning, score_breakdown = self._calculate_confidence(
                query=query,
                answer=answer,
                chunks=top_chunks
            )

            # ADAPTIVE RETRY: If confidence is low and we have more chunks available, retry with 20 chunks.
            # Disabled by default (ENABLE_ADAPTIVE_RETRY=false in .env) because on CPU-bound models
            # like qwen3:30b, a second LLM call doubles latency (30–90s extra) and can push total
            # response time past the Akamai 5-minute threshold.
            _adaptive_retry_enabled = getattr(settings, "enable_adaptive_retry", False)
            if _adaptive_retry_enabled and confidence < 0.5 and len(chunks) >= 20 and max_chunks < 20:
                logger.info(
                    f"[AnswerGenerator] Adaptive retry triggered: confidence={confidence:.2f} < 0.5, "
                    f"retrying with 20 chunks (ENABLE_ADAPTIVE_RETRY=true)"
                )
                return self.generate_answer(query, chunks, max_chunks=20)
            elif not _adaptive_retry_enabled and confidence < 0.5:
                logger.info(
                    f"[AnswerGenerator] Adaptive retry SKIPPED (disabled): confidence={confidence:.2f} "
                    f"— set ENABLE_ADAPTIVE_RETRY=true to enable"
                )

            generation_time_ms = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(
                f"Answer generated: confidence={confidence:.2f}, "
                f"chunks_used={len(top_chunks)}, time={generation_time_ms:.0f}ms"
            )
            logger.info(f"[AnswerGenerator] answer_preview='{(answer or '')[:220]}'")

            return {
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "reasoning": reasoning,
                "generation_time_ms": generation_time_ms,
                "chunks_used": len(top_chunks),
                "score_breakdown": score_breakdown
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return {
                "answer": "An error occurred while generating the answer.",
                "confidence": 0.0,
                "sources": [],
                "reasoning": f"Error: {str(e)}",
                "generation_time_ms": 0,
                "chunks_used": 0,
                "score_breakdown": {}
            }

    def _generate_answer_with_llm(self, query: str, context: str) -> str:
        """
        Generate answer using LLM.

        Args:
            query: User query
            context: Formatted context from chunks

        Returns:
            Generated answer text (think-tag-stripped)
        """
        try:
            # Format the answer generation prompt
            prompt_text = self.answer_prompt.format(
                query=query,
                context=context
            )

            # Invoke LLM — time this specifically to isolate model latency
            _llm_t0 = datetime.now()
            response = self.llm.invoke(prompt_text)
            _llm_ms = (datetime.now() - _llm_t0).total_seconds() * 1000
            logger.info(f"[AnswerGenerator] LLM invoke done in {_llm_ms:.0f}ms")

            # --- Extract raw text from response object ---
            # ChatOllama / ChatOpenAI return AIMessage with .content.
            # Qwen3 (qwen3:30b-a3b) puts its chain-of-thought in <think>…</think>
            # blocks. When the model uses extended thinking, LangChain sometimes
            # delivers an AIMessage whose .content is '' while the think block
            # sits in additional_kwargs['reasoning_content'] or in str(response).
            if hasattr(response, 'content'):
                raw = response.content or ""
            else:
                raw = str(response)

            # Strip <think>…</think> to get clean answer prose
            answer = _strip_think_tags(raw).strip()

            # Fallback 1: check reasoning_content in additional_kwargs
            # qwen3:30b-a3b puts chain-of-thought in additional_kwargs['reasoning_content']
            # and sometimes the actual answer is embedded there after </think>
            if not answer and hasattr(response, 'additional_kwargs'):
                ak = response.additional_kwargs or {}
                # Check reasoning_content — extract text AFTER </think> if present
                reasoning = ak.get("reasoning_content", "") or ak.get("thinking", "")
                if reasoning:
                    # Sometimes actual answer is after the </think> block
                    after_think = _strip_think_tags(reasoning).strip()
                    if after_think:
                        answer = after_think
                        logger.debug("[AnswerGenerator] Extracted answer from reasoning_content")

            # Fallback 2: check other additional_kwargs fields
            if not answer and hasattr(response, 'additional_kwargs'):
                ak = response.additional_kwargs or {}
                for key in ("content", "text", "answer"):
                    v = ak.get(key, "")
                    if v and isinstance(v, str):
                        cleaned = _strip_think_tags(v).strip()
                        if cleaned:
                            answer = cleaned
                            logger.debug(f"[AnswerGenerator] Used additional_kwargs['{key}'] fallback")
                            break

            # Fallback 3: check response_metadata for any content
            if not answer and hasattr(response, 'response_metadata'):
                rm = response.response_metadata or {}
                msg = rm.get("message", {})
                msg_content = msg.get("content", "") if isinstance(msg, dict) else ""
                if msg_content and isinstance(msg_content, str):
                    cleaned = _strip_think_tags(msg_content).strip()
                    if cleaned:
                        answer = cleaned
                        logger.debug("[AnswerGenerator] Used response_metadata.message.content fallback")

            if not answer:
                logger.warning(
                    f"[AnswerGenerator] LLM returned empty answer after all fallbacks. "
                    f"eval_count={getattr(response, 'response_metadata', {}).get('eval_count', 'unknown')} "
                    f"done_reason={getattr(response, 'response_metadata', {}).get('done_reason', 'unknown')}"
                )
                answer = ""

            return answer

        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return "An error occurred while generating the answer."

    def _build_document_summaries(
        self,
        chunks: List[Dict[str, Any]],
        max_chars: int = 800,
    ) -> Dict[str, str]:
        """
        Group chunks by source filename and build a merged text preview per document.

        All chunks belonging to the same filename are joined (in retrieval order)
        and the combined text is capped at max_chars. This produces a coherent,
        document-level summary instead of a single mid-sentence chunk fragment.

        Args:
            chunks:    Retrieved chunks (already ranked / ordered by relevance).
            max_chars: Maximum characters for the merged preview (default 800,
                       roughly 120 words — enough for a 3-4 sentence paragraph).

        Returns:
            Dict mapping filename -> merged summary string.
        """
        from collections import defaultdict

        doc_texts: Dict[str, List[str]] = defaultdict(list)
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            filename = (
                meta.get("filename")
                or meta.get("file_name")
                or "Unknown Document"
            )
            text = (chunk.get("text") or "").strip()
            if text:
                doc_texts[filename].append(text)

        summaries: Dict[str, str] = {}
        for filename, texts in doc_texts.items():
            combined = " [...] ".join(texts)
            summaries[filename] = (
                combined[:max_chars] + "..." if len(combined) > max_chars else combined
            )
        return summaries

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source documents from chunks for AnswerSource schema.

        Each source entry now carries a *document-level* merged summary (up to
        800 chars) built from all retrieved chunks that belong to the same file,
        instead of a hard 200-char truncation of a single chunk.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of dictionaries matching AnswerSource schema
        """
        # Build per-document merged summaries first so every card gets
        # a coherent, complete preview of the whole source document.
        doc_summaries = self._build_document_summaries(chunks, max_chars=800)

        sources = []
        seen_chunks: set = set()

        for idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("id", "unknown")

            # Avoid duplicate chunks
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)

            # Create source object matching AnswerSource schema
            metadata = dict(chunk.get("metadata", {}))
            filename = metadata.get("filename", "")

            # Inject download URL if filename exists
            if filename:
                metadata["download_url"] = f"/neaps/download/{filename}"

            metadata["chunk_id_raw"] = str(chunk_id)
            metadata["chunk_number"] = idx

            # Use the merged document summary so the card shows a full,
            # readable paragraph instead of a mid-sentence chunk fragment.
            display_text = doc_summaries.get(
                filename or "Unknown Document",
                (chunk.get("text") or ""),
            )

            source = {
                "chunk_id": str(idx),
                "text": display_text,
                "score": chunk.get("score", 0.0),
                "metadata": metadata,
                "vector_score": chunk.get("vector_score"),
                "bm25_score": chunk.get("bm25_score"),
                "dictionary_boost": chunk.get("dictionary_boost"),
            }
            sources.append(source)

        return sources



    def _calculate_confidence(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculate confidence score for generated answer.

        Args:
            query: User query
            answer: Generated answer
            chunks: Chunks used for generation (for score breakdown)

        Returns:
            Tuple of (confidence_score, reasoning, score_breakdown)
        """
        # Calculate score breakdown from chunks
        if chunks:
            avg_vector_score = sum(chunk.get("vector_score", 0.0) for chunk in chunks) / len(chunks)
            avg_bm25_score = sum(chunk.get("bm25_score", 0.0) for chunk in chunks) / len(chunks)
            avg_dictionary_boost = sum(chunk.get("dictionary_boost", 0.0) for chunk in chunks) / len(chunks)
            avg_fused_score = sum(chunk.get("fused_score", chunk.get("score", 0.0)) for chunk in chunks) / len(chunks)
        else:
            avg_vector_score = avg_bm25_score = avg_dictionary_boost = avg_fused_score = 0.0

        score_breakdown = {
            "vector_score": avg_vector_score,
            "bm25_score": avg_bm25_score,
            "dictionary_boost": avg_dictionary_boost,
            "fused_score": avg_fused_score
        }

        refusal_phrases = [
            "not found in provided documents",
            "cannot find information",
            "i cannot find",
            "unable to find",
            "no information found",
            "does not contain information",
            "not mentioned in the provided documents"
        ]
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            return 0.0, "No relevant information found in context", score_breakdown

        rerank_scores = [
            float(chunk.get("rerank_score"))
            for chunk in chunks
            if chunk.get("rerank_score") is not None
        ]
        if rerank_scores:
            top_rerank = max(rerank_scores)
            confidence = 1.0 / (1.0 + math.exp(-top_rerank))
            logger.info(f"[AnswerGenerator] confidence_source=rerank top_rerank={top_rerank:.4f} confidence={confidence:.4f}")
            return confidence, "Derived from top reranker score", score_breakdown

        fused_scores = [float(chunk.get("fused_score", chunk.get("score", 0.0))) for chunk in chunks] if chunks else []
        if fused_scores:
            confidence = max(0.0, min(1.0, max(fused_scores)))
            logger.info(f"[AnswerGenerator] confidence_source=fused top_fused={max(fused_scores):.4f} confidence={confidence:.4f}")
            return confidence, "Derived from top retrieval fusion score", score_breakdown

        logger.info("[AnswerGenerator] confidence_source=none confidence=0.0")
        return 0.0, "No scored chunks available", score_breakdown

    def _parse_confidence_score(self, assessment: str) -> float:
        """Parse confidence score from LLM assessment."""
        try:
            # Look for "Overall Confidence: X.X"
            for line in assessment.split('\n'):
                if 'overall confidence' in line.lower():
                    # Extract number
                    parts = line.split(':')
                    if len(parts) > 1:
                        score_str = parts[1].strip()
                        score = float(score_str)
                        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

            # Fallback: parse individual scores and average
            scores = []
            for line in assessment.split('\n'):
                if '/10' in line:
                    try:
                        score = float(line.split(':')[1].split('/')[0].strip())
                        scores.append(score / 10)  # Convert to 0-1 scale
                    except:
                        continue

            if scores:
                return sum(scores) / len(scores)

            return 0.5  # Default medium confidence

        except Exception as e:
            logger.debug(f"Failed to parse confidence score: {e}")
            return 0.5

    def _extract_reasoning(self, assessment: str) -> str:
        """Extract reasoning from LLM assessment."""
        try:
            for line in assessment.split('\n'):
                if 'reasoning' in line.lower():
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        return parts[1].strip()
            return "Assessment completed"
        except:
            return "No reasoning provided"

    def _heuristic_confidence(self, answer: str) -> float:
        """
        Calculate confidence using simple heuristics when LLM scoring fails.

        Args:
            answer: Generated answer

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence

        # Adjust based on answer characteristics
        if len(answer) > 50:
            confidence += 0.1

        if len(answer) > 150:
            confidence += 0.1

        # Lower confidence if answer indicates uncertainty
        uncertainty_phrases = [
            "i don't have",
            "not enough information",
            "cannot determine",
            "unclear",
            "insufficient"
        ]

        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3

        return max(0.0, min(1.0, confidence))

    def should_return_answer(self, confidence: float) -> bool:
        """
        Determine if answer should be returned based on confidence.

        Args:
            confidence: Confidence score

        Returns:
            True if confidence meets threshold
        """
        return confidence >= self.min_confidence


# Global answer generator instance (lazy loading)
_answer_generator: Optional[AnswerGenerator] = None
# Lock ensures only one thread initialises the singleton even under concurrent
# cold-start requests (double-checked locking pattern).
_answer_generator_lock: threading.Lock = threading.Lock()


def get_answer_generator() -> AnswerGenerator:
    """Get or create global answer generator instance (thread-safe)."""
    global _answer_generator
    # Fast-path: already initialised — no lock needed
    if _answer_generator is not None:
        return _answer_generator
    # Slow-path: first call — acquire lock and check again inside
    with _answer_generator_lock:
        if _answer_generator is None:
            _answer_generator = AnswerGenerator()
            logger.info("[AnswerGenerator] Created new answer generator instance")
    return _answer_generator


def invalidate_answer_generator():
    """Invalidate cached answer generator to force reload."""
    global _answer_generator
    with _answer_generator_lock:
        _answer_generator = None
    logger.info("Answer generator cache invalidated")