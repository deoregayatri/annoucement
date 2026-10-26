"""
Answer generation module using Ollama (qwen3:30b-a3b by default).
Generates coherent answers from retrieved chunks with confidence scoring.
"""

import logging
import math
import re as _re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_core.prompts import PromptTemplate

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain abbreviation map  (add new entries here as needed)
# ---------------------------------------------------------------------------
# Format:  ABBREVIATION (upper) -> full form(s)
# The expander appends the full form in parentheses so the LLM sees both.
# Keys are matched case-insensitively on word boundaries.
# ---------------------------------------------------------------------------
ABBREVIATION_MAP: Dict[str, str] = {
    # ── Vehicle / loan product abbreviations ─────────────────────────────
    "CV":   "commercial vehicle",
    "CVS":  "commercial vehicles",
    "PL":   "personal loan",
    "HL":   "home loan",
    "GL":   "gold loan",
    "LAP":  "loan against property",
    "MFI":  "microfinance",
    "SME":  "small and medium enterprise",
    "MSME": "micro small and medium enterprise",
    "TW":   "two wheeler",
    "PW":   "passenger vehicle",
    "CE":   "construction equipment",
    "HCV":  "heavy commercial vehicle",
    "LCV":  "light commercial vehicle",
    "SCV":  "small commercial vehicle",
    # ── Financial / P&L abbreviations ────────────────────────────────────
    "OPEX":    "operational expenditure / operating expenses",
    "CAPEX":   "capital expenditure",
    "EBITDA":  "earnings before interest taxes depreciation and amortisation",
    "EBIT":    "earnings before interest and taxes",
    "PAT":     "profit after tax",
    "PBT":     "profit before tax",
    "NII":     "net interest income",
    "NIM":     "net interest margin",
    "ROA":     "return on assets",
    "ROE":     "return on equity",
    "ROCE":    "return on capital employed",
    "GNP":     "gross non-performing assets",
    "GNPA":    "gross non-performing assets",
    "NNP":     "net non-performing assets",
    "NNPA":    "net non-performing assets",
    "AUM":     "assets under management",
    "AUM":     "assets under management",
    "GLP":     "gross loan portfolio",
    "DPD":     "days past due",
    "PCR":     "provision coverage ratio",
    "CAR":     "capital adequacy ratio",
    "CRAR":    "capital to risk-weighted assets ratio",
    "LCR":     "liquidity coverage ratio",
    "NSFR":    "net stable funding ratio",
    "CET1":    "common equity tier 1",
    "T1":      "tier 1 capital",
    "T2":      "tier 2 capital",
    "RWA":     "risk-weighted assets",
    "NPA":     "non-performing assets",
    "SMA":     "special mention account",
    "OTR":     "one-time restructuring",
    "ECL":     "expected credit loss",
    "LGD":     "loss given default",
    "PD":      "probability of default",
    "EAD":     "exposure at default",
    "YOY":     "year on year",
    "QOQ":     "quarter on quarter",
    "MOM":     "month on month",
    "FY":      "financial year",
    "H1":      "first half",
    "H2":      "second half",
    "Q1":      "first quarter",
    "Q2":      "second quarter",
    "Q3":      "third quarter",
    "Q4":      "fourth quarter",
    # ── Capital market / corporate abbreviations ──────────────────────────
    "IPO":  "initial public offering",
    "FPO":  "further public offering",
    "OFS":  "offer for sale",
    "QIP":  "qualified institutional placement",
    "NCD":  "non-convertible debenture",
    "CP":   "commercial paper",
    "MTN":  "medium term note",
    "AGM":  "annual general meeting",
    "EGM":  "extraordinary general meeting",
    "BOD":  "board of directors",
    "MD":   "managing director",
    "CFO":  "chief financial officer",
    "CEO":  "chief executive officer",
    "CTO":  "chief technology officer",
    "COO":  "chief operating officer",
    "ESG":  "environmental social governance",
    "CSR":  "corporate social responsibility",
    "MOU":  "memorandum of understanding",
    "LOI":  "letter of intent",
}

# Pre-compile a regex for whole-word matching of all abbreviation keys
_ABBREV_PATTERN = _re.compile(
    r'\b(' + '|'.join(_re.escape(k) for k in sorted(ABBREVIATION_MAP, key=len, reverse=True)) + r')\b',
    _re.IGNORECASE
)


class AnswerGenerator:
    """
    Generate coherent answers from retrieved document chunks using LLM reasoning.
    Includes confidence scoring and source attribution.
    """

    ANSWER_GENERATION_PROMPT = """You are a financial document intelligence assistant.
Use ONLY the provided documents.
Answer clearly and concisely.
Cite document name and chunk number for every fact.
Do not invent data.

IMPORTANT - Abbreviation Matching:
Treat abbreviations and their full forms as equivalent when searching for information.
For example: 'CV' = 'CVs' = 'Commercial Vehicle' = 'Commercial Vehicles'.
'PL' = 'Personal Loan'. 'GL' = 'Gold Loan'. Match case-insensitively.
If the query is an abbreviation, look for both the abbreviation AND its expanded form in the context.
Only respond with "Not found in provided documents" if NEITHER the abbreviation NOR its expanded form appears anywhere in the context chunks below.

Required citation format for every factual statement:
(Document: <file_name>, Chunk: <number>, Page: <page>, Announcement Date: <ann_created_date>)

Question:
{query}

Retrieved context:
{context}

Answer:"""

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
        self.min_confidence  = min_confidence or settings.min_answer_confidence
        self.model_name      = model_name or settings.ollama_model_answer
        self.temperature     = settings.ollama_temperature_answer
        logger.info(f"Initialized answer generator with model: {self.model_name}")

        # Create prompt templates
        self.answer_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=self.ANSWER_GENERATION_PROMPT
        )

        self.confidence_prompt = PromptTemplate(
            input_variables=["context", "query", "answer"],
            template=self.CONFIDENCE_SCORING_PROMPT
        )

    # -----------------------------------------------------------------------
    # Abbreviation expansion
    # -----------------------------------------------------------------------
    @staticmethod
    def _expand_abbreviations(query: str) -> Tuple[str, bool]:
        """
        Expand known domain abbreviations in *query* so the LLM receives both
        the short form and the full form.

        Examples
        --------
        'CV loan growth'   -> 'CV (commercial vehicle) loan growth'
        'OPEX for FY24'    -> 'OPEX (operational expenditure / operating expenses) for FY (financial year) 24'
        'total NPA ratio'  -> 'total NPA (non-performing assets) ratio'

        Returns
        -------
        (expanded_query, was_expanded)
        """
        expanded_parts: Dict[str, str] = {}  # abbrev -> replacement already made

        def _replace(m: _re.Match) -> str:
            token = m.group(0)         # original casing from query
            key   = token.upper()      # look up in map
            full  = ABBREVIATION_MAP.get(key)
            if full and token not in expanded_parts:
                expanded_parts[key] = full
                return f"{token} ({full})"
            return token               # already expanded or unknown

        expanded = _ABBREV_PATTERN.sub(_replace, query)
        was_expanded = bool(expanded_parts)
        if was_expanded:
            logger.info(
                f"[AnswerGenerator] Abbreviation expansion: "
                + ", ".join(f"{k}->{v}" for k, v in expanded_parts.items())
            )
        return expanded, was_expanded

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
            text = chunk.get("text", "")

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

            # --- Abbreviation expansion ---
            # Expand known domain abbreviations so the LLM sees both short form
            # and full form (e.g. 'OPEX' -> 'OPEX (operational expenditure)').
            # The PROMPT TEMPLATE is unchanged; only the query string is enriched.
            llm_query, was_expanded = self._expand_abbreviations(query or "")
            if was_expanded:
                logger.info(f"[AnswerGenerator] Expanded query for LLM: '{llm_query[:200]}'")

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

            # Generate answer — pass EXPANDED query to LLM
            answer = self._generate_answer_with_llm(llm_query, context)

            # Extract sources
            sources = self._extract_sources(top_chunks)

            # Calculate confidence with refusal detection
            confidence, reasoning, score_breakdown = self._calculate_confidence(
                query=query,
                answer=answer,
                chunks=top_chunks
            )

            # ADAPTIVE RETRY: If confidence is low and we have more chunks available, retry with 20 chunks
            if confidence < 0.5 and len(chunks) >= 20 and max_chunks < 20:
                logger.info(f"Low confidence ({confidence:.2f}), retrying with 20 chunks")
                return self.generate_answer(query, chunks, max_chunks=20)

            generation_time_ms = (datetime.now() - start_time).total_seconds() * 1000 / 60000 

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
        Generate answer using llm_client (shared httpx pool, 180s timeout, 3 retries).
        Called from a thread executor so call_llm_sync is correct here.
        """
        from agents.llm_client import call_llm_sync

        prompt_text = self.answer_prompt.format(query=query, context=context)

        raw = call_llm_sync(
            prompt=prompt_text,
            model=self.model_name,
            temperature=self.temperature,
        )

        if raw.startswith("[LLM Error:"):
            logger.error(f"[AnswerGenerator] LLM permanently failed: {raw}")
            return "An error occurred while generating the answer."

        return raw.strip()


    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source documents from chunks for AnswerSource schema.

        Merges consecutive chunks from the same document (same filename + page)
        so the source card shows complete text instead of a 350-char fragment.
        This fixes the "text cut off mid-sentence" issue in announcement source cards
        without requiring re-ingestion of documents.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of dictionaries matching AnswerSource schema
        """
        sources    = []
        seen_ids   = set()

        # Group chunks by document key (filename + page) to merge adjacent ones
        # Key: (filename, page) → merged source dict
        merged_map: Dict[str, Dict] = {}
        merge_order = []   # preserve rank order for display

        for idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("id", "unknown")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            metadata = dict(chunk.get("metadata", {}))
            filename = metadata.get("filename", "")
            page     = (
                metadata.get("page")
                or metadata.get("page_number")
                or metadata.get("page_no")
                or metadata.get("source_page")
                or ""
            )

            # Inject download URL
            if filename:
                metadata["download_url"] = f"/neaps/download/{filename}"

            metadata["chunk_id_raw"] = str(chunk_id)
            metadata["chunk_number"] = idx

            # Merge key — same file + same page → merge text together
            merge_key = f"{filename}||{page}"

            if merge_key in merged_map:
                # Append text to existing source (avoid duplicate overlap text)
                existing_text = merged_map[merge_key]["text"]
                new_text      = chunk.get("text", "")
                # Only append if not already present (handles chunk_overlap duplicates)
                if new_text and new_text not in existing_text:
                    merged_map[merge_key]["text"] = existing_text + " " + new_text
                # Keep highest score
                existing_score = merged_map[merge_key].get("score", 0.0)
                new_score      = chunk.get("score", 0.0)
                if new_score > existing_score:
                    merged_map[merge_key]["score"] = new_score
            else:
                # First chunk for this document/page
                source = {
                    "chunk_id":         str(idx),
                    "text":             chunk.get("text", ""),
                    "score":            chunk.get("score", 0.0),
                    "metadata":         metadata,
                    "vector_score":     chunk.get("vector_score"),
                    "bm25_score":       chunk.get("bm25_score"),
                    "dictionary_boost": chunk.get("dictionary_boost"),
                }
                merged_map[merge_key] = source
                merge_order.append(merge_key)

        # Return in original rank order
        sources = [merged_map[k] for k in merge_order]
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
            top_fused = max(fused_scores)
            # RRF scores are tiny (0.005–0.02 range); normalise via sigmoid so that
            # a hit present in both dense+sparse legs (~0.015) maps to ~0.65+.
            # Classic fused scores (already 0–1) pass through sigmoid near 0.5+
            # which is still honest. We scale RRF by 200 so the typical range
            # 0.006–0.020 maps to sigmoid(1.2)–sigmoid(4.0) ≈ 0.77–0.98.
            if top_fused < 0.1:  # RRF-style score — needs scaling
                confidence = 1.0 / (1.0 + math.exp(-top_fused * 200))
                logger.info(
                    f"[AnswerGenerator] confidence_source=fused_rrf "
                    f"top_fused={top_fused:.6f} confidence={confidence:.4f}"
                )
            else:  # already a 0–1 cosine-style score — clamp directly
                confidence = max(0.0, min(1.0, top_fused))
                logger.info(
                    f"[AnswerGenerator] confidence_source=fused "
                    f"top_fused={top_fused:.4f} confidence={confidence:.4f}"
                )
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


def get_answer_generator() -> AnswerGenerator:
    """Get or create global answer generator instance."""
    global _answer_generator
    if _answer_generator is None:
        _answer_generator = AnswerGenerator()
        logger.info("Created new answer generator instance")
    return _answer_generator


def invalidate_answer_generator():
    """Invalidate cached answer generator to force reload."""
    global _answer_generator
    _answer_generator = None
    logger.info("Answer generator cache invalidated")