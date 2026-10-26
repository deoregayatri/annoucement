"""
query_classifier.py
===================
Lightweight intent classifier that runs BEFORE the Gardle (RAG graph) is triggered.

Categories:
  - small_talk    : conversational greetings / sign-offs ("Hi", "Hello", "Thanks")
  - basic_info    : simple factual requests not in documents ("time", "weather", "who are you")
  - complex_query : everything else → goes to Gardle

Rules:
  1. If the query contains ANY financial / document keyword → always complex_query
     (handles edge cases like "Hey, analyze this document" or "Hello, what is TCS revenue?")
  2. Check length: queries > 40 chars with no small-talk opener → complex_query
  3. Match against small_talk patterns
  4. Match against basic_info patterns
  5. Default → complex_query

No LLM calls. No I/O. Decision is O(n) where n = len(query).
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

logger = logging.getLogger(__name__)

QueryCategory = Literal["small_talk", "basic_info", "complex_query"]

# ---------------------------------------------------------------------------
# Domain / financial keyword overrides
# If ANY of these appear in the query, it is ALWAYS routed to Gardle
# regardless of greeting words.
# ---------------------------------------------------------------------------
_FINANCIAL_KEYWORDS = re.compile(
    r"\b("
    r"revenue|profit|loss|earnings|eps|dividend|acquisition|merger|announcement"
    r"|filing|disclosure|quarterly|annual|report|results|balance\s+sheet"
    r"|cash\s+flow|nse|bse|sebi|ipo|fpo|rights\s+issue|buyback|delisting"
    r"|shareholding|promoter|analyst|guidance|outlook|target|order\s+book"
    r"|margin|ebitda|pat|debt|leverage|investments|capex|expansion"
    r"|subsidiary|board\s+meeting|agm|egm|compliance|regulatory|audit"
    r"|neaps|document|analyze|analysis|search|find|show|tell.*about"
    r"|what.*happen|when|why|how\s+much|details|summarize|compare"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Small-talk patterns
# Must match the FULL (stripped) query or be the only meaningful content.
# ---------------------------------------------------------------------------
_SMALL_TALK_EXACT = {
    "hi", "hello", "hey", "hiya", "helo", "heya",
    "bye", "goodbye", "good bye", "see you", "cya",
    "thanks", "thank you", "thank u", "ty",
    "ok", "okay", "k", "sure", "alright", "cool", "nice",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how r u", "how are you doing", "how are u",
    "what's up", "whats up", "sup",
    "help", "?", "...",
}

_SMALL_TALK_PATTERNS = re.compile(
    r"^("
    r"hi+\s*[!.]*|h[ae]+y+\s*[!.]*|hello+\s*[!.]*|hola\s*[!.]*"
    r"|good\s+(morning|afternoon|evening|night)\s*[!.]*"
    r"|how\s+are\s+you(\s+doing|\s+today)?\s*[?!.]*"
    r"|what'?s\s*up\s*[?!.]*"
    r"|(bye|goodbye|cya|see\s+you)\s*[!.]*"
    r"|thank\s*(you|u|s)\s*[!.]*"
    r"|(ok|okay|sure|alright|cool|nice)\s*[!.]*"
    r"|you're\s+welcome\s*[!.]*"
    r"|no\s+problem\s*[!.]*"
    r"|please\s*[!.]*"
    r")$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Basic-info patterns  (time, date, weather, bot identity)
# ---------------------------------------------------------------------------
_BASIC_INFO_PATTERNS = re.compile(
    r"^("
    r"what\s+(is\s+)?(the\s+)?(current\s+)?(time|date|day|month|year)\s*[?!.]*"
    r"|what\s+time\s+is\s+it\s*[?!.]*"
    r"|what'?s\s+the\s+(time|date)\s*[?!.]*"
    r"|tell\s+me\s+the\s+(time|date)\s*[?!.]*"
    r"|current\s+time\s*[?!.]*"
    r"|today'?s?\s+date\s*[?!.]*"
    r"|what\s+is\s+(the\s+)?weather\s*[?!.]*"
    r"|how'?s?\s+(the\s+)?weather\s*[?!.]*"
    r"|weather\s+(today|now|forecast)\s*[?!.]*"
    r"|who\s+are\s+you\s*[?!.]*"
    r"|what\s+are\s+you\s*[?!.]*"
    r"|what\s+can\s+you\s+do\s*[?!.]*"
    r"|what\s+do\s+you\s+do\s*[?!.]*"
    r"|are\s+you\s+(a\s+)?(bot|ai|assistant|robot)\s*[?!.]*"
    r"|tell\s+me\s+about\s+yourself\s*[?!.]*"
    r")\s*$",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    category: QueryCategory
    confidence: float   # 1.0 = rule-based certainty
    direct_answer: str  # pre-built response for small_talk / basic_info (empty for complex_query)


def _build_time_answer() -> str:
    """Return the current server time as a human-readable string."""
    now = datetime.now()
    return (
        f"The current server time is **{now.strftime('%I:%M %p')}** "
        f"on **{now.strftime('%A, %B %d, %Y')}**."
    )


def classify_query(query: str) -> ClassificationResult:
    """
    Classify a user query and return its category + a pre-built answer
    if applicable.

    Args:
        query: raw user input (pre-sanitised by guardrails)

    Returns:
        ClassificationResult with category, confidence, and direct_answer.
    """
    stripped = (query or "").strip()
    lower    = stripped.lower()

    # -- Guard: financial / document keywords always win  --------------------
    if _FINANCIAL_KEYWORDS.search(stripped):
        logger.info(
            f"[QueryClassifier] COMPLEX (financial keyword override): '{stripped[:80]}'"
        )
        return ClassificationResult(
            category="complex_query",
            confidence=1.0,
            direct_answer="",
        )

    # -- Exact-match small talk  ---------------------------------------------
    if lower in _SMALL_TALK_EXACT:
        logger.info(
            f"[QueryClassifier] SMALL_TALK (exact match): '{stripped}'"
        )
        return ClassificationResult(
            category="small_talk",
            confidence=1.0,
            direct_answer=(
                "Hello! I'm the **Announcement Intelligence** assistant. "
                "I analyse corporate announcements and regulatory filings. "
                "Select a company from the top bar and ask me about their disclosures."
            ),
        )

    # -- Regex small talk  ---------------------------------------------------
    if _SMALL_TALK_PATTERNS.match(lower):
        logger.info(
            f"[QueryClassifier] SMALL_TALK (pattern match): '{stripped}'"
        )
        return ClassificationResult(
            category="small_talk",
            confidence=1.0,
            direct_answer=(
                "Hello! I'm the **Announcement Intelligence** assistant. "
                "I specialise in analysing corporate announcements and exchange filings. "
                "How can I help you with a company's disclosures today?"
            ),
        )

    # -- Regex basic info  ---------------------------------------------------
    if _BASIC_INFO_PATTERNS.match(lower):
        # Time/date queries get a real answer; identity queries get a canned response.
        if re.search(r"\b(time|date|day|month|year)\b", lower):
            answer = _build_time_answer()
        elif re.search(r"\bweather\b", lower):
            answer = (
                "I don't have access to live weather data. "
                "Please check a weather service such as weather.com for current conditions."
            )
        else:
            # "who are you", "what can you do", etc.
            answer = (
                "I'm the **Announcement Intelligence** assistant — a RAG-powered system "
                "built to analyse NSE/BSE corporate announcements and NEAPS exchange filings. "
                "Select a company from the top bar and ask me about their financial disclosures, "
                "board decisions, or regulatory filings."
            )
        logger.info(
            f"[QueryClassifier] BASIC_INFO (pattern match): '{stripped}'"
        )
        return ClassificationResult(
            category="basic_info",
            confidence=1.0,
            direct_answer=answer,
        )

    # -- Guard: long queries are almost certainly complex (if not small talk or basic info) --
    if len(stripped) > 40:
        logger.info(
            f"[QueryClassifier] COMPLEX (length > 40): '{stripped[:80]}'"
        )
        return ClassificationResult(
            category="complex_query",
            confidence=0.9,
            direct_answer="",
        )

    # -- Default: send to Gardle  -------------------------------------------
    logger.info(
        f"[QueryClassifier] COMPLEX (default): '{stripped[:80]}'"
    )
    return ClassificationResult(
        category="complex_query",
        confidence=0.8,
        direct_answer="",
    )
