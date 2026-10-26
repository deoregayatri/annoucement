"""
Input Guardrail System for NEAPS Announcement Search API.

Provides multi-layer input validation and security checks for all incoming
search queries before they reach the vector store, BM25 retriever, or LLM.

Layers:
    1. SQL Injection Detection
    2. XSS / HTML Injection Detection
    3. Shell / OS Command Injection Detection
    4. Prompt Injection Detection
    5. Data Exfiltration Protection
    6. Content Moderation (hate, violence, illegal content)
    7. Query Intent Classification
    8. Query Sanitization (strips residual unsafe chars from allowed queries)
"""

import re
import logging
import html
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result returned by InputGuardrail.validate()."""
    allowed: bool
    sanitized_query: str
    threat_type: Optional[str] = None   # e.g. "sql_injection", "prompt_injection"
    intent: str = "legitimate"          # "legitimate" | "jailbreak" | "system_probing" | "data_extraction"
    reason: Optional[str] = None        # Human-readable rejection reason
    http_status: int = 200              # 200 if allowed, 400 or 403 otherwise
    warnings: List[str] = field(default_factory=list)  # Non-blocking observations


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

# -- 1. SQL Injection --------------------------------------------------------
_SQL_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bDROP\s+TABLE\b",
        r"\bDROP\s+DATABASE\b",
        r"\bDELETE\s+FROM\b",
        r"\bTRUNCATE\s+TABLE\b",
        r"\bINSERT\s+INTO\b",
        r"\bUPDATE\s+\w+\s+SET\b",
        r"\bSELECT\s+\*\s+FROM\b",
        r"\bUNION\s+(ALL\s+)?SELECT\b",
        r"\bOR\s+['\"]?\w*['\"]?\s*=\s*['\"]?\w*['\"]?",      # OR '1'='1'
        r"\bAND\s+['\"]?\w*['\"]?\s*=\s*['\"]?\w*['\"]?",     # AND '1'='1'
        r";\s*(DROP|DELETE|INSERT|UPDATE|EXEC|EXECUTE|CREATE|ALTER|GRANT|REVOKE)",
        r"\bEXEC(UTE)?\s*\(",
        r"\bxp_cmdshell\b",
        r"\bsp_executesql\b",
        r"--\s*$",                                              # SQL comment at end
        r"/\*.*?\*/",                                           # SQL block comment
        r"\b(CHAR|NCHAR|VARCHAR|NVARCHAR)\s*\(\s*\d+\s*\)",
        r"\bCONVERT\s*\(\s*(INT|VARCHAR|CHAR)",
        r"\bCAST\s*\(",
        r"0x[0-9a-fA-F]{4,}",                                  # Hex literals
        r"\bINFORMATION_SCHEMA\b",
        r"\bSYS\.(TABLES|COLUMNS|OBJECTS)\b",
        r"\bPG_SLEEP\b",
        r"\bBENCHMARK\s*\(",
        r"\bWAITFOR\s+DELAY\b",
        r"'\s*OR\s*'1'\s*=\s*'1",
        r'"\s*OR\s*"1"\s*=\s*"1',
    ]
]

# -- 2. XSS / HTML Injection -------------------------------------------------
_XSS_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in [
        r"<script[^>]*>.*?</script>",
        r"<script[^>]*/>",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"data\s*:\s*text/html",
        r"on\w+\s*=\s*[\"'][^\"']*[\"']",   # onerror="...", onclick="..."
        r"on\w+\s*=\s*\w+\s*\(",             # onerror=alert(
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
        r"<img[^>]+src\s*=\s*[\"']?\s*javascript",
        r"expression\s*\(",                   # CSS expression()
        r"url\s*\(\s*javascript",
        r"&#x?[0-9a-fA-F]+;",               # HTML entities used to bypass
        r"<.*?javascript.*?>",
    ]
]

_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)

# -- 3. Shell / OS Command Injection -----------------------------------------
_SHELL_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r";\s*(rm|del|rmdir|rd|format|mkfs|dd)\s",
        r"\|\s*(bash|sh|cmd|powershell|python|perl|ruby|php)\b",
        r"`[^`]+`",                          # Backtick execution
        r"\$\([^)]+\)",                      # $(command)
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bsystem\s*\(",
        r"\bpopen\s*\(",
        r"\bsubprocess\b",
        r"\bos\.system\b",
        r"\bos\.popen\b",
        r"&&\s*(rm|del|cat|curl|wget|nc|ncat|netcat)",
        r"\bnc\s+-[el]",                     # netcat
        r"\bcurl\s+.*(http|ftp|file)://",
        r"\bwget\s+",
        r"\bchmod\s+[0-7]+",
        r"\bsudo\s+",
        r"\bsu\s+-",
        r">\s*/etc/",
        r">\s*/dev/",
        r"cat\s+/etc/passwd",
        r"cat\s+/etc/shadow",
    ]
]

# -- 4. Prompt Injection -----------------------------------------------------
_PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context)",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?|context|everything)",
        r"override\s+(your\s+)?(system\s+)?(instructions?|prompt|rules?)",
        r"\bdan\s+mode\b",
        r"\bjailbreak\b",
        r"\bdo\s+anything\s+now\b",
        r"\byou\s+are\s+now\s+(a|an)\s+\w+",      # "you are now a hacker"
        r"\bact\s+as\s+(a|an)\s+\w+",              # "act as an AI with no rules"
        r"\bpretend\s+(you\s+are|to\s+be)\s+",
        r"\byour\s+new\s+persona\s+is\b",
        r"\bno\s+restrictions?\b",
        r"\bno\s+(ethical\s+)?(limits?|boundaries|rules?|guidelines?)\b",
        r"\byou\s+must\s+(obey|follow|comply)\b",
        r"\brespond\s+as\s+if\b",
        r"\byour\s+true\s+self\b",
        r"\bsystem\s*:\s*you\s+are\b",             # System: you are...
        r"\[system\]",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\<system\>",
        r"\btoken\s+injection\b",
        r"new\s+instructions?\s*:",
        r"updated\s+instructions?\s*:",
        r"final\s+instructions?\s*:",
        r"\bdevMode\s*=\s*true\b",
    ]
]

# -- 5. Data Exfiltration ----------------------------------------------------
_EXFILTRATION_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\blist\s+all\s+(documents?|files?|records?|data|entries|chunks?)\b",
        r"\bdump\s+(the\s+)?(entire\s+)?(knowledge\s*base|database|db|corpus|index|collection)\b",
        r"\bshow\s+(all|every|entire)\s+(documents?|files?|records?|data|secrets?)\b",
        r"\bexport\s+(all|every|entire|everything|the\s+database)\b",
        r"\bprint\s+all\s+(documents?|records?|data|files?)\b",
        r"\bretrieve\s+all\s+(documents?|records?|data|entries)\b",
        r"\bextract\s+all\s+(documents?|data|records?|entries)\b",
        r"\bget\s+all\s+(documents?|data|records?|files?|chunks?)\b",
        r"\binternal\s+(company\s+)?secrets?\b",
        r"\bconfidential\s+(documents?|data|files?|records?)\b",
        r"\bproprietary\s+(information|data|documents?)\b",
        r"\bcompany\s+(secrets?|confidential)\b",
        r"\bknowledge\s+base\s+(dump|export|extract|full|entire|all|contents?)\b",
        r"\bvector\s+(store|database|db)\s+(dump|export|contents?|all)\b",
        r"\bshow\s+(me\s+)?everything\s+(in|about)\s+(the\s+)?(system|database|index)\b",
        r"\bwhat\s+documents?\s+(do\s+you\s+have|are\s+(in|stored))\b",
        r"\bhow\s+many\s+(total\s+)?documents?\s+(are\s+)?(in\s+)?(your|the)\s+(database|system|index)\b",
    ]
]

# -- 6. Content Moderation ---------------------------------------------------
_HATE_SPEECH_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhate\s+speech\b",
        r"\bracist\s+(slur|attack|content)\b",
        r"\bethnic\s+cleansing\b",
        r"\bgenocide\s+(instructions?|how\s+to)\b",
    ]
]

_VIOLENCE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhow\s+to\s+(make\s+a?\s+)?(bomb|explosive|weapon|poison|bioweapon)\b",
        r"\bhow\s+to\s+kill\s+(someone|people|a\s+person)\b",
        r"\bhow\s+to\s+(stab|shoot|murder|assassinate)\b",
        r"\bstep[s\s]+to\s+(murder|kill|attack)\b",
        r"\bbuild\s+(a\s+)?(bomb|explosive|weapon)\b",
        r"\bsynthesize\s+(nerve\s+agent|poison|fentanyl|meth)\b",
    ]
]

_ILLEGAL_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhow\s+to\s+(hack|crack|break\s+into|penetrate|exploit)\s+(a\s+|an\s+|any\s+)?\w+(\s+\w+)?\b",
        r"\bhow\s+to\s+(bypass)\s+(a\s+|an\s+|any\s+)?(system|server|firewall|security|account|password)\b",
        r"\bhow\s+to\s+(launder|wash)\s+money\b",
        r"\bhow\s+to\s+sell\s+(drugs|narcotics|controlled\s+substances?)\b",
        r"\bhow\s+to\s+buy\s+(illegal\s+)?(weapons?|guns?)\s+illegally\b",
        r"\bchild\s+(pornography|exploitation|abuse\s+material)\b",
        r"\bcsam\b",
        r"\bhow\s+to\s+traffick\b",
        r"\bfraud\s+instructions?\b",
        r"\bphishing\s+(kit|template|email)\b",
    ]
]

# -- 7. System Probing -------------------------------------------------------
_PROBING_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwhat\s+is\s+your\s+(system\s+)?(prompt|instruction|configuration|config)\b",
        r"\bshow\s+(me\s+)?your\s+(system\s+)?(prompt|instructions?|rules?|configuration)\b",
        r"\brepeat\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?)\b",
        r"\bprint\s+your\s+(system\s+)?(prompt|instructions?|initial\s+message)\b",
        r"\bwhat\s+(model|llm|embedding)\s+(are\s+you|is\s+this)\s+(using|running|based\s+on)\b",
        r"\bwhat\s+is\s+your\s+(architecture|backend|infrastructure)\b",
        r"\btell\s+me\s+(about\s+)?your\s+(system|backend|database|index)\s+(architecture|structure)\b",
        r"\bwhat\s+(vector|embedding|llm)\s+(model|database|store)\s+(do\s+you|are\s+you)\s+use\b",
        r"\bqdrant|chromadb|weaviate|pinecone\s+(configuration|credentials|api\s+key)\b",
        r"\bapi\s+key\b",
        r"\bconnection\s+string\b",
        r"\bpassword\s+for\b",
        r"\bcredentials?\s+(for|of)\b",
    ]
]

# -- 8. Safe char set for final sanitization ----------------------------------
# Allow alphanumerics, spaces, common punctuation used in financial queries
_SAFE_CHARS_RE = re.compile(r"[^\w\s\.\,\?\!\-\'\"\:\;\&\@\#\%\(\)\/\\\+\=\<\>\~\`\|\[\]\{\}]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


# ---------------------------------------------------------------------------
# Main guardrail class
# ---------------------------------------------------------------------------

class InputGuardrail:
    """
    Multi-layer input validation and security guardrail.

    Usage::

        guardrail = InputGuardrail()
        result = guardrail.validate(user_query)
        if not result.allowed:
            raise HTTPException(status_code=result.http_status, detail=result.reason)
        query = result.sanitized_query
    """

    # Maximum query length allowed (characters)
    MAX_QUERY_LENGTH: int = 2000

    # Maximum number of retrieval results that will be served to any single request
    # (Data Exfiltration Protection - enforced as a recommendation, the route enforces actual cap)
    MAX_RESULTS_ALLOWED: int = 20

    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: When True, any marginal / ambiguous intent is rejected.
                         When False (default), only clearly malicious queries are rejected.
        """
        self.strict_mode = strict_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, query: str, user_role: str = "anonymous") -> GuardrailResult:
        """
        Run all guardrail layers on *query* and return a GuardrailResult.

        Args:
            query:      Raw user-supplied query string.
            user_role:  Optional role string ("admin", "user", "anonymous").
                        Future: used to relax or tighten specific checks.

        Returns:
            GuardrailResult with allowed=True and a sanitized query when safe,
            or allowed=False with an HTTP status code and rejection reason.
        """
        # -- Pre-checks --------------------------------------------------
        if not query or not query.strip():
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="empty_query",
                reason="Query cannot be empty.",
                http_status=400,
            )

        query = query.strip()

        # Length guard
        if len(query) > self.MAX_QUERY_LENGTH:
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="query_too_long",
                reason=f"Query exceeds maximum allowed length of {self.MAX_QUERY_LENGTH} characters.",
                http_status=400,
            )

        # -- Layer 1 : SQL Injection --------------------------------------
        hit, matched = self._check_patterns(query, _SQL_PATTERNS)
        if hit:
            logger.warning(f"[Guardrail] SQL injection detected. Pattern: '{matched}' | Query: {query[:80]!r}")
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="sql_injection",
                reason="Query contains SQL injection patterns and cannot be processed.",
                http_status=400,
            )

        # -- Layer 2 : XSS / HTML Injection ------------------------------
        hit, matched = self._check_patterns(query, _XSS_PATTERNS)
        if hit:
            logger.warning(f"[Guardrail] XSS/HTML injection detected. Pattern: '{matched}' | Query: {query[:80]!r}")
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="xss_injection",
                reason="Query contains HTML/script injection patterns and cannot be processed.",
                http_status=400,
            )

        # -- Layer 3 : Shell / OS Command Injection -----------------------
        hit, matched = self._check_patterns(query, _SHELL_PATTERNS)
        if hit:
            logger.warning(f"[Guardrail] Shell injection detected. Pattern: '{matched}' | Query: {query[:80]!r}")
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="shell_injection",
                reason="Query contains shell/OS command patterns and cannot be processed.",
                http_status=400,
            )

        # -- Layer 4 : Prompt Injection -----------------------------------
        hit, matched = self._check_patterns(query, _PROMPT_INJECTION_PATTERNS)
        if hit:
            logger.warning(f"[Guardrail] Prompt injection detected. Pattern: '{matched}' | Query: {query[:80]!r}")
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="prompt_injection",
                reason="Query appears to be a prompt injection attempt and cannot be processed.",
                http_status=400,
            )

        # -- Layer 5 : Data Exfiltration ----------------------------------
        hit, matched = self._check_patterns(query, _EXFILTRATION_PATTERNS)
        if hit:
            logger.warning(f"[Guardrail] Data exfiltration attempt detected. Pattern: '{matched}' | Query: {query[:80]!r}")
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="data_exfiltration",
                intent="data_extraction",
                reason="Query appears to request bulk data extraction, which is not permitted.",
                http_status=403,
            )

        # -- Layer 6 : Content Moderation ---------------------------------
        moderation_result = self._check_content_moderation(query)
        if moderation_result:
            category, matched_text = moderation_result
            logger.warning(f"[Guardrail] Content moderation: {category}. Pattern: '{matched_text}' | Query: {query[:80]!r}")
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type=f"content_moderation_{category}",
                reason=f"Query contains restricted content ({category}) and cannot be processed.",
                http_status=400,
            )

        # -- Layer 7 : Intent Classification ------------------------------
        intent = self._classify_intent(query)
        if intent == "system_probing":
            logger.warning(f"[Guardrail] System probing detected. Query: {query[:80]!r}")
            return GuardrailResult(
                allowed=False,
                sanitized_query="",
                threat_type="system_probing",
                intent="system_probing",
                reason="Query appears to probe internal system configuration, which is not permitted.",
                http_status=403,
            )

        # -- Layer 8 : Query Sanitization ---------------------------------
        sanitized = self._sanitize(query)

        # Log the final decision
        logger.debug(f"[Guardrail] Query allowed. Intent: {intent} | Query: {query[:80]!r}")

        warnings: List[str] = []
        if sanitized != query:
            warnings.append("Query was sanitized before processing.")
            logger.info(f"[Guardrail] Query sanitized: {query[:60]!r}  {sanitized[:60]!r}")

        return GuardrailResult(
            allowed=True,
            sanitized_query=sanitized,
            intent=intent,
            http_status=200,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_patterns(text: str, patterns: List[re.Pattern]) -> Tuple[bool, Optional[str]]:
        """Return (True, matched_pattern_str) if any pattern matches, else (False, None)."""
        for pattern in patterns:
            m = pattern.search(text)
            if m:
                return True, m.group(0)
        return False, None

    @staticmethod
    def _check_content_moderation(query: str) -> Optional[Tuple[str, str]]:
        """
        Check query against content moderation categories.

        Returns:
            (category, matched_text) if flagged, else None.
        """
        hit, matched = InputGuardrail._check_patterns(query, _HATE_SPEECH_PATTERNS)
        if hit:
            return "hate_speech", matched

        hit, matched = InputGuardrail._check_patterns(query, _VIOLENCE_PATTERNS)
        if hit:
            return "violence", matched

        hit, matched = InputGuardrail._check_patterns(query, _ILLEGAL_PATTERNS)
        if hit:
            return "illegal_content", matched

        return None

    @staticmethod
    def _classify_intent(query: str) -> str:
        """
        Classify query intent.

        Returns one of:
            "legitimate"       Normal information-seeking query.
            "jailbreak"        Attempts to override AI behaviour (caught earlier but double-check).
            "system_probing"   Tries to extract system/infrastructure details.
            "data_extraction"  Bulk data extraction (caught earlier).
        """
        # System probing signals
        hit, _ = InputGuardrail._check_patterns(query, _PROBING_PATTERNS)
        if hit:
            return "system_probing"

        # Secondary jailbreak signals not caught by prompt injection layer
        jailbreak_intents = [
            re.compile(p, re.IGNORECASE) for p in [
                r"\bunlimited\s+(power|access|capability)\b",
                r"\bno\s+guardrails?\b",
                r"\bunfiltered\s+(response|answer|mode)\b",
                r"\braw\s+mode\b",
                r"\bdeveloper\s+mode\b",
            ]
        ]
        hit, _ = InputGuardrail._check_patterns(query, jailbreak_intents)
        if hit:
            return "jailbreak"

        return "legitimate"

    @staticmethod
    def _sanitize(query: str) -> str:
        """
        Final sanitization pass on queries that passed all security checks.

        Steps:
            1. Unescape HTML entities (allow legitimate & -> &)
            2. Strip raw HTML tags (should have been caught earlier, but belt-and-braces)
            3. Collapse excessive whitespace
            4. Strip leading/trailing whitespace
        """
        # Unescape HTML entities so they display correctly
        sanitized = html.unescape(query)

        # Strip any residual HTML tags
        sanitized = _HTML_TAG_RE.sub("", sanitized)

        # Collapse multiple whitespace characters
        sanitized = _MULTI_SPACE_RE.sub(" ", sanitized)

        return sanitized.strip()


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_guardrail_instance: Optional[InputGuardrail] = None


def get_guardrail(strict_mode: bool = False) -> InputGuardrail:
    """Return the singleton InputGuardrail instance (lazy init)."""
    global _guardrail_instance
    if _guardrail_instance is None:
        _guardrail_instance = InputGuardrail(strict_mode=strict_mode)
        logger.info("[Guardrail] InputGuardrail initialised.")
    return _guardrail_instance
