"""
Input Sanitization & Security Utilities.

Provides strict validation and cleaning against common injection vectors
(SQLi, XSS, XXE) to protect backend queries and the Oracle database.
"""

import re
import html
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class InputSanitizer:
    """
    Centralized utility for securely parsing and cleaning API inputs.
    """

    # Basic strict regex for company symbols (alphanumeric, max 15 chars)
    SYMBOL_PATTERN = re.compile(r"^[A-Z0-9_\-\.]{1,15}$", re.IGNORECASE)

    # Basic regex to catch obvious SQL injection attempts or dangerous script tags
    DANGEROUS_SQL_PATTERN = re.compile(
        r"(?i)\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|EXECUTE|DECLARE|CAST|WAITFOR|DELAY|SYSOBJECTS|SYSCOLUMNS)\b.*?--|"
        r"(;.*?--)|\bOR\b\s+1\s*=\s*1|\bAND\b\s+1\s*=\s*1",
        re.IGNORECASE
    )
    
    DANGEROUS_HTML_PATTERN = re.compile(r"<(script|iframe|object|embed|applet|form|body)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)

    @classmethod
    def sanitize_query(cls, text: str) -> str:
        """
        Cleans a standard search or analytical query payload.
        Strips control characters, normalizes whitespace, escapes HTML entities.
        """
        if not text:
            return ""
        
        # 1. Strip out zero-width chars and unprintable control characters
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 2. Escape dangerous HTML/XML entities (prevents XXE / XSS)
        cleaned = html.escape(cleaned, quote=False)
        
        # 3. Prevent obvious SQL payloads
        # Note: We rely on BIND variables in the DB, but we strip obvious malicious intent to save LLM tokens and log noise
        if cls.DANGEROUS_SQL_PATTERN.search(cleaned):
            logger.warning(f"[Security] Potential SQLi payload detected and neutralized in query.")
            cleaned = cls.DANGEROUS_SQL_PATTERN.sub("[REDACTED]", cleaned)

        if cls.DANGEROUS_HTML_PATTERN.search(cleaned):
            logger.warning(f"[Security] Potential XSS payload detected and neutralized in query.")
            cleaned = cls.DANGEROUS_HTML_PATTERN.sub("[REDACTED]", cleaned)

        # 4. Normalize whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned

    @classmethod
    def validate_symbol(cls, symbol: str) -> bool:
        """
        Validates a company ticker symbol for strict adherence to alphanumeric format.
        """
        if not symbol:
            return False
        return bool(cls.SYMBOL_PATTERN.match(symbol))

    @classmethod
    def clean_symbol(cls, symbol: str) -> Optional[str]:
        """
        Cleans and upper-cases a symbol. If completely invalid, returns None.
        """
        if not symbol:
            return None
        
        sym = symbol.strip().upper()
        # Keep only alphanumeric and standard ticker chars
        sym = re.sub(r"[^A-Z0-9_\-\.]", "", sym)
        if len(sym) > 15 or len(sym) < 1:
            return None
        return sym
