"""
Text chunking utilities - Stage 3 upgrade: token-based chunking.

Adds:
  - chunk_text_tokens()  : tiktoken-based token-aware chunking (new primary API)
  - chunk_documents()    : LangChain Document list chunker with metadata enrichment
  - Section-boundary preservation via Markdown heading detection

Retains backward-compat:
  - chunk_text()                  : original character-based chunker
  - create_chunks_with_metadata() : original metadata wrapper
  - create_chunks_from_pages()    : original page-level chunker
  - validate_chunks()             : original validator
"""

import logging
import re
from typing import List, Dict, Any, Optional

from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 350    # tokens  (target)
DEFAULT_OVERLAP    = 50     # tokens  (~10%)

# Matches section boundaries in financial PDFs:
#   - Markdown headings:              ## Revenue Summary
#   - ALL-CAPS titles:                BALANCE SHEET
#   - **Bold markdown** titles:       **Key Highlights**
#   - SCHEDULE / ANNEXURE labels:     SCHEDULE - I, ANNEXURE A
#   - Table / Figure / Chart labels:  Table 1: Revenue Data
#   - Roman-numeral sections:         II. Risk Factors
#   - Numeric sections:               1. Financial Summary, 2.3 Revenue
#   - Titled sections with colon:     Key Financial Highlights:
_HEADING_RE = re.compile(
    r"""(?x)^(
        \#{1,6}\s.+                                      # ## Markdown heading
        |[A-Z][A-Z\s]{4,}:?\s*                          # ALL-CAPS TITLE
        |\*\*[A-Z][A-Za-z\s]{3,}\*\*                   # **Bold Title**
        |(SCHEDULE|ANNEXURE|APPENDIX|EXHIBIT)\s*[-\u2013]?\s*[A-Z0-9]+  # SCHEDULE - I
        |(Table|Figure|Chart)\s+\d+\s*[:\-\u2013]       # Table 1:
        |[IVXLCDM]{1,6}\.\s+[A-Z][A-Za-z\s]{3,}        # II. Risk Factors
        |\d+\.\s+[A-Z][A-Za-z\s]{3,}                   # 1. Financial Summary
        |\d+\.\d+\s+[A-Z][A-Za-z\s]{2,}                # 2.3 Revenue Breakdown
        |\d+\)\s+[A-Z][A-Za-z0-9\s]{2,}                # 32) FAQ QUESTION
        |[A-Z][A-Za-z\s]{4,}:\s*                        # Key Highlights:
    )$""",
    re.MULTILINE | re.VERBOSE,
)


# ===========================================================================
# NEW: Token-based chunker (LangChain + tiktoken)
# ===========================================================================

def _split_by_headings(text: str) -> List[str]:
    """Split text at heading boundaries keeping each heading with its following body."""
    positions = [m.start() for m in _HEADING_RE.finditer(text)]
    if not positions:
        return [text]

    sections = []
    if positions[0] > 0:
        sections.append(text[: positions[0]].strip())
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        sections.append(text[start:end].strip())
    return [s for s in sections if s]


def chunk_text_tokens(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    preserve_sections: bool = True,
) -> List[str]:
    """
    Split text into token-bounded chunks using tiktoken (cl100k_base).

    Args:
        text: Source text
        chunk_size: Target tokens per chunk (default 500)
        overlap: Overlap tokens between chunks (default 50, ~10%)
        preserve_sections: Avoid splitting mid-section where possible

    Returns:
        List of non-empty, deduplicated chunk strings
    """
    if not text or not text.strip():
        return []

    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        encoding_name="cl100k_base",
    )

    if preserve_sections:
        sections = _split_by_headings(text)
        raw: List[str] = []
        for section in sections:
            raw.extend(splitter.split_text(section))
    else:
        raw = splitter.split_text(text)

    # Deduplicate consecutive identical chunks (artefact of section reassembly)
    chunks, prev = [], None
    for c in raw:
        c = c.strip()
        if c and c != prev:
            chunks.append(c)
            prev = c

    logger.debug(
        f"[TextChunker] Token chunks: {len(chunks)} from {len(text)} chars "
        f"(size={chunk_size}t, overlap={overlap}t, sections={preserve_sections})"
    )
    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    preserve_sections: bool = True,
) -> List[Document]:
    """
    Split a list of LangChain Documents into token-bounded sub-documents.
    Parent metadata is preserved and enriched with chunk_index / total_chunks.

    Args:
        documents: LangChain Document list
        chunk_size: Target tokens per chunk
        overlap: Overlap tokens
        preserve_sections: Section-boundary preservation

    Returns:
        Flat list of chunked LangChain Documents
    """
    result: List[Document] = []
    for doc in documents:
        chunks = chunk_text_tokens(
            doc.page_content,
            chunk_size=chunk_size,
            overlap=overlap,
            preserve_sections=preserve_sections,
        )
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            meta = dict(doc.metadata)
            meta["chunk_index"] = idx
            meta["total_chunks"] = total
            result.append(Document(page_content=chunk, metadata=meta))

    logger.info(
        f"[TextChunker] chunk_documents: {len(documents)} docs  {len(result)} chunks "
        f"(size={chunk_size}t, overlap={overlap}t)"
    )
    return result


def create_token_chunks_from_pages(
    pages: List[Dict[str, Any]],
    filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    preserve_sections: bool = True,
    **additional_metadata: Any,
) -> List[Dict[str, Any]]:
    """Create token-based chunks from page dicts, preserving page_number."""
    all_chunks: List[Dict[str, Any]] = []
    total_idx = 0
    
    # We stitch the pages together with a special separator to run chunk_text_tokens on the whole document,
    # OR we can process page by page. Since preserve_sections relies on finding headings, processing
    # page by page is easier for preserving page_number, though it forces a chunk boundary at page ends.
    # To avoid breaking sections across pages, we'll process page by page.
    for page in pages:
        page_num = page.get("page_number", 0)
        text = page.get("text", "")
        if not text.strip():
            continue
            
        chunks = chunk_text_tokens(
            text, 
            chunk_size=chunk_size, 
            overlap=overlap, 
            preserve_sections=preserve_sections
        )
        
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": total_idx,
                    "chunk_size": len(chunk),
                    **additional_metadata,
                },
            })
            total_idx += 1
            
    logger.info(f"[TextChunker] {len(all_chunks)} token-chunks from {len(pages)} pages for '{filename}'")
    return all_chunks


# ===========================================================================
# LEGACY: Character-based chunker (kept for backward compatibility)
# ===========================================================================

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Legacy character-based chunker. Kept for backward compatibility.
    Prefer chunk_text_tokens() for new code.
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            search_start = end - int(chunk_size * 0.2)
            sentence_break = max(
                text.rfind(". ", search_start, end),
                text.rfind("! ", search_start, end),
                text.rfind("? ", search_start, end),
                text.rfind("\n", search_start, end),
            )
            if sentence_break != -1:
                end = sentence_break + 1
            else:
                space_pos = text.rfind(" ", search_start, end)
                if space_pos != -1:
                    end = space_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap if end < text_length else text_length

    logger.info(f"[TextChunker] Legacy: {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_chunks_with_metadata(
    text: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **additional_metadata: Any,
) -> List[Dict[str, Any]]:
    """Legacy: Create character-based chunks with metadata dict."""
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "text": chunk,
            "metadata": {
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                **additional_metadata,
            },
        })
    logger.info(f"[TextChunker] {len(result)} chunks with metadata for '{filename}'")
    return result


def create_chunks_from_pages(
    pages: List[Dict[str, Any]],
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **additional_metadata: Any,
) -> List[Dict[str, Any]]:
    """Legacy: Create character-based chunks from page dicts, preserving page_number."""
    all_chunks: List[Dict[str, Any]] = []
    total_idx = 0
    for page in pages:
        page_num = page.get("page_number", 0)
        text = page.get("text", "")
        if not text.strip():
            continue
        for chunk in chunk_text(text, chunk_size, chunk_overlap):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": total_idx,
                    "chunk_size": len(chunk),
                    **additional_metadata,
                },
            })
            total_idx += 1
    logger.info(f"[TextChunker] {len(all_chunks)} chunks from {len(pages)} pages for '{filename}'")
    return all_chunks


def validate_chunks(chunks: List[str], min_chunk_size: int = 10) -> List[str]:
    """Validate and filter out too-short chunks."""
    valid = [c for c in chunks if c and len(c.strip()) >= min_chunk_size]
    dropped = len(chunks) - len(valid)
    if dropped:
        logger.warning(f"[TextChunker] Filtered {dropped} chunks below min_size={min_chunk_size}")
    return valid
