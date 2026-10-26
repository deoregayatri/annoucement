"""
JSON Processor -- Stage 2.

Fetches article content via the configured NSE fetch API
(POST {"url": <article_url>} to settings.nse_article_fetch_url),
normalises the response, chunks into ~500-token chunks using
LangChain's RecursiveCharacterTextSplitter (pure Python, air-gap safe),
generates 1024-dim embeddings via the shared EmbeddingModel singleton,
and stores chunks in announcements_json collection.

Fetch strategy:
  1. POST  settings.nse_article_fetch_url  { "url": "<article_url>" }  (primary)
  2. GET   <article_url> directly                                      (fallback)

All retry, back-off, and size-cap logic applies to both attempts.
The fetch API URL is controlled entirely through .env (NSE_ARTICLE_FETCH_URL).
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# ~4 chars per token on average for English text
# 500 tokens * 4 = 2000 chars, 50 tokens * 4 = 200 chars overlap
CHUNK_SIZE_CHARS    = 2000   # equivalent to ~500 tokens
CHUNK_OVERLAP_CHARS = 200    # equivalent to ~50 tokens overlap
MAX_RETRIES         = 3
RETRY_BASE_WAIT_S   = 2     # exponential backoff base


class JsonIngestionProcessor:
    """
    Fetches JSON from a URL and indexes it into announcements_json.

    LangChain components used:
      - TokenTextSplitter       : token-aware chunking
      - LangChain Document      : unified chunk representation
      - EmbeddingModel singleton: 1024-dim BAAI/bge-large-en-v1.5
      - VectorStore (json_store): Qdrant upsert
    """

    def __init__(self) -> None:
        # Singletons loaded once across all processor instances
        from core.embeddings import get_embedding_model
        from core.collection_manager import get_json_store

        self.embedder   = get_embedding_model()
        self.json_store = get_json_store()

        # RecursiveCharacterTextSplitter: pure Python, no internet access needed.
        # Replaces TokenTextSplitter (tiktoken) which tries to download GPT-2 vocab
        # from openaipublic.blob.core.windows.net -- blocked on air-gapped servers.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_CHARS,
            chunk_overlap=CHUNK_OVERLAP_CHARS,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        logger.info(
            f"[JsonIngestionProcessor] Ready -- "
            f"chunk_size={CHUNK_SIZE_CHARS}chars (~500t), overlap={CHUNK_OVERLAP_CHARS}chars"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_url(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch URL  parse JSON  chunk  embed  index.

        Args:
            row: IngestionRow.to_dict() with keys:
                 row_id, url, company_name, company_symbol,
                 title, platform, mentioned_date, url_hash

        Returns:
            Summary dict with chunk_count, document_id, status
        """
        url       = row["url"]
        row_id    = row["row_id"]
        url_hash  = row.get("url_hash", hashlib.sha256(url.encode()).hexdigest())

        logger.info(f"[JsonIngestionProcessor] Processing url={url[:80]} row_id={row_id}")

        # --- 1. Fetch JSON with retries ---
        json_data = self._fetch_with_retry(url)
        if json_data is None:
            return {"status": "FAILED", "error": "fetch_failed", "row_id": row_id}

        # --- 2. Normalise to text ---
        text = self._extract_text(json_data)
        if not text.strip():
            logger.warning(f"[JsonIngestionProcessor] Empty text extracted from {url}")
            return {"status": "FAILED", "error": "empty_text", "row_id": row_id}

        # --- 3. Chunk via LangChain ---
        lc_documents = self._chunk(text, row, url_hash)

        # --- 4. Embed ---
        texts      = [doc.page_content for doc in lc_documents]
        embeddings = self.embedder.embed(texts)

        # --- 5. Build metadata list ---
        metadata_list = [doc.metadata for doc in lc_documents]

        # --- 6. Upsert into Qdrant ---
        ids = self.json_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata_list,
            file_hash=url_hash,
            source_type="json",
        )

        # (Legacy BM25 invalidation removed)

        logger.info(
            f"[JsonIngestionProcessor] Indexed {len(ids)} chunks for row_id={row_id} "
            f"company={row['company_symbol']}"
        )
        return {
            "status"      : "SUCCESS",
            "row_id"      : row_id,
            "document_id" : url_hash,
            "chunk_count" : len(ids),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(self, url: str) -> Optional[Any]:
        """
        Fetch article content with retry and exponential back-off.

        Primary:  POST settings.nse_article_fetch_url { "url": url }
        Fallback: GET  url  (used if the fetch API is unavailable)

        Returns parsed JSON (dict/list/str) or None on total failure.
        """
        # Security: reject non-HTTP(S) schemes
        if not (url.startswith("http://") or url.startswith("https://")):
            logger.error(f"[JsonIngestionProcessor] Rejected non-HTTP URL: {url[:80]}")
            return None

        from config import settings
        fetch_api_url = settings.nse_article_fetch_url   # e.g. https://www.com/api/fetch-article

        HEADERS = {
            "Accept"      : "application/json",
            "Content-Type": "application/json",
            "User-Agent"  : "AnnouncementRAG/1.0",
        }
        MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB cap

        def _read_response(response) -> Optional[Any]:
            """Read, size-cap, and JSON-parse a requests.Response."""
            content = response.raw.read(MAX_RESPONSE_BYTES + 1)
            if len(content) > MAX_RESPONSE_BYTES:
                logger.warning(f"[JsonIngestionProcessor] Response >10 MB, truncating for {url[:60]}")
                content = content[:MAX_RESPONSE_BYTES]
            return __import__("json").loads(content)

        # ------------------------------------------------------------------
        # Primary: POST to NSE fetch API
        # ------------------------------------------------------------------
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    f"[JsonIngestionProcessor] Fetch attempt {attempt}/{MAX_RETRIES}: "
                    f"POST {fetch_api_url} url={url[:60]}"
                )
                response = requests.post(
                    fetch_api_url,
                    json={"url": url},
                    timeout=30,
                    headers=HEADERS,
                    stream=True,
                )
                logger.info(
                    f"[JsonIngestionProcessor] NSE API HTTP {response.status_code} "
                    f"for url={url[:60]}"
                )
                response.raise_for_status()
                return _read_response(response)

            except requests.exceptions.HTTPError as exc:
                logger.warning(f"[JsonIngestionProcessor] NSE API HTTP error attempt {attempt}: {exc}")
            except requests.exceptions.ConnectionError as exc:
                logger.warning(f"[JsonIngestionProcessor] NSE API connection error attempt {attempt}: {exc}")
            except requests.exceptions.Timeout:
                logger.warning(f"[JsonIngestionProcessor] NSE API timeout on attempt {attempt}")
            except Exception as exc:
                logger.error(f"[JsonIngestionProcessor] NSE API unexpected error attempt {attempt}: {exc}")

            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_WAIT_S ** attempt
                logger.info(f"[JsonIngestionProcessor] Retrying in {wait}s ...")
                time.sleep(wait)

        # ------------------------------------------------------------------
        # Fallback: direct GET on the article URL
        # ------------------------------------------------------------------
        logger.warning(
            f"[JsonIngestionProcessor] NSE fetch API failed after {MAX_RETRIES} attempts - "
            f"falling back to direct GET: {url[:80]}"
        )
        GET_HEADERS = {"Accept": "application/json", "User-Agent": "AnnouncementRAG/1.0"}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"[JsonIngestionProcessor] Fallback GET attempt {attempt}/{MAX_RETRIES}: {url[:80]}")
                response = requests.get(url, timeout=30, headers=GET_HEADERS, stream=True)
                logger.info(f"[JsonIngestionProcessor] Fallback HTTP {response.status_code}")
                response.raise_for_status()
                return _read_response(response)

            except requests.exceptions.HTTPError as exc:
                logger.warning(f"[JsonIngestionProcessor] Fallback HTTP error attempt {attempt}: {exc}")
            except requests.exceptions.ConnectionError as exc:
                logger.warning(f"[JsonIngestionProcessor] Fallback connection error attempt {attempt}: {exc}")
            except requests.exceptions.Timeout:
                logger.warning(f"[JsonIngestionProcessor] Fallback timeout on attempt {attempt}")
            except Exception as exc:
                logger.error(f"[JsonIngestionProcessor] Fallback unexpected error attempt {attempt}: {exc}")

            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_WAIT_S ** attempt
                logger.info(f"[JsonIngestionProcessor] Fallback retry in {wait}s ...")
                time.sleep(wait)

        logger.error(f"[JsonIngestionProcessor] All fetch strategies exhausted for {url}")
        return None

    @staticmethod
    def _extract_text(json_data: Any) -> str:
        """
        Normalise JSON to a plain-text string for embedding.
        Handles: dict, list, str, or nested structures.
        """
        if isinstance(json_data, str):
            return json_data

        if isinstance(json_data, dict):
            parts = []
            for key, val in json_data.items():
                if isinstance(val, (dict, list)):
                    parts.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")
                else:
                    parts.append(f"{key}: {val}")
            return "\n".join(parts)

        if isinstance(json_data, list):
            return "\n\n".join(
                JsonIngestionProcessor._extract_text(item) for item in json_data
            )

        return str(json_data)

    def _chunk(self, text: str, row: Dict[str, Any], url_hash: str) -> List[Document]:
        """Split text into token-bounded LangChain Documents with full metadata."""
        now_ts = int(datetime.now().timestamp())
        mentioned_ts = None
        if row.get("mentioned_date"):
            try:
                mentioned_ts = int(
                    datetime.fromisoformat(row["mentioned_date"]).timestamp()
                )
            except (ValueError, TypeError):
                pass

        base_meta = {
            "company_name"        : row.get("company_name", ""),
            "company_symbol"      : row.get("company_symbol", ""),
            "company_id"          : row.get("company_id", row.get("company_symbol", "")),
            "title"               : row.get("title", ""),
            "platform"            : row.get("platform", ""),
            "url"                 : row.get("url", ""),
            "excel_row_id"        : row.get("row_id", ""),
            "document_id"         : url_hash,
            "source_type"         : "json",
            "active"              : True,
            "mentioned_timestamp" : mentioned_ts or now_ts,
            "announcement_date"   : mentioned_ts or now_ts,
            # Extra columns from Excel (optional; empty string if absent)
            "description"         : row.get("description", ""),
            "sensex_announcement" : row.get("sensex_announcement", ""),
        }

        chunks = self.splitter.split_text(text)
        documents = []
        for idx, chunk in enumerate(chunks):
            meta = dict(base_meta)
            meta["chunk_index"] = idx
            meta["total_chunks"] = len(chunks)
            documents.append(Document(page_content=chunk, metadata=meta))

        logger.debug(
            f"[JsonIngestionProcessor] Created {len(documents)} chunks "
            f"for document_id={url_hash[:12]}..."
        )
        return documents
