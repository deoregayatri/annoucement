"""
neaps_sparse.py
===============
Thread-safe incremental BM25 inverted index persisted locally to a .pkl file.

Concurrency model (production-safe):
  - ONE dedicated writer thread consumes a queue.Queue — zero concurrent writes.
  - Readers acquire threading.RLock for the dirty-flag check + iteration.
  - add_chunks() is fully non-blocking (enqueue-and-return).
  - flush() blocks until all queued writes are applied and the index is saved.
"""

import logging
import math
import pickle
import queue
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_INDEX_DIR  = Path(__file__).resolve().parent.parent / "indexes"
_INDEX_FILE = _INDEX_DIR / "neaps_bm25_index.pkl"

# Financial-domain stopwords
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on",
    "at", "by", "is", "are", "was", "were", "be", "been", "being",
    "with", "as", "from", "that", "this", "it", "its", "has", "have",
    "had", "not", "but", "if", "so", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could",
    "about", "into", "than", "all", "any", "no", "such", "also",
    "per", "re", "via", "ie", "eg",
})


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1 and t not in _STOPWORDS]


class LocalIncrementalBM25:
    """
    Incremental BM25 index with production-safe concurrency.

    Writer contract:
      - Only _writer_loop() mutates inverted_index / df / documents / N / total_dl.
      - All other code treats those as read-only while holding _rlock.

    Reader contract:
      - search() acquires _rlock to guard against dict size changes during iteration.
      - _rlock is re-entrant so helpers called from search() are safe.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.N       = 0
        self.total_dl = 0

        self.df:             Dict[str, int]              = {}
        self.idf:            Dict[str, float]            = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.documents:      List[Dict[str, Any]]        = []
        # O(1) prefix lookup for short-token expansion (built at load time)
        # Replaces the O(V) linear scan in _expand_token.
        self._prefix_index:  Dict[str, List[str]]        = {}

        # Concurrency primitives
        self._rlock     = threading.RLock()   # readers + dirty-flag guard
        self._idf_dirty = False
        self._write_q   = queue.Queue()       # single-writer queue

        # Single dedicated writer thread
        self._writer = threading.Thread(
            target  = self._writer_loop,
            name    = "bm25-writer",
            daemon  = True,
        )
        self._writer.start()
        logger.debug("[BM25] Single-writer thread started.")

    # ------------------------------------------------------------------
    # Public API — callable from any thread
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Dict[str, Any]], save: bool = True) -> None:
        """
        Non-blocking enqueue.  Returns immediately; actual mutation happens in
        the dedicated writer thread.

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'.
            save:   If True (default), persists to disk after applying this batch.
        """
        if not chunks:
            return
        self._write_q.put(("add", chunks, save))

    def flush(self, timeout: float = 60.0) -> bool:
        """
        Block until all pending writes are applied and the index is saved to disk.
        Returns True if flushed successfully, False if timed out.
        """
        done_event = threading.Event()
        self._write_q.put(("flush", done_event))
        finished = done_event.wait(timeout=timeout)
        if not finished:
            logger.warning(f"[BM25] flush() timed out after {timeout}s")
        return finished

    def search(
        self,
        query:          str,
        company_symbol: Optional[str] = None,
        top_k:          int = 20,
    ) -> List[Dict[str, Any]]:
        """Thread-safe search — acquires _rlock to guard index iteration."""
        t0 = time.perf_counter()
        with self._rlock:
            if self._idf_dirty:
                self._recompute_idf()
                self._idf_dirty = False

            if self.N == 0:
                return []

            results = self._search_internal(query, company_symbol, top_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"[BM25] search returned {len(results)} results in {elapsed_ms:.1f}ms")
        return results

    def save(self, path: Path = None) -> None:
        """
        Persist index to disk safely (atomic tmp → replace).
        Call flush() first if you need guaranteed write-consistency.
        """
        path = path or _INDEX_FILE
        with self._rlock:
            if self._idf_dirty:
                self._recompute_idf()
                self._idf_dirty = False
            data = {
                "documents":      list(self.documents),
                "inverted_index": {k: list(v) for k, v in self.inverted_index.items()},
                "df":       dict(self.df),
                "idf":      dict(self.idf),
                "N":        self.N,
                "total_dl": self.total_dl,
                "k1":       self.k1,
                "b":        self.b,
            }
        self._write_to_disk(data, path)

    def load(self, path: Path = None) -> bool:
        """Load index from disk (called at startup, before writer thread is busy)."""
        path = path or _INDEX_FILE
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Direct assignment — safe here because we're in __init__ / load,
            # no concurrent writers yet.
            self.documents      = data["documents"]
            self.inverted_index = data["inverted_index"]
            self.df             = data["df"]
            self.idf            = data["idf"]
            self.N              = data["N"]
            self.total_dl       = data["total_dl"]
            self.k1             = data.get("k1", self.k1)
            self.b              = data.get("b",  self.b)
            # Build O(1) prefix index after loading vocabulary
            self._build_prefix_index()
            logger.info(
                f"[BM25] Loaded {self.N} docs from {path} "
                f"vocab={len(self.inverted_index)} prefix_keys={len(self._prefix_index)}"
            )
            return True
        except Exception as exc:
            logger.error(f"[BM25] Failed to load index from {path}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Private: writer thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        """
        Single consumer — serialises ALL writes to inverted_index.
        Never runs concurrently with another writer.
        """
        while True:
            try:
                cmd = self._write_q.get(timeout=30)
            except queue.Empty:
                continue

            kind = cmd[0]

            if kind == "add":
                _, chunks, save = cmd
                t0 = time.perf_counter()
                try:
                    self._apply_chunks(chunks)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    logger.info(
                        f"[BM25] Added {len(chunks)} chunks "
                        f"(index={self.N} docs, apply={elapsed_ms:.1f}ms)"
                    )
                    if save:
                        t_save = time.perf_counter()
                        self._flush_and_save()
                        logger.info(f"[BM25] Save completed in {(time.perf_counter()-t_save)*1000:.0f}ms")
                except Exception as exc:
                    logger.error(f"[BM25] _apply_chunks failed: {exc}", exc_info=True)
                finally:
                    self._write_q.task_done()

            elif kind == "flush":
                _, done_event = cmd
                self._flush_and_save()
                done_event.set()
                self._write_q.task_done()

    def _apply_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Mutate the index — ONLY called from _writer_loop().
        Requires _rlock to prevent dict resizing while search() iterates.
        """
        for chunk in chunks:
            text     = chunk.get("text", "")
            meta     = chunk.get("metadata", {})
            chunk_id = meta.get("chunk_id") or chunk.get("id")

            doc_idx = len(self.documents)
            tokens  = _tokenize(text)
            doc_len = len(tokens)
            tf = Counter(tokens)

            with self._rlock:
                doc_idx = len(self.documents)
                self.documents.append({
                    "id":       chunk_id,
                    "text":     text,
                    "metadata": meta,
                    "length":   doc_len,
                })

                for token, count in tf.items():
                    self.df[token] = self.df.get(token, 0) + 1
                    self.inverted_index.setdefault(token, []).append((doc_idx, count))

                self.N        += 1
                self.total_dl += doc_len

        # Signal readers to recompute IDF before next search
        with self._rlock:
            self._idf_dirty = True
            # Rebuild O(1) prefix index to include any new tokens added by this batch
            self._build_prefix_index()

    def _flush_and_save(self) -> None:
        """
        Recompute IDF (under _rlock) then pickle to disk.
        Called only from the writer thread.
        """
        with self._rlock:
            self._recompute_idf()
            self._idf_dirty = False
            # Snapshot under lock — fast shallow copies
            data = {
                "documents":      list(self.documents),
                "inverted_index": {k: list(v) for k, v in self.inverted_index.items()},
                "df":       dict(self.df),
                "idf":      dict(self.idf),
                "N":        self.N,
                "total_dl": self.total_dl,
                "k1":       self.k1,
                "b":        self.b,
            }
        # Pickle outside the lock — expensive, but no readers blocked
        self._write_to_disk(data)

    # ------------------------------------------------------------------
    # Private: internal search (called under _rlock held by search())
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Private: prefix index for O(1) short-token expansion
    # ------------------------------------------------------------------

    def _build_prefix_index(self) -> None:
        """
        Pre-compute prefix -> [tokens] mapping for fast abbreviation expansion.
        Replaces the O(V) linear scan that was called on every BM25 query.

        Called once after load() and after each _apply_chunks() batch.
        Must be called under _rlock or during single-threaded init.
        """
        from collections import defaultdict
        prefix_idx: Dict[str, List[str]] = defaultdict(list)
        for token in self.inverted_index:
            # Index all prefixes of length 1-3 for short-token lookup
            for prefix_len in range(1, min(4, len(token) + 1)):
                prefix_idx[token[:prefix_len]].append(token)
        self._prefix_index = dict(prefix_idx)

    def _expand_token(self, token: str) -> List[str]:
        """
        For short tokens (len <= 3), return all indexed tokens that start with
        that prefix so that 'cv' matches 'cvs', 'cv' etc.
        Uses O(1) pre-built prefix index instead of O(V) linear scan.
        For longer tokens, just return the token itself (exact match only).
        """
        if len(token) <= 3:
            matches = self._prefix_index.get(token)
            return matches if matches else [token]
        return [token]

    def _search_internal(
        self,
        query:          str,
        company_symbol: Optional[str],
        top_k:          int,
    ) -> List[Dict[str, Any]]:
        tokens  = _tokenize(query)
        scores: Dict[int, float] = {}
        avg_dl  = self.total_dl / self.N if self.N > 0 else 1.0

        for token in set(tokens):
            # For short abbreviations (e.g. "cv"), expand to prefix matches so
            # "cv" hits "cvs", "commercial-vehicle" stems, etc. in the index.
            candidate_tokens = self._expand_token(token)
            for candidate in candidate_tokens:
                if candidate not in self.inverted_index:
                    continue
                idf = self.idf.get(candidate, 0.0)
                # Apply a mild dampening factor for prefix-expanded matches
                # so exact matches still rank higher when both are present.
                expansion_factor = 1.0 if candidate == token else 0.85
                for entry in self.inverted_index[candidate]:
                    # Handle both old format (plain doc_idx) and new format (doc_idx, tf)
                    if isinstance(entry, int):
                        doc_idx = entry
                        tf = 1.0  # assume TF=1 for old flat indices
                    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                        doc_idx, tf = entry
                    else:
                        continue  # skip invalid entries

                    dl  = self.documents[doc_idx]["length"]
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * dl / avg_dl)
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + expansion_factor * idf * (num / den)

        results = []
        for doc_idx, score in scores.items():
            if score <= 0:
                continue
            doc  = self.documents[doc_idx]
            meta = doc.get("metadata", {})

            if company_symbol:
                doc_sym = meta.get("company_symbol", "").upper()
                if isinstance(company_symbol, list):
                    syms = [s.upper() for s in company_symbol if s]
                    if syms and doc_sym not in syms:
                        continue
                elif isinstance(company_symbol, dict) and "$in" in company_symbol:
                    syms = [s.upper() for s in (company_symbol["$in"] or []) if s]
                    if syms and doc_sym not in syms:
                        continue
                elif isinstance(company_symbol, str):
                    if doc_sym != company_symbol.upper():
                        continue

            if meta.get("active") is False:
                continue

            results.append({
                "id":          doc["id"],
                "score":       float(score),
                "text":        doc["text"],
                "metadata":    meta,
                "source":      "sparse",
                "source_type": meta.get("source_type", "neaps_document"),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Private: disk I/O
    # ------------------------------------------------------------------

    def _recompute_idf(self) -> None:
        """Recompute IDF dict from current df. Must be called under _rlock."""
        self.idf.clear()
        for token, freq in self.df.items():
            self.idf[token] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def _write_to_disk(self, data: dict, path: Path = None) -> None:
        """Atomic write: serialize to .tmp then rename."""
        path = path or _INDEX_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            logger.info(f"[BM25] Index saved to {path} ({data['N']} docs).")
        except Exception as exc:
            logger.error(f"[BM25] Save failed: {exc}")


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_index: Optional[LocalIncrementalBM25] = None
_index_lock = threading.Lock()


def get_bm25_index() -> LocalIncrementalBM25:
    """Return the global BM25 singleton, loading from disk on first access."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = LocalIncrementalBM25()
                _index.load()
    return _index
 