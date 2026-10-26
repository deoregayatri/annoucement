"""
pipeline/document_pipeline.py

Centralized document processing orchestrator - a clean facade over
`core.ingestion.IngestionPipeline` that normalises the async/sync
interface and provides a simple synchronous entry point for workers.

Usage:
    from pipeline.document_pipeline import DocumentPipeline, process_document

    # Sync (from WorkerManager thread pool):
    result = process_document(file_path="/data/report.pdf", metadata={...})

    # Async (from FastAPI route):
    pipeline = DocumentPipeline()
    result   = await pipeline.process_document(file_path=..., metadata=...)
"""

import asyncio
import concurrent.futures
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    Facade over `core.ingestion.IngestionPipeline`.

    Responsibilities:
    - Delegates all processing to `core.ingestion.IngestionPipeline`.
    - Provides both async and sync entry points so routers and workers
      can call the same object without worrying about event-loop context.
    """

    def __init__(self) -> None:
        from core.ingestion import IngestionPipeline
        self._inner = IngestionPipeline()

    async def process_document(
        self,
        file_path: Union[str, Path],
        file_content: Optional[bytes] = None,
        original_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Async interface - for use inside FastAPI routes.

        Returns:
            {
              "chunks"      : List[str],
              "metadata"    : List[dict],
              "doc_metadata": dict,
              "chunk_ids"   : List[str],
              "skipped"     : bool,
            }
        """
        return await self._inner.process_document(
            file_path=file_path,
            file_content=file_content,
            original_filename=original_filename,
            metadata=metadata,
        )

    def process_document_sync(
        self,
        file_path: Union[str, Path],
        file_content: Optional[bytes] = None,
        original_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous interface - safe to call from WorkerManager thread/process pools.

        Automatically handles the case where an event loop is already running
        (e.g., when called from a ThreadPoolExecutor that shares a loop with FastAPI).
        """
        coro = self._inner.process_document(
            file_path=file_path,
            file_content=file_content,
            original_filename=original_filename,
            metadata=metadata,
        )

        if inspect.iscoroutine(coro):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Running inside an already-live event loop - delegate to a thread
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        return pool.submit(asyncio.run, coro).result(timeout=600)
                else:
                    return loop.run_until_complete(coro)
            except RuntimeError:
                return asyncio.run(coro)
        # Shouldn't happen, but guard against future sync inner implementations
        return coro  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Module-level convenience function (mirrors workers/handlers.py call pattern)
# ---------------------------------------------------------------------------

def process_document(
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    original_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Top-level synchronous helper - preferred entry point for workers.

    Args:
        file_path:         Absolute path to the document on disk.
        metadata:          Optional metadata dict (company_name, company_symbol, etc.)
        original_filename: Original filename if file_path is a temp path.

    Returns:
        Same dict as IngestionPipeline.process_document().
    """
    pipeline = DocumentPipeline()
    result   = pipeline.process_document_sync(
        file_path=file_path,
        original_filename=original_filename,
        metadata=metadata,
    )
    logger.info(
        f"[DocumentPipeline] processed '{original_filename or file_path}': "
        f"chunks={len(result.get('chunks', []))} skipped={result.get('skipped', False)}"
    )
    return result
