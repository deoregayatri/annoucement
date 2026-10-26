"""
Document management router.

Endpoints:
  POST   /documents                             - add raw text documents to the index
  POST   /upload                                - upload a file (PDF/DOCX/TXT)
  GET    /documents/count                       - chunk and document counts
  GET    /documents/list                        - list all indexed documents
  DELETE /company/{company_id}/document/{doc_id} - soft-delete a document

Bug fixes from audit:
- NameError: `add_result` -> correctly uses `len(point_ids)` and `point_ids`
- AddDocumentsResponse: `vector_ids` field now properly populated
- /upload log line referencing undefined `add_result` is fixed
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from api.schemas import (
    AddDocumentsRequest,
    AddDocumentsResponse,
    UploadResponse,
    DocumentCountResponse,
    DocumentListResponse,
    DocumentInfo,
)
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents", response_model=AddDocumentsResponse)
async def add_documents(request: AddDocumentsRequest):
    """
    Add raw text documents to the NEAPS Qdrant index.

    Fix: response now correctly includes the `vector_ids` required field.
    """
    try:
        from vector_store.neaps_qdrant_store import get_neaps_qdrant_store
        from embeddings.neaps_hybrid_embedder import embed_chunks_hybrid

        store = get_neaps_qdrant_store()

        safe_metadata = request.metadata or ([{}] * len(request.documents))
        neaps_chunks = [
            {"text": text, "metadata": meta or {}}
            for text, meta in zip(request.documents, safe_metadata)
        ]
        embeddings = embed_chunks_hybrid(request.documents)
        point_ids = store.upsert_chunks(neaps_chunks, embeddings)

        logger.info(f"Added {len(point_ids)} documents to NEAPS store.")
        return AddDocumentsResponse(
            documents_added=len(point_ids),
            vector_ids=point_ids,
            vocabulary_entries_extracted=0,
        )

    except Exception as e:
        logger.error(f"[/documents] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add documents: {str(e)}",
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and ingest a document file (PDF / DOCX / TXT).

    Steps:
    1. Validate extension and file size.
    2. SHA-256 deduplication check.
    3. Save file to data directory.
    4. Run IngestionPipeline (OCR -> chunk -> embed -> store).
    5. Return ingestion summary.

    Bug fixes: removed reference to undefined `add_result` variable.
    """
    raw_filename = file.filename or "upload"
    filename = os.path.basename(raw_filename.replace("\\", "/"))
    file_ext = Path(filename).suffix.lower()
    allowed = [f".{e.strip()}" for e in settings.allowed_extensions.split(",")]

    if file_ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed)}",
        )

    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)

    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large: {file_size_mb:.2f}MB. Maximum: {settings.max_file_size_mb}MB",
        )

    file_hash = hashlib.sha256(file_content).hexdigest()
    logger.info(f"[/upload] hash={file_hash[:12]}... filename={filename}")

    # ------ Deduplication check -------------------------------------------
    try:
        from core.collection_manager import get_doc_store

        if get_doc_store().document_hash_exists(file_hash):
            logger.info(f"[/upload] Duplicate skipped: {filename}")
            return UploadResponse(
                success=True,
                message=f"Document '{filename}' already exists and was skipped.",
                filename=filename,
                chunks_created=0,
                documents_added=0,
                vocabulary_extracted=0,
                already_exists=True,
                file_hash=file_hash,
            )
    except Exception as e:
        logger.warning(f"[/upload] Dedup check failed (non-critical): {e}")

    # ------ Save to data directory ----------------------------------------
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / filename

    try:
        with open(file_path, "wb") as fh:
            fh.write(file_content)

        # ------ Run ingestion pipeline -------------------------------------
        from core.ingestion import IngestionPipeline

        pipeline = IngestionPipeline()
        result = await pipeline.process_document(
            file_path=file_path,
            original_filename=filename,
            metadata={
                "company_name": "Direct Upload",
                "company_symbol": "UPLOAD",
                "source": "manual_upload",
            },
        )

        if not result["chunks"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the file.",
            )

        # ------ Store in Qdrant -------------------------------------------
        from vector_store.neaps_qdrant_store import get_neaps_qdrant_store
        from embeddings.neaps_hybrid_embedder import embed_chunks_hybrid

        store = get_neaps_qdrant_store()
        neaps_chunks = []
        for text, meta in zip(result["chunks"], result["metadata"]):
            meta["file_hash"] = file_hash
            meta["source_type"] = "user_document"
            neaps_chunks.append({"text": text, "metadata": meta})

        embeddings = embed_chunks_hybrid(result["chunks"])
        point_ids = store.upsert_chunks(neaps_chunks, embeddings)

        # ------ Update BM25 local index (critical for keyword search) ------
        # Without this, sparse/BM25 search returns 0 results for uploaded docs,
        # causing short keyword queries like "cv" to show "no results found".
        try:
            from embeddings.neaps_sparse import get_bm25_index

            bm25_index_chunks = [
                {"text": chunk["text"], "metadata": chunk["metadata"]}
                for chunk in neaps_chunks
            ]
            get_bm25_index().add_chunks(bm25_index_chunks, save=True)
            logger.info(
                f"[/upload] BM25 index updated with {len(bm25_index_chunks)} chunks for '{filename}'"
            )
        except Exception as bm25_err:
            logger.warning(f"[/upload] BM25 index update failed (non-critical): {bm25_err}")

        doc_meta = result.get("doc_metadata", {})
        company = doc_meta.get("company_name", "Unknown")
        document_date = doc_meta.get("announcement_date", "Unknown")
        logger.info(
            f"[/upload] Indexed {filename}: chunks={len(point_ids)} "
            f"company='{company}' date='{document_date}'"
        )

        return UploadResponse(
            success=True,
            message="Document uploaded and indexed successfully",
            filename=filename,
            chunks_created=len(result["chunks"]),
            documents_added=len(point_ids),
            vocabulary_extracted=0,
            already_exists=False,
            file_hash=file_hash,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Clean up partial file
        try:
            if file_path.exists():
                os.unlink(file_path)
        except Exception as ex:
            logger.warning(f"[/upload] Cleanup failed for {file_path}: {ex}")
        logger.error(f"[/upload] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}",
        )


@router.get("/documents/count", response_model=DocumentCountResponse)
async def get_document_count():
    """Return chunk and unique-document counts from both vector stores."""
    try:
        from retrieval.hybrid_retriever import get_hybrid_retriever

        stats = get_hybrid_retriever().get_stats()
        vector_stats = stats.get("vector_store", {})
        bm25_stats = stats.get("bm25", {})

        return DocumentCountResponse(
            vector_db_count=vector_stats.get("unique_document_count", 0),
            bm25_count=bm25_stats.get("unique_document_count", 0),
            vector_db_chunks=vector_stats.get("points_count", 0),
            bm25_chunks=bm25_stats.get("num_documents", 0),
        )

    except Exception as e:
        logger.error(f"[/documents/count] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document count: {str(e)}",
        )


@router.get("/documents/list", response_model=DocumentListResponse)
async def get_document_list():
    """Return list of all indexed documents."""
    try:
        from core.collection_manager import get_doc_store

        doc_store = get_doc_store()
        documents = doc_store.get_document_list()
        infos = [DocumentInfo(**doc) for doc in documents]
        return DocumentListResponse(documents=infos, total_documents=len(infos))

    except Exception as e:
        logger.error(f"[/documents/list] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document list: {str(e)}",
        )


@router.delete("/company/{company_id}/document/{doc_id}")
async def delete_company_document(company_id: str, doc_id: str):
    """Soft-delete all chunks of a document from both JSON and DOC collections."""
    try:
        from core.collection_manager import get_json_store, get_doc_store

        json_ok = get_json_store().delete_by_document_id(doc_id, soft=True)
        doc_ok = get_doc_store().delete_by_document_id(doc_id, soft=True)

        logger.info(
            f"[DELETE /company/{company_id}/document/{doc_id}] json={json_ok} doc={doc_ok}"
        )
        return {
            "success": json_ok or doc_ok,
            "document_id": doc_id,
            "company_id": company_id,
            "soft_deleted": True,
            "message": "Soft-deleted. Hard deletion scheduled for weekly compaction.",
        }

    except Exception as exc:
        logger.error(f"[DELETE /company/document] Failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(exc)}",
        )
