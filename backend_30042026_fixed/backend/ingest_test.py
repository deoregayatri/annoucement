import sys
import uuid
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from vector_store.neaps_qdrant_store import get_neaps_qdrant_store
from embeddings.neaps_hybrid_embedder import embed_chunks_hybrid
from utils.document_processor import extract_pages_from_file
from utils.text_chunker import create_token_chunks_from_pages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_ingest")

def ingest_pdf(pdf_path: str, company_symbol: str = "JASH", company_name: str = "Jash Engineering Limited"):
    local_path = Path(pdf_path)
    if not local_path.exists():
        logger.error(f"File not found: {local_path}")
        return

    logger.info(f"Extracting text from {local_path.name}...")
    pages = extract_pages_from_file(local_path)
    if not pages:
        logger.error("No pages extracted from PDF.")
        return

    file_size_mb = local_path.stat().st_size / (1024 * 1024)

    logger.info("Chunking text...")
    raw_chunks = create_token_chunks_from_pages(
        pages=pages,
        filename=local_path.name,
        chunk_size=settings.neaps_chunk_size,
        overlap=settings.neaps_chunk_overlap,
        file_type="pdf",
        file_size_mb=file_size_mb,
    )

    if not raw_chunks:
        logger.error("Chunker produced zero chunks.")
        return

    logger.info(f"Created {len(raw_chunks)} raw chunks. Enriching metadata...")
    ann_seq = 999999 # dummy seq for testing
    
    enriched = []
    for i, chunk in enumerate(raw_chunks):
        meta = chunk.setdefault("metadata", {})
        meta["company_symbol"] = company_symbol
        meta["company_id"] = company_symbol
        meta["active"] = True
        meta["source_type"] = "neaps_document"
        meta["ann_seq"] = ann_seq
        meta["ANN_SEQ"] = ann_seq
        meta["company_name"] = company_name
        meta["SYMB_SYMBOL"] = company_symbol
        meta["SYMB_COMPANY_NAME"] = company_name
        meta["chunk_id"] = str(uuid.uuid4())
        enriched.append(chunk)

    logger.info("Generating hybrid embeddings...")
    texts = [c["text"] for c in enriched]
    hybrid_embs = embed_chunks_hybrid(texts)

    logger.info("Upserting to Qdrant...")
    store = get_neaps_qdrant_store()
    ids = store.upsert_chunks(enriched, hybrid_embs)
    
    logger.info(f"Successfully upserted {len(ids)} chunks to Qdrant.")
    
    # Try updating BM25 index as well
    try:
        from embeddings.neaps_sparse import get_bm25_index
        bm25 = get_bm25_index()
        bm25.add_chunks(enriched, save=True)
        logger.info(f"BM25 index updated with {len(enriched)} chunks.")
    except Exception as exc:
        logger.warning(f"Failed to update BM25 index: {exc}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_test.py <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    ingest_pdf(pdf_path)
