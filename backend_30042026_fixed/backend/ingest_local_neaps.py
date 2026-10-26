"""

Temporary ingestion script for local corporate announcements.

Reads metadata from 'announcementss' in Oracle DB, finds documents in 

/mnt/Announcement/neaps_document/Ann_docs_all_test/, and ingests 

into NEAPS Qdrant collection skipping duplicates.

"""

import logging

import time

from pathlib import Path

from typing import Any, Dict, List

import oracledb

# Reuse existing pipeline modules

from config import settings

from vector_store.neaps_qdrant_store import get_neaps_qdrant_store

from embeddings.neaps_hybrid_embedder import embed_chunks_hybrid

from utils.document_processor import extract_pages_from_file, extract_text_from_file

from utils.text_chunker import create_token_chunks_from_pages, create_chunks_with_metadata

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",

    datefmt="%Y-%m-%d %H:%M:%S",

)

logger = logging.getLogger("local_neaps_ingest")


LOCAL_DOC_FOLDER = Path("/mnt/Announcement/neaps_document/Ann_docs_all_test")

TARGET_COMPANIES = ["RELIANCE", "POONAWALLA", "ADANIENT"]

def fetch_oracle_metadata() -> List[Dict[str, Any]]:

    """Fetch metadata from the Oracle announcementss table for specific companies."""

    user = settings.neaps_oracle_user

    password = settings.neaps_oracle_password

    dsn = settings.neaps_oracle_dsn

    logger.info(f"Connecting to Oracle DSN: {dsn}...")

    # Init thick mode if enabled in settings

    if settings.neaps_oracle_thick_mode:

        try:

            lib_dir = settings.neaps_oracle_client_lib_dir or ""

            kwargs = {"lib_dir": lib_dir} if lib_dir else {}

            oracledb.init_oracle_client(**kwargs)

        except Exception:

            pass

    conn = oracledb.connect(user=user, password=password, dsn=dsn)

    try:

        cur = conn.cursor()

        # Query distinct symbols as requested by user constraints to verify structure

        cur.execute("SELECT DISTINCT symb_symbol FROM announcementss")

        symbols = [row[0] for row in cur.fetchall()]

        logger.info(f"Found distinct symbols in announcementss: {len(symbols)}")

        # Fetch metadata for the target companies

        target_symbols_sql = ", ".join([f"'{sym}'" for sym in TARGET_COMPANIES])

        query = f"""

            SELECT ANN_SEQ, ANN_SYMBOL_CD, SYMB_COMPANY_NAME, ANN_DT, 

                   ANN_FILE_ATTCH, ANN_CREAT_DT, ANN_SERVER_FILE_ATTCH, 

                   ANN_TEXT, ANN_STATUS, ANN_CD, SYMB_SYMBOL 

            FROM announcementss 

            WHERE UPPER(SYMB_SYMBOL) IN ({target_symbols_sql})

        """

        cur.execute(query)

        columns = [col[0].lower() for col in cur.description]

        rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        logger.info(f"Fetched {len(rows)} announcement records for {TARGET_COMPANIES}")

        return rows

    finally:

        conn.close()


def process_and_ingest(records: List[Dict[str, Any]]):

    store = get_neaps_qdrant_store()

    stats = {

        "processed": 0,

        "skipped_dups": 0,

        "failed": 0,

        "total_chunks_stored": 0

    }

    for idx, row in enumerate(records, 1):

        ann_seq = int(row.get("ann_seq", 0))

        company_sym = str(row.get("symb_symbol", "")).upper()

        # 1. DUPLICATE PROTECTION: Check if ann_seq is already in Qdrant

        if store.ann_seq_exists(ann_seq):

            logger.info(f"[{idx}/{len(records)}] SKIPPING {ann_seq} - Duplicate (already exists in vector DB)")

            stats["skipped_dups"] += 1

            continue

        # Locate the local file

        file_name = row.get("ann_server_file_attch") or row.get("ann_file_attch")

        if not file_name:

            logger.warning(f"[{idx}/{len(records)}] SKIPPING {ann_seq} - No filename found in metadata")

            stats["failed"] += 1

            continue

        local_path = LOCAL_DOC_FOLDER / file_name

        if not local_path.exists():

            logger.warning(f"[{idx}/{len(records)}] SKIPPING {ann_seq} - File not found at {local_path}")

            stats["failed"] += 1

            continue

        logger.info(f"[{idx}/{len(records)}] Processing {ann_seq} for {company_sym} from {file_name}")

        try:

            file_ext = local_path.suffix.lower()

            file_size_mb = local_path.stat().st_size / (1024 * 1024)

            raw_chunks = []

            # Extract and chunk document

            if file_ext == ".pdf":

                pages = extract_pages_from_file(local_path)

                if not pages:

                    raise ValueError("No pages extracted from PDF")

                # Use token chunking for PDFs
                raw_chunks = create_token_chunks_from_pages(
                    pages=pages,
                    filename=filename,
                    chunk_size=settings.neaps_chunk_size,
                    overlap=settings.neaps_chunk_overlap,
                    preserve_sections=True,
                    **ann_meta
                )
            else:

                text, _ = extract_text_from_file(local_path)

                if not text.strip():

                    raise ValueError("No text extracted from document")

                raw_chunks = create_chunks_with_metadata(

                    text=text,

                    filename=file_name,

                    chunk_size=settings.neaps_chunk_size,

                    chunk_overlap=settings.neaps_chunk_overlap,

                    file_type=file_ext.replace(".", ""),

                    file_size_mb=file_size_mb

                )

            if not raw_chunks:

                raise ValueError("Chunker produced no chunks")

            # Enrich chunks with ALL Oracle metadata as requested

            enriched_chunks = []

            for chunk in raw_chunks:

                meta = chunk.setdefault("metadata", {})

                # Qdrant filtering requirements (must have company_symbol)

                meta["company_symbol"] = company_sym

                meta["company_id"] = company_sym

                meta["active"] = True

                meta["source_type"] = "neaps_document"

                # Add all requested Oracle metadata

                meta["ANN_SEQ"] = ann_seq

                meta["SYMB_SYMBOL"] = company_sym

                meta["SYMB_COMPANY_NAME"] = row.get("symb_company_name")

                # Dates need string conversion if they are datetime objects

                ann_dt = row.get("ann_dt")

                meta["ANN_DT"] = ann_dt.isoformat() if hasattr(ann_dt, "isoformat") else str(ann_dt)

                ann_creat_dt = row.get("ann_creat_dt")

                meta["ANN_CREAT_DT"] = ann_creat_dt.isoformat() if hasattr(ann_creat_dt, "isoformat") else str(ann_creat_dt)

                meta["ANN_FILE_ATTCH"] = row.get("ann_file_attch")

                meta["ANN_SERVER_FILE_ATTCH"] = row.get("ann_server_file_attch")

                meta["ANN_STATUS"] = row.get("ann_status")

                meta["ANN_CD"] = row.get("ann_cd")

                enriched_chunks.append(chunk)

            # Generate Hybrid Embeddings

            texts = [c["text"] for c in enriched_chunks]

            hybrid_embs = embed_chunks_hybrid(texts)

            # Upsert into Qdrant

            ids = store.upsert_chunks(enriched_chunks, hybrid_embs)

            logger.info(f"  -> SUCCESS: Ingested {len(ids)} chunks for {ann_seq}")

            stats["processed"] += 1

            stats["total_chunks_stored"] += len(ids)

        except Exception as e:

            logger.error(f"  -> FAILED: Error processing {ann_seq} '{file_name}': {str(e)}")

            stats["failed"] += 1

    # Final reporting

    print("\n" + "="*50)

    print(" INGESTION COMPLETE")

    print("="*50)

    print(f" Documents processed successfully  : {stats['processed']}")

    print(f" Documents skipped (duplicates)    : {stats['skipped_dups']}")

    print(f" Documents failed                  : {stats['failed']}")

    print(f" Total new chunks stored in Qdrant : {stats['total_chunks_stored']}")

    print("="*50 + "\n")


if __name__ == "__main__":

    records = fetch_oracle_metadata()

    if records:

        process_and_ingest(records)

    else:

        logger.warning("No records found for target companies.")

 