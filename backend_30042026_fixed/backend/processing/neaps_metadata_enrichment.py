"""
neaps_metadata_enrichment.py
=============================
Attaches NEAPSDBA-sourced metadata to each chunk produced by the chunker.

Fields added to every chunk's metadata dict:
  ann_seq         - Oracle primary key for the announcement
  company_symbol  - NSE/BSE ticker (e.g. 'TCS')
  ann_date        - Announcement date string (YYYY-MM-DD)
  ann_date_ts     - Unix timestamp of announcement date (for Qdrant Range filters)
  file_name       - Original file name from NEAPS server
  source_type     - always 'neaps_document'
  active          - True (soft-delete flag, Qdrant convention)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _to_timestamp(date_val: Any) -> Optional[int]:
    """Convert various date formats to Unix int timestamp."""
    if date_val is None:
        return None
    if isinstance(date_val, (int, float)):
        return int(date_val)
    if isinstance(date_val, datetime):
        return int(date_val.timestamp())
    if isinstance(date_val, str):
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y"):
            try:
                return int(datetime.strptime(date_val.strip(), fmt).timestamp())
            except ValueError:
                continue
    return None


def enrich_chunks(
    chunks: List[Dict[str, Any]],
    ann_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Enrich a list of chunk dicts with NEAPS announcement metadata.

    Args:
        chunks:     Output of neaps_chunker.chunk_document() - list of
                    {'text': ..., 'metadata': {...}} dicts.
        ann_record: Oracle announcement row -
                    {'ann_seq', 'ann_symbol_cd', 'ann_dt',
                     'ann_file_attach', 'ann_server_file_attach'}

    Returns:
        Same chunks list with metadata enriched in-place.
    """
    ann_seq = int(ann_record.get("ann_seq", 0))
    symbol = str(ann_record.get("ann_symbol_cd", "")).upper()
    company_name = str(ann_record.get("symb_company_name") or "").strip()
    ann_dt_raw = ann_record.get("ann_dt")
    ann_dt_str = str(ann_dt_raw)[:10] if ann_dt_raw else ""
    ann_ts = _to_timestamp(ann_dt_raw)
    ann_created_raw = ann_record.get("ann_creat_dt")
    ann_created_str = str(ann_created_raw) if ann_created_raw else ""
    ann_created_ts = _to_timestamp(ann_created_raw)
    ann_file_attach = ann_record.get("ann_file_attch") or ann_record.get("ann_file_attach") or ""
    ann_server_file_attch = ann_record.get("ann_server_file_attch") or ann_record.get("ann_server_file_attach") or ""
    file_name = ann_file_attach or ann_server_file_attch
    ann_status = ann_record.get("ann_status")
    ann_cd = ann_record.get("ann_cd")
    ann_text = ann_record.get("ann_text")

    for chunk in chunks:
        meta = chunk.setdefault("metadata", {})
        meta.update(
            {
                "ann_seq"        : ann_seq,
                "company_symbol" : symbol,
                "company_id"     : symbol,
                "ann_date"       : ann_dt_str,
                "announcement_date": ann_dt_str,
                "ann_date_ts"    : ann_ts,
                "ann_created_date": ann_created_str,
                "ann_created_ts" : ann_created_ts,
                "file_name"      : file_name,
                "filename"       : file_name,
                "company_name"   : company_name,
                "symb_company_name": company_name,
                "ann_status"     : ann_status,
                "ann_cd"         : ann_cd,
                "ann_text"       : ann_text,
                "ann_file_attch" : ann_file_attach,
                "ann_server_file_attch": ann_server_file_attch,
                "ANN_FILE_ATTCH" : ann_file_attach,
                "ANN_SERVER_FILE_ATTCH": ann_server_file_attch,
                "source_type"    : "neaps_document",
                "active"         : True,
            }
        )

    logger.debug(
        f"[NeapsEnrich] Enriched {len(chunks)} chunks - "
        f"ann_seq={ann_seq}, symbol={symbol}, date={ann_dt_str}"
    )
    return chunks
