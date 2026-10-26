"""
app/services/sat/service.py
============================
Scraper for the Securities Appellate Tribunal (SAT) orders search at
https://satweb.sat.gov.in/orders — "Date Wise" tab only.

Verified live behaviour (from direct HTML/JS/network inspection):
  - GET  /orders           – loads the search form; response HTML contains a
                             hidden `#security_token` input required on every POST.
  - POST /get-orders-by-date (root path, NOT /orders/get-orders-by-date)
                             – form fields: apl_type (1=SEBI, 2=IRDA, 3=PFRDA),
                             startDate, endDate (both DD-MM-YYYY), security_token.
                             Requires the PHPSESSID cookie from the GET above.
                             Returns JSON: {"content": "<html table>", "token": "<new security_token>"}.
                             The token rotates on every call — the returned token
                             must be reused on the next request in the same session.
  - GET  /crefresh_val     – returns {"image": "<img ...>", "word": "<plaintext answer>"}.
                             Captcha verification is client-side only: orders.js
                             (`setupCaptcha`) merely compares the typed value against
                             this hidden word to enable the Search button — the value
                             is never included in the POST to get-orders-by-date.
                             No OCR/captcha-solving is performed or required.
  - No server-side pagination for date-wise search — a single request can return
    thousands of rows for wide ranges, so wide ranges are split into chunks here.
"""

import asyncio
import logging
import random
import re
from datetime import date, datetime, timedelta

import httpx
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

from app.core.cache import TTLCache
from app.core.http_client import fetch_with_proxy, COMMON_HEADERS

log = logging.getLogger(__name__)

BASE_URL = "https://satweb.sat.gov.in"
ORDERS_PAGE_URL = f"{BASE_URL}/orders"
DATE_ENDPOINT = f"{BASE_URL}/get-orders-by-date"

APL_TYPE_MAP: dict[str, str] = {
    "sebi": "1",
    "irda": "2",
    "pfrda": "3",
}

CHUNK_MONTHS = 3          # split wide ranges into <=3 month windows per request
MAX_CHUNKS = 40           # safety cap: 40 x 3 months = 10 years

TTL_SHORT_S = 10 * 60       # 10 minutes — to_date is today or future
TTL_LONG_S  = 7 * 24 * 3600  # 7 days    — to_date is fully in the past

# Module-level singleton: shared across all calls within the process.
_cache: TTLCache = TTLCache(default_ttl_seconds=TTL_SHORT_S)

SAT_HEADERS = {
    **COMMON_HEADERS,
    "Referer": ORDERS_PAGE_URL,
    "Origin": BASE_URL,
    "X-Requested-With": "XMLHttpRequest",
}

_TOKEN_RE = re.compile(r'''id=["']security_token["']\s+value=["']([^"']+)["']''')


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def _parse_input_date(value: str) -> date:
    """Accept DD-MM-YYYY, YYYY-MM-DD, or DD/MM/YYYY."""
    if not value or not value.strip():
        raise ValueError("Date value is required")
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date '{value}'. Use DD-MM-YYYY or YYYY-MM-DD.")


def _iter_date_chunks(start: date, end: date) -> list[tuple[str, str]]:
    """Split [start, end] into <=CHUNK_MONTHS windows, formatted DD-MM-YYYY for SAT."""
    chunks: list[tuple[str, str]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + relativedelta(months=CHUNK_MONTHS) - timedelta(days=1), end)
        chunks.append((cur.strftime("%d-%m-%Y"), chunk_end.strftime("%d-%m-%Y")))
        cur = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

_VS_SPLIT_RE = re.compile(r"\s+vs\s+", re.I)


def _parse_table(html: str) -> list[dict]:
    """
    Parse the date-wise results fragment into normalised record dicts.

    Table columns: S.No. | AL No. | Appeal No | Appellant vs Respondent |
                   Court | Order Date | View Order (anchor)
    """
    if not html or not html.strip():
        return []

    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 7:
            continue  # header rows use <th>, not <td>

        sl_no = cells[0].get_text(strip=True)
        al_no = cells[1].get_text(strip=True)
        appeal_no = cells[2].get_text(strip=True)
        parties = cells[3].get_text(" ", strip=True)
        court = cells[4].get_text(strip=True)
        order_date = cells[5].get_text(strip=True)

        anchor = cells[6].find("a")
        view_order_url = anchor.get("href", "").strip() if anchor else ""

        parts = _VS_SPLIT_RE.split(parties, maxsplit=1)
        appellant = parts[0].strip() if parts else parties
        respondent = parts[1].strip() if len(parts) == 2 else ""

        records.append({
            "sl_no": sl_no,
            "al_no": al_no,
            "appeal_no": appeal_no,
            "circular_no": None,
            "parties": parties,
            "appellant": appellant,
            "respondent": respondent,
            "court": court,
            "order_date": order_date,
            "view_order_url": view_order_url,
        })

    return records


# ---------------------------------------------------------------------------
# Core HTTP flow (stateful cookie session, like NCDEX)
# ---------------------------------------------------------------------------

async def _fetch_security_token(client: httpx.AsyncClient, via_proxy: bool) -> str:
    """GET /orders and extract the hidden security_token, establishing the session cookie."""
    resp = await fetch_with_proxy(
        client,
        ORDERS_PAGE_URL,
        headers={**COMMON_HEADERS, "Referer": BASE_URL},
        timeout=30,
        via_proxy=via_proxy,
    )
    resp.raise_for_status()

    match = _TOKEN_RE.search(resp.text)
    if not match:
        raise RuntimeError("[SAT] Could not extract security_token from /orders page")
    return match.group(1)


async def _fetch_date_chunk(
    client: httpx.AsyncClient,
    apl_code: str,
    start_str: str,
    end_str: str,
    token: str,
    via_proxy: bool,
) -> tuple[list[dict], str]:
    """POST one date-range chunk and return (records, rotated_token)."""
    resp = await fetch_with_proxy(
        client,
        DATE_ENDPOINT,
        method="POST",
        data={
            "apl_type": apl_code,
            "startDate": start_str,
            "endDate": end_str,
            "security_token": token,
        },
        headers=SAT_HEADERS,
        timeout=60,
        via_proxy=via_proxy,
    )
    resp.raise_for_status()

    payload = resp.json()
    records = _parse_table(payload.get("content", ""))
    new_token = payload.get("token") or token
    return records, new_token


# ---------------------------------------------------------------------------
# Public scraper function
# ---------------------------------------------------------------------------

async def scrape_sat_orders_datewise(req) -> dict:
    """
    Scrape SAT orders for a date range via the "Date Wise" search tab.

    Wide ranges are split into <=CHUNK_MONTHS windows, capped at MAX_CHUNKS,
    and merged into a single record list.

    Results are cached in-process:
      - to_date >= today  → 10-minute TTL (orders may be added later the same day)
      - to_date < today   → 7-day TTL (past order listings are immutable)
    """
    apl_type = (req.apl_type or "").strip().lower()
    if apl_type not in APL_TYPE_MAP:
        raise ValueError(
            f"Unknown apl_type '{req.apl_type}'. Valid: {sorted(APL_TYPE_MAP)}"
        )

    start_date = _parse_input_date(req.from_date)
    end_date = _parse_input_date(req.to_date)
    if start_date > end_date:
        raise ValueError("from_date must not be after to_date")

    chunks = _iter_date_chunks(start_date, end_date)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"Date range too large — resolves to {len(chunks)} chunks of "
            f"{CHUNK_MONTHS} months each (max {MAX_CHUNKS}). Narrow the range."
        )

    via_proxy = getattr(req, "via_proxy", False)
    apl_code = APL_TYPE_MAP[apl_type]

    # ------------------------------------------------------------------
    # Cache check
    # ------------------------------------------------------------------
    cache_key = TTLCache.make_key(
        "sat_orders", apl_type, start_date.isoformat(), end_date.isoformat()
    )
    cached = _cache.get(cache_key)
    if cached is not None:
        log.info(
            f"[SAT] cache hit: apl_type={apl_type!r} "
            f"{start_date}..{end_date} count={cached['count']}"
        )
        return cached

    log.info(
        f"[SAT] cache miss: apl_type={apl_type!r} "
        f"{start_date}..{end_date} — fetching from upstream"
    )

    # ------------------------------------------------------------------
    # Upstream fetch (chunked)
    # ------------------------------------------------------------------
    log.info(
        f"[SAT] date-wise scrape: apl_type={apl_type!r} "
        f"{start_date}..{end_date} chunks={len(chunks)} via_proxy={via_proxy}"
    )

    all_records: list[dict] = []
    async with httpx.AsyncClient() as client:
        token = await _fetch_security_token(client, via_proxy)

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            try:
                records, token = await _fetch_date_chunk(
                    client, apl_code, chunk_start, chunk_end, token, via_proxy
                )
                all_records.extend(records)
                log.debug(
                    f"[SAT] chunk {i + 1}/{len(chunks)} {chunk_start}..{chunk_end} "
                    f"-> {len(records)} records"
                )
            except Exception:
                log.exception(
                    f"[SAT] chunk fetch failed for {chunk_start}..{chunk_end}; skipping"
                )

            if i < len(chunks) - 1:
                await asyncio.sleep(random.uniform(0.5, 1.2))

    result = {
        "source": "sat",
        "url": ORDERS_PAGE_URL,
        "filters_applied": {
            "apl_type": apl_type,
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
        },
        "chunks_fetched": len(chunks),
        "count": len(all_records),
        "records": all_records,
    }

    # ------------------------------------------------------------------
    # Cache store — TTL depends on whether the range is fully in the past
    # ------------------------------------------------------------------
    today = date.today()
    ttl = TTL_SHORT_S if end_date >= today else TTL_LONG_S
    _cache.set(cache_key, result, ttl_seconds=ttl)
    log.info(
        f"[SAT] cached result: apl_type={apl_type!r} {start_date}..{end_date} "
        f"ttl={ttl}s ({'short' if end_date >= today else 'long'})"
    )

    return result
