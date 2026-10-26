"""
app/services/sat/pdflogic.py
============================
Standalone service and module for downloading and extracting PDF files from
Securities Appellate Tribunal (SAT) order links (e.g. view-order URLs).

Handles SAT stateful cookie session (PHPSESSID) and security_token rotation,
extracting binary PDFs via the upstream `view-order-document` endpoint.

Can be run as a standalone FastAPI service (like downloader/main.py) on UAT/AWS:
    python -m app.services.sat.pdflogic
    or
    uvicorn app.services.sat.pdflogic:app --host 0.0.0.0 --port 8083

Or imported into existing routers / scrapers.
"""

import asyncio
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sat_pdflogic")

BASE_URL = "https://satweb.sat.gov.in"
ORDERS_PAGE_URL = f"{BASE_URL}/orders"
VIEW_DOC_ENDPOINT = f"{BASE_URL}/view-order-document"

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": BASE_URL,
    "Referer": ORDERS_PAGE_URL,
    "X-Requested-With": "XMLHttpRequest",
}

TOKEN_RE = re.compile(r'''id=["']security_token["']\s+value=["']([^"']+)["']''')
ORDER_ID_RE = re.compile(r"/view-order/[a-zA-Z0-9_-]+/(\d+)")
TRAILING_ID_RE = re.compile(r"/(\d+)/?$")


# ==========================================================
# SCHEMAS
# ==========================================================

class SinglePDFRequest(BaseModel):
    url: Optional[str] = Field(None, description="SAT view-order URL, e.g. https://satweb.sat.gov.in/view-order/<hash>/<order_id>")
    view_order_url: Optional[str] = Field(None, description="Alternative alias for view-order URL")
    order_id: Optional[str] = Field(None, description="Direct order ID if already known, e.g. 44189")
    appeal_no: Optional[str] = Field(None, description="Optional appeal number for naming the saved file")
    force: bool = Field(False, description="Re-download even if already exists on disk")


class BatchPDFRequest(BaseModel):
    urls: Optional[list[str]] = Field(None, description="List of view-order URLs")
    records: Optional[list[dict[str, Any]]] = Field(None, description="Crawler output records list")
    source: Optional[str] = Field("sat", description="Source identifier")
    force: bool = Field(False, description="Re-download even if already exists on disk")


class FilePDFRequest(BaseModel):
    file_path: str = Field(..., description="Path to crawler JSON file on disk")
    force: bool = Field(False, description="Re-download even if already exists on disk")


# ==========================================================
# CORE DOWNLOADER LOGIC
# ==========================================================

class SATPDFDownloader:
    def __init__(self, download_dir: Optional[str] = None):
        base = download_dir or os.getenv("DOWNLOAD_DIR", "data/raw")
        self.save_dir = Path(base) / "sat"
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_order_id(value: Union[str, int]) -> str:
        """
        Extract numeric order_id from URL or plain string.
        Examples:
            'https://satweb.sat.gov.in/view-order/6beef1.../44189' -> '44189'
            'https://satweb.sat.gov.in/view-order/6beef1.../44189/' -> '44189'
            '44189' -> '44189'
        """
        val_str = str(value).strip()
        if val_str.isdigit():
            return val_str

        m = ORDER_ID_RE.search(val_str)
        if m:
            return m.group(1)

        m = TRAILING_ID_RE.search(val_str)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract order_id from: '{val_str}'")

    @staticmethod
    def _clean_name(value: str) -> str:
        value = str(value).strip()
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        return "".join(c if c in allowed else "_" for c in value)

    def get_target_file_path(self, order_id: str, appeal_no: Optional[str] = None) -> Path:
        if appeal_no:
            safe_appeal = self._clean_name(appeal_no)
            filename = f"sat_{safe_appeal}_{order_id}.pdf"
        else:
            filename = f"sat_order_{order_id}.pdf"
        return self.save_dir / filename

    async def fetch_security_token(self, client: httpx.AsyncClient) -> str:
        """GET /orders to establish PHPSESSID and retrieve initial security_token."""
        resp = await client.get(
            ORDERS_PAGE_URL,
            headers={**COMMON_HEADERS, "Referer": BASE_URL},
            timeout=30.0,
        )
        resp.raise_for_status()
        match = TOKEN_RE.search(resp.text)
        if not match:
            raise RuntimeError("Could not extract security_token from SAT /orders page")
        token = match.group(1)
        logger.debug(f"[SAT] Session established, token: {token[:10]}...")
        return token

    async def download_order_pdf(
        self,
        client: httpx.AsyncClient,
        token: str,
        order_id: str,
        appeal_no: Optional[str] = None,
        original_url: Optional[str] = None,
        force: bool = False,
    ) -> tuple[dict[str, Any], str]:
        """
        Fetch a single PDF using active session and rotating token.
        Returns: (result_dict, new_token)
        """
        file_path = self.get_target_file_path(order_id, appeal_no)

        # Skip if already downloaded and not forcing
        if file_path.exists() and file_path.stat().st_size > 0 and not force:
            logger.info(f"[SAT] File already exists for order {order_id}: {file_path}")
            return {
                "order_id": order_id,
                "appeal_no": appeal_no,
                "original_url": original_url,
                "file_name": file_path.name,
                "file_path": str(file_path.resolve()),
                "download_url": f"/api/v1/sat/pdf/{order_id}",
                "file_size_bytes": file_path.stat().st_size,
                "status": "already_exists",
                "error": None,
            }, token

        # Post to view-order-document
        resp = await client.post(
            VIEW_DOC_ENDPOINT,
            data={"order": order_id, "security_token": token},
            headers=COMMON_HEADERS,
            timeout=60.0,
        )
        resp.raise_for_status()

        data = resp.json()
        new_token = data.get("token") or token
        status = data.get("status")

        if status != "success" or not data.get("content"):
            err_msg = data.get("message") or f"Upstream SAT returned status '{status}'"
            logger.warning(f"[SAT] Failed to retrieve document for order {order_id}: {err_msg}")
            return {
                "order_id": order_id,
                "appeal_no": appeal_no,
                "original_url": original_url,
                "file_name": file_path.name,
                "file_path": None,
                "download_url": None,
                "file_size_bytes": 0,
                "status": "failed",
                "error": err_msg,
            }, new_token

        # Decode base64 PDF
        pdf_bytes = base64.b64decode(data["content"])
        file_path.write_bytes(pdf_bytes)

        logger.info(f"[SAT] Successfully saved order {order_id} ({len(pdf_bytes)} bytes) -> {file_path}")
        return {
            "order_id": order_id,
            "appeal_no": appeal_no,
            "original_url": original_url,
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "download_url": f"/api/v1/sat/pdf/{order_id}",
            "file_size_bytes": len(pdf_bytes),
            "status": "downloaded",
            "error": None,
        }, new_token

    async def download_single(
        self,
        url_or_id: str,
        appeal_no: Optional[str] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Download one order PDF by URL or order ID."""
        order_id = self.extract_order_id(url_or_id)
        original_url = url_or_id if str(url_or_id).startswith("http") else None

        file_path = self.get_target_file_path(order_id, appeal_no)
        if file_path.exists() and file_path.stat().st_size > 0 and not force:
            return {
                "order_id": order_id,
                "appeal_no": appeal_no,
                "original_url": original_url,
                "file_name": file_path.name,
                "file_path": str(file_path.resolve()),
                "download_url": f"/api/v1/sat/pdf/{order_id}",
                "file_size_bytes": file_path.stat().st_size,
                "status": "already_exists",
                "error": None,
            }

        async with httpx.AsyncClient(headers=COMMON_HEADERS, follow_redirects=True, timeout=60.0) as client:
            token = await self.fetch_security_token(client)
            result, _ = await self.download_order_pdf(
                client=client,
                token=token,
                order_id=order_id,
                appeal_no=appeal_no,
                original_url=original_url,
                force=force,
            )
            return result

    async def download_batch(
        self,
        items: list[Union[str, dict[str, Any]]],
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Download multiple orders in a single session with rotating token.
        `items` can be a list of URLs or list of crawler record dicts.
        """
        results: list[dict[str, Any]] = []

        if not items:
            return {
                "status": "completed",
                "source": "sat",
                "total": 0,
                "downloaded": 0,
                "already_exists": 0,
                "failed": 0,
                "pdf_list": [],
            }

        async with httpx.AsyncClient(headers=COMMON_HEADERS, follow_redirects=True, timeout=60.0) as client:
            token = await self.fetch_security_token(client)

            for i, item in enumerate(items):
                url = None
                appeal_no = None

                if isinstance(item, str):
                    url = item
                elif isinstance(item, dict):
                    url = (
                        item.get("view_order_url")
                        or item.get("url")
                        or item.get("attachment_url")
                        or item.get("order_id")
                    )
                    appeal_no = item.get("appeal_no") or item.get("al_no")
                else:
                    results.append({
                        "order_id": None,
                        "status": "failed",
                        "error": f"Invalid item format: {type(item)}",
                    })
                    continue

                if not url:
                    results.append({
                        "order_id": None,
                        "status": "skipped",
                        "error": "Missing URL or order_id in record",
                    })
                    continue

                try:
                    order_id = self.extract_order_id(url)
                    result, token = await self.download_order_pdf(
                        client=client,
                        token=token,
                        order_id=order_id,
                        appeal_no=appeal_no,
                        original_url=url,
                        force=force,
                    )
                    results.append(result)
                except Exception as exc:
                    logger.exception(f"[SAT] Error processing item {url}: {exc}")
                    # In case token died or session expired, attempt token refresh for next items
                    try:
                        token = await self.fetch_security_token(client)
                    except Exception:
                        pass
                    results.append({
                        "order_id": None,
                        "original_url": url,
                        "status": "failed",
                        "error": str(exc),
                    })

                # Slight polite delay between requests
                if i < len(items) - 1:
                    await asyncio.sleep(0.3)

        downloaded = sum(1 for r in results if r.get("status") == "downloaded")
        already_exists = sum(1 for r in results if r.get("status") == "already_exists")
        failed = sum(1 for r in results if r.get("status") == "failed")

        return {
            "status": "completed",
            "source": "sat",
            "total": len(items),
            "downloaded": downloaded,
            "already_exists": already_exists,
            "failed": failed,
            "pdf_list": results,
        }

    async def download_from_file(self, file_path: str, force: bool = False) -> dict[str, Any]:
        """Read crawler JSON file from disk and download all orders."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("records", [])
        else:
            raise ValueError(f"Unrecognized JSON structure in {file_path}")

        result = await self.download_batch(items=items, force=force)
        return {
            "file_path": str(path.resolve()),
            **result,
        }


# Module instance
downloader = SATPDFDownloader()


# ==========================================================
# FASTAPI ROUTER (can be mounted in app/router/sat.py)
# ==========================================================

sat_pdf_router = APIRouter()


@sat_pdf_router.get("/health")
async def health():
    return {"status": "healthy", "service": "SAT PDF Downloader & Extractor"}


@sat_pdf_router.post("/extract-pdfs")
@sat_pdf_router.post("/pdf-list")
@sat_pdf_router.post("/download/batch")
async def extract_pdfs(request: Request):
    """
    Extract and download PDF files from SAT order links.
    Accepts:
      - Raw list of URLs: ["https://satweb.sat.gov.in/view-order/.../44189", ...]
      - Object with urls: {"urls": ["https://.../44189"]}
      - Crawler output: {"records": [{"view_order_url": "...", "appeal_no": "..."}]}
      - Single object: {"view_order_url": "https://.../44189"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    force = False
    items: list[Any] = []

    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        force = bool(body.get("force", False))
        if "records" in body and isinstance(body["records"], list):
            items = body["records"]
        elif "urls" in body and isinstance(body["urls"], list):
            items = body["urls"]
        elif "url" in body or "view_order_url" in body or "order_id" in body:
            items = [body]
        elif "file_path" in body:
            try:
                return await downloader.download_from_file(body["file_path"], force=force)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Download failed: {exc}")
        else:
            raise HTTPException(
                status_code=422,
                detail="Request must contain 'records', 'urls', 'url', 'view_order_url', or a list of URLs",
            )
    else:
        raise HTTPException(status_code=400, detail="Payload must be a JSON array or object")

    try:
        result = await downloader.download_batch(items=items, force=force)
        return result
    except Exception as exc:
        logger.exception("[SAT] Batch PDF extraction failed")
        raise HTTPException(status_code=502, detail=f"Extraction failed: {exc}")


@sat_pdf_router.post("/download")
@sat_pdf_router.post("/download/single")
async def download_single(request: SinglePDFRequest):
    """Download single SAT order PDF from URL or order ID."""
    target = request.url or request.view_order_url or request.order_id
    if not target:
        raise HTTPException(status_code=422, detail="Provide 'url', 'view_order_url', or 'order_id'")

    try:
        result = await downloader.download_single(
            url_or_id=target,
            appeal_no=request.appeal_no,
            force=request.force,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("[SAT] Single download failed")
        raise HTTPException(status_code=502, detail=f"Download failed: {exc}")


@sat_pdf_router.post("/download/from-file")
async def download_from_file(request: FilePDFRequest):
    """Read crawler JSON file saved on disk and download all attachments."""
    try:
        return await downloader.download_from_file(request.file_path, force=request.force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"File download failed: {exc}")


@sat_pdf_router.get("/pdf/{order_id}")
async def get_pdf_by_id(order_id: str, force: bool = Query(False)):
    """
    Directly serve or stream the SAT PDF for a given order ID.
    If already cached on disk, serves immediately; otherwise downloads and streams.
    """
    try:
        clean_id = downloader.extract_order_id(order_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    file_path = downloader.get_target_file_path(clean_id)
    if not file_path.exists() or force:
        try:
            await downloader.download_single(clean_id, force=force)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch PDF upstream: {exc}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found for order {clean_id}")

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=file_path.name,
    )


@sat_pdf_router.get("/pdf")
async def get_pdf_by_query(
    url: Optional[str] = Query(None, description="SAT view-order URL"),
    order_id: Optional[str] = Query(None, description="SAT order ID"),
    force: bool = Query(False),
):
    """Stream PDF by providing `url` or `order_id` in query parameters."""
    target = url or order_id
    if not target:
        raise HTTPException(status_code=400, detail="Provide 'url' or 'order_id' query param")

    try:
        clean_id = downloader.extract_order_id(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return await get_pdf_by_id(clean_id, force=force)


# ==========================================================
# STANDALONE FASTAPI APP
# ==========================================================

app = FastAPI(
    title="SAT PDF Downloader & Extractor",
    description=(
        "Downloads and extracts .pdf order files from SAT (Securities Appellate Tribunal) "
        "session-protected view-order URLs and provides direct download endpoints."
    ),
    version="1.0.0",
)

# Mount routes at both /api/v1/sat and root for maximum convenience
app.include_router(sat_pdf_router, prefix="/api/v1/sat", tags=["SAT PDF Extractor"])
app.include_router(sat_pdf_router, tags=["Root Endpoints"])


@app.get("/")
async def root():
    return {
        "service": "SAT PDF Downloader & Extractor",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "extract_pdfs_batch": "POST /api/v1/sat/extract-pdfs",
            "download_single": "POST /api/v1/sat/download",
            "download_from_file": "POST /api/v1/sat/download/from-file",
            "stream_pdf": "GET /api/v1/sat/pdf/{order_id} or GET /api/v1/sat/pdf?url=...",
        },
        "sample_curl": (
            "curl -X POST 'http://localhost:8083/api/v1/sat/extract-pdfs' "
            "-H 'Content-Type: application/json' "
            "-d '{\"urls\": [\"https://satweb.sat.gov.in/view-order/6beef1b6aadd650a5288deda0f6279e35c33f48e284efbc043bdaab2dd000cfa/44189\"]}'"
        ),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8083"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting SAT PDF Downloader on {host}:{port}...")
    uvicorn.run(app, host=host, port=port, reload=False)
