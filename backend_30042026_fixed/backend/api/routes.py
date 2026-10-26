"""
API routes aggregator - thin facade that combines all sub-routers into
the single `router` object that main.py imports.

The monolithic 1,132-line routes.py has been split into:
  api/routers/search_router.py     - POST /search, POST /search/v2
  api/routers/document_router.py   - POST /documents, POST /upload, GET /documents/*, DELETE /company/...
  api/routers/stats_router.py      - GET /health, GET /stats, GET /companies
  api/routers/ingestion_router.py  - POST /ingest/*, GET /jobs/status
  api/routers/compliance_router.py - POST /compliance-timeline

main.py continues to import `router` from this module - no external change required.
"""

import logging

from fastapi import APIRouter

from api.routers.search_router     import router as _search_router
from api.routers.document_router   import router as _document_router
from api.routers.stats_router      import router as _stats_router
from api.routers.ingestion_router  import router as _ingestion_router
from api.routers.compliance_router import router as _compliance_router
from api.routers.cancel_router     import router as _cancel_router
from api.routers.news_router_bkpp       import router as _news_router

logger = logging.getLogger(__name__)

# Single aggregated router exported for main.py
router = APIRouter()

router.include_router(_search_router)
router.include_router(_document_router)
router.include_router(_stats_router)
router.include_router(_ingestion_router)
router.include_router(_compliance_router)
router.include_router(_cancel_router)
router.include_router(_news_router)

logger.info("[API] All sub-routers registered: search, document, stats, ingestion, compliance, cancel, news")
