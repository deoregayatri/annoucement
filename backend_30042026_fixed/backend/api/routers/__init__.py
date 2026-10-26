"""
API routers package — aggregates all sub-routers.
"""

from api.routers.search_router     import router as search_router
from api.routers.document_router   import router as document_router
from api.routers.stats_router      import router as stats_router
from api.routers.ingestion_router  import router as ingestion_router
from api.routers.compliance_router import router as compliance_router

__all__ = [
    "search_router",
    "document_router",
    "stats_router",
    "ingestion_router",
    "compliance_router",
]
