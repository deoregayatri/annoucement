"""
Stats / health router.

Endpoints:
  GET /health    - service health check
  GET /stats     - system statistics (vector store, BM25, embeddings)
"""

import logging

from fastapi import APIRouter, HTTPException, status

from api.schemas import (
    SystemStatsResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
health_logger = logging.getLogger("health")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check - returns healthy/degraded/unhealthy based on component availability."""
    try:
        from workers.system_state import get_system_state

        busy, reason, since = get_system_state().is_busy()
        if busy:
            health_logger.info(f"Health check: ingestion running ({reason})")

        from retrieval.hybrid_retriever import get_hybrid_retriever

        retriever = get_hybrid_retriever()
        stats = retriever.get_stats()

        neaps_ok = False
        try:
            from vector_store.neaps_qdrant_store import get_neaps_qdrant_store

            neaps_info = get_neaps_qdrant_store().get_collection_info()
            neaps_ok = neaps_info.get("status") == "active"
        except Exception:
            pass

        services = {
            "qdrant": stats["vector_store"].get("status") == "green" or neaps_ok,
            "bm25": stats["bm25"].get("index_size_bytes", 0) >= 0,
            "dictionary": stats["dictionary"].get("total_entries", 0) > 0,
            "embeddings": True,
            "ingestion_idle": not busy,
        }

        core_ok = services["qdrant"] and services["bm25"] and services["embeddings"]
        overall = "healthy" if (core_ok and not busy) else "degraded"
        health_logger.info(f"Health check: {overall} | {services}")
        return HealthResponse(status=overall, services=services)

    except Exception as e:
        health_logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            services={
                "qdrant": False,
                "bm25": False,
                "dictionary": False,
                "embeddings": False,
            },
        )


@router.get("/stats", response_model=SystemStatsResponse)
async def get_stats():
    """Return system statistics from the hybrid retriever."""
    try:
        from retrieval.hybrid_retriever import get_hybrid_retriever

        retriever = get_hybrid_retriever()
        stats = retriever.get_stats()

        # Augment with NEAPS collection details
        try:
            from vector_store.neaps_qdrant_store import get_neaps_qdrant_store

            neaps_info = get_neaps_qdrant_store().get_collection_info()
            stats["vector_store"]["neaps_collection"] = neaps_info.get(
                "collection", "neaps_docs"
            )
            stats["vector_store"]["neaps_total_vectors"] = neaps_info.get(
                "total_vectors", 0
            )
            stats["vector_store"]["neaps_active_vectors"] = neaps_info.get(
                "active_vectors", 0
            )
        except Exception as ne:
            logger.debug(f"NEAPS stats unavailable: {ne}")

        return SystemStatsResponse(**stats)

    except Exception as e:
        logger.error(f"[/stats] Failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )
