"""
Main application entry point for the Announcement Intelligence System.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import py_eureka_client.eureka_client as eureka_client
from api.routes import router
from api.neaps_api import router as neaps_router
from config import settings
from utils import setup_logging
from utils.encryption_middleware import EncryptionMiddleware
import os
os.environ["SSL_CERT_FILE"] = '/mnt/ssl_test/ca.crt'
os.environ["REQUESTS_CA_BUNDLE"] = '/mnt/ssl_test/ca.crt'
# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
logging.getLogger("py_eureka_client").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("=" * 60)
    logger.info("ANNOUNCEMENT INTELLIGENCE SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Embedding Model: {settings.embedding_model_name}")
    logger.info(f"Reranker Model: {settings.reranker_model_name}")
    logger.info(f"LLM: Ollama ({settings.ollama_model_search})")
    logger.info(f"Vector DB: Qdrant ({settings.qdrant_host}:{settings.qdrant_port})")
    logger.info(f"Dictionary: SQLite ({settings.sqlite_db_path})")
    logger.info(
        f"Query Expansion: {'Enabled' if settings.is_query_expansion_enabled else 'Disabled'}"
    )
    logger.info("=" * 60)

    # Initialize components
    # ORDER IS CRITICAL:
    #   1. Invalidate first (clears any stale objects from a previous reload)
    #   2. Eagerly pre-load AFTER invalidation so the first request is fast
    # Bug fix: previously we invalidated but never pre-loaded, so the first
    # /api/v1/search call had to pay the full 228s cold-load cost.
    from retrieval.hybrid_retriever import (
        get_hybrid_retriever,
        invalidate_hybrid_retriever,
    )
    from core.embeddings import invalidate_embedding_model

    # Step 1: Invalidate stale cached singletons
    invalidate_embedding_model()
    invalidate_hybrid_retriever()
    logger.info("[Startup] Stale component caches invalidated.")

    # Step 2: Eagerly pre-load all heavy models in thread executor
    # This eliminates the 228s cold-load on the very first search query.
    import asyncio as _asyncio
    _loop = _asyncio.get_running_loop()

    async def _preload_models():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = []

            # Embedding model (bge-large-en-v1.5) — ~30s cold load
            try:
                from core.embeddings import get_embedding_model
                futures.append(pool.submit(get_embedding_model))
                logger.info("[Startup] Queued: embedding model pre-load")
            except Exception as e:
                logger.warning(f"[Startup] Embedding pre-load queue failed: {e}")

            # Cross-encoder reranker (ms-marco-MiniLM) — ~5-8s cold load
            try:
                from core.reranker import get_reranker
                futures.append(pool.submit(get_reranker))
                logger.info("[Startup] Queued: reranker pre-load")
            except Exception as e:
                logger.warning(f"[Startup] Reranker pre-load queue failed: {e}")

            # BM25 sparse index (.pkl) — variable, depends on corpus size
            try:
                from embeddings.neaps_sparse import get_bm25_index
                futures.append(pool.submit(get_bm25_index))
                logger.info("[Startup] Queued: BM25 index pre-load")
            except Exception as e:
                logger.warning(f"[Startup] BM25 pre-load queue failed: {e}")

            # Wait for all to complete (or fail gracefully)
            for f in futures:
                try:
                    f.result(timeout=300)
                except Exception as ex:
                    logger.warning(f"[Startup] Pre-load task failed (non-fatal): {ex}")

    await _preload_models()
    logger.info("[Startup] All models pre-loaded — first query will be fast.")

    try:
        retriever = get_hybrid_retriever()
        stats = retriever.get_stats()
        logger.info(
            f"Vector Store: {stats.get('vector_store', {}).get('points_count', 0)} documents"
        )
        logger.info(
            f"BM25 Index: {stats.get('bm25', {}).get('num_documents', 0)} documents"
        )
        logger.info(
            f"Dictionary: {stats.get('dictionary', {}).get('total_entries', 0)} entries"
        )

        # Also log NEAPS collection
        try:
            from vector_store.neaps_qdrant_store import get_neaps_qdrant_store

            neaps_info = get_neaps_qdrant_store().get_collection_info()
            neaps_count = neaps_info.get("total_vectors", 0)
            logger.info(
                f"NEAPS Collection ({neaps_info.get('collection', 'neaps_docs')}): "
                f"{neaps_count} chunks (active={neaps_info.get('active_vectors', 0)})"
            )
        except Exception as ne:
            logger.debug(f"NEAPS collection not available: {ne}")

        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

    #  Start DB-backed job queue worker
    try:
        from workers.worker_manager import get_worker_manager

        _wm = get_worker_manager()
        _wm.start()
        logger.info(
            f"[JobQueue] WorkerManager started "
            f"(poll={settings.worker_poll_interval_s}s, "
            f"io_workers={settings.max_workers_io}, "
            f"cpu_workers={settings.max_workers_cpu})"
        )
    except Exception as _wm_err:
        logger.warning(
            f"[JobQueue] WorkerManager failed to start: {_wm_err}. Jobs will run synchronously."
        )
        _wm = None

    try:
        logger.info(f"Initiated Registered with Eureka at {settings.eureka_server_url} as {settings.eureka_app_name}")
        await eureka_client.init_async(
            eureka_server=settings.eureka_server_url,
            app_name=settings.eureka_app_name,
            instance_port=settings.api_port,
            instance_ip=settings.eureka_instance_ip,
        )
        logger.info(f"Registered with Eureka at {settings.eureka_server_url} as {settings.eureka_app_name}")
    except Exception as e:
        logger.error(f"Failed to register with Eureka: {e}")

    yield
    
    # Deregister from Eureka
    try:
        await eureka_client.stop_async()
        logger.info("Deregistered from Eureka")
    except Exception as e:
        logger.error(f"Error deregistering from Eureka: {e}")

    # Shutdown
    logger.info("Shutting down Announcement Intelligence System")
    if _wm is not None:
        _wm.stop(wait_s=30.0)
        logger.info("[JobQueue] WorkerManager stopped.")


# Create FastAPI app
app = FastAPI(
    title="Announcement Intelligence API",
    description="Hybrid retrieval system for NSE/NEAPS announcement analysis",
    version="1.0.0",
    lifespan=lifespan,
    root_path='/announcement-service'
)
from utils.encryption_middleware import decrypt_aes
app.add_middleware(EncryptionMiddleware)

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure as needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Add simplified request logging for user actions only
@app.middleware("http")
async def log_user_actions(request, call_next):
    """Log all HTTP requests with method, path, status and duration."""
    import time as time_module

    start_time = time_module.time()
    response = await call_next(request)
    duration_ms = (time_module.time() - start_time) * 1000

    status_flag = "[OK]" if response.status_code < 400 else "[FAIL]"
    logger.info(
        f"{status_flag} {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration_ms:.0f}ms"
    )
    return response


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["search"])
app.include_router(neaps_router)  # NEAPS ingestion + query endpoints

# Mount static files for document downloads
from fastapi.staticfiles import StaticFiles
import os

_neaps_dir = settings.neaps_local_storage or str(settings.base_dir / "neaps_downloads")
os.makedirs(_neaps_dir, exist_ok=True)
app.mount("/files", StaticFiles(directory=_neaps_dir), name="files")

# from fastapi.staticfiles import StaticFiles
# import os
# data_dir = os.path.join(os.path.dirname(__file__), "data")
# os.makedirs(data_dir, exist_ok=True)
# app.mount("/files", StaticFiles(directory=data_dir), name="files")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Announcement Intelligence API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        reload_excludes=[
            "logs/*",
            "*.log",
            "indexes/*",
            "*.pkl",
            "database/*",
            "*.db",
            "*.db-journal",
            "__pycache__/*",
            ".pytest_cache/*",
        ],
        log_level=settings.log_level.lower(),
    )
