# Production Priority Fixes & Action Plan
**Target Architecture:** 4 LLM Backend Servers + 1 Load Balancer + FastSearch API / Ingestion

---

## Executive Summary & Fix Priorities

| Priority | Component | Issue | Fix | Estimated Effort | Impact |
|---|---|---|---|---|---|
| **P1 (Immediate)** | `llm_client.py` | 4 LLM servers underutilized; hardcoded to 2 workers; burst delay | Scale concurrency to 8–10, increase HTTP pool to 30, remove token-bucket delay | 5 Mins | **Unblocks concurrent users** (drops max delay from 134s to <40s) |
| **P1 (Immediate)** | `neaps_api.py` | First query takes 40–200s (cold start) | Pre-warm dense & sparse embedding models on server startup | 5 Mins | **Eliminates cold-start freeze** on first user search |
| **P1 (Immediate)** | `worker_manager.py` | Ingestion workers risk 100% CPU/RAM starvation | Set worker `hard_cap=6`, reserve 4GB RAM for live user search | 2 Mins | **Prevents server lockups & OOM kills** |
| **P2 (High)** | `neaps_api.py` | Every query makes 2 sequential LLM calls | Merge routing & summarization into 1 unified prompt | 15 Mins | **Cuts single-user latency by ~50%** (from 40s to ~20s) |
| **P2 (High)** | `neaps_api.py` | Zero-chunk / low-confidence queries hang or return empty | Add immediate fallback response when retrieved chunks = 0 | 10 Mins | **Prevents silent failures & user timeouts** |
| **P3 (Stability)** | Ingestion Pipeline | Memory leaks during long-running PDF batch jobs | Worker recycling (`max_tasks_per_child=50`) and bulk DB checks | 30 Mins | Long-term pipeline stability |

---

## 🔴 Priority 1: Immediate Hotfixes (Apply First)

### Fix 1.1: Unlock 4-LLM Server Concurrency & Connection Pool
**File:** `backend/agents/llm_client.py`

#### Changes:
1. **Increase HTTP connection pool for Load Balancer (Line 86–90):**
```python
# OLD:
limits=httpx.Limits(max_connections=8, max_keepalive_connections=4, keepalive_expiry=60.0)

# NEW:
limits=httpx.Limits(max_connections=30, max_keepalive_connections=15, keepalive_expiry=60.0)
```

2. **Scale concurrency to utilize all 4 LLM nodes (Line 104 & 117):**
```python
# OLD:
_current_concurrency: int = 2

# NEW:
_current_concurrency: int = 8  # 2 concurrent streams per LLM server x 4 servers
```

3. **Remove artificial Token-Bucket throttling (Line 155):**
```python
# OLD:
def __init__(self, rate: float = 0.5, capacity: int = 4):

# NEW:
def __init__(self, rate: float = 10.0, capacity: int = 20):
```

---

### Fix 1.2: Pre-Warm Search Models on Startup
**File:** `backend/api/neaps_api.py`

In the FastAPI startup event / `lifespan`:
```python
# Add inside lifespan / startup function:
try:
    from embeddings.neaps_dense import get_dense_embedder
    from embeddings.neaps_sparse import get_sparse_embedder
    
    logger.info("[Startup] Pre-warming embedding models...")
    get_dense_embedder()
    get_sparse_embedder()
    logger.info("[Startup] Embedding models ready in RAM.")
except Exception as e:
    logger.warning(f"[Startup] Pre-warm warning: {e}")
```

---

### Fix 1.3: Ingestion Worker CPU/RAM Safety Cap
**File:** `backend/workers/worker_manager.py` (Lines 52–58)

```python
def _compute_worker_count(
    ram_per_worker_gb: float = 1.6,
    system_reserve_gb: float = 4.0,  # Reserve 4GB for live user searches
    cpu_headroom: float = 0.60,      # Leave 40% CPU cores free for search API
    hard_cap: int = 6,               # Cap ingestion at 6 workers max
    min_workers: int = 1,
) -> int:
```

---

## ⚡ Linux Quick-Deployment Script for Priority 1

Run these commands directly on the production host:

```bash
#!/bin/bash
set -e

echo "=== Applying Priority 1 Hotfixes ==="

# 1. Patch llm_client.py
sed -i 's/_current_concurrency: int = 2/_current_concurrency: int = 8/g' backend/agents/llm_client.py
sed -i 's/max_connections=8/max_connections=30/g' backend/agents/llm_client.py
sed -i 's/rate: float = 0.5, capacity: int = 4/rate: float = 10.0, capacity: int = 20/g' backend/agents/llm_client.py

# 2. Patch worker_manager.py
sed -i 's/hard_cap: int = 12/hard_cap: int = 6/g' backend/workers/worker_manager.py
sed -i 's/system_reserve_gb: float = 2.0/system_reserve_gb: float = 4.0/g' backend/workers/worker_manager.py

# 3. Restart Backend Service
echo "Restarting backend service..."
systemctl restart backend || pm2 restart backend

echo "=== Priority 1 Hotfixes Applied Successfully ==="
```

---

## 🟡 Priority 2: Latency Reduction & Edge Case Handling

### Fix 2.1: Consolidate Double LLM Calls into One Prompt
**File:** `backend/api/neaps_api.py`

* **Issue:** Every query executes 2 round-trips to the LLM (1 classification/routing call + 1 answer generation call).
* **Fix:** Combine context extraction and answer formatting into a single prompt.
* **Benefit:** Reduces average single-user query response time from **~40.8s to ~20–22s**.

### Fix 2.2: Handle Zero-Chunk / Low Confidence Edge Cases
**File:** `backend/api/neaps_api.py`

When hybrid search returns 0 chunks:
```python
if not retrieved_chunks:
    return {
        "status": "success",
        "answer": "No relevant announcements or documents were found for this query.",
        "sources": [],
        "confidence": 0.0,
        "duration_ms": elapsed_ms
    }
```

---

## 🟢 Priority 3: Long-Term Stability (Batch Pipeline)

1. **Worker Process Recycling:** Ensure `max_tasks_per_child=50` is active in `ProcessPoolExecutor` to clear leaked RAM after 50 PDFs.
2. **Oracle Bulk Duplicate Check:** Replace N+1 queries in `neaps_duplicate_checker.py` with `SELECT ANN_SEQ WHERE ANN_SEQ IN (...)`.
3. **Decouple BM25 Disk Writes:** Save the BM25 index on a background 15-minute timer rather than blocking the ingestion process after every file.

---

## 🧪 Post-Fix Verification Plan

After applying Priority 1, execute the verification runner against the API:

```bash
# Test 1: Verify single user response time (Target: < 25s, no cold-start delay)
python3 uat_scenario_runner.py --url http://127.0.0.1:8000 --scenario 1

# Test 2: Verify 10 concurrent users across the 4 LLM nodes (Target: Max < 45s, 0 timeouts)
python3 uat_scenario_runner.py --url http://127.0.0.1:8000 --scenario 2 --workers 10
```
