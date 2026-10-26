# Comprehensive Ingestion Pipeline Optimization Plan

This document serves as the master blueprint for fixing crashes, memory leaks, and performance bottlenecks in the NEAPS ingestion pipeline. 

## 1. Resilience & Auto-Resume (Preventing Stuck Symbols)

**The Problem:** The API tracks running tasks in a Python dictionary. If the server patches at midnight or crashes, this memory is wiped, and users have to click "Process" manually again.
**The Fix:**
- **Wire the API to the DB:** In `api/neaps_api.py`, change the `/start-ingestion` endpoint to execute `get_job_store().insert_job(job_type="document_ingestion", payload={"symbol": "SKYGOLD"})`.
- **Remove In-Memory State:** Delete the `_ingestion_tasks` dictionary entirely. The existing `WorkerManager` will automatically see the new row in Oracle and start processing. If a crash happens, it will auto-resume upon restart.

## 2. Worker Memory Leaks (The 16GB / 32GB Crash)

**The Problem:** The `bge-large` model and PDF parsers gradually leak RAM over 9 hours, causing the server to crash (OOM).
**The Fix:**
- **Worker Recycling:** In `ingestion/neaps_ingestion_service.py`, switch from `ProcessPoolExecutor` to `multiprocessing.Pool` and explicitly pass `maxtasksperchild=50`. This tells the OS to silently destroy and recreate a worker after 50 PDFs, instantly freeing all its bloated RAM.
- **Unlock the Hard Cap:** In `workers/worker_manager.py`, change `hard_cap=6` to `hard_cap=12`. 
- **Bypass psutil Check (Offline Server):** Also in `worker_manager.py`, under `except ImportError:`, change the fallback from `2` workers to explicitly return `10` workers so your new 32GB RAM server is actually utilized.

## 3. The N+1 Database Query (Massive Slowdown)

**The Problem:** To check if 1,000 PDFs are already processed, the system opens an Oracle connection and runs `SELECT ...` 1,000 separate times sequentially.
**The Fix:** 
- In `ingestion/neaps_duplicate_checker.py`, create a new function `get_processed_sequences(symbols)`. 
- Have it execute a single bulk query: `SELECT ANN_SEQ FROM AI_NEAPS_TRACKER WHERE ANN_SEQ IN (...) AND PROCESS_STATUS='COMPLETED'`.
- In `run_ingestion`, filter the `pending` list using this bulk result instead of calling `.is_processed()` in a loop.

## 4. The IPC Memory Trap (Freezing the Main API)

**The Problem:** Worker processes are sending Megabytes of parsed text chunks back to the main process via `return {"chunks_data": enriched}`. This forces Python to serialize and push massive data through a tiny Inter-Process Communication pipe, causing memory spikes and freezing.
**The Fix:**
- Inside `_process_file_task_top_level`, instead of returning the text data, save it to a temporary JSON file on disk (e.g., `temp/skygold_ann123.json`).
- Return *only* the file path: `return {"status": "success", "file_path": "temp/skygold_ann123.json"}`.
- The main orchestrator reads the file directly from disk when it's ready.

## 5. Main Process RAM Bloat (OOM Kill)

**The Problem:** The main process collects the text of every single chunk for the entire company (potentially hundreds of thousands of chunks) into a single array `all_enriched_chunks` before updating BM25 at the very end.
**The Fix:**
- **Batch BM25 Updates:** Update the BM25 index incrementally (e.g., after every 20 PDFs finish) and clear the array, rather than hoarding all chunks in memory until the very end.

## 6. BM25 Disk Freezing & Qdrant Latency

**The Problem:** `save=True` writes a Gigabyte-sized BM25 index to disk after every single company finishes, freezing the pipeline. Furthermore, pushing 100 chunks at a time to Qdrant creates unnecessary network lag.
**The Fix:**
- **BM25 Saving:** Change `save=True` to `save=False` in `run_ingestion`. Create a scheduled background thread that periodically saves the BM25 index to disk every 15 minutes, decoupling the disk write from the processing loop.
- **Qdrant Payload:** In `_process_file_task_top_level`, change `BATCH_SIZE = 100` to `BATCH_SIZE = 500` to reduce network round trips by 80%.

---

## PART B: UAT Bugfixes & Enhancements (Deadline: UAT by 21st Sep, Live by 25th Sep)

### Bugfixes

**Bug 1. Stats column does not show companies under processing.**
- **Backend Fix:** Update `/neaps/company-stats` and `/neaps/dashboard` in `neaps_api.py`. Currently only shows `COMPLETED` companies. Modify the Oracle query to also return counts for `PENDING`, `PROCESSING`, and `FAILED` states.

**Bug 2. Selected announcements are not summarized** (e.g., searching "JIO IPO" shows announcements but they are not summarized).
- **Backend Fix:** In `neaps_api.py` → `_generate_answer()`, enforce a stricter system prompt: *"You MUST summarize ALL provided documents. Do not omit any provided context."* Also verify the context string is not being silently truncated before reaching the LLM.

**Bug 3. Selected news articles are not summarized** (same issue as Bug 2 but for news articles).
- **Backend Fix:** Same LLM prompt fix as Bug 2. Ensure news article chunks are included in the context passed to the LLM and not filtered out before summarization.

**Bug 4. News summarized do not provide relevant sources** (e.g., references Article 6, Article 7 but only 5 articles were provided).
- **Backend Fix:** The LLM is hallucinating citations. Update the prompt in `_generate_answer()` to explicitly list the valid chunk IDs and instruct: *"ONLY cite documents from the list above. Do NOT reference any document not provided."*

**Bug 5. Error while unable to download PDF files. Media article not found in output.**
- **Backend Fix:** In `neaps_api.py` → `/download/{filename}`, improve exception handling to return a clear 404 JSON response instead of a crash. Add a fallback mechanism if the SFTP fetch fails.

**Bug 6. Chunks in source summary are up to 8-9 but PDF shows only 5 (top 10, basis latest one).**
- **Backend Fix:** Mismatch between `top_k` and `final_k`. In `query_neaps()`, strictly slice `reranked = reranked[:request.final_k]` to guarantee only the exact requested number of chunks reach the LLM and the source list.

**Bug 7. If post-patching python stopped, check how we can process the symbols which user selected but python stopped.**
- **Backend Fix:** This is solved by **Point 1 (Auto-Resume)**. Wiring the API to the Oracle `job_queue` table ensures that any symbol the user selected is persisted in the database and automatically resumed when the server restarts after patching.

**Bug 8. CMYK file processing.**
- **Backend Fix:** PyMuPDF (`fitz`) sometimes fails to extract text from CMYK color-space PDFs. In `utils/document_processor.py`, add a fallback that forces the document to convert to RGB colorspace, or uses an OCR fallback (like Tesseract) if the extracted text is empty or garbled.

### Enhancements

**Enhancement 9. Relevant announcements not displayed** (e.g., searching "JIO IPO" in announcement intelligence does not show the relevant Reliance announcement).
- **Backend Fix:** This is a search recall issue. Adjust the Hybrid Search weights (BM25 vs Dense) in `retrieval/neaps_hybrid_search.py` to give more weight to exact keyword matches (BM25) for specific terms like "JIO IPO".

**Enhancement 10. Timings are in milliseconds — convert to normal format.**
- **Backend Fix:** In `neaps_api.py`, wherever `duration_ms`, `parse_ms` etc. are returned, convert them to a human-readable format (e.g., `"2m 35s"`) before sending the JSON response to the frontend.

**Enhancement 11. Confidence 0% still news is visible.**
- **Backend Fix:** In `neaps_api.py` → `query_neaps()`, filter out any retrieved chunks where `chunk.get('score', 0) == 0` or below a minimum threshold (e.g., 0.15) before sending them to the LLM or the frontend.

**Enhancement 12. Announcement Insights → header needed.**
- **Frontend Fix:** This requires a UI change to add a header component above the insights section. No backend change needed.

**Enhancement 13. Failed documents tracing and under processing symbols.**
- **Backend Fix:** Add a new endpoint `/neaps/failed-documents` that queries `AI_NEAPS_TRACKER` for `PROCESS_STATUS = 'FAILED'` and returns the exact error messages and filenames. Also include "under processing" symbols in the dashboard API (ties into Bug 1).

**Enhancement 14. Scrollbar in ask question — drag based.**
- **Frontend Fix:** This requires a UI CSS/JS update to enable custom drag-based scrollbars in the chat/question component. No backend change needed.
