# NEAPS Ingestion Pipeline Optimization Plan

This document details the critical architecture flaws, memory leaks, and performance bottlenecks identified in the current NEAPS ingestion pipeline, along with their respective solutions.

---

## 1. Resilience & Auto-Resume (Preventing Stuck Symbols)

**The Problem:** 
Currently, when a user starts processing a symbol from the UI, the API tracks this job in a temporary Python dictionary in memory (`_ingestion_tasks`). If the server reboots for midnight patching or crashes due to memory limits, this memory is wiped. The system "forgets" that the processing was running, leaving the process permanently stuck, and requiring the user to manually click "Process" again.

**The Fix:**
- **Wire the API to the Database:** In `api/neaps_api.py`, modify the `/start-ingestion` endpoint so it executes `get_job_store().insert_job(job_type="document_ingestion", payload={"symbol": "SKYGOLD"})`. This safely records the user's request into the robust Oracle `job_queue` table.
- **Remove In-Memory State:** Delete the fragile `_ingestion_tasks` dictionary entirely. The existing background `WorkerManager` will automatically detect the new row in the Oracle database and begin processing. 
- **Auto-Resume:** Because the job is stored in the database, if the server patches at midnight and restarts, the background worker will instantly resume the job where it left off automatically.

---

## 2. Worker Memory Leaks (The 16GB / 32GB Crash)

**The Problem:** 
The heavy AI embedding model (`bge-large`) and the PDF extraction libraries gradually leak RAM over many hours of continuous processing. Even with 16GB or 32GB of RAM, running for 9.5 hours causes the worker processes to bloat until the Operating System kills the server with an Out-Of-Memory (OOM) exception.

**The Fix:**
- **Worker Recycling (`maxtasksperchild`):** In `ingestion/neaps_ingestion_service.py`, replace `ProcessPoolExecutor` with Python's native `multiprocessing.Pool` and configure it with `maxtasksperchild=50`. This brilliant parameter forces the Operating System to silently destroy and recreate a worker process completely fresh after it processes 50 PDFs. This instantly flushes all bloated RAM back to the OS, completely eliminating the memory leak.
- **Unlock the Hard Cap:** In `workers/worker_manager.py`, change the hardcoded `hard_cap=6` limit to `hard_cap=12` so the code is actually allowed to fully utilize the new 32GB RAM server.
- **Bypass psutil Check (Offline Server):** Also in `worker_manager.py`, change the fallback from `2` workers to explicitly return `10` workers so the system doesn't accidentally throttle itself due to missing internet packages.

---

## 3. The N+1 Database Query (Massive Slowdown)

**The Problem:** 
To check if 1,000 PDFs for a single company are already processed, the duplicate checker opens an Oracle connection and runs a `SELECT` statement 1,000 separate times sequentially in a loop. This creates severe network latency and drastically slows down the start of ingestion.

**The Fix:** 
- In `ingestion/neaps_duplicate_checker.py`, create a new function `get_processed_sequences(symbols)`.
- Execute a single bulk query instead: `SELECT ANN_SEQ FROM AI_NEAPS_TRACKER WHERE ANN_SEQ IN (...) AND PROCESS_STATUS='COMPLETED'`. This reduces 1,000 network round-trips to just 1.

---

## 4. The IPC Memory Trap (Freezing the Main API)

**The Problem:** 
Worker processes are taking the parsed text from PDFs (Megabytes of data) and returning it back to the main process via `return {"chunks_data": enriched}`. This forces Python to serialize and push massive amounts of text data through a tiny Inter-Process Communication (IPC) pipe. This notoriously causes huge CPU spikes, extreme memory bloat on both sides, and freezes the main thread.

**The Fix:**
- Inside `_process_file_task_top_level`, instead of returning the heavy text data, save it to a temporary JSON file on the local disk (e.g., `temp/skygold_ann123.json`).
- Return *only* the file path string: `return {"status": "success", "file_path": "temp/skygold_ann123.json"}`.
- The main orchestrator can then rapidly read the file directly from disk when it is ready.

---

## 5. Main Process RAM Bloat (OOM Kill)

**The Problem:** 
The main orchestrator collects the text of *every single chunk* for the *entire company* (potentially hundreds of thousands of chunks) into a single array (`all_enriched_chunks`) before trying to update the BM25 index at the very end.

**The Fix:**
- Update the BM25 index incrementally (e.g., after every 20 PDFs finish) and clear the array, rather than hoarding all chunks in memory until the very end.

---

## 6. BM25 Disk Freezing & Qdrant Latency

**The Problem:** 
`save=True` is currently writing a Gigabyte-sized BM25 index to the hard drive every single time a company finishes, freezing the pipeline. Furthermore, pushing chunks to Qdrant only 100 at a time creates unnecessary network lag.

**The Fix:**
- **BM25 Saving:** Change `save=True` to `save=False` in `run_ingestion`. Create a scheduled background thread that periodically saves the BM25 index to disk every 15 minutes, decoupling the heavy disk write from the fast processing loop.
- **Qdrant Payload:** In `_process_file_task_top_level`, change `BATCH_SIZE = 100` to `BATCH_SIZE = 500` to reduce network round trips to the Qdrant database by 80%.

---

# PART B: UAT Bugfixes & Enhancements

**Deadline: UAT by 21st Sep | Live by 25th Sep**

---

## Bugfixes

### Bug 1. Stats column does not show companies under processing
**The Problem:** The stats column only shows the list of companies that have been fully processed. Companies currently being processed or queued are invisible.
**The Fix:** Update `/neaps/company-stats` and `/neaps/dashboard` in `neaps_api.py`. Modify the Oracle query to also return counts for `PENDING`, `PROCESSING`, and `FAILED` states, not just `COMPLETED`.

---

### Bug 2. Selected announcements are not summarized
**Example:** Reliance Industries announced IPO of Jio Platforms Ltd. on 19-Jun-2026. However, upon searching "JIO IPO", few announcements were selected but not summarized.
**The Fix:** In `neaps_api.py` → `_generate_answer()`, enforce a stricter system prompt: *"You MUST summarize ALL provided documents. Do not omit any provided context."* Also verify the context string is not being silently truncated before reaching the LLM.

---

### Bug 3. Selected news articles are not summarized
**Example:** Same as Bug 2 but for news articles — selected but not summarized.
**The Fix:** Same LLM prompt fix as Bug 2. Ensure news article chunks are included in the context passed to the LLM and not filtered out before summarization.

---

### Bug 4. News summarized do not provide relevant sources
**Example:** HDFC Bank Chairman Atanu Chakraborty resigned. The news insights summarized multiple articles but did not provide the sources. References were given to Article 6, Article 7, however, only 5 articles were provided.
**The Fix:** The LLM is hallucinating citations. Update the prompt in `_generate_answer()` to explicitly list the valid chunk IDs and instruct: *"ONLY cite documents from the list above. Do NOT reference any document not provided."*

---

### Bug 5. Error while unable to download PDF files
**The Problem:** Media article not found in output when PDF download fails.
**The Fix:** In `neaps_api.py` → `/download/{filename}`, improve exception handling to return a clear 404 JSON response instead of a crash stack trace. Add a fallback mechanism if the SFTP fetch fails.

---

### Bug 6. Chunks in source summary are up to 8-9 but PDF shows only 5
**The Problem:** The source list shows more chunks than what is actually displayed in the PDF viewer (top 10, basis latest one).
**The Fix:** Mismatch between `top_k` and `final_k`. In `query_neaps()`, strictly slice `reranked = reranked[:request.final_k]` to guarantee only the exact requested number of chunks reach the LLM and the source list.

---

### Bug 7. Post-patching python stopped — symbols lost
**The Problem:** If post patching python stopped, there is no way to process the symbols which user selected but python stopped.
**The Fix:** This is solved by **Point 1 (Auto-Resume)** above. Wiring the API to the Oracle `job_queue` table ensures that any symbol the user selected is persisted in the database and automatically resumed when the server restarts after patching.

---

### Bug 8. CMYK file processing
**The Problem:** PyMuPDF (`fitz`) sometimes fails to extract text properly from CMYK color-space PDFs.
**The Fix:** In `utils/document_processor.py`, add a fallback that forces the document to convert to RGB colorspace, or uses an OCR fallback (like Tesseract) if the extracted text is empty or garbled.

---

## Enhancements

### Enhancement 9. Relevant announcements not displayed
**Example:** Reliance Industries announced IPO of Jio Platforms Ltd. on 19-Jun-2026. However, upon searching "JIO IPO", the said announcement was not displayed in the "announcement intelligence".
**The Fix:** This is a search recall issue. Adjust the Hybrid Search weights (BM25 vs Dense) in `retrieval/neaps_hybrid_search.py` to give more weight to exact keyword matches (BM25) for specific terms like "JIO IPO".

---

### Enhancement 10. Timings are in milliseconds — convert to normal format
**The Problem:** All timing values displayed in the UI are in raw milliseconds, which is not user-friendly.
**The Fix:** In `neaps_api.py`, wherever `duration_ms`, `parse_ms` etc. are returned, convert them to a human-readable format (e.g., `"2m 35s"`) before sending the JSON response to the frontend.

---

### Enhancement 11. Confidence 0% still news is visible
**The Problem:** News items with 0% confidence score are still displayed to the user.
**The Fix:** In `neaps_api.py` → `query_neaps()`, filter out any retrieved chunks where `chunk.get('score', 0) == 0` or below a minimum threshold (e.g., 0.15) before sending them to the LLM or the frontend.

---

### Enhancement 12. Announcement Insights → header needed
**The Fix:** **Frontend Fix.** This requires a UI change to add a header component above the insights section. No backend change needed.

---

### Enhancement 13. Failed documents tracing and under processing symbols
**The Fix:** Add a new endpoint `/neaps/failed-documents` that queries `AI_NEAPS_TRACKER` for `PROCESS_STATUS = 'FAILED'` and returns the exact error messages and filenames so admins can trace failures. Also include "under processing" symbols in the dashboard API (ties into Bug 1).

---

### Enhancement 14. Scrollbar in ask question — drag based
**The Fix:** **Frontend Fix.** This requires a UI CSS/JS update to enable custom drag-based scrollbars in the chat/question component. No backend change needed.
