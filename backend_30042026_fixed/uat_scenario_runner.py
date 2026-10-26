#!/usr/bin/env python3
"""
uat_scenario_runner.py
======================
Generates production-equivalent scenarios in UAT to surface:

  SCENARIO 1 - Cold-Start Timing
      Hits the API immediately after server restart.
      Catches the 228s embedding cold-load before Fix 2 is applied.

  SCENARIO 2 - Concurrent Semaphore Stress (10 parallel users)
      Fires 10 simultaneous /neaps/query requests.
      Triggers the asyncio Semaphore deadlock (Fix 1) if not patched.

  SCENARIO 3 - Zero-Chunk Edge Cases
      Sends queries designed to return 0 chunks (unusual tickers, empty context).
      Catches the score-filter bug (Fix 3).

  SCENARIO 4 - Sustained Load (30 sequential requests, 2s apart)
      Simulates 30 users over ~1 minute.
      Surfaces memory leaks, BM25 slowdown, worker saturation.

  SCENARIO 5 - Mixed Concurrent Load (3 companies, 3 users each)
      Replicates real production traffic patterns.

Usage:
    python3 uat_scenario_runner.py --url http://UAT_HOST:8000
    python3 uat_scenario_runner.py --url http://UAT_HOST:8000 --scenario 2
    python3 uat_scenario_runner.py --url http://UAT_HOST:8000 --company RELIANCE
"""

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

try:
    import requests
except ImportError:
    print("[ERROR] requests not installed. Run: pip install requests")
    sys.exit(1)

DEFAULT_URL     = "http://localhost:8000"
REQUEST_TIMEOUT = 300
NEAPS_ENDPOINT  = "/neaps/query"

QUERIES = [
    "What are the key financial highlights for the latest quarter?",
    "Summarize the recent board of directors changes",
    "What dividend has been declared in the last announcement?",
    "Is there any merger or acquisition activity mentioned?",
    "What are the risk factors disclosed in recent filings?",
    "Summarize the latest annual report findings",
    "Has the company issued any new shares recently?",
    "What is the current debt position of the company?",
    "Are there any regulatory penalties or notices?",
    "Provide a summary of recent NEAPS exchange filings",
]

EDGE_CASE_QUERIES = [
    "JIO IPO latest details",
    "xyznonexistent123 announcement",
    "What is the weather forecast for Mumbai?",
    "Insider trading investigation status",
    "Quarterly results for subsidiary ABC Ltd",
]

COMPANIES = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO"]


def build_payload(query, company, final_k=5):
    return {"query": query, "companies": [company], "top_k": 20, "final_k": final_k, "use_reranker": True}


def call_api(url, payload, label=""):
    t0 = time.perf_counter()
    result = {"label": label, "query": payload.get("query","")[:60],
              "company": payload.get("companies",["?"])[0],
              "status": None, "http_code": None, "latency_s": None,
              "chunks": None, "error": None, "answer_len": None, "success": False}
    try:
        resp = requests.post(url + NEAPS_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT,
                             headers={"Content-Type": "application/json"})
        result["latency_s"] = round(time.perf_counter() - t0, 2)
        result["http_code"] = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            result["status"]  = "OK"
            result["success"] = True
            result["chunks"]  = len(data.get("sources", [])) or data.get("chunk_count", 0)
            answer = data.get("answer") or data.get("executive_summary") or ""
            result["answer_len"] = len(answer)
            if "error" in answer.lower() or len(answer) < 20:
                result["status"] = "EMPTY_ANSWER"
                result["success"] = False
        else:
            result["status"] = f"HTTP_{resp.status_code}"
            result["error"]  = resp.text[:200]
    except requests.exceptions.Timeout:
        result["latency_s"] = round(time.perf_counter() - t0, 2)
        result["status"] = "TIMEOUT"
        result["error"]  = f"Timed out after {REQUEST_TIMEOUT}s"
    except Exception as exc:
        result["latency_s"] = round(time.perf_counter() - t0, 2)
        result["status"] = "EXCEPTION"
        result["error"]  = str(exc)[:200]
    return result


def print_result(r, prefix="  "):
    icon = "OK  " if r["success"] else "FAIL"
    chunks_str = f"chunks={r['chunks']}" if r['chunks'] is not None else "chunks=?"
    err = f"  ERR:{r['error'][:60]}" if r["error"] else ""
    print(f"{prefix}[{icon}] [{r['label']:12s}] {str(r['status']):15s} {r['latency_s']:6.1f}s  {chunks_str}{err}")


def summarise(results):
    ok  = [r for r in results if r["success"]]
    lat = [r["latency_s"] for r in results if r["latency_s"] is not None]
    cks = [r["chunks"]    for r in results if r["chunks"] is not None]
    lat_sorted = sorted(lat)
    p95_idx = max(0, int(0.95 * len(lat_sorted)) - 1) if lat_sorted else 0
    return {
        "total":       len(results),
        "success":     len(ok),
        "failed":      len(results) - len(ok),
        "success_pct": round(100 * len(ok) / max(len(results), 1), 1),
        "lat_min":     round(min(lat), 2)            if lat else None,
        "lat_max":     round(max(lat), 2)            if lat else None,
        "lat_p50":     round(statistics.median(lat), 2) if lat else None,
        "lat_p95":     round(lat_sorted[p95_idx], 2) if lat_sorted else None,
        "chunks_avg":  round(statistics.mean(cks), 1) if cks else None,
        "chunks_zero": sum(1 for c in cks if c == 0),
    }


def verdict(s, thresholds):
    issues = []
    if s["success_pct"] < thresholds.get("min_success_pct", 80):
        issues.append(f"Success {s['success_pct']}% < {thresholds['min_success_pct']}%")
    if s["lat_p95"] and s["lat_p95"] > thresholds.get("max_p95_s", 30):
        issues.append(f"P95={s['lat_p95']}s > {thresholds['max_p95_s']}s")
    if s["chunks_zero"] > thresholds.get("max_zero_chunks", 0):
        issues.append(f"{s['chunks_zero']} requests got 0 chunks")
    return "PASS" if not issues else "FAIL -- " + " | ".join(issues)


def _print_summary(s, v):
    print(f"""
  +-- Summary -----------------------------------------------------------+
  |  Total:{s['total']:3d}  Success:{s['success']:3d} ({s['success_pct']:5.1f}%)  Failed:{s['failed']:3d}            |
  |  Latency  min={str(s['lat_min'])+'s':8s}  p50={str(s['lat_p50'])+'s':8s}  p95={str(s['lat_p95'])+'s':8s}  max={str(s['lat_max'])+'s':8s} |
  |  Chunks   avg={str(s['chunks_avg']):5s}   zero-chunk requests={s['chunks_zero']:3d}                  |
  |  Verdict: {v:<60s}|
  +----------------------------------------------------------------------+""")


def check_health(url):
    try:
        r = requests.get(url + "/", timeout=10)
        if r.status_code == 200:
            print(f"  [OK] Server reachable: {url}")
            return True
        print(f"  [FAIL] HTTP {r.status_code}")
        return False
    except Exception as exc:
        print(f"  [FAIL] Cannot reach server: {exc}")
        return False


# ── Scenarios ──────────────────────────────────────────────────────────────

def scenario_1_cold_start(url, company):
    print("\n" + "="*65)
    print("  SCENARIO 1 -- Cold-Start Timing")
    print("  Catches: 228s embedding cold-load (Fix 2)")
    print("  Pass   : First response < 30s (model pre-warmed at startup)")
    print("="*65)
    payload = build_payload("Summarize the latest board announcement", company)
    print(f"  Single request -> {company} ...")
    r = call_api(url, payload, label="cold-start")
    print_result(r)
    if r["latency_s"]:
        if r["latency_s"] > 60:
            print(f"\n  [WARNING] {r['latency_s']}s -- embedding model NOT pre-warmed. Deploy Fix 2 (main.py).")
        else:
            print(f"\n  [OK] {r['latency_s']}s -- model is pre-warmed correctly.")
    s = summarise([r])
    v = verdict(s, {"min_success_pct": 100, "max_p95_s": 30, "max_zero_chunks": 0})
    _print_summary(s, v)
    return s


def scenario_2_concurrent_semaphore(url, company, workers=10):
    print("\n" + "="*65)
    print(f"  SCENARIO 2 -- Concurrent Semaphore Stress ({workers} parallel users)")
    print("  Catches: asyncio Semaphore deadlock -> 300s timeouts (Fix 1)")
    print("  Pass   : All requests finish < 60s, no 300s hangs")
    print("="*65)
    tasks = [(url, build_payload(QUERIES[i % len(QUERIES)], company), f"user-{i+1:02d}") for i in range(workers)]
    results = []
    print(f"  Firing {workers} requests simultaneously...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(call_api, *t): t[2] for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print_result(r)
    wall = round(time.perf_counter() - t0, 2)
    print(f"\n  Wall-clock time for {workers} concurrent requests: {wall}s")
    deadlocks = [r for r in results if r["latency_s"] and r["latency_s"] > 120]
    if deadlocks:
        print(f"\n  [DEADLOCK] {len(deadlocks)} requests took >120s -- Semaphore bug active. Deploy Fix 1.")
    s = summarise(results)
    v = verdict(s, {"min_success_pct": 80, "max_p95_s": 60, "max_zero_chunks": 2})
    _print_summary(s, v)
    return s


def scenario_3_zero_chunk_edge_cases(url, company):
    print("\n" + "="*65)
    print("  SCENARIO 3 -- Zero-Chunk Edge Cases")
    print("  Catches: score<0.01 filter drops all chunks -> empty answers (Fix 3)")
    print("  Pass   : Fallback activates, 0 requests get 0 chunks")
    print("="*65)
    results = []
    for i, q in enumerate(EDGE_CASE_QUERIES):
        r = call_api(url, build_payload(q, company), label=f"edge-{i+1}")
        results.append(r)
        print_result(r)
        time.sleep(0.5)
    zero = [r for r in results if r["chunks"] == 0]
    if zero:
        print(f"\n  [WARNING] {len(zero)} queries got 0 chunks. Fix 3 (neaps_api.py fallback) needed.")
    s = summarise(results)
    v = verdict(s, {"min_success_pct": 60, "max_p95_s": 60, "max_zero_chunks": 0})
    _print_summary(s, v)
    return s


def scenario_4_sustained_load(url, company, count=30, interval_s=2.0):
    print("\n" + "="*65)
    print(f"  SCENARIO 4 -- Sustained Load ({count} requests, {interval_s}s apart)")
    print("  Catches: BM25 O(n) slowdown, memory leaks, worker saturation (Fix 8)")
    print("  Pass   : Latency stays stable over time (no progressive slowdown)")
    print("="*65)
    results = []
    for i in range(count):
        q = QUERIES[i % len(QUERIES)]
        r = call_api(url, build_payload(q, company), label=f"req-{i+1:03d}")
        results.append(r)
        if (i+1) % 5 == 0 or not r["success"]:
            print_result(r, prefix=f"  [{i+1:3d}/{count}] ")
        time.sleep(interval_s)
    first10 = [r["latency_s"] for r in results[:10]  if r["latency_s"]]
    last10  = [r["latency_s"] for r in results[-10:] if r["latency_s"]]
    if first10 and last10:
        delta = round(statistics.mean(last10) - statistics.mean(first10), 2)
        if delta > 5:
            print(f"\n  [WARNING] Progressive slowdown: +{delta}s. Check BM25 / memory (Fix 8).")
        else:
            print(f"\n  [OK] Latency stable (delta={delta:+.1f}s over {count} requests).")
    s = summarise(results)
    v = verdict(s, {"min_success_pct": 90, "max_p95_s": 45, "max_zero_chunks": 1})
    _print_summary(s, v)
    return s


def scenario_5_mixed_concurrent(url, users_per_company=3):
    total = users_per_company * len(COMPANIES)
    print("\n" + "="*65)
    print(f"  SCENARIO 5 -- Mixed Concurrent Load ({users_per_company} users x {len(COMPANIES)} companies = {total} requests)")
    print("  Catches: Multi-tenant race conditions, company filter bugs")
    print("  Pass   : >80% success, P95 < 45s")
    print("="*65)
    tasks = []
    for ci, co in enumerate(COMPANIES):
        for ui in range(users_per_company):
            q = QUERIES[(ci * users_per_company + ui) % len(QUERIES)]
            tasks.append((url, build_payload(q, co), f"{co[:3]}-u{ui+1}"))
    results = []
    print(f"  Firing {total} mixed requests simultaneously...")
    with ThreadPoolExecutor(max_workers=total) as pool:
        futures = {pool.submit(call_api, *t): t[2] for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            print_result(r)
    s = summarise(results)
    v = verdict(s, {"min_success_pct": 80, "max_p95_s": 45, "max_zero_chunks": 2})
    _print_summary(s, v)
    return s


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NEAPS UAT Scenario Runner")
    parser.add_argument("--url",      default=DEFAULT_URL)
    parser.add_argument("--company",  default="RELIANCE")
    parser.add_argument("--scenario", type=int, default=0, help="1-5 or 0=all")
    parser.add_argument("--workers",  type=int, default=10)
    parser.add_argument("--count",    type=int, default=30)
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  NEAPS UAT SCENARIO RUNNER")
    print(f"  Target  : {args.url}")
    print(f"  Company : {args.company}")
    print(f"  Time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65)

    if not check_health(args.url):
        print("\n  Start the UAT server first, then re-run.")
        sys.exit(1)

    run_all = args.scenario == 0
    summaries = {}

    if run_all or args.scenario == 1:
        summaries["1_cold_start"] = scenario_1_cold_start(args.url, args.company)
    if run_all or args.scenario == 2:
        summaries["2_concurrent"] = scenario_2_concurrent_semaphore(args.url, args.company, args.workers)
    if run_all or args.scenario == 3:
        summaries["3_zero_chunk"] = scenario_3_zero_chunk_edge_cases(args.url, args.company)
    if run_all or args.scenario == 4:
        summaries["4_sustained"]  = scenario_4_sustained_load(args.url, args.company, args.count)
    if run_all or args.scenario == 5:
        summaries["5_mixed"]      = scenario_5_mixed_concurrent(args.url)

    print("\n" + "="*65)
    print("  FINAL REPORT")
    print("="*65)
    for name, s in summaries.items():
        icon = "[PASS]" if s["success_pct"] >= 80 else "[FAIL]"
        print(f"  {icon} {name:20s}  success={s['success_pct']:5.1f}%  p95={str(s['lat_p95'])+'s':8s}  zero_chunks={s['chunks_zero']}")
    print()

if __name__ == "__main__":
    main()
