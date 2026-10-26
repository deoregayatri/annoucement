import asyncio
import aiohttp
import time
import json
import argparse
from typing import List

# Default test queries representing common company searches
TEST_QUERIES = [
    "What are the latest financial results for Tata Motors?",
    "Show me board meeting outcomes for Reliance Industries",
    "Did Infosys announce any dividends recently?",
    "Resignation of directors at Wipro",
    "Quarterly earnings report for HDFC Bank",
    "New product launches by ITC Limited",
    "Acquisitions or mergers involving L&T",
    "Shareholding pattern updates for TCS"
]

async def send_search_request(session: aiohttp.ClientSession, url: str, query: str, worker_id: int):
    """Send a single search request and track latency."""
    start_time = time.time()
    payload = {
        "query": query,
        "top_k": 5
    }
    
    try:
        async with session.post(url, json=payload) as response:
            status = response.status
            # Read response to ensure it completed
            data = await response.json()
            latency = time.time() - start_time
            
            # Simple check if answer was generated (if answer generation is enabled)
            has_answer = bool(data.get("answer"))
            
            return {
                "worker": worker_id,
                "status": status,
                "latency": latency,
                "success": status == 200,
                "has_answer": has_answer,
                "retrieved_chunks": len(data.get("results", []))
            }
    except Exception as e:
        latency = time.time() - start_time
        return {
            "worker": worker_id,
            "status": "ERROR",
            "latency": latency,
            "success": False,
            "error": str(e)
        }

async def run_load_test(url: str, concurrent_users: int, total_requests: int):
    """Run the concurrent load test."""
    print(f"Starting Load Test...")
    print(f"Target URL: {url}")
    print(f"Concurrent Users: {concurrent_users}")
    print(f"Total Requests: {total_requests}")
    print("-" * 50)
    
    start_time = time.time()
    results = []
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        # Create tasks for all requests
        tasks = []
        for i in range(total_requests):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            tasks.append(send_search_request(session, url, query, i))
            
            # If we hit the concurrency limit, wait for the batch to finish
            if len(tasks) >= concurrent_users:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                tasks = []
                print(f"Completed {len(results)} / {total_requests} requests...")
                
        # Process any remaining tasks
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"Completed {len(results)} / {total_requests} requests...")

    end_time = time.time()
    print_report(results, end_time - start_time)

def print_report(results: List[dict], total_time: float):
    """Print the load test summary report."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if successful:
        avg_latency = sum(r["latency"] for r in successful) / len(successful)
        max_latency = max(r["latency"] for r in successful)
        min_latency = min(r["latency"] for r in successful)
        answers_generated = sum(1 for r in successful if r.get("has_answer"))
    else:
        avg_latency = max_latency = min_latency = answers_generated = 0

    print("\n" + "=" * 50)
    print("LOAD TEST RESULTS")
    print("=" * 50)
    print(f"Total Time Taken:    {total_time:.2f} seconds")
    print(f"Total Requests:      {len(results)}")
    print(f"Successful Requests: {len(successful)} ({(len(successful)/len(results))*100:.1f}%)")
    print(f"Failed/Errors:       {len(failed)} ({(len(failed)/len(results))*100:.1f}%)")
    print(f"Requests Per Second: {len(results) / total_time:.2f} req/s")
    print("-" * 50)
    print(f"Average Latency:     {avg_latency:.2f} seconds")
    print(f"Min Latency:         {min_latency:.2f} seconds")
    print(f"Max Latency:         {max_latency:.2f} seconds")
    print(f"Answers Generated:   {answers_generated} / {len(successful)}")
    print("=" * 50)
    
    if failed:
        print("\nSample Errors:")
        for r in failed[:5]:
            print(f"- Status: {r.get('status')} | Error: {r.get('error', 'N/A')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Load Tester for Hybrid Search")
    parser.add_argument("--url", type=str, default="http://localhost:8092/api/v1/search", help="Target API URL")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("-n", "--requests", type=int, default=50, help="Total number of requests to make")
    
    args = parser.parse_args()
    
    # Check if aiohttp is installed
    try:
        import aiohttp
    except ImportError:
        print("Error: aiohttp library is required. Install it using: pip install aiohttp")
        exit(1)
        
    asyncio.run(run_load_test(args.url, args.concurrency, args.requests))
