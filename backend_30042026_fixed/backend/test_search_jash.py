"""
Script to test hybrid search for symbol JASH.
Query: 'total operation cost'
"""

import sys
import os
import logging

# Configure basic logging so we can see pipeline logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Ensure backend is in python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from retrieval.neaps_hybrid_search import search

def run_search():
    query_text = "total operation cost"
    symbol = "JASH"
    
    print(f"Running search for query: '{query_text}' with symbol: '{symbol}'\n")
    
    # Run the hybrid search
    results = search(
        query=query_text,
        company_symbol=symbol,
        top_k=5,
        final_top_k=5,
    )
    
    print(f"\nFound {len(results)} results.")
    for idx, r in enumerate(results):
        print(f"\n--- Result {idx + 1} ---")
        print(f"Score: {r.get('score')}")
        print(f"Retrieval Method: {r.get('retrieval_method')}")
        text = r.get('text', '')
        print(f"Text snippet: {text[:300]}...")
        
        # Optional: print specific metadata
        meta = r.get('metadata', {})
        print(f"Filename: {meta.get('file_name', 'N/A')}")
        print(f"Chunk Index: {meta.get('chunk_index', 'N/A')}")
        print(f"Announcement Seq: {meta.get('ann_seq', 'N/A')}")

if __name__ == "__main__":
    run_search()
