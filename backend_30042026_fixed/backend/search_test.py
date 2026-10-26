import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# Add backend to path so we can import from config
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from vector_store.neaps_qdrant_store import get_neaps_qdrant_store
from embeddings.neaps_hybrid_embedder import embed_chunks_hybrid
import json

def run_search(query_text: str, limit: int = 3):
    print(f"Loading embedding model and encoding query: '{query_text}' ...")
    
    # We can reuse the hybrid embedder from the ingestion pipeline
    # We pass it as a single chunk to get the dense vector
    embs = embed_chunks_hybrid([query_text])
    if not embs:
        print("Failed to encode query.")
        return
        
    dense_vector = embs[0]["dense"]
    
    store = get_neaps_qdrant_store()
    print(f"Searching Qdrant collection '{store.collection_name}'...")
    
    results = store.client.query_points(
        collection_name=store.collection_name,
        query=dense_vector,
        using="dense",
        limit=limit,
        with_payload=True
    ).points

    print("\n" + "="*50)
    print("  SEARCH RESULTS")
    print("="*50)
    
    for i, res in enumerate(results, 1):
        # Handle both dict returns and ScoredPoint objects
        if isinstance(res, dict):
            score = res.get("score", 0.0)
            payload = res.get("payload", {})
        else:
            score = getattr(res, "score", 0.0)
            payload = getattr(res, "payload", {})
            
        page = payload.get("page_number", "N/A")
        text = payload.get("text", "").strip()
        
        print(f"\n[{i}] Score: {score:.4f} | Page: {page}")
        print("-" * 50)
        print(text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Qdrant Search")
    parser.add_argument("query", type=str, help="The search query text")
    parser.add_argument("--limit", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()
    
    run_search(args.query, args.limit)
