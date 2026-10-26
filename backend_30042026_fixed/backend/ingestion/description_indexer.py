import uuid
import requests
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "company_descriptions"
MODEL_PATH = "BAAI/bge-large-en"


class DescriptionIndexer:
    def __init__(self):
        print("🚀 Initializing Description Indexer...")
        # Force bypass proxy for localhost to prevent 403 errors
        os.environ["no_proxy"] = "localhost,127.0.0.1"

        self.client = QdrantClient(url="http://localhost:6333", timeout=60)
        self.model = SentenceTransformer(MODEL_PATH)
        self.session = requests.Session()

        # Add your NSE proxy/cookie headers here
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            # ... (your other headers) ...
        })

        # 🔥 The optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=500,
            length_function=len,
        )

        self._create_collection()

    def _create_collection(self):
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )

    def is_url_indexed(self, url):
        # Check if the URL already exists in Qdrant
        results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter={"must": [{"key": "url", "match": {"value": url}}]},
            limit=1
        )
        return len(results) > 0

    def fetch_description(self, url):
        # Fetch the actual text from the NSE API
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            # Assuming the API returns JSON with a 'description' field. Adjust if needed.
            return response.json().get('description', '')
        except Exception as e:
            print(f"⚠️ Failed to fetch URL {url}: {e}")
            return None

    def process_single(self, url, metadata_company, matched_company, symbol):
        if self.is_url_indexed(url):
            print(f"⏭️ Already indexed in Qdrant: {url}")
            return

        description = self.fetch_description(url)
        if not description:
            print(f"⚠️ No description found for: {url}")
            return

        # ✂️ Split the description into chunks
        chunks = self.text_splitter.split_text(description)
        points = []

        for i, chunk_text in enumerate(chunks):
            embedding = self.model.encode(chunk_text).tolist()
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}_chunk_{i}"))

            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "url": url,
                        "text": chunk_text,
                        "chunk_text": chunk_text,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "csv_company_name": metadata_company,
                        "matched_company_name": matched_company,
                        "symbol": symbol,
                        "company_symbol": symbol,
                        "source_type": "news",
                    }
                )
            )

        # 📦 Store all chunks in Qdrant
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(f"✅ Indexed {len(chunks)} chunks for: {url}")