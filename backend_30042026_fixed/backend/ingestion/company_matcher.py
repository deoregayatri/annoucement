import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from ingestion.neaps_oracle_client import NeapsOracleClient

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_PATH = "/mnt/Models/HF_bge_large"
COLLECTION_NAME = "company_match"

REBUILD_INDEX = False  # ⚠️ keep False after first run


# ---------------------------------------------------------
# FETCH DATA
# ---------------------------------------------------------

def fetch_tbl_symbols():
    client = NeapsOracleClient()
    conn = client._get_conn()

    if conn is None:
        print("❌ DB connection failed")
        return []

    cur = conn.cursor()

    cur.execute("""
        SELECT symb_symbol, symb_company_name
        FROM tbl_symbolss
    """)

    rows = cur.fetchall()
    cur.close()

    print(f"✅ Loaded {len(rows)} companies from DB")

    return rows


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

def normalize_name(name: str):
    import re

    name = name.lower()

    # ---------------------------------------------------------
    # Expand common abbreviations
    # ---------------------------------------------------------
    replacements = {
        r"\bltd\.?\b": "limited",
        r"\bpvt\.?\b": "private",
        r"\bco\.?\b": "company",
        r"\bfin\b": "financial",
        r"\bserv\b": "services",
        r"\bind\b": "india",
        r"\bi\b": "india",
    }

    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)

    # ---------------------------------------------------------
    # Handle brackets like (I)
    # ---------------------------------------------------------
    name = re.sub(r"\(\s*i\s*\)", " india ", name)

    # ---------------------------------------------------------
    # Remove punctuation
    # ---------------------------------------------------------
    name = re.sub(r"[^\w\s]", " ", name)

    # ---------------------------------------------------------
    # Remove extra spaces
    # ---------------------------------------------------------
    name = re.sub(r"\s+", " ", name)

    return name.strip()


# ---------------------------------------------------------
# MATCHER
# ---------------------------------------------------------

class CompanyMatcher:

    def __init__(self):
        print("🚀 Loading embedding model...")
        self.model = SentenceTransformer(MODEL_PATH)

        self.client = QdrantClient(host="localhost", port=6333)

        self.meta_map = {}
        self.db_names = []

        self._prepare_index()

    # ---------------------------------------------------------
    # BUILD / LOAD INDEX
    # ---------------------------------------------------------
    def _prepare_index(self):
        rows = fetch_tbl_symbols()

        if not rows:
            print("❌ No data fetched -> matcher disabled")
            return

        self.meta_map = {
            i: {"symbol": r[0], "name": r[1]}
            for i, r in enumerate(rows) if r[1]
        }

        self.db_names = [r[1] for r in rows if r[1]]

        existing = [c.name for c in self.client.get_collections().collections]

        # ✅ Skip rebuild if already exists
        if COLLECTION_NAME in existing and not REBUILD_INDEX:
            print("✅ Using existing Qdrant index")
            return

        print("⚙️ Building fresh Qdrant index...")

        if COLLECTION_NAME in existing:
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            )
        )

        print("⏳ Encoding embeddings (first run only)...")

        embeddings = self.model.encode(
            self.db_names,
            batch_size=16,
            convert_to_numpy=True
        )

        print("📤 Uploading vectors...")

        BATCH_SIZE = 200

        for i in range(0, len(self.db_names), BATCH_SIZE):
            batch = [
                PointStruct(
                    id=j,
                    vector=embeddings[j].tolist(),
                    payload=self.meta_map[j]
                )
                for j in range(i, min(i + BATCH_SIZE, len(self.db_names)))
            ]

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )

            print(f"Uploaded {i} - {i + len(batch)}")

        print("✅ Index ready")

    # ---------------------------------------------------------
    # MATCH FUNCTION
    # ---------------------------------------------------------
    def match(self, company_name: str, threshold=0.85):
        norm = normalize_name(company_name)

        # ---------------------------------------------------------
        # 1. EXACT MATCH
        # ---------------------------------------------------------
        for meta in self.meta_map.values():
            if normalize_name(meta["name"]) == norm:
                return [{
                    "matched_name": meta["name"],
                    "symbol": meta["symbol"],
                    "score": 1.0,
                    "type": "exact"
                }]

        # ---------------------------------------------------------
        # 2. TOKEN MATCH
        # ---------------------------------------------------------
        input_tokens = set(norm.split())

        best = None
        best_overlap = 0

        for meta in self.meta_map.values():
            db_tokens = set(normalize_name(meta["name"]).split())
            overlap = len(input_tokens & db_tokens)

            if overlap > best_overlap:
                best_overlap = overlap
                best = meta

        if best:
            token_score = best_overlap / max(len(input_tokens), len(db_tokens))

            if token_score >= 0.90:
                return [{
                    "matched_name": best["name"],
                    "symbol": best["symbol"],
                    "score": round(token_score, 2),
                    "type": "token"
                }]

        # ---------------------------------------------------------
        # 3. SEMANTIC MATCH
        # ---------------------------------------------------------
        vec = self.model.encode(company_name, convert_to_numpy=True).tolist()

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=1
        )

        if results.points:
            hit = results.points[0]

            if hit.score >= threshold:
                return [{
                    "matched_name": hit.payload["name"],
                    "symbol": hit.payload["symbol"],
                    "score": round(hit.score, 2),
                    "type": "semantic"
                }]

        # ❌ No confident match
        return []