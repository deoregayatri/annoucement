"""
API schemas using Pydantic.
"""

from typing import List, Dict, Any, Optional,Union
from pydantic import BaseModel, Field, field_validator
from utils.security import InputSanitizer

# ============== REQUEST SCHEMAS ==============

class SearchRequest(BaseModel):
    """Search request schema."""

    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(
        default=5, ge=1, le=50, description="Number of results"
    )
    content_type: Optional[str] = Field(default="general", description="Content type")
    use_reranker: Optional[bool] = Field(default=True, description="Use reranker")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters"
    )
    # Company and date filters
    companies: Optional[List[str]] = Field(
        default=None, description="Filter by companies"
    )
    date_from: Optional[str] = Field(
        default=None, description="Filter from date (YYYY-MM-DD)"
    )
    date_to: Optional[str] = Field(
        default=None, description="Filter to date (YYYY-MM-DD)"
    )
    use_graph: Optional[bool] = Field(
        default=False, description="Use LangGraph RAG agent"
    )
    company_id: Optional[str] = Field(
        default=None, description="Company identifier (graph mode)"
    )

    # ADD after line: company_id: Optional[str] = Field(...)
    # search_mode: Optional[str] = Field(
    #     default="announcements",
    #     description="Which corpus to search: 'announcements', 'news', or 'both'"
    # )

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_search_query(cls, v: str) -> str:
        try:
            return InputSanitizer.sanitize_query(v)
        except Exception:
            # If sanitization fails, return empty string to allow request to proceed
            return "" if v is None else str(v)

class AddDocumentsRequest(BaseModel):
    """Add documents request schema."""

    documents: List[str] = Field(..., description="List of document texts")
    metadata: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Document metadata"
    )
    extract_vocabulary: bool = Field(default=True, description="Extract vocabulary")
    doc_ids: Optional[List[str]] = Field(
        default=None, description="Optional document IDs"
    )

class VocabularyQueryRequest(BaseModel):
    """Vocabulary query request schema."""

    term: str = Field(..., description="Term to lookup")

class VocabularyExtractRequest(BaseModel):
    """Request schema for manual vocabulary extraction."""

    document_text: str = Field(..., description="Document text to analyze")
    doc_id: Optional[str] = Field(default=None, description="Optional document ID")
    domain: str = Field(
        default="auto", description="Domain hint or 'auto' for detection"
    )

class VocabularyUpdateRequest(BaseModel):
    """Request schema for updating vocabulary entry."""

    canonical_term: str = Field(..., description="Term to update")
    short_forms: Optional[List[str]] = None
    long_forms: Optional[List[str]] = None
    synonyms: Optional[List[str]] = None
    verified: Optional[bool] = None
    notes: Optional[str] = None

# ============== RESPONSE SCHEMAS ==============

class SearchResult(BaseModel):
    """Single search result."""

    id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float
    bm25_score: float
    dictionary_boost: float
    fused_score: float
    rerank_score: Optional[float] = None

class AnswerSource(BaseModel):
    """Source chunk used for answer generation."""

    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dictionary_boost: Optional[float] = None

class SearchResponse(BaseModel):
    """Search response schema."""

    query: str
    results: List[SearchResult]
    total_results: int
    expanded_query: Optional[Dict[str, Any]] = None
    retrieval_time_ms: float
    # Answer generation fields (optional)
    answer: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[List[AnswerSource]] = None
    reasoning: Optional[str] = None
    generation_time_ms: Optional[float] = None
    score_breakdown: Optional[Dict[str, float]] = None
    guardrail_warning: Optional[str] = None
    # Explicit signal when search yields nothing — UI should surface this
    no_results_reason: Optional[str] = None
    # Graph/intelligence mode fields
    executive_summary: Optional[str] = None
    neaps_intelligence: Optional[str] = None
    latencies_ms: Optional[Dict[str, float]] = None
    query_id: Optional[str] = None
    # ADD at the end of SearchResponse, after query_id field
    # news_results: Optional[List[SearchResult]] = None
    # news_answer: Optional[str] = None

class AddDocumentsResponse(BaseModel):
    """Add documents response schema."""

    documents_added: int
    vector_ids: List[str]
    extraction_metadata: Optional[List[Dict[str, Any]]] = None

class SystemStatsResponse(BaseModel):
    """System statistics response."""

    vector_store: Dict[str, Any]
    bm25: Dict[str, Any]
    dictionary: Dict[str, Any]
    embedding_cache: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "1.0.0"
    services: Dict[str, bool]

class UploadResponse(BaseModel):
    """File upload response schema."""

    success: bool
    message: str
    filename: str
    chunks_created: int
    documents_added: int
    extraction_metadata: Optional[List[Dict[str, Any]]] = None
    # Deduplication fields
    already_exists: bool = False
    file_hash: Optional[str] = None

class DocumentCountResponse(BaseModel):
    """Document count statistics."""

    vector_db_count: int
    bm25_count: int
    vector_db_chunks: int
    bm25_chunks: int

class DocumentInfo(BaseModel):
    """Document information."""

    filename: str
    file_type: str
    file_size_mb: float
    total_chunks: int

class DocumentListResponse(BaseModel):
    """Document list response."""

    documents: List[DocumentInfo]
    total_documents: int

class CompaniesListResponse(BaseModel):
    """Companies list response."""

    companies: List[str]
    total: int

# ============== COMPLIANCE SCHEMAS ==============

class ComplianceTimelineRequest(BaseModel):
    """Compliance timeline request schema."""

    query: str = Field(..., description="Query for timeline reconstruction")

    @field_validator("query", mode="before")
    @classmethod
    def sanitize_compliance_query(cls, v: str) -> str:
        try:
            return InputSanitizer.sanitize_query(v)
        except Exception:
            # If sanitization fails, return empty string to allow request to proceed
            return "" if v is None else str(v)

class ComplianceTimelineResponse(BaseModel):
    """Compliance timeline response schema."""

    report: str
    refined_query: str
    company_symbol: str
    latencies_ms: Dict[str, float]
    total_documents_analyzed: int
    loops_executed: int

 