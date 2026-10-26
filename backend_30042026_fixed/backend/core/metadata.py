"""
Core metadata definitions for the vocabulary backend.
Defines the structure of metadata extracted from legal and financial documents.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pypdf import PdfReader
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    """
    Metadata associated with a single document chunk.
    These fields are stored in the Qdrant payload for filtering.
    """
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    page_number: int = Field(..., description="Page number of the chunk")
    total_pages: int = Field(..., description="Total pages in the document")
    
    # Core Identity
    company_name: str = Field(..., description="Full legal name of the company")
    company_symbol: str = Field(..., description="Stock ticker symbol (e.g., TCS)")
    
    # Temporal Data
    announcement_date: str = Field(..., description="ISO 8601 Date (YYYY-MM-DD)")
    financial_year: Optional[str] = Field(None, description="Financial Year (e.g., FY24)")
    
    # Classification
    subject: str = Field(..., description="Subject of the announcement (e.g., 'Outcome of Board Meeting')")
    
    # Extra Context (Optional)
    chunk_index: int = Field(..., description="Index of this chunk in sequence")
    chunk_size: int = Field(..., description="Size of chunk in characters")

class ExtractionResult(BaseModel):
    """
    Raw output from the LLM extraction process.
    """
    company_name: str
    company_symbol: str
    subject: str
    document_date: Optional[str] = Field(None, description="Date found in document content")
