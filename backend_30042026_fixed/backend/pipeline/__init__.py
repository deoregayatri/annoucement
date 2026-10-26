"""
Pipeline package — centralized document processing orchestrator.
"""

from pipeline.document_pipeline import DocumentPipeline, process_document

__all__ = ["DocumentPipeline", "process_document"]
