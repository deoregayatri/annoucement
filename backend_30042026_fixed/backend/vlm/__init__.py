# vlm/ — VLM (Vision Language Model) image processing pipeline.
# Sub-modules:
#   image_extractor   — extract embedded images from PDF pages via PyMuPDF
#   image_deduplicator — hash-based dedup + in-memory result cache
#   vlm_pipeline       — orchestrates extraction → dedup → VLM call
