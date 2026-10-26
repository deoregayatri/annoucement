"""
Compliance router - POST /compliance-timeline endpoint.

Runs the LangGraph Compliance Agent for market manipulation timeline detection.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from api.schemas import ComplianceTimelineRequest, ComplianceTimelineResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/compliance-timeline", response_model=ComplianceTimelineResponse)
async def compliance_timeline(request: ComplianceTimelineRequest):
    """
    Execute the LangGraph Compliance Agent.

    Uses iterative sufficiency evaluation and chronological sorting to reconstruct
    a timeline of events that may indicate market manipulation risk.

    Returns: { report, refined_query, company_symbol, latencies_ms,
                total_documents_analyzed, loops_executed }
    """
    from agents.compliance_graph import run_compliance_query

    try:
        result = run_compliance_query(request.query)
        return ComplianceTimelineResponse(
            report=result.get("report", ""),
            refined_query=result.get("refined_query", ""),
            company_symbol=result.get("company_symbol", ""),
            latencies_ms=result.get("latencies_ms", {}),
            total_documents_analyzed=result.get("total_documents_analyzed", 0),
            loops_executed=result.get("loops_executed", 0),
        )
    except Exception as exc:
        logger.error(f"[/compliance-timeline] Failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
