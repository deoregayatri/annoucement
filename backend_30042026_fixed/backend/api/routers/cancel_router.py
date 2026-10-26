"""
cancel_router.py
================
Endpoint to cancel an in-progress query task.

POST /api/v1/cancel-task
  Body: { "task_id": "<query_task_id>" }
  Returns:
    200: { "success": true, "cancelled": true/false, "message": "..." }
    404: if task_id is unknown

IMPORTANT:
  This router ONLY manages query task cancellations stored in
  `search_router._query_tasks`.  It does NOT touch the ingestion
  executor or `neaps_api._ingestion_tasks` — those are fully isolated.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class CancelTaskRequest(BaseModel):
    task_id: str = Field(..., description="Query task_id returned by /search")


class CancelTaskResponse(BaseModel):
    success: bool
    task_id: str
    cancelled: bool
    already_done: bool = False
    message: str


@router.post("/cancel-task", response_model=CancelTaskResponse)
async def cancel_query_task(request: CancelTaskRequest):
    """
    Cancel an active query task by its task_id.

    Only cancels RAG-graph query tasks — ingestion tasks are unaffected.
    If the task has already completed, returns already_done=True.
    """
    from api.routers.search_router import _query_tasks, _query_tasks_lock

    task_id = request.task_id.strip()
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="task_id must not be empty",
        )

    with _query_tasks_lock:
        task = _query_tasks.get(task_id)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query task not found: {task_id}",
        )

    task_status = task.get("status", "unknown")

    # Already finished — nothing to cancel
    if task_status in {"completed", "failed", "cancelled"}:
        logger.info(
            f"[CancelRouter] task_id={task_id[:8]} already in terminal state={task_status}"
        )
        return CancelTaskResponse(
            success=True,
            task_id=task_id,
            cancelled=False,
            already_done=True,
            message=f"Task already in state: {task_status}",
        )

    # Signal cancellation via the threading.Event stored in the task
    cancel_event = task.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()

    # Mark task as cancelled in the registry
    with _query_tasks_lock:
        if task_id in _query_tasks:
            _query_tasks[task_id]["status"] = "cancelled"

    logger.info(f"[CancelRouter] Cancelled query task_id={task_id[:8]}")
    return CancelTaskResponse(
        success=True,
        task_id=task_id,
        cancelled=True,
        already_done=False,
        message="Query task cancellation signal sent.",
    )
