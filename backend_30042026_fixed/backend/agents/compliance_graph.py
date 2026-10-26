"""
LangGraph Compliance Agent for Market Manipulation Timeline Detection.

This advanced Agentic RAG pipeline enforces Chronological Sorting and
Context Sufficiency loops to detect vague 'hints' dropped before official announcements.

Note: Uses a low LLM temperature (0.1) for deterministic compliance analysis.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from agents.llm_utils import _strip_think_tags  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class ComplianceGraphState(TypedDict, total=False):
    # Input
    original_query: str
    
    # NER & Revisions
    company_symbol: str
    action_intent: str
    refined_query: str
    
    # Loop & Search params
    top_k: int
    loop_count: int
    
    # Data & Evaluation
    raw_chunks: List[Dict[str, Any]]
    is_sufficient: bool
    evaluator_feedback: str
    
    # Output
    sorted_chunks: List[Dict[str, Any]]
    timeline_report: str
    
    latencies_ms: Dict[str, float]
    error: Optional[str]

# ---------------------------------------------------------------------------
# LLM Helper — uses shared utility but overrides temperature for compliance
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, json_format: bool = False) -> str:
    """
    Compliance-specific LLM caller with low temperature (0.1) for deterministic output.
    Wraps agents.llm_client.call_llm_sync with overridden temperature.
    """
    try:
        from agents.llm_client import call_llm_sync
        return call_llm_sync(prompt, json_format=json_format, temperature=0.1)
    except Exception as exc:
        logger.error(f"[Compliance LLM] Call failed: {exc}")
        return "{}" if json_format else f"[LLM Error: {exc}]"

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def query_rewriter_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """Uses LLM to perform NER, correct spelling, and structure the search."""
    t0 = time.perf_counter()
    query = state.get("original_query", "")

    prompt = f"""You are an expert financial named entity recognizer (NER).
Analyze the following user query: "{query}"

Extract the target company symbol (e.g., RELIANCE, TCS, POONAWALLA) and the core action or product intent they are asking about (e.g., "Entering Commercial Vehicles", "Acquisition of startup").
Return your answer ONLY in the following JSON format:
{{
    "company_symbol": "TICKER",
    "action_intent": "The resolved action intent without typos",
    "refined_search_query": "company_symbol action_intent"
}}
Ensure the refined_search_query is perfect for a vector database keyword + semantic search.
"""
    response_json = _call_llm(prompt, json_format=True)
    
    try:
        parsed = json.loads(response_json)
        state["company_symbol"] = parsed.get("company_symbol", "")
        state["action_intent"] = parsed.get("action_intent", query)
        state["refined_query"] = parsed.get("refined_search_query", query)
    except Exception as e:
        logger.warning(f"[QueryRewriter] Failed to parse JSON: {e}")
        state["company_symbol"] = ""
        state["action_intent"] = query
        state["refined_query"] = query

    logger.info(f" [QueryRewriter] User Query: '{query}'")
    logger.info(f" [QueryRewriter] NER Executing Strict Search: '{state['refined_query']}'")

    state["loop_count"] = 0
    state["top_k"] = 15
    state["latencies_ms"] = {"query_rewriter_ms": (time.perf_counter() - t0) * 1000}
    return state


def retriever_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """Executes the hybrid search on NEAPS Qdrant."""
    t0 = time.perf_counter()
    try:
        from retrieval.neaps_hybrid_search import search as neaps_search
        
        # We start with state["top_k"] which increases if the loop triggers
        top_k = state.get("top_k", 15)
        company = state.get("company_symbol", None)
        if not company: company = None
        
        chunks = neaps_search(
            query=state["refined_query"],
            company_symbol=company,
            top_k=top_k * 2,
            final_top_k=top_k
        )
        state["raw_chunks"] = chunks
        logger.info(f" [ComplianceRetriever] Retrieved {len(chunks)} chunks (top_k={top_k}).")
        
    except Exception as exc:
        logger.error(f"[ComplianceRetriever] Failed: {exc}")
        state["raw_chunks"] = []

    lat = state.get("latencies_ms", {})
    lat["retriever_node_ms"] = (time.perf_counter() - t0) * 1000
    state["latencies_ms"] = lat
    return state


def sufficiency_evaluator_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """Evaluates if the timeline is complete (contains both early hint and explicit action)."""
    t0 = time.perf_counter()
    chunks = state.get("raw_chunks", [])
    
    if not chunks:
        state["is_sufficient"] = True # No chunks, stop looping
        return state

    # Extract brief context for LLM to decide
    brief_context = "\n".join([f"[{c.get('metadata', {}).get('announcement_date', 'Unknown')}] {c.get('text', '')[:200]}..." for c in chunks])
    
    prompt = f"""You are a compliance evaluator checking if we have retrieved the full timeline spanning context for an event.
Query Intent: {state['action_intent']} for {state['company_symbol']}

We need to reconstruct a timeline to catch market manipulation. This requires BOTH:
1. The earliest possible mention/hint of the intent (even if vague).
2. The final official formal announcement of the intent.

Here are the snippets of retrieved documents:
{brief_context}

Evaluate if the context contains enough chronological variance to cover both the early hint and the final launch.
Reply strictly in JSON:
{{
    "is_sufficient": true/false,
    "feedback": "Why it is or isn't sufficient"
}}
"""
    resp = _call_llm(prompt, json_format=True)
    try:
        parsed = json.loads(resp)
        state["is_sufficient"] = parsed.get("is_sufficient", True)
        state["evaluator_feedback"] = parsed.get("feedback", "")
    except Exception:
        state["is_sufficient"] = True
    
    logger.info(f" [SufficiencyEvaluator] Sufficient Data? {state['is_sufficient']} | Feedback: {state.get('evaluator_feedback')}")
    
    lat = state.get("latencies_ms", {})
    lat["evaluator_ms"] = (time.perf_counter() - t0) * 1000
    state["latencies_ms"] = lat
    return state


def expansion_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """Increases retrieval radius if context is insufficient."""
    state["top_k"] = state.get("top_k", 15) + 20
    state["loop_count"] = state.get("loop_count", 0) + 1
    logger.info(f" [ExpansionNode] Context insufficient. Increasing top_k to {state['top_k']} (Loop {state['loop_count']}).")
    return state


def temporal_sorter_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """Sorts retrieved chunks by chronological announcement date."""
    chunks = state.get("raw_chunks", [])
    
    # Simple deduplication by text
    seen = set()
    deduped = []
    for c in chunks:
        txt = c.get("text", "")[:200]
        if txt not in seen:
            seen.add(txt)
            deduped.append(c)

    # Sort primarily by announcement_date (oldest to newest)
    def extract_date(meta):
        # We expect ISO dates like '2023-12-05'
        val = meta.get("announcement_date", "")
        return val if isinstance(val, str) else ""

    deduped.sort(key=lambda x: extract_date(x.get("metadata", {})))
    
    state["sorted_chunks"] = deduped
    logger.info(f" [TemporalSorter] Sorted {len(deduped)} chunks chronologically.")
    return state


def analyzer_node(state: ComplianceGraphState) -> ComplianceGraphState:
    """Generates the final compliance timeline report."""
    t0 = time.perf_counter()
    chunks = state.get("sorted_chunks", [])
    
    if not chunks:
        state["timeline_report"] = "No relevant official filings discovered for the query."
        return state

    context = ""
    for idx, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        dt = meta.get("announcement_date", "Unknown Date")
        pg = meta.get("page_number", "Unspecified")
        src = meta.get("source_type", "Document")
        
        context += f"\n--- [Date: {dt} | Page {pg} | {src}] ---\n{c.get('text', '')}\n"

    prompt = f"""You are a senior regulatory compliance detective specializing in detecting stock market manipulation.
Your task is to analyze the following chronologically-ordered snippets from official exchange fillings for {state['company_symbol']} regarding {state['action_intent']}.

Focus on identifying:
1. When the company first gave a vague, indirect "hint" about their intentions.
2. When the company finally made the official, concrete announcement.
3. The time gap between the hint and the action, and any potential market manipulation risk this phrasing style presents.

DOCUMENTS (Oldest to Newest):
{context}

Generate a formal COMPLIANCE TIMELINE REPORT using this exact markdown structure:

###  Executive Summary
[Brief verdict on whether manipulation risks or problematic disclosure gaps exist]

###  The Timeline
[Bulleted list of events in chronological order. YOU MUST explicitly cite the Exact Date and the Exact Page Number (e.g. `[Oct 14, 2023 - Page 4]`) for every single point].

###  Risk Assessment
[Analysis of the time gap between the earliest hints and final decisions, describing the risk of artificial price inflation due to vague disclosures]
"""
    state["timeline_report"] = _call_llm(prompt)
    logger.info(f" [AnalyzerNode] Generated report ({len(state['timeline_report'])} chars).")
    
    lat = state.get("latencies_ms", {})
    lat["analyzer_ms"] = (time.perf_counter() - t0) * 1000
    state["latencies_ms"] = lat
    return state

# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

def should_loop(state: ComplianceGraphState) -> str:
    """Conditions to loop back or proceed."""
    if state.get("is_sufficient", True):
        return "proceed"
    if state.get("loop_count", 0) >= 2: # Max 2 expansions to prevent infinite loops
        return "proceed"
    return "expand"

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_compliance_graph():
    from langgraph.graph import StateGraph, END
    
    builder = StateGraph(ComplianceGraphState)
    
    builder.add_node("query_rewriter", query_rewriter_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("evaluator", sufficiency_evaluator_node)
    builder.add_node("expansion", expansion_node)
    builder.add_node("temporal_sorter", temporal_sorter_node)
    builder.add_node("analyzer", analyzer_node)
    
    builder.set_entry_point("query_rewriter")
    builder.add_edge("query_rewriter", "retriever")
    builder.add_edge("retriever", "evaluator")
    
    builder.add_conditional_edges(
        "evaluator",
        should_loop,
        {
            "expand": "expansion",
            "proceed": "temporal_sorter"
        }
    )
    
    builder.add_edge("expansion", "retriever")
    builder.add_edge("temporal_sorter", "analyzer")
    builder.add_edge("analyzer", END)
    
    return builder.compile()

# Global Cache
_compliance_graph = None

def get_compliance_graph():
    global _compliance_graph
    if _compliance_graph is None:
        _compliance_graph = build_compliance_graph()
    return _compliance_graph

def run_compliance_query(query: str) -> Dict[str, Any]:
    """Execute the compliance-focused Agentic RAG Pipeline."""
    graph = get_compliance_graph()
    result = graph.invoke({"original_query": query})
    return {
        "report": result.get("timeline_report", "No report generated."),
        "refined_query": result.get("refined_query", ""),
        "company_symbol": result.get("company_symbol", ""),
        "latencies_ms": result.get("latencies_ms", {}),
        "total_documents_analyzed": len(result.get("sorted_chunks", [])),
        "loops_executed": result.get("loop_count", 0)
    }
