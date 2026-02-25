from typing import Any, Dict, List, TypedDict


# Shared state passed between all agent nodes.
class AgentState(TypedDict):
    input: str
    db_results: List[Dict[str, Any]]
    found_parts: List[str]
    rag_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    final_answer: str
