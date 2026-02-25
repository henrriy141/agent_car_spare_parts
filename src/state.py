from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    query: str
    db_results: List[Dict[str, Any]]
    found_parts: List[str]
    rag_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    final_answer: str
