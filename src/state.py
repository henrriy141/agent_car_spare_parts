from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """State shared across all agents in the LangGraph workflow."""

    # The original user query
    query: str

    # Parts found in the local SQLite database (Agent 1 output)
    db_results: List[Dict[str, Any]]

    # Part names found in DB – the "saved variable" passed to subsequent agents
    found_parts: List[str]

    # Results from the PDF catalog RAG search (Agent 2 output)
    rag_results: List[Dict[str, Any]]

    # Results from the web search (Agent 3 output)
    web_results: List[Dict[str, Any]]

    # Final compiled answer produced by the synthesis agent
    final_answer: str
