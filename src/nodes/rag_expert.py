from src.state import AgentState
from src.tools.rag_tool import search_documents


def rag_expert_node(state: AgentState) -> AgentState:
    state["rag_results"] = search_documents(state["query"])
    return state


run = rag_expert_node
