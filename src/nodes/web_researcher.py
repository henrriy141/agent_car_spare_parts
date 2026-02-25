from src.state import AgentState
from src.tools.search_tool import web_search


def web_researcher_node(state: AgentState) -> AgentState:
    state["web_results"] = web_search(state["query"])
    return state


run = web_researcher_node
