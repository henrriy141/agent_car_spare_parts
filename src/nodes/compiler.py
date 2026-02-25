from src.state import AgentState


def compiler_node(state: AgentState) -> AgentState:
    lines = [
        f"Query: {state['query']}",
        f"DB results: {len(state['db_results'])}",
        f"RAG results: {len(state['rag_results'])}",
        f"Web results: {len(state['web_results'])}",
    ]
    state["final_answer"] = "\n".join(lines)
    return state


run = compiler_node
