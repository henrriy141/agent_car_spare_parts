from src.state import AgentState


def compiler_node(state: AgentState) -> AgentState:
    lines = [
        f"Query: {state['query']}",
        f"DB results: {state['db_results']}",
        f"RAG results: {state['rag_results']}",
        f"Web results: {state['web_results']}",
    ]
    state["final_answer"] = "\n".join(lines)
    return state


run = compiler_node
