from langgraph.graph import END, StateGraph

from src.nodes.compiler import compiler_node
from src.nodes.db_specialist import db_specialist_node
from src.nodes.rag_expert import rag_expert_node
from src.nodes.web_researcher import web_researcher_node
from src.state import AgentState


# Define the linear agent workflow.
workflow = StateGraph(AgentState)

workflow.add_node("db_agent", db_specialist_node)
workflow.add_node("rag_agent", rag_expert_node)
workflow.add_node("web_agent", web_researcher_node)
workflow.add_node("compiler_agent", compiler_node)

workflow.set_entry_point("db_agent")
workflow.add_edge("db_agent", "rag_agent")
workflow.add_edge("rag_agent", "web_agent")
workflow.add_edge("web_agent", "compiler_agent")
workflow.add_edge("compiler_agent", END)

app = workflow.compile()


# Execute the compiled workflow for a single query.
def run_graph(query: str) -> AgentState:
    state: AgentState = {
        "query": query,
        "db_results": [],
        "found_parts": [],
        "rag_results": [],
        "web_results": [],
        "final_answer": "",
    }
    return app.invoke(state)
