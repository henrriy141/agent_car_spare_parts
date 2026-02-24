"""Assembles the Langgraph state machine for the spare-parts agent."""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import call_model, call_tools, should_continue


def build_graph():
    """Build and compile the spare-parts agent graph.

    Graph topology
    --------------
    START → model → (tools → model)* → END

    The agent calls the LLM, optionally invokes tools, feeds results back
    to the LLM, and repeats until the model produces a final answer without
    any tool calls.
    """
    graph = StateGraph(AgentState)

    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)

    graph.set_entry_point("model")

    graph.add_conditional_edges(
        "model",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "model")

    return graph.compile()
