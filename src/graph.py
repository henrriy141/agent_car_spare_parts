"""
LangGraph workflow for the car spare-parts search system.

Flow:
  query → db_search_agent → rag_search_agent → web_search_agent
        → compile_results_agent → final_answer
"""

from typing import Any, Dict, List

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from .database import search_parts_db
from .rag import FAISS, search_catalog_rag
from .state import AgentState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


def _fmt_db_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No matching parts found in local database."
    lines = []
    for r in results:
        lines.append(
            f"  • {r['name']} (#{r['part_number']}) – ${r['price']} "
            f"[{r['availability']}]\n"
            f"    Compatible: {r['compatible_models']}"
        )
    return "\n".join(lines)


def _fmt_rag_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No relevant information found in PDF catalog."
    lines = []
    for i, r in enumerate(results, 1):
        snippet = r["content"].strip().replace("\n", " ")[:300]
        lines.append(f"  [{i}] (page {r['page']}) {snippet}…")
    return "\n".join(lines)


def _fmt_web_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No web results available."
    return "\n".join(r["content"] for r in results)


# ---------------------------------------------------------------------------
# Agent 1 – Database search
# ---------------------------------------------------------------------------

def db_search_agent(state: AgentState) -> Dict[str, Any]:
    """
    Search the local SQLite database for car parts that match the query.
    Saves the found part names in ``found_parts`` so later agents can
    reference them.
    """
    query = state["query"]
    print(f"\n[DB Agent] Searching local database for: {query!r}")

    results = search_parts_db(query)
    found_parts = [r["name"] for r in results]

    if results:
        print(f"  → Found {len(results)} part(s): {', '.join(found_parts)}")
    else:
        print("  → No parts found in local database.")

    return {
        "db_results": results,
        "found_parts": found_parts,
    }


# ---------------------------------------------------------------------------
# Agent 2 – RAG search (PDF catalog)
# ---------------------------------------------------------------------------

def _make_rag_agent(vectorstore: FAISS):
    """
    Return a LangGraph node function that has ``vectorstore`` in its closure.
    """

    def rag_search_agent(state: AgentState) -> Dict[str, Any]:
        """
        Search the PDF catalog vector store using the user query enriched with
        any part names already found in the database (``found_parts``).
        """
        query = state["query"]
        found_parts: List[str] = state.get("found_parts", [])

        # Enrich the semantic query with already-found part names
        if found_parts:
            enriched_query = f"{query}. Related parts: {', '.join(found_parts)}"
        else:
            enriched_query = query

        print(f"\n[RAG Agent] Searching PDF catalog for: {enriched_query!r}")

        results = search_catalog_rag(enriched_query, vectorstore)

        if results:
            print(f"  → Found {len(results)} relevant catalog chunk(s).")
        else:
            print("  → No relevant chunks found in PDF catalog.")

        return {"rag_results": results}

    return rag_search_agent


# ---------------------------------------------------------------------------
# Agent 3 – Web search (tool-calling ReAct agent with DuckDuckGo)
# ---------------------------------------------------------------------------

def web_search_agent(state: AgentState) -> Dict[str, Any]:
    """
    Use a Gemini-powered ReAct agent with a DuckDuckGo search tool to find
    up-to-date pricing and availability for the requested car parts.
    """
    query = state["query"]
    found_parts: List[str] = state.get("found_parts", [])

    llm = _get_llm()
    search_tool = DuckDuckGoSearchRun(name="web_search")

    # Build a focused web-search prompt
    parts_hint = ""
    if found_parts:
        parts_hint = f" Parts already identified: {', '.join(found_parts[:5])}."

    web_query = (
        f"Find pricing and availability for car spare parts: {query}.{parts_hint} "
        "Include part numbers, prices (USD), compatible vehicle models, and where "
        "to buy online."
    )

    print(f"\n[Web Agent] Searching the web for: {query!r}")

    try:
        agent = create_react_agent(llm, [search_tool])
        result = agent.invoke({"messages": [HumanMessage(content=web_query)]})
        # The last message in the response is the agent's final answer
        final_message = result["messages"][-1].content
    except (ValueError, RuntimeError, OSError) as exc:
        # Broad catch: network errors, rate-limiting, tool or LLM failures all
        # map to different exception types; we degrade gracefully in all cases.
        final_message = f"Web search could not be completed: {exc}"
        print(f"  -> Web search error: {exc}")

    print("  → Web search complete.")
    return {"web_results": [{"content": final_message, "source": "web_search"}]}


# ---------------------------------------------------------------------------
# Agent 4 – Results compilation
# ---------------------------------------------------------------------------

def compile_results_agent(state: AgentState) -> Dict[str, Any]:
    """
    Use Gemini to synthesise results from all three search agents into a
    single, customer-friendly answer.
    """
    print("\n[Compile Agent] Synthesising results with Gemini…")

    query = state["query"]
    db_section = _fmt_db_results(state.get("db_results", []))
    rag_section = _fmt_rag_results(state.get("rag_results", []))
    web_section = _fmt_web_results(state.get("web_results", []))

    prompt = (
        f"You are a professional car parts advisor.\n\n"
        f"Customer query: {query}\n\n"
        f"=== LOCAL DATABASE RESULTS ===\n{db_section}\n\n"
        f"=== PDF CATALOG RESULTS (RAG) ===\n{rag_section}\n\n"
        f"=== WEB SEARCH RESULTS ===\n{web_section}\n\n"
        "Based on all the above information, please provide a clear and helpful "
        "response that:\n"
        "1. Lists all relevant parts found with their part numbers and prices.\n"
        "2. Confirms vehicle compatibility where available.\n"
        "3. Highlights stock availability.\n"
        "4. Includes any useful information from the web search.\n"
        "Be concise and well-structured."
    )

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])

    print("  → Final answer ready.")
    return {"final_answer": response.content}


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def create_graph(vectorstore: FAISS):
    """
    Build and compile the LangGraph workflow.

    Args:
        vectorstore: Pre-initialised FAISS vector store for the RAG agent.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to be invoked.
    """
    rag_agent_node = _make_rag_agent(vectorstore)

    workflow = StateGraph(AgentState)

    workflow.add_node("db_search", db_search_agent)
    workflow.add_node("rag_search", rag_agent_node)
    workflow.add_node("web_search", web_search_agent)
    workflow.add_node("compile_results", compile_results_agent)

    workflow.add_edge(START, "db_search")
    workflow.add_edge("db_search", "rag_search")
    workflow.add_edge("rag_search", "web_search")
    workflow.add_edge("web_search", "compile_results")
    workflow.add_edge("compile_results", END)

    return workflow.compile()
