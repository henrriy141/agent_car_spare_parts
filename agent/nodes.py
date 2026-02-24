"""Node functions executed by the Langgraph graph."""

import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

from agent.state import AgentState
from agent.tools import TOOLS

_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

_llm = ChatOpenAI(model=_MODEL_NAME, temperature=0).bind_tools(TOOLS)
_tools_by_name = {t.name: t for t in TOOLS}


def call_model(state: AgentState) -> dict:
    """Invoke the LLM with the current conversation and return its response."""
    response = _llm.invoke(state["messages"])
    return {"messages": [response]}


def call_tools(state: AgentState) -> dict:
    """Execute any tool calls requested by the last LLM message."""
    last_message = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    new_results: list[dict] = list(state.get("search_results", []))

    for tool_call in last_message.tool_calls:
        tool = _tools_by_name.get(tool_call["name"])
        if tool is None:
            tool_messages.append(
                ToolMessage(
                    content=f"Error: unknown tool '{tool_call['name']}'",
                    tool_call_id=tool_call["id"],
                )
            )
            continue
        result = tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                new_results.extend(parsed)
            else:
                new_results.append(parsed)
        except json.JSONDecodeError:
            pass

    return {"messages": tool_messages, "search_results": new_results}


def should_continue(state: AgentState) -> str:
    """Routing function: continue to tools or end the graph."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
