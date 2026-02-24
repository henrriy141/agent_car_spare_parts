from typing import Annotated, Any
from typing_extensions import TypedDict, NotRequired
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State shared across all nodes in the spare-parts agent graph."""

    # Conversation history (messages are appended, never overwritten)
    messages: Annotated[list, add_messages]

    # Raw search results collected by the search tool (optional, defaults to [])
    search_results: NotRequired[list[dict[str, Any]]]
