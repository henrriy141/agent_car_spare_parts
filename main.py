"""Entry point for the car spare-parts agent."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from agent import build_graph  # noqa: E402 – import after load_dotenv


def main():
    graph = build_graph()

    print("Car Spare Parts Agent")
    print("Type your query (or 'quit' to exit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        last_message = result["messages"][-1]
        print(f"Agent: {last_message.content}\n")


if __name__ == "__main__":
    main()
