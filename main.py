"""
Entry point for the Car Spare Parts Agentic System.

Usage:
    python main.py

Set GOOGLE_API_KEY in a .env file (or as an environment variable) before
running.  See .env.example for the required format.

On the first run the system will:
  1. Create the local SQLite database with sample spare-parts data.
  2. Generate a sample PDF catalog (data/catalog.pdf).
  3. Build a FAISS vector store from the PDF (data/vectorstore/).

Subsequent runs reuse the existing database and vector store.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from src.database import init_database  # noqa: E402
from src.graph import create_graph      # noqa: E402
from src.rag import init_rag            # noqa: E402

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    "I need an oil filter and brake pads for a Toyota Camry 2020",
    "Looking for spark plugs and air filter for Honda Civic 2018",
    "Need an alternator for Ford Fusion 2015",
]


def run_query(graph, query: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  QUERY: {query}")
    print("=" * 65)

    initial_state = {
        "query": query,
        "db_results": [],
        "found_parts": [],
        "rag_results": [],
        "web_results": [],
        "final_answer": "",
    }

    result = graph.invoke(initial_state)

    print(f"\n  Parts found in DB ({len(result['db_results'])}): "
          f"{result['found_parts'] or 'none'}")

    print("\n--- FINAL ANSWER ---")
    print(result["final_answer"])


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set.  "
            "Copy .env.example to .env and add your key."
        )

    print("=== Car Spare Parts Agentic System ===\n")

    print("[Setup] Initialising local database…")
    init_database()

    print("[Setup] Initialising RAG system (first run builds vector store)…")
    vectorstore = init_rag()

    print("[Setup] Building LangGraph workflow…")
    graph = create_graph(vectorstore)

    print("\n[Ready] Running example queries…")
    for query in EXAMPLE_QUERIES:
        run_query(graph, query)


if __name__ == "__main__":
    main()
