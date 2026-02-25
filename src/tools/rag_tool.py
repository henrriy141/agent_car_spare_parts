from pathlib import Path
from typing import Any, Dict, List

DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"


def init_rag_store() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def search_documents(query: str) -> List[Dict[str, Any]]:
    init_rag_store()
    return [
        {
            "content": f"No indexed vectors yet for query: {query}",
            "source": "local_documents",
            "page": "N/A",
        }
    ]
