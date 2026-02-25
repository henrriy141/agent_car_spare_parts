from typing import Any, Dict, List


def web_search(query: str) -> List[Dict[str, Any]]:
    return [
        {
            "title": "Web search placeholder",
            "snippet": f"No web provider configured yet for query: {query}",
            "url": "",
            "source": "web",
        }
    ]
