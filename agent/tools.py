"""Tools available to the spare-parts search agent."""

import json
from langchain_core.tools import tool


@tool
def search_spare_parts(query: str) -> str:
    """Search for car spare parts matching the given query.

    Args:
        query: Natural-language description of the part (e.g.
               "front brake pads for Toyota Corolla 2019").

    Returns:
        JSON string with a list of matching parts, each containing
        ``name``, ``part_number``, ``price``, ``availability`` and
        ``supplier`` keys.
    """
    # Placeholder implementation – replace with a real API call.
    results = [
        {
            "name": f"Result for: {query}",
            "part_number": "N/A",
            "price": "N/A",
            "availability": "unknown",
            "supplier": "N/A",
        }
    ]
    return json.dumps(results, ensure_ascii=False)


@tool
def get_part_details(part_number: str) -> str:
    """Retrieve detailed information about a specific spare part by its part number.

    Args:
        part_number: The manufacturer's part number.

    Returns:
        JSON string with detailed part information including compatibility,
        dimensions, weight and warranty.
    """
    # Placeholder implementation – replace with a real API call.
    details = {
        "part_number": part_number,
        "description": "Detailed information not yet available.",
        "compatibility": [],
        "dimensions": {},
        "weight_kg": None,
        "warranty_months": None,
    }
    return json.dumps(details, ensure_ascii=False)


TOOLS = [search_spare_parts, get_part_details]
