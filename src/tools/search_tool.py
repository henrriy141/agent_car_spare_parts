# Web search helper utilities.
from typing import Any, Dict, List
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
import os
from src.state import AgentState

# Execute a targeted web search for pricing and availability context.
def web_search(query: str):
    """
    Enhanced search specifically tuned to find retail prices and 
    competitor stock levels for spare parts.
    """
    # 1. Initialize the search tool that returns multiple snippets
    # 'max_results' helps get a broader view of the market
    search = DuckDuckGoSearchResults(max_results=5)
    
    # 2. Refine the search query for 'Shopping' intent
    # We add specific keywords to trigger price listings and marketplace results
    refined_query = (
        f'{query} price shop online stock availability '
        f'-(site:youtube.com | site:facebook.com)' # Exclude social media for cleaner data
    )
    
    print(f"--- 🌐 SEARCHING WEB FOR: {refined_query} ---")
    
    try:
        # 3. Execute the search
        # This returns a string containing multiple titles, snippets, and links
        search_results = search.run(refined_query)
        
        # If the search is too specific and returns nothing, try a broader one
        if not search_results or "snippet" not in search_results.lower():
            search_results = search.run(f"average price of {query} spare part")
            
    except Exception as e:
        print(f"Web search failed: {e}")
        search_results = "No external market pricing could be retrieved at this moment."
        
    return search_results