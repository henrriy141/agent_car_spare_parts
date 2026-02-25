from typing import Any, Dict, List
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from src.state import AgentState

def web_search(query: str):
    # 1. Initialize the free search tool
    search = DuckDuckGoSearchRun()
    
    # 2. Build a smart search query
    # We use the Part Name/Number found by the DB agent for accuracy
    search_query = f"latest price and reviews for {query}"
    
    try:
        # 3. Execute the free search
        search_results = search.run(search_query)
    except Exception as e:
        print(f"Web search failed: {e}")
        search_results = "No internet data could be retrieved at this moment."
        
    return search_results