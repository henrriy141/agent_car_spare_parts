from src.state import AgentState
from src.tools.search_tool import web_search
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def web_researcher_node(state: AgentState):
    
    # 4. Use Gemini to summarize the web findings
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    search_results = web_search(state["query"])
    
    web_prompt = f"""
    You are a Market Research Assistant.
    
    INTERNAL DATA: {state['db_data']}
    INTERNET SEARCH RESULTS: {search_results}
    
    Based on the search results, provide a brief summary of the current market context 
    (like average online price or recent customer feedback) for this part.
    
    Rules:
    - Provide the answer in plain text.
    - No markdown or special formatting.
    - Be concise (max 3-4 sentences).
    - If no relevant info is found, simply say 'No external market data found'.
    """
    
    response = llm.invoke(web_prompt)
    state["web_results"] = response.content
    return state

run = web_researcher_node
