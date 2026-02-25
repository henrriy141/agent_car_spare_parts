"""
Simple script to demonstrate how to read and query the FAISS vectorstore.
"""

from src.state import AgentState
from src.tools.search_tool import web_search
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
# 4. Use Gemini to summarize the web findings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

def web_researcher_node():
    search_results = web_search("I need Oil Filter Standard")
    
    web_prompt = f"""
    ### ROLE
You are a Strategic Sales Analyst for a spare parts company. Your goal is to compare our internal offers with the current internet market to highlight our competitive advantages.

### DATA
INTERNAL DATA:The Oil Filter Standard is available. Its part number is OF-001 and it costs 12.99. It is in stock.
INTERNET SEARCH RESULTS: {search_results}

### TASK
1. Compare our price, part number, and availability with the search results found online.
2. Identify at least one reason why the customer should choose us (e.g., our price is lower, we have it in stock while others don't, or our part is a confirmed OEM match).
3. Summarize the market context and provide a brief "Value Proposition" explaining our advantage.

### RULES
- Provide the answer in plain text only.
- Absolutely NO markdown, NO bolding (**), and NO symbols.
- Be concise (max 4-5 sentences).
- If the internet price is lower, focus on our "immediate availability" or "guaranteed compatibility."
- If no relevant info is found, simply say: No external market data found to compare.
    """
    
    response = llm.invoke(web_prompt)
    print(response.content)
    return 'finished'



if __name__ == "__main__":
    web_researcher_node()
