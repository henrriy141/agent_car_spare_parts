from src.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools.rag_tool import search_documents
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# LLM used to synthesize answers from retrieved documents.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GOOGLE_API_KEY)


def rag_expert_node(state: AgentState) -> AgentState:
    """RAG expert node: search vectorstore for relevant documents."""
    query = state["query"]
    print("--- EJECUTANDO AGENTE RAG (RAG EXPERT) ---")
    rag_results = search_documents(llm=llm)
    state["rag_results"] = rag_results.invoke(query)
    #print(f"RAG results: {state['rag_results']}")
    return state


run = rag_expert_node
