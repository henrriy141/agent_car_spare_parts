from src.state import AgentState
from src.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai
import os
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

from langchain_google_genai import ChatGoogleGenerativeAI
# LLM used to rewrite the final answer into a natural response.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GOOGLE_API_KEY)

def compiler_node(state: AgentState) -> AgentState:
    base_answer = state.get("final_answer", "").strip()
    if not base_answer:
        lines = [
            f"Query: {state['input']}",
            f"DB results: {state['db_results']}",
            f"RAG results: {state['rag_results']}",
            f"Web results: {state['web_results']}",
        ]
        base_answer = "\n".join(lines)

    # Ask the LLM to rewrite the summary into a clear, user-friendly response.
    prompt = (
        "You are a helpful assistant. Rewrite the input into a natural, concise response "
        "that explains the process and summarizes the results for the user.\n\n"
        f"Input:\n{base_answer}\n\n"
        "Response:"
    )

    response = llm.invoke(prompt)
    state["final_answer"] = response.content.strip()
    return state


run = compiler_node
