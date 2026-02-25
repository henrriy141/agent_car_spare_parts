from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from src.state import AgentState
from src.tools.sql_tool import get_sql_toolkit
import google.generativeai as genai
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Optional

class PartDetails(BaseModel):
    part_number: str = Field(description="The unique identifier for the spare part")
    name: str = Field(description="The formal name of the part")
    price: float = Field(description="The internal retail price")
    status: str = Field(description="Availability: in_stock, out_of_stock, or low_stock")
    compatibility: list[str] = Field(description="List of compatible car models")

load_dotenv()
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GOOGLE_API_KEY)
structured_llm = llm.with_structured_output(PartDetails)

def db_specialist_node(state: AgentState):
    print("--- EJECUTANDO AGENTE SQL (DB SPECIALIST) ---")
    query = state["input"]
    
    toolkit = get_sql_toolkit(llm)

    # Instruction prompt for the SQL agent.
    SQL_AGENT_PROMPT = """You are an agent designed to interact with a SQL database.
        Given an input question, identify the product or products that match the query 
        and create a syntactically correct sqlite query to run,
        then look at the results of the query and return the answer. Unless the user
        specifies a specific number of examples they wish to obtain, always limit your
        query to at most 5 results.

        You need to always include price, part number and availability information.

        give the answer in a plain text format, and do not include any markdown or code formatting in your answer.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        If no data is found, explain which tables you consulted and why no results were returned.
        """
    
    # Agent executor
    executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        agent_type="tool-calling",
        prefix=SQL_AGENT_PROMPT,
    )
    
 
    try:
        # Step A: Get raw info from DB using the agent
        response = executor.invoke({"input": query})
        raw_output = response["output"]

        # Handle Gemini metadata if present
        if isinstance(raw_output, list) and len(raw_output) > 0:
            raw_text = raw_output[0].get('text', str(raw_output[0]))
        else:
            raw_text = str(raw_output)

        # Step B: Parse the raw DB text into Structured Output (Pydantic)
        print("--- STRUCTURING OUTPUT WITH PYDANTIC ---")
        structured_data = structured_llm.invoke(
            f"Extract part details from this database record: {raw_text}"
        )

        # Convert Pydantic object to string for the RAG/Web agents or 
        # store the object itself if your state allows it.
        formatted_result = (
            f"Part Name: {structured_data.name} | "
            f"ID: {structured_data.part_number} | "
            f"Price: {structured_data.price} | "
            f"Status: {structured_data.status} | "
            f"Compatibility: {', '.join(structured_data.compatibility)}"
        )

        return {"db_results": formatted_result}
    
    except Exception as e:
        print(f"Error: {e}")
        return {"db_results": f"No structured data found. Error: {str(e)}"}
    
run = db_specialist_node