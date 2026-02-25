from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from src.state import AgentState
from src.tools.sql_tool import get_sql_toolkit
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GOOGLE_API_KEY)

def db_specialist_node(state: AgentState):
    print("--- EJECUTANDO AGENTE SQL (DB SPECIALIST) ---")
    query = state["query"]
    
    toolkit = get_sql_toolkit(llm)

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
    
    # Agente ejecutor
    executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="tool-calling",
        prefix=SQL_AGENT_PROMPT,
    )
    
    # Ejecutar la consulta
    response = executor.invoke({"input": query})
    try:
        output = response["output"]

        # 1. Si el output es una lista (como en tu caso)
        if isinstance(output, list) and len(output) > 0:
            # Extraemos el campo 'text' del primer elemento
            clean_text = output[0].get('text', str(output[0]))
        
        # 2. Si el output ya es un string directo
        elif isinstance(output, str):
            clean_text = output
            
        else:
            clean_text = str(output)

        print(f"--- clean result: {clean_text} ---")
        return {"db_data": clean_text}
    
    except Exception as e:
        return {"db_data": f"Error al procesar: {str(e)}"}
   