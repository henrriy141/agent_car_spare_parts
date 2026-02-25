import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# Cargar la base de datos local
db = SQLDatabase.from_uri("sqlite:///data/spare_parts.db")

def get_sql_toolkit(llm):
    return SQLDatabaseToolkit(db=db, llm=llm)