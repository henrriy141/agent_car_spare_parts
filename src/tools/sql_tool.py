import sqlite3
from pathlib import Path
from typing import Any, Dict, List
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
import os


# SQL helper utilities for the local inventory database.
# Load the local database connection for SQL tools.
db = SQLDatabase.from_uri("sqlite:///data/spare_parts.db")

# Build a toolkit wrapper for the SQL agent.
def get_sql_toolkit(llm):
    return SQLDatabaseToolkit(db=db, llm=llm)