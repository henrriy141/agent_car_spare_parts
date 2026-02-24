# Agent Car Spare Parts

An agentic system built with **Python**, **LangChain**, and **LangGraph** that
automates multi-source search for car spare parts.  
Google **Gemini** (via `langchain-google-genai`) is used as the LLM throughout.

---

## Architecture

The system is a linear LangGraph workflow with four nodes:

```
User Query
    │
    ▼
[Agent 1] DB Search      – keyword search in a local SQLite database
    │           └─ saves found part names in `found_parts` state variable
    ▼
[Agent 2] RAG Search     – semantic search in a PDF catalog (FAISS + Gemini Embeddings)
    │
    ▼
[Agent 3] Web Search     – Gemini ReAct agent using DuckDuckGo tool
    │
    ▼
[Agent 4] Compile        – Gemini synthesises all results into a final answer
    │
    ▼
Final Answer
```

| Agent | Tool / Technique | Output state key |
|---|---|---|
| DB Search | SQLite keyword search | `db_results`, `found_parts` |
| RAG Search | FAISS vector store (PDF catalog) | `rag_results` |
| Web Search | DuckDuckGo (tool-calling ReAct agent) | `web_results` |
| Compile | Gemini LLM | `final_answer` |

---

## Project Structure

```
agent_car_spare_parts/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/                   # auto-created on first run
│   ├── spare_parts.db      # SQLite database (20 sample parts)
│   ├── catalog.pdf         # generated PDF parts catalog
│   └── vectorstore/        # FAISS index files
├── src/
│   ├── __init__.py
│   ├── state.py            # AgentState TypedDict
│   ├── database.py         # SQLite init & search
│   ├── rag.py              # PDF generation, FAISS init & RAG search
│   └── graph.py            # LangGraph workflow (all four agents)
└── main.py                 # entry point
```

---

## Setup

### Prerequisites

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)

### Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# open .env and set GOOGLE_API_KEY=<your key>
```

### Run

```bash
python main.py
```

**First run** automatically:
1. Creates `data/spare_parts.db` with 20 sample spare parts.
2. Generates `data/catalog.pdf` – a rich-text PDF parts catalog.
3. Builds `data/vectorstore/` – a FAISS index from the PDF.

Subsequent runs reuse the existing database and vector store.

---

## Customisation

| What | Where |
|---|---|
| Add/change sample DB parts | `src/database.py` → `SAMPLE_PARTS` |
| Add/change PDF catalog content | `src/rag.py` → `CATALOG_SECTIONS` |
| Use your own PDF | Replace `data/catalog.pdf` and delete `data/vectorstore/` |
| Change LLM model | `src/graph.py` → `_get_llm()` |
| Change embedding model | `src/rag.py` → `_get_embeddings()` |
