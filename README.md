# agent_car_spare_parts

Agentic system built with **Langchain** and **Langgraph** to automate the search for car spare parts.

## Project structure

```
agent_car_spare_parts/
├── agent/
│   ├── __init__.py   # Package entry point
│   ├── state.py      # Shared graph state (AgentState)
│   ├── tools.py      # Langchain tools (search_spare_parts, get_part_details)
│   ├── nodes.py      # Graph node functions (call_model, call_tools)
│   └── graph.py      # Langgraph graph assembly (build_graph)
├── main.py           # Interactive CLI entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

## Usage

```bash
python main.py
```

The agent accepts natural-language queries such as:

> "Find front brake pads for a Toyota Corolla 2019"

It uses the configured LLM together with the `search_spare_parts` and
`get_part_details` tools to find and present matching parts.

## Extending the agent

* **Add real search APIs** – replace the placeholder implementations in
  `agent/tools.py` with calls to real spare-parts data sources.
* **Add new tools** – define additional `@tool` functions in `agent/tools.py`
  and add them to the `TOOLS` list.
* **Adjust the graph** – modify `agent/graph.py` to add new nodes or
  conditional routing logic.
