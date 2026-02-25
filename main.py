from dotenv import load_dotenv

from src.graph import run_graph



# Run a single example query through the graph.
def main() -> None:
    load_dotenv()
    query = "I need  an Alternator 130A Remanufactured"
    result = run_graph(query)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
