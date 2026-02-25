from dotenv import load_dotenv

from src.graph import run_graph



def main() -> None:
    load_dotenv()
    query = "I need  Oil Filter Standard"
    result = run_graph(query)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
