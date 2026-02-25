"""
Simple script to demonstrate how to read and query the FAISS vectorstore.
"""

from src.tools.rag_tool import load_vectorstore, search_documents


def main():
    print("=" * 70)
    print("READING FAISS VECTORSTORE")
    print("=" * 70)
    
    # Load vectorstore
    print("\n1. Loading vectorstore...")
    vectorstore = load_vectorstore()
    print(f"   ✓ Vectorstore loaded with {vectorstore.index.ntotal} vectors")
    
    # Query examples
    queries = [
        "brake pads",
        "oil filter",
        "engine parts",
        "electrical components",
    ]
    
    print("\n2. Running example queries...\n")
    for query in queries:
        print(f"   Query: '{query}'")
        results = search_documents(query, k=3)
        for i, result in enumerate(results, 1):
            print(f"   Result {i} (Page {result['page']}):")
            print(f"     {result['content'][:100]}...")
        print()


if __name__ == "__main__":
    main()
