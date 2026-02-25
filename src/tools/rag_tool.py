from pathlib import Path
from typing import Any, Dict, List
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Location of the persisted FAISS vectorstore.
VECTORSTORE_PATH = Path(__file__).resolve().parents[2] / "data" / "vectorstore"


# Load the FAISS vectorstore from disk using the same embeddings.
def load_vectorstore() -> FAISS:
    """Load FAISS vectorstore from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore

# Join retrieved documents into a single context string.
def format_docs(docs: Iterable[LCDocument]):
    return "\n\n".join(doc.page_content for doc in docs)

# Build a RAG chain that uses the vectorstore retriever and an LLM.
def search_documents( llm=None):
    """Search the vectorstore for documents similar to the query."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    
    prompt = PromptTemplate.from_template(
        "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:\n"
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
        
