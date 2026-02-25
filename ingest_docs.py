from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import pdfplumber
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def load_pdf_with_pdfplumber(file_path: str) -> list[LCDocument]:
    """Extract text from PDF using pdfplumber."""
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                docs.append(
                    LCDocument(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num},
                    )
                )
    return docs


def ingest_documents():
    FILE_PATH = r"C:\Users\henry.garcia\Documents\agent_spare_car_parts\agent_car_spare_parts\data\catalog.pdf"
    
    # Load PDF
    print(f"Loading PDF from {FILE_PATH}...")
    docs = load_pdf_with_pdfplumber(FILE_PATH)
    print(f"Loaded {len(docs)} pages from PDF")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks")

    # Create embeddings using HuggingFace (no API key needed)
    print("Loading HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Create FAISS vectorstore
    vectorstore_path = Path(FILE_PATH).parent / "vectorstore"
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(str(vectorstore_path))
    print(f"✓ Vectorstore saved to {vectorstore_path}")


if __name__ == "__main__":
    ingest_documents()