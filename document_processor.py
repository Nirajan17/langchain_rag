from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from config import PDF_FILE_PATH

def load_and_process_documents(embeddings):
    if not Path(PDF_FILE_PATH).exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_FILE_PATH}")
    
    loader = PyPDFLoader(PDF_FILE_PATH)
    pages = loader.load()
    
    vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
    return vector_store.as_retriever() 