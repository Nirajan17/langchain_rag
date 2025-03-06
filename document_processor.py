from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore

def load_and_process_documents(embeddings, pdf_path):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
    return vector_store.as_retriever() 