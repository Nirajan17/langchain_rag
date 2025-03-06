import warnings
from model_setup import initialize_model, initialize_embeddings
from document_processor import load_and_process_documents
from chatbot import Chatbot

warnings.filterwarnings("ignore")

def main():
    # Initialize components
    llm = initialize_model()
    embeddings = initialize_embeddings()
    retriever = load_and_process_documents(embeddings)
    
    # Create and run chatbot
    chatbot = Chatbot(llm, retriever)
    chatbot.chat()

if __name__ == "__main__":
    main()
