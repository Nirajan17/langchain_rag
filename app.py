import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import warnings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_model():
    """Initialize the language model and embeddings"""
    load_dotenv()
    
    # Get HuggingFace API token
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not HUGGINGFACEHUB_API_TOKEN:
        st.error("HuggingFace API token not found. Please set it in your .env file.")
        st.stop()

    # Initialize LLM
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=512,
        temperature=0.5,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

    # Initialize embeddings
    embeddings = HuggingFaceEndpointEmbeddings()
    
    return llm, embeddings

def process_pdf(uploaded_file, embeddings):
    """Process the uploaded PDF file and create vector store"""
    # Save uploaded file temporarily
    with st.spinner("Processing PDF..."):
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load and process the PDF
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        
        # Create vector store
        vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return vector_store

def create_qa_chain(llm, vector_store):
    """Create the question-answering chain"""
    template = """
    Using the following context, please provide a clear and accurate answer to the question. 
    If the answer cannot be found in the context, please say so.

    Context: {context}

    Question: {question}

    Answer: Let me help you with that.
    """

    prompt = PromptTemplate.from_template(template)
    
    retriever = vector_store.as_retriever()
    
    chain = {
        "context": retriever,
        "question": RunnablePassthrough()
    } | prompt | llm
    
    return chain

def display_chat_history():
    """Display the chat history in a conversational format"""
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        # User message
        message(question, is_user=True, key=f"user_msg_{i}")
        # Assistant message
        message(answer, key=f"assistant_msg_{i}")

def message(text, is_user=False, key=None):
    """Display a message in the chat interface"""
    if is_user:
        message_alignment = "flex-end"
        background_color = "#D3E3FD"
        avatar = "👤"
    else:
        message_alignment = "flex-start"
        background_color = "#F0F2F6"
        avatar = "🤖"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment}">
            <div style="background: {background_color}; padding: 10px 15px; border-radius: 20px; max-width: 70%;">
                <span style="font-size: 20px; margin-right: 5px;">{avatar}</span>
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.title("📚 PDF Question Answering System")

    # Add sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload your PDF document using the file uploader
        2. Wait for the document to be processed
        3. Type your question in the text input
        4. Click 'Send' or press Enter to get a response
        
        **Note:** The system will analyze the document content and provide relevant answers based on the information available in the PDF.
        """)
        
        # File upload section in sidebar
        uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
        
        if uploaded_file:
            # Initialize model and embeddings
            llm, embeddings = initialize_model()
            
            # Process PDF and create vector store
            if st.session_state.vector_store is None or uploaded_file.name != st.session_state.get('last_file', ''):
                st.session_state.vector_store = process_pdf(uploaded_file, embeddings)
                st.session_state.last_file = uploaded_file.name
                st.session_state.chain = create_qa_chain(llm, st.session_state.vector_store)
                st.success("PDF processed successfully! You can now ask questions.")

    # Main chat interface
    if st.session_state.vector_store is not None:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            display_chat_history()

        # Question input section
        st.markdown("<div style='position: fixed; bottom: 0; width: 100%; background-color: white; padding: 20px;'>", unsafe_allow_html=True)
        col1, col2 = st.columns([6, 1])
        with col1:
            question = st.text_input("Ask a question:", key="question_input")
        with col2:
            send_button = st.button("Send")

        if send_button and question:
            with st.spinner("Generating answer..."):
                try:
                    response = st.session_state.chain.invoke(question)
                    # Add to chat history
                    st.session_state.chat_history.append((question, response))
                    # Clear input
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Please upload a PDF document to start asking questions.")

if __name__ == "__main__":
    main() 