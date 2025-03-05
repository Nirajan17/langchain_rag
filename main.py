import getpass
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

load_dotenv()

# loading huggingface access token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

# loading model using huggingfaceendpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=512,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# generating the prompt template for the model
template = """
Using the following context, please provide a clear and accurate answer to the question. 
If the answer cannot be found in the context, please say so.

Context: {context}

Question: {question}

Answer: Let me help you with that.
"""

prompt = PromptTemplate.from_template(template)

# generating embeddings
embeddings  = HuggingFaceEndpointEmbeddings()

file_path = "paper-crash-detection.pdf"
if not Path(file_path).exists():
    raise FileNotFoundError(f"PDF file not found: {file_path}")

loader = PyPDFLoader(file_path)
pages = loader.load()

# creating vector stores, for now inmemory vectore store
vector_store = InMemoryVectorStore.from_documents(pages, embeddings)

# retreiver
retreiver = vector_store.as_retriever()

# creating language chain
chain = {
    "context": retreiver,  
    "question": RunnablePassthrough() 
} | prompt | llm 

# question and invoking the chain
question = input("Ask the question about the Crash Detection Paper: \n")
response = chain.invoke(question)

print(response)