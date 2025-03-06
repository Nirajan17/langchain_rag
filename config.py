import os
from dotenv import load_dotenv
import getpass

load_dotenv()

# Model configurations
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_LENGTH = 512
TEMPERATURE = 0.5
PDF_FILE_PATH = "paper-crash-detection.pdf"

# HuggingFace configuration
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

# System prompt template
SYSTEM_PROMPT = """
You are a helpful assistant. Use the following context to answer the question.

Context: {context}

Question: {question}

Answer: Let me help you with that.
""" 