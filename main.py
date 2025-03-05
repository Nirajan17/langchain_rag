import getpass
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import warnings

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
    max_length=128,
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# generating the prompt template for the model
template = """
question: {question}
context: {context}
You are an smart AI assistant and should reply to the {question} given according to the {context} given.
"""

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = input("Enter the question: \n")
context = input("In which context the question is being asked? \n")

response = chain.invoke({"question": question, "context": context})

print(response)