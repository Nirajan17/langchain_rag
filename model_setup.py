from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from config import REPO_ID, MAX_LENGTH, TEMPERATURE, HUGGINGFACEHUB_API_TOKEN

def initialize_model():
    return HuggingFaceEndpoint(
        repo_id=REPO_ID,
        max_length=MAX_LENGTH,
        temperature=TEMPERATURE,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )

def initialize_embeddings():
    return HuggingFaceEndpointEmbeddings() 