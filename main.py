from transformers import pipeline
from transformers.utils.logging import set_verbosity_error
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


set_verbosity_error()

# Use a pipeline as a high-level helper
from transformers import pipeline

model = pipeline("question-answering", model="deepset/roberta-base-squad2")

template = "generate about {question} in {context}."
prompt = PromptTemplate.from_template(template)
parser = StrOutputParser()

chain = prompt | model | parser

# Get user input.
topic = input("Enter the topic you want a description of: ")
context = input("Enter the context you are talking about.")

# Invoke the chain with a dictionary of prompt variables.
response = chain.invoke({"question": topic, "context": context})

print("Response:", response)
