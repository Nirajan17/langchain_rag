from transformers import pipeline
from transformers.utils.logging import set_verbosity_error
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

set_verbosity_error()

pipe = pipeline("text-generation", model="gpt2")

model = HuggingFacePipeline(pipeline=pipe)

template = "Explain the topic {question} in {context} country in detail."
prompt = PromptTemplate.from_template(template)

chain = prompt | model

# Get user input.
topic = input("Enter the topic you want a description of: ")
context = "Nepal"

# Invoke the chain with a dictionary of prompt variables.
response = chain.invoke({"question": topic, "context": context})

print("Response:", response)
