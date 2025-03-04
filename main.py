from transformers import pipeline
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

model = pipeline("summarization", model="facebook/bart-large-cnn", use_auth_token=False)

text = input("Enter the text to be summarized: ")

# Generate summary
response = model(text, max_length=130, min_length=30, do_sample=False)

# Print the summary
print("Summary:", response[0]['summary_text'])
