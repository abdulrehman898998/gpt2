from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoModelForSequenceClassification
import torch
import streamlit as st

# Load the default GPT-2 tokenizer and model for text generation
model_name = "gpt2"  # Using the default GPT-2 model

# Load GPT-2 model for text generation
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load the tokenizer for GPT-2
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

# Set the pad token to be the same as eos token
tokenizer.pad_token = tokenizer.eos_token

# Load your classification model (if it's a separate one)
classification_model = AutoModelForSequenceClassification.from_pretrained(
    "abdulrehman89OK/spam_non_sapm_classifier", local_files_only=True
)

# Function to generate text with GPT-2
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function for spam/non-spam classification
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "Spam" if predicted_class == 1 else "Non-Spam"

# Streamlit App UI

# Title of the app
st.title("Text Generation and Spam Classification")

# Add a text input box for text generation
st.header("GPT-2 Text Generation")
prompt = st.text_input("Enter a prompt for text generation:")
if st.button("Generate Text"):
    if prompt:
        generated_text = generate_text(prompt)
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt to generate text.")

# Add a text input box for spam classification
st.header("Spam/Non-Spam Classification")
input_text = st.text_area("Enter text for spam classification:")
if st.button("Classify Text"):
    if input_text:
        result = classify_text(input_text)
        st.subheader("Classification Result:")
        st.write(result)
    else:
        st.warning("Please enter text for classification.")
