from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch

def load_models():
    """
    Load the text generation (GPT-2) and spam/ham classification models.
    """
    # Text generation model (GPT-2)
    tg_model_name = "gpt2"  # GPT-2 for text generation
    tg_model = AutoModelForCausalLM.from_pretrained(tg_model_name)
    tg_tokenizer = AutoTokenizer.from_pretrained(tg_model_name)

    # Spam classification model
    cls_model_name = "abdulrehman89OK/spam_non_sapm_classifier"  # Replace with your Hugging Face model repo
    cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name)
    cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)

    return tg_model, tg_tokenizer, cls_model, cls_tokenizer

def generate_text(model, tokenizer, prompt, max_length=50):
    """
    Generate text using GPT-2.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def classify_text(model, tokenizer, text):
    """
    Classify text as spam or ham using the classification model.
    """
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "spam" if predicted_class == 1 else "ham"
