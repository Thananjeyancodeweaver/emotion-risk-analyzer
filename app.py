import streamlit as st
import re
import emoji
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Emotion & Risk Analyzer", layout="centered")

st.title("🧠 Emotion & Risk Analyzer")
st.write("Analyze text with emojis using the trained sentiment model")

MODEL_PATH = "saved_models/roberta_sentiment"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def clean_text(text):
    text = emoji.demojize(str(text), delimiters=(" ", " "))
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = float(probs[0][pred])

    label_map = {0: "Negative", 1: "Positive"}
    return label_map[pred], conf

def simple_risk_logic(clean_text_value, sentiment):
    high_risk_terms = [
        "want to die", "kill myself", "end my life", "suicide",
        "don t want to live", "no reason to live"
    ]
    moderate_terms = [
        "alone", "hopeless", "worthless", "depressed", "wasted",
        "tired", "empty", "lost"
    ]

    if any(term in clean_text_value for term in high_risk_terms):
        return "Critical"
    if any(term in clean_text_value for term in moderate_terms):
        return "Moderate"
    if sentiment == "Negative":
        return "Moderate"
    return "Low"

user_input = st.text_area("Enter text with emojis:", height=160)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        clean = clean_text(user_input)
        sentiment, confidence = predict_sentiment(clean)
        risk = simple_risk_logic(clean, sentiment)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", sentiment)
        c2.metric("Risk Level", risk)
        c3.metric("Confidence", round(confidence, 3))

        st.subheader("Processed Text")
        st.write(clean)
