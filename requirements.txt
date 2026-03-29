import streamlit as st
import re
import emoji

st.set_page_config(page_title="Emotion Analyzer", layout="centered")

st.title("🧠 Emotion & Risk Analyzer")

def clean_text(text):
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s:]", " ", text)
    return text

def analyze(text):
    clean = clean_text(text)

    if "die" in clean:
        return "Critical Risk", "Negative"
    elif "sad" in clean or "alone" in clean:
        return "Moderate Risk", "Negative"
    else:
        return "Low Risk", "Positive"

text = st.text_area("Enter text with emojis:")

if st.button("Analyze"):
    risk, sentiment = analyze(text)

    col1, col2 = st.columns(2)
    col1.metric("Sentiment", sentiment)
    col2.metric("Risk Level", risk)

    st.write("Processed Text:", clean_text(text))
