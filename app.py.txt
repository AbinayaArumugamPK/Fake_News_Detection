import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "Model/fake_news_lstm.keras"
    )

    with open("Model/tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer
model, tokenizer = load_model()
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def predict_news(news):
    cleaned = clean_text(news)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=300,
        padding="post",
        truncating="post"
    )
    probability = model.predict(
        padded,
        verbose=0
    )[0][0]
    if probability >= 0.5:
        result = "REAL NEWS"
        confidence = probability * 100
    else:
        result = "FAKE NEWS"
        confidence = (1 - probability) * 100
    return result, confidence
st.title("📰 Fake News Detection")
st.write(
    "Enter a news article below to classify it "
    "using a Deep Learning LSTM model."
)
news_text = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("Please enter a news article.")
    else:
        result, confidence = predict_news(news_text)
        st.subheader("Prediction")
        if result == "REAL NEWS":
            st.success(f"✅ {result}")

        else:
            st.error(f"⚠️ {result}")
        st.write(
            f"Model Confidence: **{confidence:.2f}%**"
        )
st.divider()
st.caption(
    "Note: This system is a machine-learning classifier "
    "trained on a specific dataset. Its prediction and "
    "confidence do not independently verify the factual "
    "truth of a news claim."
)

