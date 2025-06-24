import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

MODEL_ID = "Akmalalfatah/analisis-review-amazon"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_ID)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "Positif 😊" if prediction == 1 else "Negatif 😞"

st.title("Analisis Sentimen Review Amazon")
st.write("Masukkan review produk, lalu model akan menebak apakah review itu **positif** atau **negatif**.")

user_input = st.text_area("Tulis review di sini...")

if st.button("Prediksi"):
    if user_input.strip():
        hasil = predict_sentiment(user_input)
        st.success(f"Hasil prediksi: **{hasil}**")
    else:
        st.warning("Tolong isi teks review terlebih dahulu.")
