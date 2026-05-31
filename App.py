import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

model = joblib.load('verifact_svm_model.pkl')
vectorizer = joblib.load('verifact_tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

st.set_page_config(page_title="VeriFact", page_icon="🔍")
st.title("🔍 VeriFact – Fake News Detection System")
st.write("Enter a news article below to check whether it is Real or Fake.")

user_input = st.text_area("News Article", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a news article.")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        confidence = round(max(model.predict_proba(features)[0]) * 100, 2)
        
        if prediction == 1:
            st.success(f"✅ Real News — Confidence: {confidence}%")
        else:
            st.error(f"❌ Fake News — Confidence: {confidence}%")
