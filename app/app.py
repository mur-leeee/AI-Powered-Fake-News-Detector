import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai

EXPECTED_PASSWORD = st.secrets.get("APP_PASSWORD", "vtu_cse_2026")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.query_params.get("pwd") == EXPECTED_PASSWORD:
    st.session_state["authenticated"] = True

if not st.session_state["authenticated"]:
    st.title("🔒 Private Portfolio Project")
    st.info("This is a secured AI Disinformation Threat tool.")
    pwd_input = st.text_input("Enter Passcode:", type="password")
    
    if pwd_input == EXPECTED_PASSWORD:
        st.session_state["authenticated"] = True
        st.rerun()
    elif pwd_input:
        st.error("Incorrect passcode.")
        
    st.stop()

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_models():
    model = pickle.load(open("models/model.pkl", "rb"))
    vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_models()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 AI-Powered News Analyzer")
st.markdown("Paste a news article below. Our baseline ML model will perform a fast scan, and you can trigger Gemma for a deep contextual breakdown.")

user_input = st.text_area("News Text:", height=200, placeholder="Enter news here...")

if st.button("Run Fast Scan (Classical ML)"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Running TF-IDF Analysis...'):
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])

            prob = model.predict_proba(vector)[0]
            pred = model.predict(vector)[0]
            confidence = max(prob)

            st.markdown("---")
            st.subheader("Baseline Model Results")

            if confidence < 0.60:
                st.warning(f"**UNCERTAIN ⚠️** (Confidence: {confidence:.2%})")
            elif pred == 1:
                st.success(f"**REAL 🟢** (Confidence: {confidence:.2%})")
            else:
                st.error(f"**FAKE 🔴** (Confidence: {confidence:.2%})")

            st.session_state['last_text'] = user_input
            st.session_state['last_pred'] = "REAL" if pred == 1 else "FAKE"

if 'last_text' in st.session_state:
    st.markdown("---")
    st.subheader("Deep Reasoning (GenAI)")
    
    if st.button("Run Deep Analysis with Gemma"):
        with st.spinner('Gemma is reading the text...'):
            try:
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                llm = genai.GenerativeModel('gemma-4-26b-a4b-it')
                
                prompt = f"""
                You are an expert Data Scientist and Fact Checker. 
                The following text was flagged by our classical ML model as {st.session_state['last_pred']}.
                
                Read the text and provide a short, 3-bullet-point analysis explaining WHY it might be classified this way. 
                Look for logical fallacies, emotional manipulation, lack of sources, or highly objective journalistic formatting.
                
                Text to analyze:
                {st.session_state['last_text'][:2000]}
                """
                
                response = llm.generate_content(prompt)
                st.write(response.text)
                
            except Exception as e:
                st.error(f"API Error: {e}")