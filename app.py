import os
import re
import nltk
import docx2txt
import PyPDF2
import streamlit as st
import tempfile
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords once
nltk.download('stopwords')

# ---------- STREAMLIT APP STYLING ----------
st.set_page_config(page_title="AI Resume Scanner", page_icon="🤖")

st.markdown(
    """
    <style>
    .stApp {
        background: url("https://imgs.search.brave.com/zqDcwD6JFkFqIjqFFN2_R43HPlNfYTDK-AQJ5yjOXB0/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/ZnJlZS12ZWN0b3Iv/ZGFyay1sb3ctcG9s/eS1iYWNrZ3JvdW5k/XzEwNDgtNzk3MS5q/cGc_c2VtdD1haXNf/aHlicmlkJnc9NzQw/JnE9ODA");
        background-size: cover;
        background-position: center;
        color: white;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #ffffff;
        color: black;
        text-color: black;
    }
    .css-1d391kg {color: black;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- TEXT EXTRACTION ----------
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    text = docx2txt.process(tmp_path)
    os.remove(tmp_path)
    return text

# ---------- TEXT PREPROCESSING ----------
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# ---------- KEYWORD SCORE ----------
def keyword_score(resume_text, jd_keywords):
    resume_text_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if kw.lower() in resume_text_lower]
    score = len(matched) / len(jd_keywords) if jd_keywords else 0
    return score, matched

# ---------- SIMILARITY CALCULATION ----------
def calculate_similarity(resume_texts, jd_text, jd_keywords):
    documents = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    final_scores = []
    matched_keywords_list = []

    # Determine weight based on JD length
    jd_len = len(jd_keywords)
    if jd_len <= 2:
        w_tfidf, w_keyword = 0, 1       # Short JD: only keywords
    elif jd_len <= 10:
        w_tfidf, w_keyword = 0.3, 0.7   # Medium JD: keyword-heavy
    else:
        w_tfidf, w_keyword = 0.7, 0.3   # Long JD: TF-IDF-heavy

    for i, text in enumerate(resume_texts):
        k_score, matched_keywords = keyword_score(text, jd_keywords)
        combined_score = w_tfidf * tfidf_scores[i] + w_keyword * k_score
        final_scores.append(combined_score)
        matched_keywords_list.append(matched_keywords)

    return final_scores, matched_keywords_list

# ---------- STREAMLIT APP ----------
st.title("🧠 AI-Powered Resume Screening System")
st.write("Upload resumes and paste a job description to find the best match!")

jd_text = st.text_area("📋 Paste Job Description")

uploaded_files = st.file_uploader("📂 Upload Resumes (PDF or DOCX)", accept_multiple_files=True)

if st.button("Analyze"):
    if not jd_text or not uploaded_files:
        st.warning("Please upload resumes and enter a job description.")
    else:
        # Extract keywords from the manual job description
        jd_words = re.sub(r'[^a-zA-Z]', ' ', jd_text).lower().split()
        jd_keywords = [w for w in jd_words if w not in stopwords.words('english')]

        resume_texts, file_names = [], []

        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                text = extract_text_from_pdf(file)
            elif file.name.endswith(".docx"):
                text = extract_text_from_docx(file)
            else:
                continue
            resume_texts.append(preprocess_text(text))
            file_names.append(file.name)

        scores, matched_keywords_list = calculate_similarity(resume_texts, preprocess_text(jd_text), jd_keywords)
        ranked = sorted(zip(file_names, scores, matched_keywords_list), key=lambda x: x[1], reverse=True)

        st.subheader("🏆 Resume Rankings:")
        for name, score, matched in ranked:
            st.write(f"📄 {name}: {score*100:.2f}% match")
            st.write(f"✅ Matched Keywords: {', '.join(matched) if matched else 'None'}")

        
