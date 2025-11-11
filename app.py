# app.py
import streamlit as st
import pickle
import re
import wikipedia
import pandas as pd
import numpy as np
from typing import Optional, Dict

# -----------------------------
# 🔧 Helper Functions
# -----------------------------

def preprocess_text(text: str) -> str:
    """Basic cleaning: remove links, punctuation, etc."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.strip()

def extract_features(text: str) -> pd.DataFrame:
    """Generate engineered + combined features to match training format."""
    df = pd.DataFrame()
    df["title_length"] = [len(text)]
    df["title_words"] = [len(text.split())]
    df["text_length"] = [len(text)]
    df["text_words"] = [len(text.split())]
    df["exclamation_count"] = [text.count("!")]
    df["question_count"] = [text.count("?")]
    df["quote_count"] = [text.count('"')]
    df["url_count"] = [len(re.findall(r"http[s]?://\S+", text))]
    df["number_count"] = [len(re.findall(r"\d+", text))]
    df["caps_ratio"] = [sum(1 for c in text if c.isupper()) / max(len(text), 1)]

    emotional_words = [
        "shocking", "amazing", "unbelievable", "secret", "exposed",
        "truth", "alert", "warning", "urgent", "breaking"
    ]
    df["emotional_words"] = [sum(text.lower().count(w) for w in emotional_words)]

    df["combined_text"] = [text]
    return df

def extract_keywords(text: str, top_n: int = 5):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]

# -----------------------------
# 🌐 Wikipedia Verification
# -----------------------------

def query_wikipedia_summary(text: str, max_chars: int = 600) -> Optional[Dict]:
    """Search Wikipedia for the given text and return best match."""
    try:
        results = wikipedia.search(text, results=1)
        if not results:
            return None
        title = results[0]
        summary = wikipedia.summary(title, sentences=3)
        return {"title": title, "summary": summary[:max_chars]}
    except Exception:
        return None

def compute_overlap_similarity(a: str, b: str) -> float:
    """Compute word overlap ratio."""
    words_a = set(re.findall(r"[a-zA-Z]+", a.lower()))
    words_b = set(re.findall(r"[a-zA-Z]+", b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a.intersection(words_b)) / len(words_a.union(words_b))

# -----------------------------
# 🧠 Prediction Logic
# -----------------------------

def model_predict(df: pd.DataFrame, vectorizer, model):
    """Run trained ML model and return predictions."""
    X_eng = df.drop(columns=["combined_text"]).fillna(0).values
    X_tfidf = vectorizer.transform(df["combined_text"])
    X = np.hstack([X_eng, X_tfidf.toarray()])
    proba = model.predict_proba(X)[0]
    label_idx = np.argmax(proba)
    label = "True" if label_idx == 1 else "Misinformation"
    conf = round(float(proba[label_idx]), 2)
    return label, conf

def rule_based_predict(text: str):
    suspicious = ["breaking", "shocking", "alert", "conspiracy", "leak"]
    score = sum(w in text.lower() for w in suspicious) / len(suspicious)
    label = "Misinformation" if score > 0.3 else "True"
    return label, round(0.5 + score / 2, 2)

def detect_misinformation(text: str, vectorizer=None, model=None):
    text = preprocess_text(text)
    if not text:
        return {"error": "Empty input"}

    # Step 1: Generate features
    df = extract_features(text)

    # Step 2: Prediction
    if vectorizer is not None and model is not None:
        label, conf = model_predict(df, vectorizer, model)
    else:
        label, conf = rule_based_predict(text)

    # Step 3: Wikipedia verification
    wiki_data = query_wikipedia_summary(text)
    wiki_similarity = 0.0
    if wiki_data:
        wiki_similarity = compute_overlap_similarity(text, wiki_data["summary"])
        if wiki_similarity >= 0.6 and label == "Misinformation":
            label = "True"
            conf = min(0.9, conf + 0.1)
        elif wiki_similarity <= 0.2 and label == "True":
            label = "Uncertain"
            conf = min(conf, 0.6)

    return {
        "label": label,
        "confidence": conf,
        "wiki": wiki_data,
        "wiki_similarity": wiki_similarity,
        "keywords": extract_keywords(text)
    }

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI Misinformation Detector", page_icon="🧠", layout="centered")
st.title("🧠 AI-Powered Misinformation Detector")
st.markdown("Detect fake or misleading information using AI + Wikipedia verification.")

# Load saved model + vectorizer
model, vectorizer = None, None
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    st.success("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    st.warning(f"⚠️ Could not load model/vectorizer: {e}")

# -----------------------------
# 💬 User Input
# -----------------------------
text_input = st.text_area(
    "Enter a statement or post:",
    height=150,
    placeholder="e.g. Joe Biden has been sniped in a recent gathering"
)

if st.button("Analyze"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            result = detect_misinformation(text_input, vectorizer, model)

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("🧩 Analysis Result")
            st.metric("Label", result["label"], f"{result['confidence']*100:.1f}% confidence")

            wiki = result.get("wiki")
            sim = result.get("wiki_similarity", 0)
            if wiki:
                st.markdown("#### 🌐 Wikipedia Cross-check:")
                st.write(f"**Matched Article:** {wiki['title']}")
                st.caption(f"Similarity score: {sim:.2f}")
                st.info(wiki['summary'])
                st.markdown(f"[🔗 View on Wikipedia](https://en.wikipedia.org/wiki/{wiki['title'].replace(' ', '_')})")
            else:
                st.warning("No relevant Wikipedia page found for this claim.")

            st.markdown("#### 🔑 Highlighted Keywords:")
            st.write(", ".join(result["keywords"]))
    else:
        st.warning("Please enter a statement to analyze.")
