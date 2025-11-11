import wikipedia
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# STEP 1 — Data Preparation
# ---------------------------
data = {
    'text': [
        "The earth revolves around the sun",
        "The moon is made of cheese",
        "Water boils at 100 degrees Celsius",
        "Humans can breathe underwater without oxygen tanks",
        "The Eiffel Tower is in Paris",
        "Mango is a fruit",
        "The Great Wall of China is visible from space"
    ],
    'label': ['real', 'fake', 'real', 'fake', 'real', 'real', 'fake']
}
df = pd.DataFrame(data)

# ---------------------------
#  STEP 2 — Model Training
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------
#  STEP 3 — Wikipedia Verification + Semantic Similarity
# ---------------------------
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def check_with_wikipedia(statement):
    try:
        search_results = wikipedia.search(statement)
        if not search_results:
            return None, 0.0, None

        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=2)
        return page_title, summary, wikipedia.page(page_title).url
    except Exception:
        return None, None, None

def verify_misinformation(statement):
    # Step 1: Predict with ML model
    input_vec = vectorizer.transform([statement])
    model_pred = model.predict(input_vec)[0]
    model_prob = model.predict_proba(input_vec)[0]
    confidence = max(model_prob)

    # Step 2: Wikipedia check
    title, summary, link = check_with_wikipedia(statement)
    if summary:
        emb1 = semantic_model.encode(clean_text(statement), convert_to_tensor=True)
        emb2 = semantic_model.encode(clean_text(summary), convert_to_tensor=True)
        similarity = float(util.cos_sim(emb1, emb2).item())
    else:
        similarity = 0.0

    # Step 3: Combined Score
    final_score = 0.6 * confidence + 0.4 * similarity

    label = "Misinformation" if final_score < 0.5 else "Likely True"

    return {
        "input": statement,
        "model_label": model_pred,
        "model_confidence": round(confidence * 100, 2),
        "wikipedia_article": title,
        "similarity_score": round(similarity, 2),
        "final_label": label,
        "final_score": round(final_score * 100, 2),
        "wiki_summary": summary,
        "wiki_link": link
    }

# ---------------------------
#  STEP 4 — Test the Model
# ---------------------------
user_input = "Breaking news Mango is a fruit"
result = verify_misinformation(user_input)

print("\n Analysis Result:")
for k, v in result.items():
    print(f"{k}: {v}")
