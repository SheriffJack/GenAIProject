"""
Misinformation Detection & Risk Scoring (with Wikipedia Fact Checking)
"""

import pandas as pd
import numpy as np
import re
import pickle
import wikipedia
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class MisinformationDetector:
    """ML pipeline for misinformation detection, scoring, and Wikipedia-based verification"""

    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.feature_names = []
    
    # -----------------------------
    # 1️⃣ Data Loading
    # -----------------------------
    def load_data(self, true_path='True.csv', fake_path='Fake.csv'):
        """Load real and fake news datasets"""
        true_df = pd.read_csv(true_path, engine='python', on_bad_lines='skip')
        fake_df = pd.read_csv(fake_path, engine='python', on_bad_lines='skip')
        
        true_df['label'] = 1
        fake_df['label'] = 0
        
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df = df.drop('subject', axis=1, errors='ignore')
        
        print(f"✅ Loaded {len(df)} total articles.")
        self.df = df
        return df
    
    # -----------------------------
    # 2️⃣ Feature Engineering
    # -----------------------------
    def extract_features(self, df):
        """Create engineered and text-based features"""
        df['title_length'] = df['title'].str.len()
        df['title_words'] = df['title'].str.split().str.len()
        df['text_length'] = df['text'].str.len()
        df['text_words'] = df['text'].str.split().str.len()
        df['exclamation_count'] = df['text'].apply(lambda x: str(x).count('!'))
        df['question_count'] = df['text'].apply(lambda x: str(x).count('?'))
        df['quote_count'] = df['text'].apply(lambda x: str(x).count('"'))
        df['url_count'] = df['text'].apply(lambda x: len(re.findall(r'http[s]?://\S+', str(x))))
        df['number_count'] = df['text'].apply(lambda x: len(re.findall(r'\d+', str(x))))
        df['caps_ratio'] = df['text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
        
        emotional_words = ['shocking', 'amazing', 'unbelievable', 'secret', 'exposed',
                           'truth', 'alert', 'warning', 'urgent', 'breaking']
        df['emotional_words'] = df['text'].apply(lambda x: sum(str(x).lower().count(word) for word in emotional_words))
        
        df['combined_text'] = df['title'] + ' ' + df['text']
        
        self.feature_names = [
            'title_length', 'title_words', 'text_length', 'text_words',
            'exclamation_count', 'question_count', 'quote_count',
            'url_count', 'number_count', 'caps_ratio', 'emotional_words'
        ]
        return df
    
    # -----------------------------
    # 3️⃣ Data Preparation
    # -----------------------------
    def prepare_data(self):
        """Combine engineered + TF-IDF features"""
        self.df = self.extract_features(self.df)
        
        X_engineered = self.df[self.feature_names].fillna(0)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_tfidf = self.vectorizer.fit_transform(self.df['combined_text'])
        
        X_combined = np.hstack([X_engineered.values, X_tfidf.toarray()])
        y = self.df['label'].values
        
        return train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)
    
    # -----------------------------
    # 4️⃣ Model Training
    # -----------------------------
    def train_models(self):
        """Train and compare ML models"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        results = {}
        for name, model in models.items():
            print(f"\n🧠 Training {name}...")
            X_train_mod = np.abs(X_train) if name == 'Naive Bayes' else X_train
            X_test_mod = np.abs(X_test) if name == 'Naive Bayes' else X_test
            
            model.fit(X_train_mod, y_train)
            y_pred = model.predict(X_test_mod)
            y_proba = model.predict_proba(X_test_mod)[:, 1]
            
            acc = np.mean(y_pred == y_test)
            auc = roc_auc_score(y_test, y_proba)
            
            results[name] = {'model': model, 'accuracy': acc, 'auc': auc}
            print(f"✅ {name} → Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        best_model_name = max(results, key=lambda x: results[x]['auc'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\n🏆 Best Model: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
    
    # -----------------------------
    # 5️⃣ Model Evaluation
    # -----------------------------
    def evaluate_model(self):
        """Evaluate the best-performing model"""
        best_model = self.best_model
        y_pred = best_model.predict(self.X_test)
        
        print(f"\n📊 Classification Report - {self.best_model_name}:")
        print(classification_report(self.y_test, y_pred, target_names=['Fake', 'True']))
        
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {self.best_model_name}")
        plt.show()
    
    # -----------------------------
    # 6️⃣ Wikipedia Verification
    # -----------------------------
    def wiki_fact_check(self, claim_text):
        """Use Wikipedia to fact-check the claim"""
        try:
            search_results = wikipedia.search(claim_text)
            if not search_results:
                return {"verdict": "Not Found", "similarity": 0.0, "summary": ""}
            
            page_title = search_results[0]
            summary = wikipedia.summary(page_title, sentences=3)
            
            # Compare semantic similarity
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform([claim_text, summary])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            if similarity > 0.6:
                verdict = "Likely True"
            elif similarity > 0.3:
                verdict = "Partially Supported"
            else:
                verdict = "Likely False"
            
            return {
                "verdict": verdict,
                "similarity": similarity,
                "summary": summary,
                "source": page_title
            }
        
        except Exception as e:
            print("⚠️ Wikipedia check failed:", e)
            return {"verdict": "Error", "similarity": 0.0, "summary": ""}
    
    # -----------------------------
    # 7️⃣ Final Risk Scoring
    # -----------------------------
    def calculate_risk_score(self, text, title=''):
        """Predict misinformation risk + verify with Wikipedia"""
        features = {
            'title_length': len(title),
            'title_words': len(title.split()),
            'text_length': len(text),
            'text_words': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'quote_count': text.count('"'),
            'url_count': len(re.findall(r'http[s]?://\S+', text)),
            'number_count': len(re.findall(r'\d+', text)),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'emotional_words': sum(text.lower().count(w) for w in 
                                   ['shocking', 'amazing', 'unbelievable', 'secret', 'breaking'])
        }
        
        X_eng = np.array([[features[f] for f in self.feature_names]])
        combined_text = title + ' ' + text
        X_tfidf = self.vectorizer.transform([combined_text])
        X_combined = np.hstack([X_eng, X_tfidf.toarray()])
        
        if self.best_model_name == 'Naive Bayes':
            X_combined = np.abs(X_combined)
        
        ml_risk = self.best_model.predict_proba(X_combined)[0][0] * 100
        
        # 🔎 Wikipedia Verification
        wiki_data = self.wiki_fact_check(title or text[:80])
        similarity = wiki_data["similarity"]
        
        # Adjust final score
        if wiki_data["verdict"] == "Likely True":
            final_risk = ml_risk * (1 - similarity * 0.5)
        elif wiki_data["verdict"] == "Likely False":
            final_risk = ml_risk + (1 - similarity) * 30
        else:
            final_risk = ml_risk
        
        final_risk = min(max(final_risk, 0), 100)
        
        print(f"\n🧩 ML Risk Score: {ml_risk:.2f}%")
        print(f"📚 Wikipedia Verdict: {wiki_data['verdict']} (Similarity: {similarity:.2f})")
        print(f"📊 Final Risk Score: {final_risk:.2f}% ({self.best_model_name})")
        if wiki_data["summary"]:
            print(f"🔗 Source: {wiki_data['source']}")
            print(f"📝 Wiki Summary: {wiki_data['summary'][:300]}...")
        
        return {
            "ml_risk": ml_risk,
            "wiki_verdict": wiki_data["verdict"],
            "similarity": similarity,
            "final_risk": final_risk,
            "wiki_summary": wiki_data["summary"],
            "source": wiki_data["source"]
        }


# -----------------------------
# 🔧 Main Execution
# -----------------------------
if __name__ == "__main__":
    detector = MisinformationDetector()
    detector.load_data()
    detector.train_models()
    detector.evaluate_model()
    with open("model.pkl", "wb") as f:
        pickle.dump(detector.best_model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(detector.vectorizer, f)

    sample = "BREAKING: Scientists reveal shocking truth about aliens discovered on Mars!"
    detector.calculate_risk_score(sample, "Shocking Discovery on Mars!")
