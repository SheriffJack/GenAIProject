import pandas as pd
import re
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def extract_features(df):
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)

    df["title_length"] = df["title"].apply(len)
    df["title_words"] = df["title"].apply(lambda x: len(x.split()))
    df["text_length"] = df["text"].apply(len)
    df["text_words"] = df["text"].apply(lambda x: len(x.split()))

    df["exclamation_count"] = df["text"].apply(lambda x: x.count("!"))
    df["question_count"] = df["text"].apply(lambda x: x.count("?"))
    df["quote_count"] = df["text"].apply(lambda x: x.count('"'))

    df["url_count"] = df["text"].apply(lambda x: len(re.findall(r"http[s]?://\S+", x)))
    df["number_count"] = df["text"].apply(lambda x: len(re.findall(r"\d+", x)))
    df["caps_ratio"] = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

    emotional_words = ["shocking", "amazing", "unbelievable", "secret", "exposed", "truth", "alert", "warning", "urgent", "breaking"]
    df["emotional_words"] = df["text"].apply(lambda x: sum(w in x.lower() for w in emotional_words))

    df["combined_text"] = df["clean_title"] + " " + df["clean_text"]
    return df
