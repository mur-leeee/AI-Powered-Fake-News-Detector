import pandas as pd
import re
import nltk
import pickle
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


# ─────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()

    # Remove Reuters byline — ISOT real news is heavily tagged with it,
    # causing the model to learn "Reuters = real" instead of actual content
    text = re.sub(r'\(reuters\)[\s\-]*', '', text)
    text = re.sub(r'reuters', '', text)
    text = re.sub(r'by [a-z]+ [a-z]+\s', '', text)  # "by john smith"

    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)


# ─────────────────────────────────────────────
# Load datasets — ISOT only
#
# Why not fake_news_dataset.csv?
#   It's dominated by health and financial misinformation, which causes the
#   model to flag ALL health/business news as fake (e.g. "FDA approved",
#   "quarterly revenue", "clinical trials" all score as fake).
#
# Why not train.csv?
#   It's AG News — topic classification (Sports/Tech/Business/World),
#   not fake vs real. No usable labels for this task.
# ─────────────────────────────────────────────
df_true = pd.read_csv("data/True.csv")[["title", "text"]]
df_true["label"] = 1

df_fake = pd.read_csv("data/Fake.csv")[["title", "text"]]
df_fake["label"] = 0

print(f"Real samples : {len(df_true)}")
print(f"Fake samples : {len(df_fake)}")

# ─────────────────────────────────────────────
# Balance & combine
# ─────────────────────────────────────────────
min_len = min(len(df_true), len(df_fake))
df_real = df_true.sample(n=min_len, random_state=42)
df_fake = df_fake.sample(n=min_len, random_state=42)

df = pd.concat([df_real, df_fake], ignore_index=True)
print(f"Balanced dataset size: {len(df)} ({min_len} real + {min_len} fake)")

# ─────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[["content", "label"]].dropna()
df = df[df["content"].str.strip() != ""]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Cleaning text...")
df["content"] = df["content"].apply(clean_text)
df = df[df["content"].str.strip() != ""]

# ─────────────────────────────────────────────
# Vectorize & train
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X = vectorizer.fit_transform(df["content"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved to models/")