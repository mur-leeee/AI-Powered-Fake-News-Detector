import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)


model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prob = model.predict_proba(vector)[0]
    pred = model.predict(vector)[0]

    fake_prob = prob[0]
    real_prob = prob[1]
    confidence = max(prob)

    if confidence < 0.60:
        label = "UNCERTAIN ⚠️"
    elif pred == 1:
        label = "REAL 🟢"
    else:
        label = "FAKE 🔴"

    return label, confidence, real_prob, fake_prob


print("=== Fake News Detector ===")
print("Type a news headline or article text. Type 'exit' to quit.\n")

while True:
    news = input("Enter news: ").strip()

    if not news:
        continue

    if news.lower() == "exit":
        print("Goodbye!")
        break

    label, confidence, real_prob, fake_prob = predict_news(news)

    print(f"\nPrediction : {label}")
    print(f"Confidence : {confidence:.2%}")
    print(f"  Real probability : {real_prob:.2%}")
    print(f"  Fake probability : {fake_prob:.2%}")
    print()