from flask import Flask, render_template, request
import json
import os
import pickle
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# ---------------- NLTK SETUP ----------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words("english"))


# ---------------- PREPROCESS FUNCTION ----------------
def preprocess(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stpwrds]
    return " ".join(words)


# ---------------- TRAIN MODEL ----------------
def train_model():
    print("Training model...")

    true_df = pd.read_csv("datasets/True.csv")
    fake_df = pd.read_csv("datasets/Fake.csv")

    true_df["label"] = 0
    fake_df["label"] = 1

    data = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine title + text while handling missing values.
    data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")
    data["content"] = data["content"].apply(preprocess)
    y = data["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data["content"],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vector = TfidfVectorizer(max_features=5000)
    X_train = vector.fit_transform(X_train_text)
    X_test = vector.transform(X_test_text)

    model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_fake": round(precision_score(y_test, y_pred, pos_label=1), 4),
        "recall_fake": round(recall_score(y_test, y_pred, pos_label=1), 4),
        "f1_fake": round(f1_score(y_test, y_pred, pos_label=1), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("vector.pkl", "wb") as vector_file:
        pickle.dump(vector, vector_file)
    with open("metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print("Model trained and saved.")
    print("Evaluation metrics:", metrics)

    return model, vector, metrics


# ---------------- LOAD OR TRAIN ----------------
if os.path.exists("model.pkl") and os.path.exists("vector.pkl"):
    print("Loading existing model...")
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vector.pkl", "rb") as vector_file:
        vector = pickle.load(vector_file)

    if os.path.exists("metrics.json"):
        with open("metrics.json", "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    else:
        metrics = None
else:
    model, vector, metrics = train_model()


# ---------------- PREDICTION FUNCTION ----------------
def fake_news_det(news):
    processed = preprocess(news)
    vect = vector.transform([processed])
    pred = model.predict(vect)
    return pred


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        message = request.form.get("news")

        if not message:
            return render_template("prediction.html", prediction_text="No input provided")

        pred = fake_news_det(message)

        if pred[0] == 1:
            result = "Prediction: Fake News"
        else:
            result = "Prediction: Real News"

        return render_template("prediction.html", prediction_text=result)

    return render_template("prediction.html")


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
